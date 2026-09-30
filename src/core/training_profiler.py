"""Shared opt-in PyTorch profiling for text and graph trainers."""
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import torch


class TrainingProfiler:
    def __init__(self, output: Path, device: str):
        self.output = output
        self.enabled = os.environ.get('ER_TRAINING_PROFILE') == '1'
        self.started = False
        self.profiler = None
        self.device = device
        self.steps = 0

    def __enter__(self):
        if self.enabled:
            self.output.mkdir(parents=True,exist_ok=True)
            activities=[torch.profiler.ProfilerActivity.CPU]
            if self.device=='cuda':
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.profiler=torch.profiler.profile(activities=activities,record_shapes=True,
                profile_memory=True,with_stack=False,
                schedule=torch.profiler.schedule(wait=0,warmup=0,active=3,repeat=1),
                on_trace_ready=self._write)
            self.profiler.__enter__()
            self.started=True
        return self

    def _write(self, profiler):
        profiler.export_chrome_trace(str(self.output/'training_trace.json'))
        sort_by = 'self_cuda_time_total' if self.device == 'cuda' else 'self_cpu_time_total'
        (self.output/'operator_summary.txt').write_text(profiler.key_averages().table(sort_by=sort_by,row_limit=100))
        (self.output/'profile_manifest.json').write_text(json.dumps({
            'device':self.device,'active_steps':min(self.steps, 3),'maximum_active_steps':3,'record_shapes':True,'profile_memory':True,
            'includes_profiling_overhead':True,'pid':os.getpid()},indent=2)+'\n')

    def step(self):
        if self.started:
            self.steps += 1
            self.profiler.step()

    @contextmanager
    def section(self,label):
        with torch.profiler.record_function(label) if self.enabled else nullcontext():
            yield

    def call(self,label,fn,*args,**kwargs):
        with self.section(label):
            return fn(*args,**kwargs)

    def close(self):
        if self.started:
            self.profiler.__exit__(None,None,None)
            self.started=False

    def __exit__(self,*args):
        self.close()

    def callback(self):
        from transformers import TrainerCallback
        owner=self
        class ProfileCallback(TrainerCallback):
            def on_train_begin(self,args,state,control,**kwargs):
                owner.__enter__()
            def on_step_end(self,args,state,control,**kwargs):
                owner.step()
            def on_train_end(self,args,state,control,**kwargs):
                owner.close()
        return ProfileCallback()
