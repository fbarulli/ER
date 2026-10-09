"""Checkpointer head of the fine-tune PERF_PATCH training-loop text.

The training-loop `@PERF_PATCH@` text is one literal split at ``class
AdversarialPerturber`` so no module crosses the 1k-line limit (mirrors the
``cli.laya_hpo_kernel_text`` head/tail split); ``FINETUNE_PERF_PATCH_SOURCE``
concatenates it with the remaining loop text byte-for-byte.
"""
from __future__ import annotations


CHECKPOINTER_TEMPLATE = '''\
import traceback


class ControlCheckpointer:
    """Per-epoch checkpoint save + resume with optimizer/scheduler state.

    The checkpoint dir is the CANONICAL, shared path on EVERY rank (rank 0
    writes, all ranks resume), deliberately separate from the per-rank final
    save dir. A ``run_tag`` is stored with each file so a resume can never
    continue from another run's checkpoint, and the best-metric weights are
    persisted separately so a resumed run that never improves ends on the
    best weights, not the last epoch.

    Only the newest ``KEEP_EPOCH_CHECKPOINTS`` epoch files are retained: a full
    resumable checkpoint carries the optimizer/scheduler state too, so keeping
    every epoch grows without bound. On a size-capped ``/kaggle/working`` with
    parallel HPO trials this filled the disk and killed the session, so the old
    epoch file is dropped BEFORE the new one is written (peak stays at one).
    """

    #: Resumable epoch checkpoints to keep on disk (``resume`` loads the newest).
    KEEP_EPOCH_CHECKPOINTS = 1

    def __init__(self, torch_module, checkpoint_dir, run_tag):
        self._torch = torch_module
        self._checkpoint_dir = checkpoint_dir
        self._run_tag = run_tag

    @classmethod
    def for_training(cls, torch_module):
        # FINETUNE_CHECKPOINT_DIR is the shared canonical dir (all ranks);
        # FINETUNE_OUTPUT_DIR is the per-rank fallback for single-process.
        directory = (globals().get("FINETUNE_CHECKPOINT_DIR")
                     or globals().get("FINETUNE_OUTPUT_DIR"))
        return cls(torch_module, directory, globals().get("FINETUNE_RUN_TAG"))

    @staticmethod
    def unwrap(model):
        return model.module if hasattr(model, "module") else model

    def directory(self):
        if not self._checkpoint_dir:
            return None
        return os.path.join(str(self._checkpoint_dir), "checkpoints")

    def _matches_run(self, state):
        return self._run_tag is None or state.get("run_tag") == self._run_tag

    @staticmethod
    def _prune_epochs(path, keep):
        """Delete epoch checkpoints beyond the newest ``keep`` (bounded disk)."""
        names = [name for name in os.listdir(path)
                 if name.startswith("epoch_") and name.endswith(".pt")
                 and name[len("epoch_"):-3].isdigit()]
        names.sort(key=lambda name: int(name[len("epoch_"):-3]))
        for name in names[:max(0, len(names) - keep)]:
            try:
                os.remove(os.path.join(path, name))
            except OSError:
                pass

    def save(self, model, optimizer, scheduler, epoch, best, bad_epochs):
        if not is_rank0():
            return
        path = self.directory()
        if not path:
            return
        os.makedirs(path, exist_ok=True)
        # Drop older epoch files FIRST so the workspace never holds more than
        # KEEP_EPOCH_CHECKPOINTS resumable files (old + new would double the
        # peak and re-hit the cap).
        self._prune_epochs(path, self.KEEP_EPOCH_CHECKPOINTS - 1)
        self._torch.save({
            "epoch": int(epoch),
            "model": self.unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best": best,
            "bad_epochs": int(bad_epochs),
            "run_tag": self._run_tag,
        }, os.path.join(path, "epoch_%d.pt" % int(epoch)))

    def save_best(self, model, metric, accuracy, epoch):
        """Persist the best early-stop-metric weights (rank 0 only)."""
        if not is_rank0():
            return
        path = self.directory()
        if not path:
            return
        os.makedirs(path, exist_ok=True)
        self._torch.save({
            "model": self.unwrap(model).state_dict(),
            "metric": metric, "accuracy": accuracy, "epoch": int(epoch),
            "run_tag": self._run_tag,
        }, os.path.join(path, "best.pt"))

    def _load_best(self, path, device):
        best_path = os.path.join(path, "best.pt")
        if not os.path.isfile(best_path):
            return None
        try:
            state = self._torch.load(best_path, map_location=device,
                                     weights_only=False)
        except Exception:
            return None
        return state if self._matches_run(state) else None

    def resume(self, model, optimizer, scheduler, device):
        """Return ``(start_epoch, best, bad_epochs, best_record)``.

        Every rank resumes from the SAME shared file so ranks stay in
        lockstep; a checkpoint from a different ``run_tag`` is ignored. The
        best record carries the saved best weights + metric so a resumed run
        that never improves still ends on the best epoch.
        """
        path = self.directory()
        if not path or not os.path.isdir(path):
            return 0, None, 0, None
        names = [name for name in os.listdir(path)
                 if name.startswith("epoch_") and name.endswith(".pt")
                 and name[len("epoch_"):-3].isdigit()]
        names.sort(key=lambda name: int(name[len("epoch_"):-3]))
        for name in reversed(names):
            try:
                state = self._torch.load(os.path.join(path, name),
                                         map_location=device,
                                         weights_only=False)
            except Exception as error:
                print("[perf-patch] resume skipped " + name + ": "
                      + str(error)[:160], flush=True)
                continue
            if not self._matches_run(state):
                continue
            self.unwrap(model).load_state_dict(state["model"])
            try:
                optimizer.load_state_dict(state["optimizer"])
                scheduler.load_state_dict(state["scheduler"])
            except Exception as error:
                print("[perf-patch] resume state skipped: "
                      + str(error)[:160], flush=True)
            best_record = self._load_best(path, device)
            print("[perf-patch] resumed " + name + " (next epoch "
                  + str(int(state.get("epoch", 0)) + 2) + ")", flush=True)
            return (int(state.get("epoch", 0)) + 1, state.get("best"),
                    int(state.get("bad_epochs", 0)), best_record)
        return 0, None, 0, None


'''
