# Data train flow

The current data-generation flow, one-command CSV rebuild, offline batching,
hybrid embedding lifecycle, file inventory, and launch instructions are
consolidated in [TRAINING_INSTRUCTIONS.md](TRAINING_INSTRUCTIONS.md).

The executable entry point is
[src/training/prepare_all.py](src/training/prepare_all.py):

```bash
PYTHONPATH=src .venv/bin/python -m training.prepare_all
```

Model definitions and comparison objectives remain in
[MODEL_TRACKS_PLAN.md](MODEL_TRACKS_PLAN.md).
