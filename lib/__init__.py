"""TRAIN — the training internals (folds, loss machinery, masking, rerank,
plots) behind the 05_train.py entry point."""

from TRAIN.folds import component_folds
from TRAIN.training import ES_PATIENCE, ES_THRESHOLD, train_one_config

__all__ = ["ES_PATIENCE", "ES_THRESHOLD", "component_folds", "train_one_config"]
