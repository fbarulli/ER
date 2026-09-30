"""Local/Colab-ready experimental worker with existing MLflow/W&B contexts.

Run: python -m graph_tracks.train --config config/graph_tracks_gnn.yaml
This worker is independent of the existing Colab launcher's dispatch.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve

from graph_tracks.config import load_config
from graph_tracks.data import census, file_hash, fit_vocabulary, load_records, load_text_cache, tensorize
from graph_tracks.model import AttributeGNN, PairScorer


def load_pairs(path: Path, records: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if set(frame.columns) != {"product_id1", "product_id2", "label", "split"}:
        raise ValueError("pairs columns must be product_id1, product_id2, label, split")
    ids = {r["product_id"]: i for i, r in enumerate(records)}
    seen = set()
    rows = {s: ([], []) for s in ("train", "dev", "test")}
    for row in frame.itertuples(index=False):
        if row.label not in {"0", "1"} or row.split not in rows:
            raise ValueError("invalid pair label/split")
        if row.product_id1 not in ids or row.product_id2 not in ids:
            raise ValueError("pair endpoint absent from listings")
        i, j = ids[row.product_id1], ids[row.product_id2]
        if i == j:
            raise ValueError("self-pairs are not a matching benchmark")
        if records[i]["split"] != row.split or records[j]["split"] != row.split:
            raise ValueError("pair crosses split boundary or contains a trained-on endpoint")
        key = tuple(sorted((i, j)))
        if key in seen:
            raise ValueError("duplicate or conflicting pair")
        seen.add(key)
        rows[row.split][0].append((i, j))
        rows[row.split][1].append(float(row.label))
    result = {s: (np.asarray(p, dtype=np.int64).reshape(-1, 2),
                  np.asarray(y, dtype=np.float32)) for s, (p, y) in rows.items()}
    for split in ("train", "dev"):
        if set(result[split][1]) != {0., 1.}:
            raise ValueError(f"{split} needs positive and negative labeled pairs")
    return result


def quality(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    precision, recall, _ = precision_recall_curve(labels, scores)
    return {"dev_pr_auc": float(average_precision_score(labels, scores)),
            "dev_p_at_r95": float(precision[recall >= .95].max()),
            "dev_positive_pairs": int(labels.sum()), "dev_negative_pairs": int((labels == 0).sum())}


def write_json(path: Path, values: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(values, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def train(config_path: Path, *, run_tag: str, resume: Path | None = None) -> Path:
    cfg = load_config(config_path)
    # Use the repository's path/config conventions; existing code is read only.
    from core.common import TRAIN_ROOT
    resolve = lambda raw: (TRAIN_ROOT / raw).resolve()
    output = Path(os.environ.get("EUROMONITOR_RESULTS_DIR", resolve(cfg.output_dir))).resolve()
    if not run_tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_tag):
        raise ValueError("run tag must contain only letters, digits, underscores, hyphens")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_manifest.json").exists() and resume is None:
        raise FileExistsError("output already contains a run; choose a new output directory")
    os.environ.setdefault("EUROMONITOR_MLRUNS_DIR", str(output / "mlruns"))
    os.environ.setdefault("WANDB_DIR", str(output / "wandb"))
    (output / "wandb").mkdir(exist_ok=True)
    from core.mlflow_ctx import MlflowCtx
    from core.wandb_ctx import WandbCtx
    from core.worker_telemetry import write_worker_live_status

    logger = logging.getLogger(f"graph_tracks.{run_tag}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.StreamHandler(), logging.FileHandler(output / "training.log")]
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
    try:
        if cfg.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("cuda configured but unavailable; no silent CPU fallback")
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        torch.use_deterministic_algorithms(True)
        records = load_records(resolve(cfg.listings))
        pairs = load_pairs(resolve(cfg.pairs), records)
        vocabulary = fit_vocabulary(records)
        support_indices = [i for i, r in enumerate(records) if r["split"] == "train"]
        support_records = [records[i] for i in support_indices]
        batch = tensorize(records, vocabulary, cfg.device)
        support = tensorize(support_records, vocabulary, cfg.device)
        text, text_metadata = None, None
        if cfg.text_cache:
            vectors, text_metadata = load_text_cache(resolve(cfg.text_cache), [r["product_id"] for r in records])
            text = torch.tensor(vectors, device=cfg.device)
        model = AttributeGNN(vocabulary, cfg.hidden_dim, cfg.output_dim,
                             0 if text is None else text.shape[1], cfg.graph_enabled).to(cfg.device)
        scorer = PairScorer(text is not None).to(cfg.device)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(scorer.parameters()),
                                      lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=TRAIN_ROOT,
                                  capture_output=True, text=True, check=True).stdout.strip()
        manifest = {"schema": "er-graph-run-v1", "track": cfg.track, "run_tag": run_tag,
                    "config": cfg.model_dump(), "git_revision": revision,
                    "listings_sha256": file_hash(resolve(cfg.listings)),
                    "pairs_sha256": file_hash(resolve(cfg.pairs)),
                    "text_cache_sha256": file_hash(resolve(cfg.text_cache)) if cfg.text_cache else None,
                    "text_metadata": text_metadata, "graph_context": "training-listings-only",
                    "selection_metric": "dev_pr_auc", "torch_version": str(torch.__version__)}
        best_metric, best_path, start_epoch = -1., None, 0
        if resume:
            marker = resume.parent / "checkpoint_manifest.json"
            if not marker.is_file() or json.loads(marker.read_text())["files"]["graph_model.pt"] != file_hash(resume):
                raise ValueError("resume checkpoint incomplete or hash mismatch")
            restored = torch.load(resume, map_location=cfg.device, weights_only=False)
            prior = restored["manifest"]
            for key in ("track", "listings_sha256", "pairs_sha256", "text_cache_sha256"):
                if prior[key] != manifest[key]:
                    raise ValueError(f"resume mismatch: {key}")
            for key, value in prior["config"].items():
                if key not in {"epochs", "output_dir", "device"} and value != manifest["config"][key]:
                    raise ValueError(f"resume config mismatch: {key}")
            # Resume must retain the prior selected checkpoint for collection.
            best_path = Path(restored["best_path"])
            if not best_path.is_file():
                # A copied checkpoint tree is portable when sibling epochs remain.
                best_path = resume.parent.parent / best_path.parent.name / "graph_model.pt"
            if not best_path.is_file():
                raise FileNotFoundError("resume requires prior best checkpoint in the restored tree")
            best_marker = best_path.parent / "checkpoint_manifest.json"
            if not best_marker.is_file() or json.loads(best_marker.read_text())["files"]["graph_model.pt"] != file_hash(best_path):
                raise ValueError("selected resume checkpoint incomplete or hash mismatch")
            # Collect the selected model inside this run even if no new epoch wins.
            selected_dir = output / "_checkpoints" / cfg.track / f"{run_tag}_f0" / best_path.parent.name
            if selected_dir.resolve() != best_path.parent.resolve():
                shutil.copytree(best_path.parent, selected_dir, dirs_exist_ok=True)
            best_path = selected_dir / "graph_model.pt"
            model.load_state_dict(restored["model"])
            scorer.load_state_dict(restored["scorer"])
            optimizer.load_state_dict(restored["optimizer"])
            torch.set_rng_state(restored["torch_rng"].cpu())
            if cfg.device == "cuda" and restored["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(restored["cuda_rng"])
            random.setstate(restored["python_rng"])
            np.random.set_state(restored["numpy_rng"])
            best_metric, start_epoch = restored["best_metric"], restored["epoch"]
            if cfg.epochs <= start_epoch:
                raise ValueError("resume epochs must exceed completed epochs")
        write_json(output / "run_manifest.json", manifest)
        write_json(output / "graph_census.json", census(records, vocabulary))
        train_pairs = torch.tensor(pairs["train"][0], device=cfg.device)
        train_labels = torch.tensor(pairs["train"][1], device=cfg.device)
        dev_pairs = torch.tensor(pairs["dev"][0], device=cfg.device)
        support_text = None if text is None else text[support_indices]
        trained_endpoints = set(pairs["train"][0].reshape(-1).tolist())
        support_ids = set(support_indices)
        pd.DataFrame([{"product_id": r["product_id"], "split": r["split"],
                       "supervised_endpoint": i in trained_endpoints,
                       "training_graph_support": i in support_ids}
                      for i, r in enumerate(records)]).to_csv(output / "listing_usage.csv", index=False)
        logger.info("[graph-train] track=%s listings=%d train_pairs=%d dev_pairs=%d device=%s",
                    cfg.track, len(records), len(train_pairs), len(dev_pairs), cfg.device)
        with MlflowCtx(run_tag) as mlflow, WandbCtx(run_tag) as wandb:
            mlflow.log_params({**cfg.model_dump(), "git_revision": revision})
            wandb.log_config(manifest)
            for epoch in range(start_epoch + 1, cfg.epochs + 1):
                started = time.monotonic()
                model.train()
                scorer.train()
                optimizer.zero_grad(set_to_none=True)
                states = model.context(support, support_text)
                embeddings = model.encode(batch, states, text)
                logits = scorer(embeddings, train_pairs, text)
                classification = F.binary_cross_entropy_with_logits(logits, train_labels)
                a, b = train_pairs.unbind(1)
                cos = (embeddings[a] * embeddings[b]).sum(-1)
                metric = (train_labels * (1 - cos) + (1 - train_labels)
                          * F.relu(cos - cfg.negative_margin)).mean()
                loss = classification + cfg.metric_weight * metric
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite loss")
                loss.backward()
                gradient_norms = {name: float(parameter.grad.norm()) for name, parameter in model.named_parameters()
                                  if parameter.grad is not None}
                if not all(np.isfinite(value) for value in gradient_norms.values()):
                    raise RuntimeError("nonfinite gradients")
                with (output / "gradient_metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps({"epoch": epoch, "parameter_gradient_norms": gradient_norms}) + "\n")
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(scorer.parameters()), cfg.max_grad_norm)
                optimizer.step()
                model.eval()
                scorer.eval()
                with torch.no_grad():
                    embeddings = model.encode(batch, model.context(support, support_text), text)
                    dev_scores = scorer(embeddings, dev_pairs, text).sigmoid().cpu().numpy()
                metrics = {"epoch": epoch, "train_loss": loss.item(),
                           "train_classification_loss": classification.item(),
                           "train_metric_loss": metric.item(), **quality(pairs["dev"][1], dev_scores),
                           "epoch_seconds": time.monotonic() - started}
                if cfg.device == "cuda":
                    metrics["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
                    metrics["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 1024**3
                improved = metrics["dev_pr_auc"] > best_metric
                checkpoint_dir = output / "_checkpoints" / cfg.track / f"{run_tag}_f0" / f"checkpoint-{epoch}"
                checkpoint_dir.mkdir(parents=True)
                checkpoint = checkpoint_dir / "graph_model.pt"
                if improved:
                    best_metric, best_path = metrics["dev_pr_auc"], checkpoint
                payload = {"schema": "er-graph-checkpoint-v1", "manifest": manifest,
                           "vocabulary": vocabulary, "support_records": support_records,
                           "support_text": None if support_text is None else support_text.detach().cpu(),
                           "text_dim": model.text_dim, "model": model.state_dict(),
                           "scorer": scorer.state_dict(), "optimizer": optimizer.state_dict(),
                           "epoch": epoch, "best_metric": best_metric, "best_path": str(best_path),
                           "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                           "numpy_rng": np.random.get_state(),
                           "cuda_rng": torch.cuda.get_rng_state_all() if cfg.device == "cuda" else None}
                torch.save(payload, checkpoint)
                write_json(checkpoint_dir / "trainer_state.json", {
                    "global_step": epoch, "best_metric": best_metric,
                    "best_model_checkpoint": str(best_path.parent)})
                # Completion marker is written LAST, matching worker conventions.
                write_json(checkpoint_dir / "checkpoint_manifest.json", {
                    "schema": "er-graph-checkpoint-v1", "files": {"graph_model.pt": file_hash(checkpoint)},
                    "epoch": epoch, "track": cfg.track})
                write_json(output / "best_checkpoint.json", {"path": str(best_path), "metric": best_metric})
                with (output / "epoch_metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(metrics, allow_nan=False) + "\n")
                mlflow.log_metrics(metrics)
                wandb.log_metrics(metrics, step=epoch)
                write_worker_live_status(target=output / "live_status.json", event="epoch_complete",
                    step=epoch, max_steps=cfg.epochs, epoch=epoch, wandb_run_id=wandb.run_id,
                    wandb_url=wandb.run_url, best_checkpoint_step=int(best_path.parent.name.split("-")[-1]),
                    best_metric=best_metric, **{k: v for k, v in metrics.items() if k != "epoch"})
                logger.info("[graph-train] epoch=%d loss=%.5f dev_pr_auc=%.5f dev_p_at_r95=%.5f best=%s",
                            epoch, loss.item(), metrics["dev_pr_auc"], metrics["dev_p_at_r95"], improved)
            for path in (output / "run_manifest.json", output / "graph_census.json",
                         output / "epoch_metrics.jsonl", output / "best_checkpoint.json",
                         output / "listing_usage.csv", output / "gradient_metrics.jsonl"):
                mlflow.log_artifact(path)
            mlflow.log_artifact(best_path)
            wandb.set_summary({"best_dev_pr_auc": best_metric, "best_checkpoint": str(best_path)})
            wandb.log_artifacts([best_path.parent, output / "run_manifest.json",
                                 output / "graph_census.json", output / "epoch_metrics.jsonl",
                                 output / "listing_usage.csv", output / "gradient_metrics.jsonl"],
                                f"{run_tag}-graph-results")
        write_json(output / "graph_worker_result.json", {"status": "ok", "best_checkpoint": str(best_path),
                    "best_dev_pr_auc": best_metric, "track": cfg.track})
        write_worker_live_status(target=output / "live_status.json", event="complete", step=cfg.epochs,
                                max_steps=cfg.epochs, epoch=cfg.epochs, best_metric=best_metric)
        logger.info("[graph-train] complete checkpoint=%s", best_path)
        return best_path
    except BaseException as error:
        logger.exception("[graph-train] failed")
        write_json(output / "graph_worker_result.json", {"status": "failed", "error": str(error)})
        raise
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-tag", default=os.environ.get("EUROMONITOR_RUN_ID", "graph-experiment"))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    train(args.config, run_tag=args.run_tag, resume=args.resume)


if __name__ == "__main__":
    main()
