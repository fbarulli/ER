"""Local/Colab-ready experimental worker with W&B tracking and isolated DVC artifacts.

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
from graph_tracks.artifacts import name, checkpoint_track
from graph_tracks.tracking import GraphWandb
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
    output = output / name(cfg.track, run_tag)
    output.mkdir(parents=True, exist_ok=True)
    if (output / name(cfg.track, "run_manifest.json")).exists() and resume is None:
        raise FileExistsError("output already contains a run; choose a new output directory")
    from core.worker_telemetry import write_worker_live_status

    logger = logging.getLogger(f"graph_tracks.{run_tag}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.StreamHandler(), logging.FileHandler(output / name(cfg.track, "training.log"))]
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
        from graph_tracks.preflight import load_inputs
        logger.info("[graph-phase] input_validation start listings=%s pairs=%s manifest=%s text_cache=%s output=%s",
                    resolve(cfg.listings), resolve(cfg.pairs), cfg.input_manifest, cfg.text_cache, output)
        input_manifest, records, pairs, vectors, text_metadata = load_inputs(cfg)
        logger.info("[graph-phase] input_validation complete split_pairs=%s text_shape=%s",
                    {split: {'positive': int(labels.sum()), 'negative': int((labels == 0).sum())}
                     for split, (_, labels) in pairs.items()}, None if vectors is None else vectors.shape)
        logger.info("[graph-phase] features start graph_enabled=%s hidden_dim=%d output_dim=%d",
                    cfg.graph_enabled, cfg.hidden_dim, cfg.output_dim)
        vocabulary = fit_vocabulary(records)
        support_indices = [i for i, r in enumerate(records) if r["split"] == "train"]
        support_records = [records[i] for i in support_indices]
        dev_indices = [i for i, record in enumerate(records) if record['split'] == 'dev']
        dev_records = [records[i] for i in dev_indices]
        support = tensorize(support_records, vocabulary, cfg.device)
        dev_batch = tensorize(dev_records, vocabulary, cfg.device)
        # Query encoding is independent per listing against training-only context.
        # Encode only supervised/evaluated populations, rather than every holdout.
        train_local = {global_id: local_id for local_id, global_id in enumerate(support_indices)}
        dev_local = {global_id: local_id for local_id, global_id in enumerate(dev_indices)}
        train_pair_indices = np.asarray([[train_local[int(a)], train_local[int(b)]]
                                        for a, b in pairs['train'][0]], dtype=np.int64)
        dev_pair_indices = np.asarray([[dev_local[int(a)], dev_local[int(b)]]
                                      for a, b in pairs['dev'][0]], dtype=np.int64)
        text = None if vectors is None else torch.tensor(vectors, device=cfg.device)
        model = AttributeGNN(vocabulary, cfg.hidden_dim, cfg.output_dim,
                             0 if text is None else text.shape[1], cfg.graph_enabled).to(cfg.device)
        scorer = PairScorer(text is not None).to(cfg.device)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(scorer.parameters()),
                                      lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        logger.info("[graph-phase] features complete listings=%d training_support=%d text_dim=%d parameters=%d",
                    len(records), len(support_records), model.text_dim,
                    sum(parameter.numel() for parameter in model.parameters()) +
                    sum(parameter.numel() for parameter in scorer.parameters()))
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=TRAIN_ROOT,
                                  capture_output=True, text=True, check=True).stdout.strip()
        manifest = {"schema": "er-graph-run-v1", "track": cfg.track, "run_tag": run_tag,
                    "config": cfg.model_dump(), "git_revision": revision,
                    "input_manifest": input_manifest,
                    "input_manifest_sha256": file_hash(resolve(cfg.input_manifest)) if cfg.input_manifest else None,
                    "listings_sha256": file_hash(resolve(cfg.listings)),
                    "pairs_sha256": file_hash(resolve(cfg.pairs)),
                    "text_cache_sha256": file_hash(resolve(cfg.text_cache)) if cfg.text_cache else None,
                    "text_metadata": text_metadata, "graph_context": "training-listings-only",
                    "selection_metric": "dev_pr_auc", "torch_version": str(torch.__version__),
                    "implementation_sha256": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                    "resume_checkpoint_sha256": file_hash(resume) if resume else None}
        best_metric, best_path, start_epoch = -1., None, 0
        logger.info("[graph-resume] mode=%s checkpoint=%s target_epochs=%d",
                    'resume' if resume else 'fresh', resume, cfg.epochs)
        if resume:
            current_manifest_path = output / name(cfg.track, "run_manifest.json")
            if current_manifest_path.exists():
                current_manifest = json.loads(current_manifest_path.read_text())
                if current_manifest.get("track") != cfg.track or current_manifest.get("run_tag") != run_tag:
                    raise ValueError("resume output belongs to a different run")
                for key in ("listings_sha256", "pairs_sha256", "text_cache_sha256", "input_manifest_sha256"):
                    if current_manifest.get(key) != manifest[key]:
                        raise ValueError(f"resume output mismatch: {key}")
            if checkpoint_track(resume) != cfg.track:
                raise ValueError("resume track mismatch")
            restored = torch.load(resume, map_location=cfg.device, weights_only=False)
            prior = restored["manifest"]
            for key in ("track", "listings_sha256", "pairs_sha256", "text_cache_sha256", "input_manifest_sha256", "implementation_sha256"):
                if prior[key] != manifest[key]:
                    raise ValueError(f"resume mismatch: {key}")
            for key, value in prior["config"].items():
                if key not in {"epochs", "output_dir", "device", "wandb", "dvc", "postprocess", "report_test", "build_index", "include_inputs", "listings", "pairs", "text_cache", "input_manifest"} and value != manifest["config"][key]:
                    raise ValueError(f"resume config mismatch: {key}")
            # Resume must retain the prior selected checkpoint for collection.
            best_path = Path(restored["best_path"])
            if not best_path.is_file():
                # A copied checkpoint tree is portable when sibling epochs remain.
                best_path = resume.parent.parent / best_path.parent.name / name(cfg.track, "graph_model.pt")
            if not best_path.is_file():
                raise FileNotFoundError("resume requires prior best checkpoint in the restored tree")
            if checkpoint_track(best_path) != cfg.track:
                raise ValueError("selected resume checkpoint track mismatch")
            # Collect the selected model inside this run even if no new epoch wins.
            selected_dir = output / "_checkpoints" / cfg.track / f"{run_tag}_f0" / best_path.parent.name
            if selected_dir.resolve() != best_path.parent.resolve():
                shutil.copytree(best_path.parent, selected_dir, dirs_exist_ok=True)
            best_path = selected_dir / name(cfg.track, "graph_model.pt")
            model.load_state_dict(restored["model"])
            scorer.load_state_dict(restored["scorer"])
            optimizer.load_state_dict(restored["optimizer"])
            torch.set_rng_state(restored["torch_rng"].cpu())
            if cfg.device == "cuda" and restored["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(restored["cuda_rng"])
            random.setstate(restored["python_rng"])
            np.random.set_state(restored["numpy_rng"])
            best_metric, start_epoch = restored["best_metric"], restored["epoch"]
            if cfg.epochs < start_epoch:
                raise ValueError("resume epochs cannot precede completed epochs")
        logger.info("[graph-resume] restored_completed_epochs=%d next_epoch=%d best_dev_pr_auc=%s selected=%s",
                    start_epoch, start_epoch + 1, best_metric if best_path else None, best_path)
        if cfg.include_inputs:
            input_dir = output / name(cfg.track, "inputs")
            input_dir.mkdir(exist_ok=True)
            input_artifacts = {}
            for key, stem in (("listings", "listings.json"), ("pairs", "pairs.csv"),
                              ("text_cache", "text_embeddings.npz"), ("input_manifest", "input_manifest.json")):
                if getattr(cfg, key):
                    target = input_dir / name(cfg.track, stem)
                    source = resolve(getattr(cfg, key))
                    if target.resolve() != source:
                        shutil.copy2(source, target)
                    input_artifacts[key] = str(target.relative_to(output))
            from graph_tracks.report_attributes import FILENAME
            report_source = resolve(cfg.listings).parent / FILENAME
            if report_source.is_file():
                target = input_dir / name(cfg.track, FILENAME)
                shutil.copy2(report_source, target)
                input_artifacts['report_attributes'] = str(target.relative_to(output))
            manifest["input_artifacts"] = input_artifacts
        write_json(output / name(cfg.track, "run_manifest.json"), manifest)
        write_json(output / name(cfg.track, "graph_census.json"), census(records, vocabulary))
        train_pairs = torch.tensor(train_pair_indices, device=cfg.device)
        train_labels = torch.tensor(pairs["train"][1], device=cfg.device)
        dev_pairs = torch.tensor(dev_pair_indices, device=cfg.device)
        support_text = None if text is None else text[support_indices]
        dev_text = None if text is None else text[dev_indices]
        logger.info("[graph-performance] encode_population train=%d dev=%d full=%d reason=independent_queries_against_training_only_context",
                    len(support_records), len(dev_records), len(records))
        trained_endpoints = set(pairs["train"][0].reshape(-1).tolist())
        support_ids = set(support_indices)
        pd.DataFrame([{"product_id": r["product_id"], "split": r["split"],
                       "supervised_endpoint": i in trained_endpoints,
                       "training_graph_support": i in support_ids}
                      for i, r in enumerate(records)]).to_csv(output / name(cfg.track, "listing_usage.csv"), index=False)
        logger.info("[graph-train] track=%s listings=%d train_pairs=%d dev_pairs=%d device=%s",
                    cfg.track, len(records), len(train_pairs), len(dev_pairs), cfg.device)
        from core.training_profiler import TrainingProfiler
        from model_tracks.incremental import ArtifactPublisher
        with ArtifactPublisher(output) as publisher, GraphWandb(cfg.wandb, cfg.track, run_tag, output) as wandb, TrainingProfiler(output / name(cfg.track,'profile'),cfg.device) as profiler:
            wandb.log_config(manifest)
            for epoch in range(start_epoch + 1, cfg.epochs + 1):
                logger.info("[graph-phase] training start epoch=%d/%d train_pairs=%d learning_rate=%s",
                            epoch, cfg.epochs, len(train_pairs), optimizer.param_groups[0]['lr'])
                started = time.monotonic()
                model.train()
                scorer.train()
                optimizer.zero_grad(set_to_none=True)
                states = profiler.call('graph/train_context',model.context,support,support_text)
                embeddings = profiler.call('graph/train_encode',model.encode,support,states,support_text)
                logits = profiler.call('graph/pair_score',scorer,embeddings,train_pairs,support_text)
                classification = F.binary_cross_entropy_with_logits(logits, train_labels)
                a, b = train_pairs.unbind(1)
                cos = (embeddings[a] * embeddings[b]).sum(-1)
                metric = (train_labels * (1 - cos) + (1 - train_labels)
                          * F.relu(cos - cfg.negative_margin)).mean()
                loss = classification + cfg.metric_weight * metric
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite loss")
                profiler.call('graph/backward',loss.backward)
                gradient_parameters = [(parameter_name, parameter) for parameter_name, parameter in model.named_parameters()
                                       if parameter.grad is not None]
                gradient_parameters.extend((f'scorer.{parameter_name}', parameter)
                                           for parameter_name, parameter in scorer.named_parameters()
                                           if parameter.grad is not None)
                # One host transfer instead of synchronizing CUDA once per parameter.
                norm_values = torch.stack([parameter.grad.norm() for _, parameter in gradient_parameters]).detach().cpu().tolist()
                gradient_norms = dict(zip((parameter_name for parameter_name, _ in gradient_parameters), norm_values))
                if not all(np.isfinite(value) for value in gradient_norms.values()):
                    raise RuntimeError("nonfinite gradients")
                with (output / name(cfg.track, "gradient_metrics.jsonl")).open("a") as handle:
                    handle.write(json.dumps({"epoch": epoch, "parameter_gradient_norms": gradient_norms}) + "\n")
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(scorer.parameters()), cfg.max_grad_norm)
                profiler.call('graph/optimizer',optimizer.step)
                logger.info("[graph-phase] dev_evaluation start epoch=%d/%d dev_pairs=%d",
                            epoch, cfg.epochs, len(dev_pairs))
                model.eval()
                scorer.eval()
                with torch.no_grad(), profiler.section('graph/dev_evaluation'):
                    embeddings = model.encode(dev_batch, model.context(support, support_text), dev_text)
                    dev_scores = scorer(embeddings, dev_pairs, dev_text).sigmoid().cpu().numpy()
                metrics = {"epoch": epoch, "train_loss": loss.item(),
                           "train_classification_loss": classification.item(),
                           "train_metric_loss": metric.item(), **quality(pairs["dev"][1], dev_scores),
                           "epoch_seconds": time.monotonic() - started}
                if cfg.device == "cuda":
                    metrics["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
                    metrics["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 1024**3
                previous_best = best_metric
                improved = metrics["dev_pr_auc"] > best_metric
                checkpoint_dir = output / "_checkpoints" / cfg.track / f"{run_tag}_f0" / f"checkpoint-{epoch}"
                checkpoint_dir.mkdir(parents=True)
                checkpoint = checkpoint_dir / name(cfg.track, "graph_model.pt")
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
                logger.info("[graph-checkpoint] write start epoch=%d path=%s selected=%s reason=%s dev_pr_auc=%.6f previous_best=%.6f",
                            epoch, checkpoint, improved,
                            'strictly higher dev_pr_auc' if improved else 'dev_pr_auc did not strictly improve',
                            metrics['dev_pr_auc'], previous_best)
                profiler.call('graph/checkpoint_write',torch.save,payload,checkpoint)
                write_json(checkpoint_dir / name(cfg.track, "trainer_state.json"), {
                    "global_step": epoch, "best_metric": best_metric,
                    "best_model_checkpoint": str(best_path.parent)})
                # Completion marker is written LAST, matching worker conventions.
                write_json(checkpoint_dir / name(cfg.track, "checkpoint_manifest.json"), {
                    "schema": "er-graph-checkpoint-v1", "files": {checkpoint.name: file_hash(checkpoint)},
                    "epoch": epoch, "track": cfg.track})
                write_json(output / name(cfg.track, "best_checkpoint.json"), {"path": str(best_path), "metric": best_metric})
                with (output / name(cfg.track, "epoch_metrics.jsonl")).open("a") as handle:
                    handle.write(json.dumps(metrics, allow_nan=False) + "\n")
                wandb.log_metrics(metrics, step=epoch)
                write_worker_live_status(target=output / name(cfg.track, "live_status.json"), event="epoch_complete",
                    step=epoch, max_steps=cfg.epochs, epoch=epoch, wandb_run_id=wandb.run_id,
                    wandb_url=wandb.run_url, best_checkpoint_step=int(best_path.parent.name.split("-")[-1]),
                    best_metric=best_metric, **{k: v for k, v in metrics.items() if k != "epoch"})
                logger.info("[graph-train] epoch=%d/%d loss=%.6f classification_loss=%.6f metric_loss=%.6f dev_pr_auc=%.6f dev_p_at_r95=%.6f seconds=%.3f best=%s selected_checkpoint=%s",
                            epoch, cfg.epochs, loss.item(), classification.item(), metric.item(),
                            metrics['dev_pr_auc'], metrics['dev_p_at_r95'], metrics['epoch_seconds'], improved, best_path)
                logger.info("[graph-checkpoint] write complete manifest=%s epoch_metrics=%s",
                            checkpoint_dir / name(cfg.track, 'checkpoint_manifest.json'),
                            output / name(cfg.track, 'epoch_metrics.jsonl'))
                profiler.step()
                publisher.submit(f'checkpoint-{epoch}', [checkpoint_dir,
                    output / name(cfg.track, 'epoch_metrics.jsonl'),
                    output / name(cfg.track, 'best_checkpoint.json')])
            logger.info("[graph-selection] training complete completed_epochs=%d selected_checkpoint=%s best_dev_pr_auc=%.6f criterion=max_dev_pr_auc",
                        cfg.epochs, best_path, best_metric)
            completion = None
            if cfg.postprocess:
                logger.info("[graph-phase] postprocess start selected=%s build_index=%s inference_batch_size=%d report_test=%s",
                            best_path, cfg.build_index, cfg.inference_batch_size, cfg.report_test)
                if not cfg.report_test:
                    logger.info("[graph-evaluation] test skipped reason=report_test_false; dev threshold remains the selection/calibration source")
                from graph_tracks.report import complete
                completion_root = output / name(cfg.track, f"completion-epoch-{cfg.epochs}")
                if resume and completion_root.exists():
                    # Preserve a partially written report; regenerate from the
                    # verified selected checkpoint in a fresh directory.
                    completion_root.rename(completion_root.with_name(
                        completion_root.name + f".interrupted-{time.time_ns()}"))
                completion_root.mkdir(parents=True)
                completion = complete(best_path, resolve(cfg.listings), resolve(cfg.pairs), completion_root, cfg,
                    text_cache=resolve(cfg.text_cache) if cfg.text_cache else None)
                logger.info("[graph-phase] postprocess complete inference=%s reports=%s report=%s artifacts=%s",
                            completion['inference'], completion['reports'], completion['report'],
                            [str(path) for path in sorted(completion_root.rglob('*')) if path.is_file()])
                for row in completion['summary']:
                    logger.info("[graph-evaluation] summary=%s", json.dumps(row, sort_keys=True))
                publisher.submit('postprocess', [completion_root])
                for row in completion["summary"]:
                    wandb.set_summary({f"{row['split']}/{key}": value for key, value in row.items()
                                       if isinstance(value, (int, float, bool))})
            if not cfg.postprocess:
                logger.info("[graph-phase] postprocess skipped reason=postprocess_false")
            wandb.set_summary({"best_dev_pr_auc": best_metric, "best_checkpoint": best_path.name,
                               "track": cfg.track, "postprocess_complete": bool(completion)})
            artifacts = [output / name(cfg.track, stem) for stem in
                ("run_manifest.json", "graph_census.json", "epoch_metrics.jsonl", "best_checkpoint.json",
                 "listing_usage.csv", "gradient_metrics.jsonl")]
            artifacts.append(output / "_checkpoints" / cfg.track)
            if cfg.include_inputs:
                artifacts.append(input_dir)
            if completion:
                artifacts.extend([completion["inference"], completion["reports"], completion["report"]])
            wandb.log_artifacts(artifacts)
            write_json(output / name(cfg.track, "graph_worker_result.json"), {
                "status": "ok", "best_checkpoint": str(best_path), "best_dev_pr_auc": best_metric,
                "track": cfg.track, "postprocess_complete": bool(completion), "wandb_run_id": wandb.run_id})
            if cfg.dvc.enabled:
                from graph_tracks.dvc import snapshot
                logger.info("[graph-dvc] snapshot started track=%s push=%s", cfg.track, cfg.dvc.push)
                project = snapshot(output, cfg.track, remote=cfg.dvc.remote, push=cfg.dvc.push,
                                   generation=f"epoch-{cfg.epochs}")
                write_json(output / name(cfg.track, "dvc_result.json"), {
                    "track": cfg.track, "project": str(project), "verified_restore": True, "pushed": cfg.dvc.push})
                logger.info("[graph-dvc] verified clean restore project=%s", project)
                wandb.set_summary({"dvc_verified_restore": True, "dvc_pushed": cfg.dvc.push,
                                   "dvc_project": project.name})
                wandb.log_artifacts([project / name(cfg.track, "dvc_manifest.json"),
                                     project / (name(cfg.track, "payload") + ".dvc")], "dvc-metadata")
        write_worker_live_status(target=output / name(cfg.track, "live_status.json"), event="complete", step=cfg.epochs,
                                max_steps=cfg.epochs, epoch=cfg.epochs, best_metric=best_metric)
        logger.info("[graph-train] complete checkpoint=%s", best_path)
        return best_path
    except BaseException as error:
        logger.exception("[graph-train] failed")
        write_json(output / name(cfg.track, "graph_worker_result.json"), {"status": "failed", "error": str(error)})
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
