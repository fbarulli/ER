from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import random
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import spacy
import torch
from huggingface_hub import HfApi
from pydantic import BaseModel
from spacy.training import Example
from spacy.util import minibatch

try:
    # Normal local/package execution: consume the shared SSOT API.
    from core.common import RESULTS, TRAIN_ROOT, ner_config
except ModuleNotFoundError:
    # Colab receives a rendered, ephemeral view of that same config rather
    # than a second YAML file or a copied config loader.
    _runtime_path = os.environ.get("NER_RUNTIME_CONFIG")
    _runtime_base = os.environ.get("NER_BASE_DIR")
    _runtime_results = os.environ.get("NER_RESULTS_DIR")
    if not (_runtime_path and _runtime_base and _runtime_results):
        raise
    TRAIN_ROOT = Path(_runtime_base).resolve()
    RESULTS = Path(_runtime_results).resolve()

    def ner_config() -> dict[str, Any]:
        return json.loads(Path(_runtime_path).read_text(encoding="utf-8"))

RESULTS_DIR = RESULTS
LOG_DIR = RESULTS_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "ner_training.log"
ARTIFACT_MANIFEST_NAME = "ner_artifacts_manifest.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_FILE,
            mode="w",
        ),
        logging.StreamHandler(),
    ],
    force=True,
)

logger = logging.getLogger(__name__)


class NERTrainingConfig(BaseModel):
    input_jsonl: str
    output_dir: str
    model_name: str

    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 2.0e-5
    dropout: float = 0.2

    train_frac: float = 0.50
    validation_frac: float = 0.25
    holdout_frac: float = 0.25

    seed: int = 42
    label: str = "BRAND"


def _resolve_config_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _resolve_config_value(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _resolve_config_value(item)
            for item in value
        ]

    if not isinstance(value, str):
        return value

    replacements = {
        "${base_dir}": str(TRAIN_ROOT),
        "${results_dir}": str(RESULTS_DIR),
    }

    resolved = value
    for key, replacement in replacements.items():
        resolved = resolved.replace(
            key,
            replacement,
        )

    resolved = os.path.expandvars(resolved)
    return resolved


def load_training_config() -> NERTrainingConfig:
    raw = ner_config()

    if not raw or "ner_training" not in raw:
        raise KeyError(
            "config/training.yaml is missing 'ner.ner_training'"
        )

    section = _resolve_config_value(
        raw["ner_training"]
    )

    return NERTrainingConfig(**section)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_records(path: str):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def _entity_tuple(entity):
    if isinstance(entity, dict):
        return (
            int(entity["start"]),
            int(entity["end"]),
            str(entity["label"]),
        )

    if (
        isinstance(entity, (list, tuple))
        and len(entity) == 3
    ):
        return (
            int(entity[0]),
            int(entity[1]),
            str(entity[2]),
        )

    raise ValueError(
        f"Unsupported entity format: {entity!r}"
    )


def build_examples(nlp, records):
    examples = []

    for record in records:
        entities = [
            _entity_tuple(entity)
            for entity in record["entities"]
        ]

        examples.append(
            Example.from_dict(
                nlp.make_doc(record["text"]),
                {"entities": entities},
            )
        )

    return examples


def split_records(
    records,
    train_frac,
    validation_frac,
    holdout_frac,
    seed,
):
    total = (
        train_frac
        + validation_frac
        + holdout_frac
    )

    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            "train_frac + validation_frac + "
            "holdout_frac must equal 1.0"
        )

    records = list(records)

    rng = random.Random(seed)
    rng.shuffle(records)

    n = len(records)

    train_end = int(
        n * train_frac
    )

    validation_end = train_end + int(
        n * validation_frac
    )

    train_records = records[:train_end]
    validation_records = records[
        train_end:validation_end
    ]
    holdout_records = records[
        validation_end:
    ]

    return (
        train_records,
        validation_records,
        holdout_records,
    )


def build_pipeline(model_name, labels):
    nlp = spacy.blank("xx")

    nlp.add_pipe(
        "transformer",
        config={
            "model": {
                "name": model_name,
                "tokenizer_config": {
                    "use_fast": True,
                    "local_files_only": True,
                },
                "transformer_config": {
                    "local_files_only": True,
                },
            }
        },
    )

    ner = nlp.add_pipe("ner")
    for label in sorted(set(labels)):
        ner.add_label(label)

    return nlp


def _gold_entities(record):
    return {
        _entity_tuple(entity)
        for entity in record["entities"]
    }


def _predicted_entities(doc):
    return {
        (
            int(ent.start_char),
            int(ent.end_char),
            str(ent.label_),
        )
        for ent in doc.ents
    }


def evaluate_records(
    nlp,
    records,
    collect_errors=False,
):
    tp = 0
    fp = 0
    fn = 0
    errors = []

    for record in records:
        text = record["text"]

        gold = _gold_entities(record)
        predicted = _predicted_entities(
            nlp(text)
        )

        tp_set = gold & predicted
        fp_set = predicted - gold
        fn_set = gold - predicted

        tp += len(tp_set)
        fp += len(fp_set)
        fn += len(fn_set)

        if collect_errors and (
            fp_set or fn_set
        ):
            gold_sorted = sorted(gold)
            pred_sorted = sorted(predicted)

            if fp_set and fn_set:
                error_type = "wrong_span"
            elif fn_set:
                error_type = "false_negative"
            else:
                error_type = "false_positive"

            errors.append(
                {
                    "error_type": error_type,
                    "text": text,
                    "expected": json.dumps(
                        gold_sorted,
                        ensure_ascii=False,
                    ),
                    "predicted": json.dumps(
                        pred_sorted,
                        ensure_ascii=False,
                    ),
                }
            )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }

    return metrics, errors


def _checkpoint_dirs(output_dir: Path):
    checkpoints = []

    for path in output_dir.glob("epoch_*"):
        if not path.is_dir():
            continue

        match = re.fullmatch(
            r"epoch_(\d+)",
            path.name,
        )

        if not match:
            continue

        checkpoints.append(
            (
                int(match.group(1)),
                path,
            )
        )

    checkpoints.sort(
        key=lambda item: item[0]
    )

    return checkpoints


def latest_checkpoint(output_dir: Path):
    checkpoints = _checkpoint_dirs(
        output_dir
    )

    if not checkpoints:
        return None

    return checkpoints[-1]


def _write_checkpoint_state(
    checkpoint_dir: Path,
    epoch: int,
    best_epoch: int | None,
    best_val_f1: float,
):
    state = {
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
    }

    with (
        checkpoint_dir
        / "training_state.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            state,
            f,
            indent=2,
        )


def _make_checkpoint_zip(
    output_dir: Path,
    checkpoint_dir: Path,
):
    zip_base = (
        output_dir.parent
        / "latest_checkpoint"
    )

    zip_path = Path(
        shutil.make_archive(
            str(zip_base),
            "zip",
            root_dir=output_dir,
            base_dir=checkpoint_dir.name,
        )
    )

    return zip_path


def _make_model_zip(
    nlp,
    destination: Path,
):
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        model_dir = tmp / "ner_model"

        nlp.to_disk(model_dir)

        zip_base = (
            destination.parent
            / destination.stem
        )

        generated = Path(
            shutil.make_archive(
                str(zip_base),
                "zip",
                root_dir=tmp,
                base_dir="ner_model",
            )
        )

    if generated != destination:
        if destination.exists():
            destination.unlink()

        generated.replace(destination)

    return destination


def _sha256_file(path: Path) -> str:
    """Return a streaming content hash for a final NER artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_artifact_manifest(artifacts: list[Path]) -> Path:
    """Publish the final-artifact hashes last for Colab transfer validation."""
    entries: dict[str, dict[str, int | str]] = {}
    for artifact in artifacts:
        if not artifact.is_file():
            raise RuntimeError(
                f"Cannot publish NER artifact manifest; missing {artifact}"
            )
        entries[artifact.name] = {
            "sha256": _sha256_file(artifact),
            "bytes": artifact.stat().st_size,
        }

    manifest_path = RESULTS_DIR / ARTIFACT_MANIFEST_NAME
    temporary = manifest_path.with_name(
        f"{manifest_path.name}.tmp-{os.getpid()}"
    )
    temporary.write_text(
        json.dumps(
            {"schema_version": "1", "artifacts": entries},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest_path


def _hf_settings():
    token = os.getenv("HF_TOKEN")
    repo_id = os.getenv("HF_NER_REPO")

    required = (
        os.getenv(
            "HF_BACKUP_REQUIRED",
            "1",
        )
        != "0"
    )

    if required and (
        not token or not repo_id
    ):
        raise RuntimeError(
            "HF checkpoint backup is required but "
            "HF_TOKEN or HF_NER_REPO is missing."
        )

    return token, repo_id, required


def prepare_hf_repo():
    token, repo_id, _required = (
        _hf_settings()
    )

    if not token or not repo_id:
        logger.warning(
            "HF backup disabled because token/repo "
            "was not supplied."
        )
        return None

    api = HfApi(token=token)

    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=True,
        exist_ok=True,
        token=token,
    )

    logger.info(
        "HF backup repo ready → %s",
        repo_id,
    )

    return api


def _upload_file_with_retry(
    api,
    *,
    local_path: Path,
    repo_id: str,
    path_in_repo: str,
    commit_message: str,
    attempts: int = 3,
):
    last_error = None

    for attempt in range(
        1,
        attempts + 1,
    ):
        try:
            api.upload_file(
                path_or_fileobj=str(
                    local_path
                ),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
            )

            return

        except Exception as exc:
            last_error = exc

            logger.exception(
                "HF upload attempt %d/%d failed "
                "for %s",
                attempt,
                attempts,
                path_in_repo,
            )

            if attempt < attempts:
                time.sleep(
                    5 * attempt
                )

    raise RuntimeError(
        f"HF upload failed after {attempts} "
        f"attempts: {path_in_repo}"
    ) from last_error


def backup_checkpoint_to_hf(
    api,
    output_dir: Path,
    checkpoint_dir: Path,
    epoch: int,
):
    if api is None:
        return

    repo_id = os.environ["HF_NER_REPO"]

    zip_path = _make_checkpoint_zip(
        output_dir,
        checkpoint_dir,
    )

    try:
        _upload_file_with_retry(
            api,
            local_path=zip_path,
            repo_id=repo_id,
            path_in_repo=(
                "checkpoints/"
                "latest_checkpoint.zip"
            ),
            commit_message=(
                f"NER checkpoint epoch {epoch}"
            ),
        )

        logger.info(
            "HF checkpoint backup complete → "
            "%s @ epoch %d",
            repo_id,
            epoch,
        )

    finally:
        zip_path.unlink(
            missing_ok=True
        )


def backup_best_model_to_hf(
    api,
    nlp,
    epoch: int,
):
    if api is None:
        return

    repo_id = os.environ["HF_NER_REPO"]
    zip_path = (
        RESULTS_DIR
        / "best_ner_model.zip"
    )

    _make_model_zip(
        nlp,
        zip_path,
    )

    try:
        _upload_file_with_retry(
            api,
            local_path=zip_path,
            repo_id=repo_id,
            path_in_repo=(
                "best/best_ner_model.zip"
            ),
            commit_message=(
                f"Best NER model epoch {epoch}"
            ),
        )

        logger.info(
            "HF best-model backup complete → "
            "%s @ epoch %d",
            repo_id,
            epoch,
        )

    finally:
        zip_path.unlink(
            missing_ok=True
        )


def upload_final_artifacts_to_hf(
    api,
    final_model_zip: Path,
    metadata_path: Path,
    errors_path: Path,
    artifact_manifest_path: Path,
):
    if api is None:
        return

    repo_id = os.environ["HF_NER_REPO"]

    artifacts = [
        (
            final_model_zip,
            "final/ner_model.zip",
        ),
        (
            metadata_path,
            "final/training_metadata.json",
        ),
        (
            errors_path,
            "final/ner_errors.csv",
        ),
        (
            artifact_manifest_path,
            "final/ner_artifacts_manifest.json",
        ),
    ]

    for local_path, remote_path in artifacts:
        if not local_path.exists():
            continue

        _upload_file_with_retry(
            api,
            local_path=local_path,
            repo_id=repo_id,
            path_in_repo=remote_path,
            commit_message=(
                "Update final NER artifacts"
            ),
        )


def write_errors_csv(
    path: Path,
    errors,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "error_type",
                "text",
                "expected",
                "predicted",
            ],
        )

        writer.writeheader()
        writer.writerows(errors)


def main():
    cfg = load_training_config()

    set_seed(cfg.seed)

    output_dir = Path(
        cfg.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "Loading NER records → %s",
        cfg.input_jsonl,
    )

    records = load_records(
        cfg.input_jsonl
    )

    (
        train_records,
        validation_records,
        holdout_records,
    ) = split_records(
        records,
        cfg.train_frac,
        cfg.validation_frac,
        cfg.holdout_frac,
        cfg.seed,
    )

    logger.info(
        "Dataset | total=%d | train=%d | "
        "validation=%d | holdout=%d",
        len(records),
        len(train_records),
        len(validation_records),
        len(holdout_records),
    )

    hf_api = prepare_hf_repo()

    restored = latest_checkpoint(
        output_dir
    )

    labels = {cfg.label}
    labels.update(
        label
        for record in records
        for _, _, label in (
            _entity_tuple(entity)
            for entity in record["entities"]
        )
    )

    if restored is None:
        start_epoch = 1

        logger.info(
            "Starting fresh NER model"
        )

        nlp = build_pipeline(
            cfg.model_name,
            labels,
        )

        train_examples = build_examples(
            nlp,
            train_records,
        )

        optimizer = nlp.initialize(
            get_examples=lambda: train_examples
        )

        optimizer.learn_rate = (
            cfg.learning_rate
        )

        best_val_f1 = -1.0
        best_epoch = None

    else:
        restored_epoch, checkpoint_dir = (
            restored
        )

        start_epoch = (
            restored_epoch + 1
        )

        logger.info(
            "Restoring checkpoint → %s",
            checkpoint_dir,
        )

        nlp = spacy.load(
            checkpoint_dir
        )
        ner = nlp.get_pipe("ner")
        for label in sorted(labels):
            ner.add_label(label)

        optimizer = nlp.resume_training()
        optimizer.learn_rate = (
            cfg.learning_rate
        )

        state_path = (
            checkpoint_dir
            / "training_state.json"
        )

        best_val_f1 = -1.0
        best_epoch = None

        if state_path.exists():
            try:
                state = json.loads(
                    state_path.read_text(
                        encoding="utf-8"
                    )
                )

                best_val_f1 = float(
                    state.get(
                        "best_val_f1",
                        -1.0,
                    )
                )

                raw_best_epoch = state.get(
                    "best_epoch"
                )

                if raw_best_epoch is not None:
                    best_epoch = int(
                        raw_best_epoch
                    )

            except Exception:
                logger.exception(
                    "Could not read checkpoint "
                    "training_state.json"
                )

    if start_epoch > cfg.epochs:
        logger.info(
            "Checkpoint already reached epoch %d; "
            "no additional training required.",
            cfg.epochs,
        )
    else:
        logger.info(
            "Training epochs: %d → %d",
            start_epoch,
            cfg.epochs,
        )

        for epoch in range(
            start_epoch,
            cfg.epochs + 1,
        ):
            epoch_loss = 0.0

            shuffled = list(
                train_records
            )

            random.shuffle(
                shuffled
            )

            train_examples = build_examples(
                nlp,
                shuffled,
            )

            batches = minibatch(
                train_examples,
                size=cfg.batch_size,
            )

            for batch_index, batch in enumerate(
                batches,
                start=1,
            ):
                losses = {}

                nlp.update(
                    batch,
                    drop=cfg.dropout,
                    sgd=optimizer,
                    losses=losses,
                )

                batch_loss = sum(
                    float(value)
                    for value
                    in losses.values()
                )

                epoch_loss += batch_loss

                if (
                    batch_index % 100
                    == 0
                ):
                    logger.info(
                        "Epoch %d/%d | batch=%d | "
                        "loss=%.4f",
                        epoch,
                        cfg.epochs,
                        batch_index,
                        epoch_loss,
                    )

            validation_metrics, _ = (
                evaluate_records(
                    nlp,
                    validation_records,
                )
            )

            logger.info(
                "Epoch %d/%d | loss=%.4f | "
                "val_P=%.4f | val_R=%.4f | "
                "val_F1=%.4f",
                epoch,
                cfg.epochs,
                epoch_loss,
                validation_metrics[
                    "precision"
                ],
                validation_metrics[
                    "recall"
                ],
                validation_metrics[
                    "f1"
                ],
            )

            if (
                validation_metrics["f1"]
                > best_val_f1
            ):
                best_val_f1 = (
                    validation_metrics["f1"]
                )

                best_epoch = epoch

                nlp.to_disk(
                    output_dir
                )

                logger.info(
                    "Saved BEST model → %s",
                    output_dir,
                )

                backup_best_model_to_hf(
                    hf_api,
                    nlp,
                    epoch,
                )

            checkpoint_dir = (
                output_dir
                / f"epoch_{epoch}"
            )

            if checkpoint_dir.exists():
                shutil.rmtree(
                    checkpoint_dir
                )

            nlp.to_disk(
                checkpoint_dir
            )

            _write_checkpoint_state(
                checkpoint_dir,
                epoch,
                best_epoch,
                best_val_f1,
            )

            logger.info(
                "Saved epoch checkpoint → %s",
                checkpoint_dir,
            )

            # Critical durability point:
            # do not start the next epoch until the
            # completed checkpoint is safely on HF.
            backup_checkpoint_to_hf(
                hf_api,
                output_dir,
                checkpoint_dir,
                epoch,
            )

    if (
        (output_dir / "config.cfg").exists()
        and (output_dir / "meta.json").exists()
    ):
        best_nlp = spacy.load(
            output_dir
        )

    else:
        restored = latest_checkpoint(
            output_dir
        )

        if restored is None:
            raise RuntimeError(
                "No trained model is available "
                "for holdout evaluation."
            )

        _, checkpoint_dir = restored
        best_nlp = spacy.load(
            checkpoint_dir
        )

    holdout_metrics, errors = (
        evaluate_records(
            best_nlp,
            holdout_records,
            collect_errors=True,
        )
    )

    logger.info(
        "HOLDOUT | P=%.4f | R=%.4f | "
        "F1=%.4f | TP=%d | FP=%d | FN=%d",
        holdout_metrics["precision"],
        holdout_metrics["recall"],
        holdout_metrics["f1"],
        holdout_metrics["tp"],
        holdout_metrics["fp"],
        holdout_metrics["fn"],
    )

    errors_path = (
        RESULTS_DIR
        / "ner_errors.csv"
    )

    write_errors_csv(
        errors_path,
        errors,
    )

    metadata = {
        "records_total": len(records),
        "train_records": len(train_records),
        "validation_records": len(
            validation_records
        ),
        "holdout_records": len(
            holdout_records
        ),
        "seed": cfg.seed,
        "model_name": cfg.model_name,
        "labels": sorted(labels),
        "epochs_requested": cfg.epochs,
        "best_epoch": best_epoch,
        "best_validation_f1": (
            best_val_f1
        ),
        "holdout": holdout_metrics,
        "hf_repo": os.getenv(
            "HF_NER_REPO"
        ),
    }

    metadata_path = (
        RESULTS_DIR
        / "training_metadata.json"
    )

    final_model_zip = (
        RESULTS_DIR
        / "ner_model_final.zip"
    )

    _make_model_zip(
        best_nlp,
        final_model_zip,
    )

    # Hash the two payload artifacts in metadata for human inspection.  The
    # small manifest below additionally hashes this metadata file itself and
    # is written LAST, making it the remote completion/integrity marker.
    metadata["artifact_hashes"] = {
        errors_path.name: _sha256_file(errors_path),
        final_model_zip.name: _sha256_file(final_model_zip),
    }
    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )
    artifact_manifest_path = _write_artifact_manifest(
        [errors_path, metadata_path, final_model_zip]
    )

    upload_final_artifacts_to_hf(
        hf_api,
        final_model_zip,
        metadata_path,
        errors_path,
        artifact_manifest_path,
    )

    logger.info(
        "Training complete. Final model zip → %s",
        final_model_zip,
    )


if __name__ == "__main__":
    main()
