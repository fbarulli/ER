#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-sku-inference-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU="${GPU:-CPU}"
TIMEOUT="${TIMEOUT:-14400}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/training_results/0916T082923217621Z/worker_1/_checkpoints/all-MiniLM-L6-v2/r0916T082923217621Z_f0/checkpoint-114}"
REMOTE_ROOT="/content/EuromonitoR"
LOCAL_OUTPUT="$ROOT/submission/sku_item_submission_original_dataset_calibrated_061.csv"

COLAB=(
  /home/opc/.local/share/uv/tools/google-colab-cli/bin/python3
  "$ROOT/src/cli/colab_cli_entry.py"
  --config "$ROOT/colab_cli_state/sessions.json"
)

if [[ ! -f "$CHECKPOINT_DIR/model.safetensors" ]]; then
  echo "checkpoint-114 model weights not found: $CHECKPOINT_DIR" >&2
  exit 1
fi
if [[ -n "$(git -C "$ROOT" status --porcelain -- COLAB_CALL.sh colab_inference_remote.py submission_inference.py)" ]]; then
  echo "commit and push the inference files before launching Colab" >&2
  exit 1
fi

REPO_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
WORK_DIR="$(mktemp -d /tmp/euromonitor-colab-inference.XXXXXX)"
SESSION_STARTED=0

cleanup() {
  if [[ "$SESSION_STARTED" == 1 ]]; then
    "${COLAB[@]}" stop --session "$SESSION" >/dev/null 2>&1 || true
  fi
  rm -rf -- "$WORK_DIR"
}
trap cleanup EXIT INT TERM

echo "[local] preparing cleaned matcher text"
PYTHONPATH="$ROOT/src" python "$ROOT/submission_inference.py" prepare \
  --output "$WORK_DIR/inference_texts.jsonl"

CHECKPOINT_NAME="$(basename "$CHECKPOINT_DIR")"
tar -czf "$WORK_DIR/checkpoint.tar.gz" \
  --exclude="$CHECKPOINT_NAME/optimizer.pt" \
  --exclude="$CHECKPOINT_NAME/scheduler.pt" \
  --exclude="$CHECKPOINT_NAME/rng_state.pth" \
  --exclude="$CHECKPOINT_NAME/trainer_state.json" \
  --exclude="$CHECKPOINT_NAME/training_args.bin" \
  -C "$(dirname "$CHECKPOINT_DIR")" "$CHECKPOINT_NAME"
split -b 16m -d -a 3 "$WORK_DIR/checkpoint.tar.gz" "$WORK_DIR/checkpoint.part-"

echo "[colab] session=$SESSION gpu=$GPU commit=$REPO_COMMIT"
if [[ "$GPU" == "CPU" ]]; then
  "${COLAB[@]}" new --session "$SESSION"
else
  "${COLAB[@]}" new --session "$SESSION" --gpu "$GPU"
fi
SESSION_STARTED=1

"${COLAB[@]}" upload --session "$SESSION" \
  "$WORK_DIR/inference_texts.jsonl" /content/inference_texts.jsonl
"${COLAB[@]}" upload --session "$SESSION" \
  "$WORK_DIR/inference_texts.json" /content/inference_texts.json
for part in "$WORK_DIR"/checkpoint.part-*; do
  "${COLAB[@]}" upload --session "$SESSION" "$part" "/content/$(basename "$part")"
done

"${COLAB[@]}" exec --session "$SESSION" --timeout "$TIMEOUT" \
  --env "REMOTE_ROOT=$REMOTE_ROOT" \
  --env "REPO_URL=https://github.com/fbarulli/ER.git" \
  --env "REPO_BRANCH=submission" \
  --env "REPO_COMMIT=$REPO_COMMIT" \
  --env "REQUESTED_GPU=$GPU" \
  -f "$ROOT/colab_inference_remote.py"

if [[ "$GPU" == "CPU" ]]; then
  echo "[colab] CPU checkout/transfer validation complete; no inference was run"
  exit 0
fi

"${COLAB[@]}" download --session "$SESSION" \
  /content/embedding_manifest.json "$WORK_DIR/embedding_manifest.json"
mapfile -t EMBEDDING_PARTS < <(
  python -c 'import json,sys; print(*json.load(open(sys.argv[1]))["parts"], sep="\n")' \
    "$WORK_DIR/embedding_manifest.json"
)
for part in "${EMBEDDING_PARTS[@]}"; do
  "${COLAB[@]}" download --session "$SESSION" "/content/$part" "$WORK_DIR/$part"
done
cat "$WORK_DIR"/embedding.part-* > "$WORK_DIR/inference_embeddings.npy"

EXPECTED_SHA="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["sha256"])' "$WORK_DIR/embedding_manifest.json")"
ACTUAL_SHA="$(sha256sum "$WORK_DIR/inference_embeddings.npy" | cut -d' ' -f1)"
if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
  echo "downloaded GPU embeddings failed checksum validation" >&2
  exit 1
fi

echo "[local] applying matcher gates and expanding to dataset.csv"
PYTHONPATH="$ROOT/src" python "$ROOT/submission_inference.py" finalize \
  --embeddings "$WORK_DIR/inference_embeddings.npy" \
  --output "$LOCAL_OUTPUT"
echo "[submission] wrote $LOCAL_OUTPUT"
