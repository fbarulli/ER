#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-sku-inference-$(date -u +%Y%m%dT%H%M%SZ)}"
GPU="${GPU:-CPU}"
TIMEOUT="${TIMEOUT:-14400}"
THRESHOLD="${THRESHOLD:-0.61}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/training_results/0916T082923217621Z/worker_1/_checkpoints/all-MiniLM-L6-v2/r0916T082923217621Z_f0/checkpoint-114}"
REMOTE_ROOT="${REMOTE_ROOT:-/content/EuromonitoR}"
OUTPUT_NAME="sku_item_submission_original_dataset_calibrated_061"
LOCAL_OUTPUT="${LOCAL_OUTPUT:-$ROOT/submission/$OUTPUT_NAME.csv}"

COLAB_PY="/home/opc/.local/share/uv/tools/google-colab-cli/bin/python3"
COLAB_ENTRY="$ROOT/src/cli/colab_cli_entry.py"
COLAB_CONFIG="$ROOT/colab_cli_state/sessions.json"
COLAB=("$COLAB_PY" "$COLAB_ENTRY" --config "$COLAB_CONFIG")

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "checkpoint directory not found: $CHECKPOINT_DIR" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT_DIR/config.json" ]]; then
  echo "checkpoint is incomplete (config.json missing): $CHECKPOINT_DIR" >&2
  exit 1
fi
if [[ -n "$(git -C "$ROOT" status --porcelain -- COLAB_CALL.sh colab_inference_remote.py submission_inference.py)" ]]; then
  echo "inference launcher files must be committed and pushed before Colab starts" >&2
  exit 1
fi

REPO_URL="${REPO_URL:-$(git -C "$ROOT" remote get-url ER)}"
REPO_BRANCH="${REPO_BRANCH:-$(git -C "$ROOT" branch --show-current)}"
REPO_COMMIT="${REPO_COMMIT:-$(git -C "$ROOT" rev-parse HEAD)}"
WORK_DIR="$(mktemp -d /tmp/euromonitor-colab-inference.XXXXXX)"
CHECKPOINT_ARCHIVE="$WORK_DIR/checkpoint-114-inference.tar.gz"
SESSION_STARTED=0

cleanup() {
  if [[ "$SESSION_STARTED" == 1 ]]; then
    "${COLAB[@]}" stop --session "$SESSION" >/dev/null 2>&1 || true
  fi
  rm -rf -- "$WORK_DIR"
}
trap cleanup EXIT INT TERM

# Inference needs model/tokenizer/config files, not the 173 MiB optimizer state.
# Keeping training-only state out also stays below Colab's single-upload limit.
CHECKPOINT_NAME="$(basename "$CHECKPOINT_DIR")"
tar -czf "$CHECKPOINT_ARCHIVE" \
  --exclude="$CHECKPOINT_NAME/optimizer.pt" \
  --exclude="$CHECKPOINT_NAME/scheduler.pt" \
  --exclude="$CHECKPOINT_NAME/rng_state.pth" \
  --exclude="$CHECKPOINT_NAME/trainer_state.json" \
  --exclude="$CHECKPOINT_NAME/training_args.bin" \
  -C "$(dirname "$CHECKPOINT_DIR")" "$CHECKPOINT_NAME"
mkdir -p "$(dirname "$LOCAL_OUTPUT")"

echo "[colab] session=$SESSION gpu=$GPU branch=$REPO_BRANCH commit=$REPO_COMMIT"
echo "[inference] checkpoint=$CHECKPOINT_DIR threshold=$THRESHOLD profile=cleaned"
if [[ "$GPU" == "CPU" ]]; then
  "${COLAB[@]}" new --session "$SESSION"
else
  "${COLAB[@]}" new --session "$SESSION" --gpu "$GPU"
fi
SESSION_STARTED=1
"${COLAB[@]}" upload --session "$SESSION" \
  "$CHECKPOINT_ARCHIVE" "/content/checkpoint-114-inference.tar.gz"

EXEC_ARGS=(
  exec --session "$SESSION" --timeout "$TIMEOUT"
  --env "REMOTE_ROOT=$REMOTE_ROOT"
  --env "REPO_URL=$REPO_URL"
  --env "REPO_BRANCH=$REPO_BRANCH"
  --env "REPO_COMMIT=$REPO_COMMIT"
  --env "REQUESTED_GPU=$GPU"
  --env "INFERENCE_THRESHOLD=$THRESHOLD"
)
if [[ -n "${BATCH_SIZE:-}" ]]; then
  EXEC_ARGS+=(--env "INFERENCE_BATCH_SIZE=$BATCH_SIZE")
fi
EXEC_ARGS+=(-f "$ROOT/colab_inference_remote.py")
"${COLAB[@]}" "${EXEC_ARGS[@]}"

REMOTE_OUTPUT="$REMOTE_ROOT/submission/$OUTPUT_NAME.csv"
"${COLAB[@]}" download --session "$SESSION" "$REMOTE_OUTPUT" "$LOCAL_OUTPUT"
"${COLAB[@]}" download --session "$SESSION" \
  "${REMOTE_OUTPUT%.csv}.json" "${LOCAL_OUTPUT%.csv}.json"
echo "[inference] downloaded $LOCAL_OUTPUT"
