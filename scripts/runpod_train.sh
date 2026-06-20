#!/usr/bin/env bash
#
# runpod_train.sh — run a training job on a RunPod pod, then AUTOMATICALLY
# stop/terminate the pod so you are not billed for idle GPU time once the run
# finishes (whether it succeeds OR hits a fatal error).
#
# Usage (from the repo root, inside /workspace/meshgraphnet-welding):
#   bash scripts/runpod_train.sh [CONFIG_JSON]
#
# Run it detached so it survives closing your laptop (see RUNPOD.md):
#   tmux new -s train
#   bash scripts/runpod_train.sh config.runpod.json      # Ctrl-b d to detach
#
# Environment knobs:
#   SHUTDOWN_ACTION   stop | remove | none   (default: stop)
#       stop    -> pod halts; GPU billing ends; resumable from the web console.
#                  (You still pay the small idle disk/volume rate.)
#       remove  -> pod is terminated; GPU + pod-disk billing ends entirely.
#                  Your code/checkpoints SURVIVE because they live on the
#                  /workspace network volume (billed separately, very cheap).
#       none    -> leave the pod running (debugging / manual inspection).
#   SHUTDOWN_GRACE    seconds to wait before shutdown (default: 30) — if you are
#                     still attached you can Ctrl-C here to cancel the shutdown.
#   POST_TRAIN_CMD    optional shell command to run AFTER training, BEFORE
#                     shutdown (e.g. a spiral evaluation). Its failure never
#                     blocks the auto-shutdown.
#
set -uo pipefail

CONFIG="${1:-config.runpod.json}"
SHUTDOWN_ACTION="${SHUTDOWN_ACTION:-stop}"
SHUTDOWN_GRACE="${SHUTDOWN_GRACE:-30}"

mkdir -p logs
RUN_LOG="logs/runpod_run_$(date +%Y%m%d_%H%M%S).log"
log () { echo "[runpod_train $(date +%H:%M:%S)] $*" | tee -a "$RUN_LOG"; }

# --- Auto-shutdown runs on ANY exit (success, training error, or sync error) ---
finish () {
  local code=$?
  log "wrapper exiting (last exit code ${code})"
  if [ "$SHUTDOWN_ACTION" = "none" ]; then
    log "SHUTDOWN_ACTION=none -> leaving pod running"; exit "$code"
  fi
  if [ -z "${RUNPOD_POD_ID:-}" ]; then
    log "RUNPOD_POD_ID not set (not on a RunPod pod?) -> skipping auto-shutdown"; exit "$code"
  fi
  if ! command -v runpodctl >/dev/null 2>&1; then
    log "runpodctl not found -> skipping auto-shutdown; stop the pod manually!"; exit "$code"
  fi
  log "auto-shutdown: will '${SHUTDOWN_ACTION}' pod ${RUNPOD_POD_ID} in ${SHUTDOWN_GRACE}s — Ctrl-C to cancel"
  sleep "$SHUTDOWN_GRACE"
  log "running: runpodctl ${SHUTDOWN_ACTION} pod ${RUNPOD_POD_ID}"
  runpodctl "${SHUTDOWN_ACTION}" pod "${RUNPOD_POD_ID}" 2>&1 | tee -a "$RUN_LOG" \
    || log "WARNING: runpodctl ${SHUTDOWN_ACTION} failed — STOP THE POD MANUALLY to avoid charges"
  exit "$code"
}
trap finish EXIT

log "config=${CONFIG} | shutdown=${SHUTDOWN_ACTION} | pod=${RUNPOD_POD_ID:-<none>}"

# 1) Reproduce the exact environment from uv.lock (pulls cu128 torch on Linux).
log "uv sync --frozen ..."
uv sync --frozen 2>&1 | tee -a "$RUN_LOG" || { log "FATAL: uv sync failed"; exit 1; }

# 2) Train. train.py writes its own timestamped logs/ + CSV history; we tee too.
log "starting training: python -m src.training.train --config ${CONFIG}"
uv run python -m src.training.train --config "${CONFIG}" 2>&1 | tee -a "$RUN_LOG"
TRAIN_CODE=${PIPESTATUS[0]}
log "training finished with exit code ${TRAIN_CODE}"

# 3) Optional post-training step (e.g. spiral eval) — never blocks shutdown.
if [ -n "${POST_TRAIN_CMD:-}" ]; then
  log "POST_TRAIN_CMD: ${POST_TRAIN_CMD}"
  bash -lc "${POST_TRAIN_CMD}" 2>&1 | tee -a "$RUN_LOG" || log "POST_TRAIN_CMD failed (continuing to shutdown)"
fi

# Exit with the training code; the EXIT trap performs the auto-shutdown.
exit "${TRAIN_CODE}"
