# Training on RunPod

Long, open-ended training run of the MeshGraphNet welding surrogate on a RunPod
GPU pod. Driven by **early stopping** (no fixed epoch count) and **resumable**
so it survives spot-instance preemption.

The dataset (`data/raw/*.npz`, ~19 MB, 20 train / 5 val sims) is committed to the
repo, so a single `git clone` brings code **and** data to the pod.

---

## 0. One-time: push this repo to GitHub (local machine)

```bash
gh repo create meshgraphnet-welding --private --source . --remote origin --push
# or, with an existing empty remote:
#   git remote add origin git@github.com:<user>/meshgraphnet-welding.git
#   git push -u origin main
```

## 1. Start a pod

- **Template:** a PyTorch CUDA 12.4 image (matches the `pytorch-cu124` index in
  `pyproject.toml`; `uv sync` selects the right wheels automatically on Linux).
- **GPU:** a single mid-range GPU is plenty for this model (`hidden_dim=128`,
  8 message-passing steps).
- **Spot/Interruptible** for cost — this run is built to resume (see §4).
- **Attach a Network Volume** (it mounts at `/workspace` by default). This is the
  **only** storage that survives a stop/preemption — the container disk is wiped.

> ### ⚠️ Persistent storage: all artifacts must live on the volume
> Everything the run *writes* must point inside the volume mount (`/workspace`),
> or you lose it when the pod dies:
> - **checkpoints** (`best_model.pt`, `last_model.pt`, `stats.pt`, …) — the model;
> - **rollout exports** (`rollout_pred_dir`);
> - **processed-graph cache** (`<data_root>/processed`) — regenerable, but caching
>   it on the volume avoids re-processing on every fresh pod.
>
> The recipe below does this two ways at once: it **clones the repo into
> `/workspace`** *and* points `--checkpoint_dir` at an absolute volume path
> **outside** the repo, so checkpoints survive even if you re-clone for a code
> update. Resume (§4) reads from the same `--checkpoint_dir`.

## 2. Set up the pod (on the volume)

```bash
# In the pod's web terminal. The network volume is mounted at /workspace.
export VOL=/workspace
mkdir -p "$VOL/checkpoints" "$VOL/output"
cd "$VOL"

git clone https://github.com/dcanales-lu/meshgraphnet-welding.git
cd meshgraphnet-welding

# Install uv, then sync deps (installs CUDA 12.4 torch wheels on Linux):
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env        # or restart the shell
uv sync

# Sanity check (fast):
uv run pytest -q
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

> The repo (and its committed `data/raw/*.npz`) now lives on the volume, so the
> processed cache it writes under `data/processed/` is persistent too.

## 3. Train (detached, writing to the volume)

Run inside **tmux** so it survives closing the web console:

```bash
apt-get update && apt-get install -y tmux    # if not already installed
tmux new -s train                            # persistent session

# inside tmux — note the absolute volume paths:
export VOL=/workspace
uv run python -m src.training.train --config config.runpod.json \
  --checkpoint_dir "$VOL/checkpoints" \
  --rollout_pred_dir "$VOL/output/rollout_pred"
```

Detach with **`Ctrl+b`** then **`d`** (training keeps running; close the tab
safely). Reattach later with `tmux attach -t train`.

What this does (`config.runpod.json`):
- runs up to **2000 epochs** with the **plateau** LR scheduler;
- validates every 2 epochs via full autoregressive rollout RMSE;
- **early-stops** after **25 validations** with no improvement;
- writes to `$VOL/checkpoints/`:
  - `best_model.pt` — best val-rollout-RMSE (the model you deploy),
  - `last_model.pt` — full optimizer/scheduler state, rewritten atomically every
    epoch (the resume anchor),
  - `config.json`, `stats.pt`, `history.json`.

Prefer no tmux? Use `nohup` (logs to a file on the volume):

```bash
nohup uv run python -m src.training.train --config config.runpod.json \
  --checkpoint_dir "$VOL/checkpoints" --rollout_pred_dir "$VOL/output/rollout_pred" \
  > "$VOL/train.log" 2>&1 &
tail -f "$VOL/train.log"
```

Tune from the CLI without editing the file (CLI overrides JSON), e.g.:

```bash
uv run python -m src.training.train --config config.runpod.json \
  --checkpoint_dir "$VOL/checkpoints" --rollout_pred_dir "$VOL/output/rollout_pred" \
  --epochs 4000 --early_stop_patience 40 --hidden_dim 192 --num_processing_steps 12
```

## 4. After a preemption — resume

Relaunch the pod with the **same network volume** attached, then rerun **with the
same `--checkpoint_dir` and `--resume`**:

```bash
export VOL=/workspace
cd "$VOL/meshgraphnet-welding"
tmux new -s train
uv run python -m src.training.train --config config.runpod.json \
  --checkpoint_dir "$VOL/checkpoints" \
  --rollout_pred_dir "$VOL/output/rollout_pred" \
  --resume
```

It reloads `$VOL/checkpoints/last_model.pt` (model + optimizer + scheduler +
epoch + best score + history) and continues where it left off. Resume targets the
`plateau`/`cosine` schedulers; `onecycle` does not resume cleanly (its LR curve
is fixed to the original total step count).

## 5. Retrieve the trained model

```bash
# RunPod CLI (gives a one-time code to receive on your machine):
runpodctl send /workspace/checkpoints/best_model.pt \
  /workspace/checkpoints/stats.pt /workspace/checkpoints/config.json
```

Or download `/workspace/checkpoints/` from the RunPod file browser. `best_model.pt`
is self-contained (bundles its `config`), and with `stats.pt` you can run
inference / rollouts locally via `src.training.rollout`.

---

### Notes
- FEM dataset generation (`src.simulation.generate_dataset`) is **CPU-bound** —
  do it locally and commit the `.npz`, not on the paid GPU pod.
- To train on a larger dataset later: generate more sims locally, commit the new
  `data/raw/*.npz`, and re-clone on the pod.
