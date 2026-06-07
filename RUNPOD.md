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
- **Strongly recommended:** attach a **Network Volume** and mount it so that
  `checkpoint_dir` lives on it (e.g. clone into the volume, or set
  `--checkpoint_dir /workspace/volume/checkpoints`). Checkpoints then survive
  pod death and `--resume` just works after a relaunch.

## 2. Set up the pod

```bash
# In the pod's web terminal:
git clone https://github.com/<user>/meshgraphnet-welding.git
cd meshgraphnet-welding

# Install uv, then sync deps (installs CUDA 12.4 torch wheels on Linux):
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env        # or restart the shell
uv sync

# Sanity check (fast):
uv run pytest -q
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## 3. Train

```bash
uv run python -m src.training.train --config config.runpod.json
```

What this does (`config.runpod.json`):
- runs up to **2000 epochs** with the **plateau** LR scheduler;
- validates every 2 epochs via full autoregressive rollout RMSE;
- **early-stops** after **25 validations** with no improvement;
- writes to `checkpoints/`:
  - `best_model.pt` — best val-rollout-RMSE (the model you deploy),
  - `last_model.pt` — full optimizer/scheduler state, rewritten atomically every
    epoch (the resume anchor),
  - `config.json`, `stats.pt`, `history.json`.

Run it in the background so it survives a dropped terminal:

```bash
nohup uv run python -m src.training.train --config config.runpod.json > train.log 2>&1 &
tail -f train.log
```

Tune from the CLI without editing the file (CLI overrides JSON), e.g.:

```bash
uv run python -m src.training.train --config config.runpod.json \
  --epochs 4000 --early_stop_patience 40 --hidden_dim 192 --num_processing_steps 12
```

## 4. After a preemption — resume

If the pod was killed, relaunch it (re-mount the same network volume, or
re-clone if `checkpoints/` was on the volume) and rerun **with `--resume`**:

```bash
uv run python -m src.training.train --config config.runpod.json --resume
```

It reloads `last_model.pt` (model + optimizer + scheduler + epoch + best score +
history) and continues where it left off. Resume targets the `plateau`/`cosine`
schedulers; `onecycle` does not resume cleanly (its LR curve is fixed to the
original total step count).

## 5. Retrieve the trained model

```bash
# RunPod CLI (gives a one-time code to receive on your machine):
runpodctl send checkpoints/best_model.pt checkpoints/stats.pt checkpoints/config.json
```

Or download `checkpoints/` from the RunPod file browser. `best_model.pt` is
self-contained (bundles its `config`), and with `stats.pt` you can run inference
/ rollouts locally via `src.training.rollout`.

---

### Notes
- FEM dataset generation (`src.simulation.generate_dataset`) is **CPU-bound** —
  do it locally and commit the `.npz`, not on the paid GPU pod.
- To train on a larger dataset later: generate more sims locally, commit the new
  `data/raw/*.npz`, and re-clone on the pod.
