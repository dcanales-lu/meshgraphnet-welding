# Launching training on RunPod — step by step

A complete walkthrough to run the MeshGraphNet welding surrogate on a RunPod GPU
pod: from the RunPod website to a training run going on the persistent volume,
detached so it survives closing the console, and resumable across spot-instance
preemption.

The dataset (`data/raw/*.npz`, ~19 MB) is committed to the repo, so a single
`git clone` brings **code + data** to the pod.

**Why a network volume?** Only the volume (mounted at `/workspace`) survives a
pod stop/preemption — the container disk is wiped. Everything the run *writes*
(checkpoints, rollout exports, the processed-graph cache) must live on it. The
steps below clone into `/workspace` **and** point `--checkpoint_dir` at an
absolute volume path outside the repo, so the model survives even a re-clone.

---

## Step 0 — Make the repo cloneable (one-time, on your machine)

If the repo is **private**, the pod can't clone it without auth. Easiest fix —
make it public (it's just code + the dataset, no secrets):

```bash
gh repo edit dcanales-lu/meshgraphnet-welding --visibility public --accept-visibility-change-consequences
```

Keeping it private? Skip this and use a token in Step 3 (variant shown there).

---

## Step 1 — Create the pod on runpod.io

1. Log in → **Pods** (or "GPU Cloud") → **Deploy**.
2. **GPU:** one mid-range card — **RTX 4090** or **A5000** is plenty
   (`hidden_dim=128`, 8 message-passing steps).
3. **Template:** a **PyTorch 2.x / CUDA 12.4** template (e.g. "RunPod PyTorch 2.4").
4. **Network Volume:** create/attach one (e.g. 20 GB). It mounts at **`/workspace`**
   — the storage that survives preemption.
5. **Instance type:** **Spot/Interruptible** (cheaper; the run resumes after a kill).
6. **Deploy**, and wait until the pod status is **Running**.

---

## Step 2 — Open the terminal

On the pod card → **Connect** → **Start Web Terminal** → **Connect to Web
Terminal**. (Or use the "SSH over exposed TCP" command for your own terminal.)

---

## Step 3 — Set up (copy-paste the whole block)

```bash
export VOL=/workspace
mkdir -p "$VOL/checkpoints" "$VOL/output"
cd "$VOL"

# Clone code + data onto the volume:
git clone https://github.com/dcanales-lu/meshgraphnet-welding.git
cd meshgraphnet-welding

# Install uv + dependencies (auto-selects CUDA 12.4 torch wheels on Linux):
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync

# Confirm the GPU is visible:
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

The last line must print `CUDA: True`. If it prints `False`, the template isn't a
CUDA one — redeploy with a PyTorch CUDA 12.4 template.

> **Private-repo variant:** replace the clone line with
> `git clone https://<TOKEN>@github.com/dcanales-lu/meshgraphnet-welding.git`
> (GitHub → Settings → Developer settings → fine-grained token, read access).

---

## Step 4 — Launch training, detached

Run inside **tmux** so it survives closing the web console:

```bash
apt-get update && apt-get install -y tmux     # if tmux isn't already there
tmux new -s train
```

Inside the tmux session, start the run (artifacts go to the volume):

```bash
export VOL=/workspace
uv run python -m src.training.train --config config.runpod.json \
  --checkpoint_dir "$VOL/checkpoints" \
  --rollout_pred_dir "$VOL/output/rollout_pred"
```

Within a minute you should see lines like:

```
DataSplit(train: 20 sims / 3790 graphs, val: 5 sims / 993 graphs, ...)
MeshGraphNet on cuda | ... params | 8 processing steps
epoch   2/2000 | train_mse 2.1e-01 | val_rollout_rmse 206.6 K  <- best
```

**Detach the console:** press **`Ctrl+b`**, then **`d`**. Training keeps
running — close the browser tab safely.

What this run does (`config.runpod.json`):
- up to **2000 epochs** with the **plateau** LR scheduler;
- validates every 2 epochs via full autoregressive-rollout RMSE;
- **early-stops** after **25 validations** with no improvement;
- writes to `$VOL/checkpoints/`: `best_model.pt` (deploy this), `last_model.pt`
  (atomic resume anchor), `config.json`, `stats.pt`, `history.json`.

Tune from the CLI without editing the file (CLI overrides JSON), e.g.
`--epochs 4000 --early_stop_patience 40 --hidden_dim 192 --num_processing_steps 12`.

Prefer no tmux? Use `nohup` (logs to the volume):

```bash
nohup uv run python -m src.training.train --config config.runpod.json \
  --checkpoint_dir "$VOL/checkpoints" --rollout_pred_dir "$VOL/output/rollout_pred" \
  > "$VOL/train.log" 2>&1 &
tail -f "$VOL/train.log"
```

---

## Step 5 — Check back / monitor

Open a web terminal any time and reattach:

```bash
tmux attach -t train          # watch live; Ctrl+b then d to detach again
```

Or peek without attaching:

```bash
cat /workspace/checkpoints/history.json
ls -la /workspace/checkpoints/
```

---

## Step 6 — If the spot pod gets killed, resume

Relaunch a pod with the **same network volume** attached, open a terminal, then
rerun with the **same `--checkpoint_dir`** plus `--resume`:

```bash
export VOL=/workspace
cd "$VOL/meshgraphnet-welding"
tmux new -s train
uv run python -m src.training.train --config config.runpod.json \
  --checkpoint_dir "$VOL/checkpoints" \
  --rollout_pred_dir "$VOL/output/rollout_pred" \
  --resume
```

You'll see `Resumed from .../last_model.pt at epoch N` and it continues. Resume
targets the `plateau`/`cosine` schedulers; `onecycle` does not resume cleanly
(its LR curve is fixed to the original total step count).

---

## Step 7 — Download the trained model

```bash
runpodctl send /workspace/checkpoints/best_model.pt \
  /workspace/checkpoints/stats.pt /workspace/checkpoints/config.json
```

This prints a one-time code; run the matching `runpodctl receive <code>` on your
laptop. (Or download `/workspace/checkpoints/` from the RunPod file browser.)
`best_model.pt` is self-contained (bundles its `config`); with `stats.pt` you can
run inference / rollouts locally via `src.training.rollout`.

---

## Step 8 — Stop the pod

When training is done and the model is downloaded, **Stop/Terminate** the pod so
you stop paying. Your artifacts remain on the network volume.

---

### Notes
- FEM dataset generation (`src.simulation.generate_dataset`) is **CPU-bound** —
  do it locally and commit the `.npz`, not on the paid GPU pod.
- To train on a larger dataset later: generate more sims locally, commit the new
  `data/raw/*.npz`, and re-clone (or `git pull`) on the pod.
- The repo lives on the volume, so the processed-graph cache it writes under
  `data/processed/` is persistent too. If you instead clone onto the container
  disk, add `--data_root "$VOL/data"` to keep that cache on the volume.
