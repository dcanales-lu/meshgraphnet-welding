# Running on RunPod (cloud GPU) — step by step

A no-Docker workflow that mirrors local training: spin up a standard GPU pod,
clone the repo onto the persistent `/workspace` volume, reproduce the env with
`uv`, run **detached**, and **auto-stop the pod** when training finishes so you
don't pay for idle GPU time. Resumable across spot-instance preemption.

> The codebase is already cloud-ready: matplotlib runs headless
> (`matplotlib.use("Agg")`), all paths are relative, `pyproject.toml` selects the
> Linux **cu128** torch wheels, and `uv.lock` + `.python-version` (3.12) pin the
> exact environment.

**Why a network volume?** Only the volume (mounted at `/workspace`) survives a
pod stop / terminate / preemption — the container disk is wiped. Everything the
run writes (checkpoints, logs, the processed-graph cache) lives on it.

---

## Step 0 — Make the repo cloneable (one-time, on your machine)

If the repo is **private**, the pod can't clone it without auth. Either make it
public (it's just code + dataset, no secrets):

```bash
gh repo edit dcanales-lu/meshgraphnet-welding --visibility public --accept-visibility-change-consequences
```

…or keep it private and use a token in Step 3 (variant shown there).

**Push the current state** so the pod gets the latest code, the new 125-sim
dataset, and the cloud scripts:

```bash
git add data/raw/*.npz config.runpod.json config.local_gpu*.json src scripts RUNPOD.md
git commit -m "Cloud workflow: refreshed runpod config + auto-shutdown wrapper"
git push
```

(The dataset is committed via a `.gitignore` exception so one `git clone` brings
**code + data**. Alternatively, skip committing data and regenerate it on the
pod — see Step 3b — since the generator is seeded and deterministic.)

---

## Step 1 — Create the pod on runpod.io

1. **Pods → Deploy.**
2. **GPU:** the model is tiny (~1.3M params); the real bottleneck is **data
   loading**, so prefer a pod with **≥8 vCPUs**. An RTX 4090 / A5000 / L40S is
   plenty. Pick a bigger GPU only if you also scale `hidden_dim` / `batch_size` /
   dataset.
3. **Template:** any official **PyTorch / CUDA 12.x** template (recent NVIDIA
   driver). `uv` pulls the **cu128** torch wheels regardless of the base image.
4. **Network Volume:** create/attach one (~20 GB) mounted at **`/workspace`**.
5. **Instance type:** **Spot/Interruptible** is cheaper and the run resumes after
   a kill (Step 6). Use On-Demand if you don't want interruptions.
6. **Deploy** and wait for **Running**.

---

## Step 2 — Open the terminal

Pod card → **Connect** → **Start Web Terminal**. (Or use the SSH-over-TCP command
for your own terminal.)

---

## Step 3 — Set up (copy-paste the whole block)

```bash
cd /workspace

# Clone code + data onto the volume:
git clone https://github.com/dcanales-lu/meshgraphnet-welding.git
cd meshgraphnet-welding

# Install uv + reproduce the EXACT env from uv.lock (pulls cu128 torch on Linux):
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync --frozen

# Confirm the GPU is visible (must print True + the card name):
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If CUDA is `False`, the driver/template is wrong — redeploy with a CUDA 12.x
template. `.venv/` lives on the volume, so on later pods `uv sync --frozen` is a
near-no-op.

> **Private-repo variant:** replace the clone line with
> `git clone https://<TOKEN>@github.com/dcanales-lu/meshgraphnet-welding.git`
> (GitHub → Settings → Developer settings → fine-grained token, read access).

**Step 3b — regenerate data on the pod (only if you did NOT commit it):**

```bash
uv run python -m src.simulation.generate_dataset --num_train 100 --num_val 25 --target_steps 300
```

The `data/processed/` graph cache builds automatically on the first training run.

---

## Step 4 — Launch training, detached + auto-shutdown

Use the wrapper `scripts/runpod_train.sh`: it runs `uv sync` → training →
optional eval, then **stops the pod on any exit** (success, crash, or sync error)
via `runpodctl` (pre-installed on pods; reads `RUNPOD_POD_ID` automatically).
Run it inside **tmux** so it survives closing the web console:

```bash
apt-get update && apt-get install -y tmux     # if tmux isn't already present
tmux new -s train

# inside tmux:
bash scripts/runpod_train.sh config.runpod.json
```

**Detach:** press **`Ctrl+b`** then **`d`** — training keeps running; close the
browser tab safely. **Reattach:** `tmux attach -t train`.

Within a minute you'll see:

```
DataSplit(train: 94 sims / ~32k graphs, val: 31 sims / ~11k graphs, ...)
MeshGraphNet on cuda | 1291777 params | 8 processing steps
epoch   5/100 | train_mse 9.7e-02 | val_rollout_rmse 147.7 K  <- best
```

What `config.runpod.json` does (tuned cloud recipe): up to **100 epochs**,
**cosine** LR decay, validates every 5 epochs via full autoregressive-rollout
RMSE, de-noised validation (`val_fraction 0.25`), `batch_size 16`,
`num_workers 8`, `progress_bar false` (clean logs). Override any field from the
CLI, e.g. `--hidden_dim 192 --num_processing_steps 12 --epochs 150`.

**No-tmux alternative (`nohup`):**

```bash
nohup bash scripts/runpod_train.sh config.runpod.json > logs/nohup.out 2>&1 &
tail -f logs/nohup.out
```

### Auto-shutdown controls (cost optimization)

The wrapper waits a 30 s grace window (Ctrl-C to cancel if attached), then:

```bash
# Default — stop the pod (GPU billing ends, resumable):
bash scripts/runpod_train.sh config.runpod.json

# Terminate fully — ends ALL pod billing; /workspace volume persists:
SHUTDOWN_ACTION=remove bash scripts/runpod_train.sh config.runpod.json

# Stay up afterwards (debugging):
SHUTDOWN_ACTION=none bash scripts/runpod_train.sh config.runpod.json

# Run a spiral eval after training, before shutdown:
POST_TRAIN_CMD='uv run python -m src.training.spiral_rollout --checkpoint checkpoints/best_model.pt --output_name spiral_runpod --device cuda && uv run python -m src.training.spiral_analysis --pred data/output/spiral_runpod.npz --prefix spiral_runpod' \
  bash scripts/runpod_train.sh config.runpod.json
```

**`stop` vs `remove`:** `stop` halts the GPU (you still pay a small idle disk
rate; resume the same pod later). `remove` terminates the pod entirely (zero pod
cost; the `/workspace` network volume is billed separately and keeps code +
checkpoints + logs). For unattended overnight runs you won't resume, `remove` is
cheapest.

> Backstop: if `runpodctl` is ever missing/unauthenticated the wrapper logs a
> warning and leaves the pod up — **verify in the console**, and set a pod
> spending/runtime limit as a safety net.

---

## Step 5 — Monitor

```bash
tail -f logs/train_*.log     # one line per epoch (+ val_rollout_rmse on val epochs)
nvidia-smi -l 5              # GPU utilization / memory
tmux attach -t train         # full live view; Ctrl+b d to detach again
```

---

## Step 6 — Resume after a spot kill

Relaunch a pod with the **same network volume**, open a terminal, set
`"resume": true` in `config.runpod.json` (or pass `--resume`), and rerun:

```bash
cd /workspace/meshgraphnet-welding
tmux new -s train
bash scripts/runpod_train.sh config.runpod.json   # with resume:true in the config
```

You'll see `Resumed from .../last_model.pt at epoch N`. Resume targets
`plateau`/`cosine`; `onecycle` does not resume cleanly (fixed LR curve).

---

## Step 7 — Download the trained model

Checkpoints/logs/plots are git-ignored, so pull them off the pod:

```bash
# On the pod — prints a one-time code:
runpodctl send checkpoints/best_model.pt checkpoints/stats.pt checkpoints/config.json

# On your laptop:
runpodctl receive <code>
```

`best_model.pt` is self-contained (bundles its `config`); with `stats.pt` you can
run rollouts locally via `src.training.rollout`. If you use
`SHUTDOWN_ACTION=remove`, fetch results via `POST_TRAIN_CMD` + `runpodctl send`
**before** the run ends, or just leave them on `/workspace` and re-mount the
volume next time.

---

## Step 8 — Cost notes

- Auto-shutdown (Step 4) means you normally **don't** stop the pod manually.
- FEM dataset generation is **CPU-bound** — do it locally and commit the `.npz`,
  or regenerate on the pod once (Step 3b); don't burn GPU hours on it.
- To scale up: bump `hidden_dim` 128→192/256, `num_processing_steps` 8→10–12,
  `batch_size`/`num_workers` to fit the GPU/vCPUs, or generate more/longer sims.
