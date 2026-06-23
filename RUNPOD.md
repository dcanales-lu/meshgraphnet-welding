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

**Push the current state** so the pod gets the latest code, configs and scripts:

```bash
git add -A && git commit -m "Cloud run" && git push
```

**The dataset is NOT in git** (the v2 corpus is ~6.8 GB raw + ~274 GB processed
cache — too large). It is **regenerated on the pod** from the seeded, deterministic
generator (Step 3b) — identical bytes, no large transfer. (Only code + configs +
`uv.lock` + docs are committed.)

---

## Step 1 — Create the pod on runpod.io

1. **Pods → Deploy.**
2. **GPU:** the model is ~1.3M params but the v2 graphs are large (≤6642 nodes)
   and the **push-forward K=2** run unrolls 2 steps. A **40–80 GB** card
   (A100 / L40S / H100) lets you raise `batch_size` (8 on 40 GB → 16+ on 80 GB).
   Prefer **≥16 vCPUs** (data generation + loading are CPU-bound).
3. **Template:** any official **PyTorch / CUDA 12.x** template. `uv` pulls the
   **cu128** torch wheels regardless of the base image.
4. **Network Volume: ≥320 GB** mounted at **`/workspace`** — the v2 corpus needs
   ~6.8 GB raw + **~274 GB** processed `.pt` cache (built on first training run),
   plus checkpoints/logs. (The cache is large because graph topology is stored
   per-timestep; a leaner on-the-fly loader is a known future optimization.)
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

# Clone the code onto the volume (data is regenerated in Step 3b, not cloned):
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

**Step 3b — regenerate the dataset on the pod (required — it is not in git):**

```bash
# Deterministic (seed 0) → identical to the local v2 corpus. ~2 h on ~16 cores.
uv run python -m src.simulation.generate_dataset --num_train 300 --num_val 60 --workers 16
```

The `data/processed/` graph cache (~274 GB, ~411k files) builds automatically on
the first training run (~30–60 min) — this is why the volume must be ≥320 GB.

---

## Step 4 — Launch training, detached + auto-shutdown

Use the wrapper `scripts/runpod_train.sh`: it runs `uv sync` → training →
optional eval, then **stops the pod on any exit** (success, crash, or sync error)
via `runpodctl` (pre-installed on pods; reads `RUNPOD_POD_ID` automatically).
Run it inside **tmux** so it survives closing the web console:

```bash
apt-get update && apt-get install -y tmux     # if tmux isn't already present
tmux new -s train

# inside tmux (the headline experiment: enthalpy GENERIC + enriched source +
# push-forward K=2):
bash scripts/runpod_train.sh config.runpod_enthalpy_pf.json
```

**Detach:** press **`Ctrl+b`** then **`d`** — training keeps running; close the
browser tab safely. **Reattach:** `tmux attach -t train`.

Within a couple of minutes (after the one-time graph-cache build) you'll see:

```
Push-forward: K=2 | ~317k training windows from 288 sims | 40000/epoch (subsampled)
GENERIC thermodynamic head enabled (physical-units).
MeshGraphNet on cuda | ~1.4M params | 8 processing steps
epoch   5/80 | train_mse ... | val_rollout_rmse ... K  <- best
```

What `config.runpod_enthalpy_pf.json` does: the thermodynamically-consistent
**enthalpy** GENERIC head with the **enriched source**, **push-forward K=2**
(multi-step training — the lever that aligns the objective with long rollouts),
plateau LR, `batch_size 8` (raise to 16+ on an 80 GB card), `max_windows_per_epoch
40000`, `max_val_sims 20`, `num_workers 16`, 80 epochs. Override any field from
the CLI, e.g. `--pushforward_steps 3 --batch_size 16 --epochs 120`.

**No-tmux alternative (`nohup`):**

```bash
nohup bash scripts/runpod_train.sh config.runpod_enthalpy_pf.json > logs/nohup.out 2>&1 &
tail -f logs/nohup.out
```

### Auto-shutdown controls (cost optimization)

The wrapper waits a 30 s grace window (Ctrl-C to cancel if attached), then:

```bash
# Default — stop the pod (GPU billing ends, resumable):
bash scripts/runpod_train.sh config.runpod_enthalpy_pf.json

# Terminate fully — ends ALL pod billing; /workspace volume persists:
SHUTDOWN_ACTION=remove bash scripts/runpod_train.sh config.runpod_enthalpy_pf.json

# Stay up afterwards (debugging):
SHUTDOWN_ACTION=none bash scripts/runpod_train.sh config.runpod_enthalpy_pf.json

# Run a spiral eval after training, before shutdown:
POST_TRAIN_CMD='uv run python -m src.training.spiral_rollout --checkpoint checkpoints_enthsrc_pf/best_model.pt --output_name spiral_runpod --device cuda && uv run python -m src.training.spiral_analysis --pred data/output/spiral_runpod.npz --prefix spiral_runpod' \
  bash scripts/runpod_train.sh config.runpod_enthalpy_pf.json
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
`"resume": true` in `config.runpod_enthalpy_pf.json` (or pass `--resume`), and rerun:

```bash
cd /workspace/meshgraphnet-welding
tmux new -s train
bash scripts/runpod_train.sh config.runpod_enthalpy_pf.json   # with resume:true in the config
```

You'll see `Resumed from .../last_model.pt at epoch N`. Resume targets
`plateau`/`cosine`; `onecycle` does not resume cleanly (fixed LR curve).

---

## Step 7 — Download the trained model

Checkpoints/logs/plots are git-ignored, so pull them off the pod:

```bash
# On the pod — prints a one-time code:
runpodctl send checkpoints_enthsrc_pf/best_model.pt checkpoints_enthsrc_pf/stats.pt checkpoints_enthsrc_pf/config.json

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
