# meshgraphnet_welding

A **MeshGraphNet** surrogate for **2D transient welding thermal simulation**.

The project learns a fast, mesh-based graph neural network surrogate that
reproduces finite-element welding temperature fields driven by a moving
**Goldak double-ellipsoid** heat source — replacing repeated expensive FEM runs
with near-real-time inference.

## Pipeline

```
scikit-fem FEM solver        PyG graph dataset          MeshGraphNet
(transient heat eq. +   -->  (nodes/edges, coord   -->  (Encoder ->        -->  RunPod
 moving Goldak source)        transforms, norm)          Processor ->            GPU training
                                                         Decoder)                & deployment
```

1. **Simulate** (`src/simulation/`) — scikit-fem transient thermal solver with
   Goldak moving-source kinematics generates ground-truth temperature fields.
2. **Build graphs** (`src/data/`) — mesh snapshots become PyTorch Geometric
   `Data` graphs with engineered node/edge features and coordinate transforms.
3. **Model** (`src/models/`) — MeshGraphNet (Encoder–Processor–Decoder) message
   passing over the mesh graph.
4. **Train** (`src/training/`) — training loop with training-noise injection for
   stable autoregressive rollouts, plus RunPod orchestration.

## Project structure

```
.
├── src/
│   ├── simulation/   # FEM transient thermal solver + Goldak heat source
│   ├── data/         # graph dataset creation, coord transforms, PyG loaders
│   ├── models/       # MeshGraphNet: Encoder, Processor, Decoder
│   └── training/     # training loop, noise injection, RunPod orchestration
├── notebooks/        # visualization & experimentation
├── tests/            # unit tests
└── pyproject.toml
```

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
```

This creates `.venv` and installs all dependencies. Run anything with `uv run`:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run pytest
```

## Quickstart

```bash
uv sync                                  # install deps into .venv
uv run pytest -q                         # sanity check (all tests pass)

# 1. Generate a dataset (CPU; FEM solver) — writes data/raw/*.npz
uv run python -m src.simulation.generate_dataset --num_train 20 --num_val 5

# 2. Train (CPU locally, or GPU on RunPod) — writes checkpoints/
uv run python -m src.training.train --config config.example.json
```

A ready-to-train dataset (20 train / 5 val sims, ~19 MB) is **committed** under
`data/raw/`, so you can skip step 1 and train immediately. The `.h5`/`.xdmf`
ParaView exports and `data/output/` are git-ignored (regenerable).

## Dataset generation

`src/simulation/generate_dataset.py` fans out randomized simulations — rectangular
/ L-shape / holed plates; straight, diagonal, sinusoidal and arc weld paths;
randomized net power, speed, Goldak geometry, convection and ambient — validating
that each weld path stays inside the material. Runs are per-sim seeded and
resumable, and each `.npz` embeds a provenance block.

```bash
uv run python -m src.simulation.generate_dataset --num_train 100 --num_val 20
```

## Training

```bash
uv run python -m src.training.train --config config.example.json
# every config field is also a CLI flag (CLI overrides JSON):
uv run python -m src.training.train --epochs 500 --noise_std 2.0 --hidden_dim 128
```

Key features of the training loop (`src/training/train.py`):

- **Simulation-level split** — whole simulations go to train/val/test (never
  snapshot-level), so scores measure real geometric/parameter generalization.
- **Training-noise injection** — Gaussian noise on the input temperature with a
  corrected target (Pfaff et al.), for stable autoregressive rollouts.
- **Validation = full autoregressive-rollout RMSE** (not single-step loss); this
  is the checkpoint-selection metric.
- **LR schedulers:** `onecycle` / `cosine` / `plateau` / `none`.
- **Early stopping** — `--early_stop_patience` (in validation units) ends long,
  open-ended runs once val-rollout-RMSE plateaus.
- **Resumable checkpoints** — `last_model.pt` (model + optimizer + scheduler +
  history) is written atomically every epoch; `--resume` continues from it
  (designed for spot/preemptible GPUs).

Outputs in `checkpoint_dir/`: `best_model.pt` (self-contained — bundles its
config), `last_model.pt`, `stats.pt` (normalization), `config.json`,
`history.json`.

## Training on a GPU (RunPod)

CUDA wheels install automatically on Linux (see below), so a pod just needs
`git clone` + `uv sync` + train. See **[`RUNPOD.md`](RUNPOD.md)** for the full
step-by-step guide: creating the pod, running detached in `tmux`, storing all
artifacts on the persistent network volume, resuming after preemption, and
downloading the trained model. Long-run defaults live in `config.runpod.json`
(plateau schedule, 2000-epoch budget, early stopping).

## Tests

```bash
uv run pytest -q
```

Covers the FEM solver, graph builder, model, split/noise utilities,
autoregressive rollout (incl. an oracle-exactness check), and the training loop
(checkpointing, early stopping, plateau scheduler, resume).

## PyTorch / CUDA

Torch wheels are selected automatically per platform via `[tool.uv.sources]`:

| Platform              | Wheels             |
| --------------------- | ------------------ |
| Windows (local dev)   | CPU                |
| Linux (RunPod / GPU)  | CUDA 12.4 (cu124)  |

So `uv sync` "just works" on both your local machine and a RunPod Linux box —
no manual index switching.

**Optional GPU acceleration:** PyTorch Geometric's core is pure-Python and
installed by default. For large graphs you can add the compiled scatter/sparse
kernels **on RunPod** (matched to the cu124 torch build), e.g.:

```bash
uv pip install pyg-lib torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu124.html
```

## License

TBD.
