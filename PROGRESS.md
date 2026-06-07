# PROGRESS.md — `meshgraphnet_welding`

> Persistent project memory. Update this file as modules land so future sessions
> can resume without re-deriving context.

_Last updated: 2026-06-01_

> **Status: training pipeline fully implemented.** The end-to-end surrogate —
> FEM solver → PyG graphs → simulation-level split + noise injection →
> MeshGraphNet → AdamW training with autoregressive-rollout-RMSE validation and
> self-contained checkpoints — is complete and trainable headless via
> `uv run python -m src.training.train --config config.json`. All **50 tests
> pass** (`uv run pytest`). Remaining work is non-blocking: RunPod job
> orchestration, scaled dataset generation, and visualization notebooks.

## 1. Architecture & Tech Stack

A **SciML Geometric Deep Learning surrogate** for **2D transient thermal
welding simulation**. A finite-element solver generates ground-truth weld
thermal fields under a moving heat source; those snapshots are converted into
graphs and used to train a **MeshGraphNet** that predicts the per-step
temperature evolution — a fast, learned replacement for repeated FEM runs.

Pipeline:

```
scikit-fem FEM solver        PyG graph dataset          MeshGraphNet
(transient heat eq. +   -->  (relative features,   -->  (Encoder ->        -->  RunPod
 moving Goldak source)        ΔT targets)                Processor->Decoder)     GPU training
```

**Tech stack**
- **Package/env manager:** `uv` (Python 3.12). Torch wheels are platform-conditional
  (CPU on Windows dev, CUDA 12.4 on Linux/RunPod) via `[tool.uv.sources]`.
- **FEM / meshing:** `scikit-fem` (12.0.1), `meshio` (+ `h5py` for XDMF), `scipy`, `numpy`.
- **Graph ML:** `torch`, `torch_geometric` (2.7).
- **Viz / IO:** `matplotlib`, XDMF time-series for ParaView, `.npz` for ML ingestion.

**Physical model (locked decisions)**
- **Top-down plate-surface 2D model**: x–y is the plate seen from above; the torch
  moves in-plane. The 3D Goldak double-ellipsoid is integrated analytically over
  through-thickness → 2D surface flux, divided by `thickness` for a volumetric source.
- θ-method time integration (default backward Euler); latent heat via the
  **apparent heat capacity** method (Gaussian bump over the solidus–liquidus range),
  making the system nonlinear → solved with Picard iteration.
- Boundary conditions: Dirichlet and Robin (convection + linearized radiation).

## 2. File Structure & Implementation Status

| Path | Purpose | Status |
|------|---------|--------|
| `src/simulation/thermal_solver.py` | FEM transient thermal solver, Goldak source, trajectories, `SimulationResult` (npz + XDMF) | ✅ **Functional** |
| `src/simulation/generate_dataset.py` | Diversified multi-sim dataset generator (randomized geometry / trajectory / process / BC) → `data/raw/*.npz`; argparse CLI + tqdm | ✅ **Functional** |
| `src/data/graph_builder.py` | FEM snapshots → PyG `Data` graphs; `WeldingGraphDataset` (lazy on-disk) + normalization | ✅ **Functional** |
| `src/models/meshgraphnet.py` | MeshGraphNet (Encoder–Processor–Decoder), `MeshGraphNetConfig` | ✅ **Functional** |
| `src/training/utils.py` | Simulation-level splitter + training-noise injection + transform-aware subsets | ✅ **Functional** |
| `src/training/rollout.py` | Autoregressive rollout evaluation engine + RMSE metrics + viz export | ✅ **Functional** |
| `src/training/train.py` | Training orchestration: config, DataLoader, AdamW + LR sched (onecycle/cosine/**plateau**), noise, rollout-RMSE checkpointing, **early stopping + resume**, CLI | ✅ **Functional** |
| `config.runpod.json` + `RUNPOD.md` | Long-run config (plateau + early stop) and RunPod deploy/resume runbook | ✅ **Functional** |
| `notebooks/` | Visualization & experimentation | ⏳ Empty (scaffold) |
| `tests/test_thermal_solver.py` | Solver unit tests + runnable demo (`__main__`) | ✅ 6 tests |
| `tests/test_graph_builder.py` | Graph pipeline tests | ✅ 9 tests |
| `tests/test_meshgraphnet.py` | Model tests | ✅ 17 tests |
| `tests/test_training_utils.py` | Split + noise-injection tests | ✅ 8 tests |
| `tests/test_rollout.py` | Autoregressive rollout tests (incl. oracle exactness) | ✅ 5 tests |
| `tests/test_train.py` | Training loop + checkpoint + CLI-config + **early-stop/plateau/resume** tests | ✅ 11 tests |

**Test status:** `uv run pytest -q` → **50 passed**.

Verified end-to-end demo: `uv run python tests/test_thermal_solver.py` writes
`.npz` + `.xdmf` + `.png` to `data/raw/` (git-ignored) and shows the expected
comet-shaped, rear-trailing weld pool.

> ⚠️ **Regenerate stale raw files.** Any `.npz` produced *before* the metadata
> change (e.g. an old `data/raw/plate_straight_weld.npz`) lacks embedded
> Goldak/BC metadata and will raise in `build_graph_sequence`. Re-run the solver.

## 3. Strict Feature Conventions

**🚫 Absolute spatial coordinates are STRICTLY FORBIDDEN as model inputs.** All
spatial information must be *relative*: the heat source enters via co-moving
coordinates in its local (tangent, normal) frame; mesh geometry enters only via
edge displacement vectors. (Absolute coords are kept in `Data.pos` for
visualization **only** — never in `x`.)

One graph per consecutive snapshot pair `(t, t+1)`.

### Node features — `x`, shape `(N, 16)` (`graph_builder.NODE_FEATURE_NAMES`)

| idx | feature | notes |
|-----|---------|-------|
| 0 | `T` | current temperature `T_i^t` |
| 1 | `q_goldak` | analytical Goldak source evaluated at the node, `q(x_i, t)` |
| 2 | `dx_local` | co-moving relative coord along trajectory **tangent** |
| 3 | `dy_local` | co-moving relative coord along trajectory **normal** |
| 4 | `net_power` | process param: `η·P` |
| 5 | `speed` | process param: welding speed `v` (per-step, from source-position diffs) |
| 6 | `a` | Goldak depth semi-axis |
| 7 | `b` | Goldak half-width |
| 8 | `c_f` | Goldak front semi-axis |
| 9 | `c_r` | Goldak rear semi-axis |
| 10 | `bc_interior` | one-hot node type |
| 11 | `bc_dirichlet` | one-hot node type |
| 12 | `bc_robin` | one-hot node type |
| 13 | `h_conv` | local convection coefficient |
| 14 | `emissivity` | local emissivity |
| 15 | `T_inf` | local ambient temperature |

- Node-type precedence: **Dirichlet > Robin > Interior** (first Robin marker wins at corners).
- Co-moving coords: `[dx', dy'] = [(x_i − p)·t̂, (x_i − p)·n̂]` where `p` is the source
  position and `(t̂, n̂)` the trajectory tangent/normal.

### Edge features — `edge_attr`, shape `(2E, 3)`

| idx | feature |
|-----|---------|
| 0 | `dx` = `(x_i − x_j).x` |
| 1 | `dy` = `(x_i − x_j).y` |
| 2 | `‖u_ij‖` (Euclidean norm) |

- `edge_index` `(2, 2E)` is **bidirectional**, built from triangle connectivity, no self-loops.

### Target — `y`, shape `(N, 1)`
- Temperature increment `ΔT = T^{t+1} − T^t`.

### Normalization
- Raw physical features are stored; `WeldingGraphDataset` computes dataset-wide
  per-feature mean/std into `processed/stats.pt`. `make_normalizer()` →
  `NormalizeTransform` z-scores continuous columns (`NORMALIZE_MASK`; one-hot
  cols 10–12 untouched) and normalizes `y`; `inverse_y()` de-normalizes ΔT at inference.

### Raw-file contract
- `SimulationResult.save_npz` embeds `metadata_json` (Goldak params, thickness,
  ambient, per-marker BC type + values) so each raw `.npz` is self-contained.

## 4. Configurable Hyperparameters (`MeshGraphNetConfig`)

The model strictly follows the **Encoder → Processor → Decoder** paradigm and is
fully config-driven (`src/models/meshgraphnet.py`):

| field | meaning | default |
|-------|---------|---------|
| `node_in_dim` | input node-feature width | 16 |
| `edge_in_dim` | input edge-feature width | 3 |
| `out_dim` | per-node output width (ΔT) | 1 |
| `hidden_dim` | latent width of all node/edge MLPs | 128 |
| `num_mlp_layers` | hidden layers per MLP (encoder/processor/decoder) | 2 |
| `num_processing_steps` | message-passing hops in the Processor | 15 |
| `activation` | `relu` / `silu` / `gelu` / `tanh` / `elu` / `leaky_relu` | `relu` |
| `use_layer_norm` | LayerNorm on encoder/processor MLP outputs (not decoder) | `True` |
| `aggregation` | receiver aggregation: `sum` / `mean` / `max` / `min` | `sum` |

- **Every hop is residual** for both edge (`e ← e + f([e, x_i, x_j])`) and node
  (`x ← x + f([x, Σ e])`) updates — stabilizes deep rollouts.
- Aggregation uses vectorized `torch_geometric.utils.scatter`.
- `forward(x, edge_index, edge_attr) -> (N, out_dim)`.

## 5. Immediate Next Steps

**Done (in `src/training/utils.py`):**
- ✅ **Simulation-level split** — `split_by_simulation` / `make_split_datasets`
  partition by *whole simulation* (never snapshot-level), with seeded
  reproducibility and small-corpus guards.
- ✅ **Training-noise injection** — `TemperatureNoiseInjection` adds
  `N(0, σ²)` to the raw input temperature and corrects the target
  (`ΔT ← ΔT − η`); applied **only** to the train fold, **before** normalization
  (noise is in physical kelvin). σ via `TrainingConfig.noise_std`.
  `TransformedSubset` lets each fold carry its own transform (train = noise +
  normalize; val/test = normalize only).
  - Note: temperature is float32 with a ~300 K baseline → recovering η from the
    input column carries ~1e-4 round-off; negligible for training.

- ✅ **Autoregressive rollout** (`src/training/rollout.py`) —
  `run_autoregressive_rollout(model, result, normalizer)` rolls predicted ΔT
  forward over a full simulation. Inputs are built with the training-identical
  `build_graph_sequence`; since the trajectory is *scheduled*, all time-dependent
  features (Goldak field, co-moving coords, speed) are precomputed and **only the
  temperature column is fed back** each step. Normalizes node features, de-normalizes
  ΔT via `inverse_y`, accumulates `T^{t+1}=T^t+ΔT`. Reports per-step + overall RMSE
  vs. FEM. `RolloutResult.to_simulation_result()` re-wraps the prediction for the
  existing npz/XDMF viz tooling. Verified exact via an oracle-model test.
  - Edge features are passed **raw** (the normalizer only standardizes node `x`/`y`),
    matching the training contract — important consistency detail.

- ✅ **Training loop** (`src/training/train.py`) — `TrainConfig` (JSON + argparse,
  CLI overrides JSON); loads `WeldingGraphDataset`, simulation-level split,
  `make_split_datasets` folds (noise on train only), PyG `DataLoader`; AdamW +
  OneCycleLR/CosineAnnealingLR; MSE on normalized ΔT; **validation = full
  autoregressive rollout RMSE** (not single-step loss) as the checkpoint metric;
  saves `best_model.pt` + `config.json` + `stats.pt` + `history.json` to
  `checkpoints/`; tqdm + clean timestamped logging for headless RunPod.
  Entry point: `uv run python -m src.training.train --config config.json`
  (sample at `config.example.json`).

**Pending:**
1. ✅ **RunPod orchestration** — long runs driven by **early stopping** (patience
   on val-rollout-RMSE) with a **plateau** LR schedule; **resumable** via atomic
   `last_model.pt` + `--resume` for spot-instance preemption. Config at
   `config.runpod.json`; deploy/resume runbook in `RUNPOD.md`. The training
   `.npz` dataset is committed to git so the pod `git clone`s code + data in one
   shot. (CUDA 12.4 wheels auto-install on Linux via `uv sync`.)
2. ✅ **Dataset generation at scale** — `src/simulation/generate_dataset.py`
   fans out N randomized simulations (rect / L-shape / holed plates; straight,
   diagonal, sinusoidal, arc paths; randomized net power, speed, Goldak axes,
   convection `h` + ambient `T_inf`, occasional Dirichlet edge). Curved paths
   use `ParametricTrajectory` with analytic tangents; weld paths are validated
   to stay inside the material. Per-sim seeded + resumable; writes
   `data/raw/sim_{train,val}_NNN.npz` with a `generation` provenance block.
   Run: `uv run python -m src.simulation.generate_dataset --num_train 20 --num_val 5`.
3. **Visualization notebooks** in `notebooks/` (rollout vs. ground-truth animations).

**Backlog / nice-to-have:**
- Generate a multi-simulation dataset (vary power, speed, Goldak geometry, paths
  including curved trajectories) for generalization.
- Visualization notebooks in `notebooks/` (rollout vs. ground-truth animations).
- Optional PyG compiled accel kernels (`pyg-lib`, `torch-scatter`) on RunPod if profiling warrants.
