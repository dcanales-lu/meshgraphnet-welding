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
