"""Graph data pipeline: FEM snapshots -> PyTorch Geometric ``Data`` graphs.

Converts the transient welding snapshots produced by
:mod:`simulation.thermal_solver` (saved as ``.npz`` :class:`SimulationResult`
files) into a sequence of :class:`torch_geometric.data.Data` graphs for the
MeshGraphNet surrogate.

Design principles (strong physical inductive biases)
----------------------------------------------------
* **No absolute spatial coordinates as node features.** All spatial information
  is *relative*: the heat-source position enters through co-moving coordinates
  expressed in the trajectory's local (tangent, normal) frame, and mesh
  geometry enters only through edge displacement vectors.
* One graph per consecutive snapshot pair ``(t, t+1)``: node features describe
  the state at ``t``; the regression target is the temperature increment
  ``ΔT = T^{t+1} − T^t``.
* Every operation (Goldak evaluation, frame rotation, edge construction) is
  vectorized.

Node feature layout (12-d, see :data:`NODE_FEATURE_NAMES`)
----------------------------------------------------------
``[T, q_Goldak, dx', dy', net_power, speed,
   onehot_interior, onehot_dirichlet, onehot_robin, h, emissivity, T_inf]``

The Goldak semi-axes ``a, b, c_f, c_r`` are *global* process constants per
simulation and are deliberately **not** exposed as separate node columns: they
are already folded into the analytical ``q_Goldak`` source field and the
co-moving coordinates ``dx', dy'``. They are still read from metadata to
*compute* those features (see :func:`_goldak_from_metadata`).

Edge feature layout (3-d): ``[dx, dy, ||u_ij||]`` with ``u_ij = x_i − x_j``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from simulation.thermal_solver import GoldakParams, SimulationResult, goldak_flux

# ---------------------------------------------------------------------------
# Feature layout constants
# ---------------------------------------------------------------------------
NODE_FEATURE_NAMES: List[str] = [
    "T",            # 0  current temperature
    "q_goldak",     # 1  analytical Goldak source at the node
    "dx_local",     # 2  co-moving relative coord along tangent
    "dy_local",     # 3  co-moving relative coord along normal
    "net_power",    # 4  process param: eta * P
    "speed",        # 5  process param: welding speed v
    "bc_interior",  # 6  one-hot node type
    "bc_dirichlet", # 7  one-hot node type
    "bc_robin",     # 8  one-hot node type
    "h_conv",       # 9  local convection coefficient
    "emissivity",   # 10 local emissivity
    "T_inf",        # 11 local ambient temperature
]
NUM_NODE_FEATURES = len(NODE_FEATURE_NAMES)
EDGE_FEATURE_NAMES: List[str] = ["dx", "dy", "dist"]
NUM_EDGE_FEATURES = len(EDGE_FEATURE_NAMES)

# Node-type codes (precedence: Dirichlet > Robin > Interior).
NODE_TYPE_INTERIOR = 0
NODE_TYPE_DIRICHLET = 1
NODE_TYPE_ROBIN = 2

# Continuous columns are z-score normalized; one-hot columns (6-8) are not.
NORMALIZE_MASK = np.ones(NUM_NODE_FEATURES, dtype=bool)
NORMALIZE_MASK[[6, 7, 8]] = False


# ---------------------------------------------------------------------------
# Static graph topology
# ---------------------------------------------------------------------------
def build_edges(cells: np.ndarray) -> np.ndarray:
    """Bidirectional ``edge_index (2, 2E)`` from triangle connectivity.

    Each triangle contributes its three edges; duplicates are removed and both
    orientations are emitted (MeshGraphNet convention). No self-loops.
    """
    faces = np.asarray(cells)
    pairs = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    pairs = np.sort(pairs, axis=1)
    undirected = np.unique(pairs, axis=0)  # (E, 2)
    directed = np.concatenate([undirected, undirected[:, ::-1]], axis=0)  # (2E, 2)
    return directed.T  # (2, 2E)


def build_edge_features(coords: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """Edge features ``[dx, dy, ||u_ij||]`` for each directed edge."""
    src, dst = edge_index[0], edge_index[1]
    u = coords[src] - coords[dst]  # (2E, 2)
    dist = np.linalg.norm(u, axis=1, keepdims=True)  # (2E, 1)
    return np.concatenate([u, dist], axis=1)


# ---------------------------------------------------------------------------
# Node-type / boundary-condition context (static across time)
# ---------------------------------------------------------------------------
def build_bc_context(result: SimulationResult):
    """Per-node type code and local BC values from masks + metadata.

    Returns ``(node_type (N,), bc_values (N, 3))`` where ``bc_values`` columns
    are ``[h_conv, emissivity, T_inf]``. Precedence: Dirichlet overrides Robin;
    at a corner shared by several Robin edges the first marker wins.
    """
    n = result.coords.shape[0]
    node_type = np.full(n, NODE_TYPE_INTERIOR, dtype=np.int64)
    bc_values = np.zeros((n, 3), dtype=np.float64)
    specs = result.metadata.get("boundary_specs", {})

    # Robin first (first marker wins at shared nodes).
    for marker, spec in specs.items():
        if spec.get("type") != "robin":
            continue
        mask = result.boundary_masks[marker]
        sel = mask & (node_type == NODE_TYPE_INTERIOR)
        node_type[sel] = NODE_TYPE_ROBIN
        bc_values[sel] = [spec["h_conv"], spec["emissivity"], spec["T_inf"]]

    # Dirichlet overrides (no convective/radiative values).
    for marker, spec in specs.items():
        if spec.get("type") != "dirichlet":
            continue
        mask = result.boundary_masks[marker]
        node_type[mask] = NODE_TYPE_DIRICHLET
        bc_values[mask] = 0.0

    return node_type, bc_values


# ---------------------------------------------------------------------------
# Per-timestep node features
# ---------------------------------------------------------------------------
def _instantaneous_speeds(result: SimulationResult) -> np.ndarray:
    """Welding speed per snapshot pair from source-position differences."""
    dpos = np.diff(result.source_position, axis=0)  # (S-1, 2)
    dt = np.diff(result.times)  # (S-1,)
    dt = np.where(dt == 0.0, 1.0, dt)
    return np.linalg.norm(dpos, axis=1) / dt  # (S-1,)


def build_node_features(
    result: SimulationResult,
    t: int,
    speed: float,
    node_type: np.ndarray,
    bc_values: np.ndarray,
    goldak: GoldakParams,
) -> np.ndarray:
    """Assemble the (N, 12) node-feature matrix for snapshot index ``t``."""
    coords = result.coords
    n = coords.shape[0]
    pos = result.source_position[t]
    tangent = result.source_tangent[t]
    normal = result.source_normal[t]
    thickness = result.metadata["thickness"]
    md = result.metadata["goldak"]

    x = np.zeros((n, NUM_NODE_FEATURES), dtype=np.float64)

    # Temperature.
    x[:, 0] = result.temperature[t]

    # Analytical Goldak source at each node (reuses the solver's flux fn). The
    # flux is gated by the recorded on/off state of the torch: during a post-weld
    # cooling tail ``source_power`` is 0, so the source feature is exactly 0
    # (consistent with the FEM, which also switches the source off). The ratio is
    # 1 while welding and degrades gracefully for any future power ramping.
    net_power = md["net_power"]
    power_ratio = (
        result.source_power[t] / net_power if net_power > 0.0 else 0.0
    )
    x[:, 1] = power_ratio * goldak_flux(
        coords[:, 0], coords[:, 1], pos, tangent, normal, goldak, thickness
    )

    # Co-moving relative coordinates in the source's (tangent, normal) frame.
    rel = coords - pos  # (N, 2)
    x[:, 2] = rel @ tangent
    x[:, 3] = rel @ normal

    # Process parameters broadcast per node.
    x[:, 4] = net_power
    x[:, 5] = speed

    # BC context: one-hot node type + local values.
    x[:, 6] = node_type == NODE_TYPE_INTERIOR
    x[:, 7] = node_type == NODE_TYPE_DIRICHLET
    x[:, 8] = node_type == NODE_TYPE_ROBIN
    x[:, 9:12] = bc_values

    return x


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------
def _goldak_from_metadata(result: SimulationResult) -> GoldakParams:
    md = result.metadata["goldak"]
    return GoldakParams(
        power=md["power"],
        efficiency=md["efficiency"],
        a=md["a"],
        b=md["b"],
        c_f=md["c_f"],
        c_r=md["c_r"],
        f_f=md["f_f"],
    )


def build_graph_sequence(
    result: SimulationResult, sim_id: int = 0
) -> List[Data]:
    """Convert a :class:`SimulationResult` into ``S-1`` PyG graphs.

    Each graph holds ``x (N,12)``, ``edge_index (2,2E)``, ``edge_attr (2E,3)``,
    ``y (N,1)`` (the temperature increment), and ``pos (N,2)`` stored separately
    for visualization only (never used as a node feature).
    """
    if not result.metadata:
        raise ValueError(
            "SimulationResult has no metadata; re-run the solver so the .npz "
            "embeds Goldak/BC parameters (see thermal_solver._build_metadata)."
        )

    coords = result.coords
    edge_index_np = build_edges(result.cells)
    edge_attr_np = build_edge_features(coords, edge_index_np)
    node_type, bc_values = build_bc_context(result)
    goldak = _goldak_from_metadata(result)
    speeds = _instantaneous_speeds(result)

    edge_index = torch.as_tensor(edge_index_np, dtype=torch.long)
    edge_attr = torch.as_tensor(edge_attr_np, dtype=torch.float32)
    pos = torch.as_tensor(coords, dtype=torch.float32)

    n_pairs = result.temperature.shape[0] - 1
    graphs: List[Data] = []
    for t in range(n_pairs):
        x_np = build_node_features(
            result, t, float(speeds[t]), node_type, bc_values, goldak
        )
        y_np = (result.temperature[t + 1] - result.temperature[t]).reshape(-1, 1)
        data = Data(
            x=torch.as_tensor(x_np, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.as_tensor(y_np, dtype=torch.float32),
            pos=pos,
        )
        data.sim_id = torch.tensor([sim_id], dtype=torch.long)
        data.time = torch.tensor([float(result.times[t])], dtype=torch.float32)
        graphs.append(data)
    return graphs


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
class NormalizeTransform:
    """Z-score transform applied to continuous node features and the target.

    Uses statistics produced by :class:`WeldingGraphDataset`. One-hot columns
    (``NORMALIZE_MASK == False``) are left untouched. :meth:`inverse_y` maps a
    normalized ΔT prediction back to physical kelvin.
    """

    def __init__(self, stats: dict):
        self.x_mean = stats["x_mean"]
        self.x_std = stats["x_std"]
        self.y_mean = stats["y_mean"]
        self.y_std = stats["y_std"]
        self.mask = stats["normalize_mask"].bool()

    def __call__(self, data: Data) -> Data:
        x = data.x.clone()
        m = self.mask
        x[:, m] = (x[:, m] - self.x_mean[m]) / self.x_std[m]
        data.x = x
        if data.y is not None:
            data.y = (data.y - self.y_mean) / self.y_std
        return data

    def inverse_y(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.y_std + self.y_mean


# ---------------------------------------------------------------------------
# PyG Dataset
# ---------------------------------------------------------------------------
class WeldingGraphDataset(Dataset):
    """Lazy on-disk PyG dataset over welding FEM simulations.

    Expects raw ``.npz`` files (``SimulationResult.save_npz`` output) under
    ``<root>/raw`` and writes one ``.pt`` graph per snapshot pair under
    ``<root>/processed``, plus a ``stats.pt`` with normalization statistics.

    Parameters
    ----------
    root:
        Dataset root (e.g. ``"data"``; raw files live in ``data/raw``).
    transform / pre_transform / pre_filter:
        Standard PyG hooks. Pass ``NormalizeTransform`` as ``transform`` to
        normalize on the fly (see :meth:`make_normalizer`).
    """

    def __init__(
        self,
        root: Union[str, Path],
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload: bool = False,
    ):
        root = Path(root)
        raw_dir = root / "raw"
        self._raw_files = sorted(p.name for p in raw_dir.glob("*.npz"))

        # Deterministic (raw_file, sim_idx, t) index so processed_file_names is
        # known up front and PyG can skip reprocessing.
        self._index = []
        for sim_idx, fname in enumerate(self._raw_files):
            with np.load(raw_dir / fname) as d:
                n_snap = int(d["times"].shape[0])
            for t in range(n_snap - 1):
                self._index.append((fname, sim_idx, t))
        self._processed_names = [
            f"data_{si}_{t}.pt" for (_, si, t) in self._index
        ] + ["stats.pt"]

        self._stats_cache: Optional[dict] = None
        super().__init__(str(root), transform, pre_transform, pre_filter,
                         force_reload=force_reload)

    # -- PyG plumbing ------------------------------------------------------
    @property
    def raw_file_names(self):
        return self._raw_files

    @property
    def processed_file_names(self):
        return self._processed_names

    def download(self):  # raw files are produced by the solver, not downloaded
        if not self._raw_files:
            raise FileNotFoundError(
                f"No .npz simulations found in {self.raw_dir}. Run the FEM "
                f"solver and save snapshots there first."
            )

    def process(self):
        # Streaming accumulators (float64) for per-feature mean/std.
        x_sum = np.zeros(NUM_NODE_FEATURES)
        x_sqsum = np.zeros(NUM_NODE_FEATURES)
        x_count = 0
        y_sum = 0.0
        y_sqsum = 0.0
        y_count = 0

        try:
            from tqdm import tqdm

            sims = tqdm(self._raw_files, desc="process sims", unit="sim")
        except ImportError:
            sims = self._raw_files

        for sim_idx, fname in enumerate(self._raw_files):
            result = SimulationResult.load_npz(Path(self.raw_dir) / fname)
            graphs = build_graph_sequence(result, sim_id=sim_idx)
            for t, data in enumerate(graphs):
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                xn = data.x.numpy().astype(np.float64)
                x_sum += xn.sum(axis=0)
                x_sqsum += (xn ** 2).sum(axis=0)
                x_count += xn.shape[0]
                yn = data.y.numpy().astype(np.float64)
                y_sum += yn.sum()
                y_sqsum += (yn ** 2).sum()
                y_count += yn.size

                torch.save(
                    data, Path(self.processed_dir) / f"data_{sim_idx}_{t}.pt"
                )
            if not isinstance(sims, list):
                sims.update(0)  # keep tqdm alive when iterating by index

        x_mean = x_sum / max(x_count, 1)
        x_var = np.maximum(x_sqsum / max(x_count, 1) - x_mean ** 2, 0.0)
        x_std = np.sqrt(x_var)
        x_std[x_std < 1e-8] = 1.0
        y_mean = y_sum / max(y_count, 1)
        y_var = max(y_sqsum / max(y_count, 1) - y_mean ** 2, 0.0)
        y_std = np.sqrt(y_var)
        y_std = y_std if y_std >= 1e-8 else 1.0

        stats = {
            "x_mean": torch.tensor(x_mean, dtype=torch.float32),
            "x_std": torch.tensor(x_std, dtype=torch.float32),
            "y_mean": torch.tensor(y_mean, dtype=torch.float32),
            "y_std": torch.tensor(y_std, dtype=torch.float32),
            "normalize_mask": torch.tensor(NORMALIZE_MASK),
            "feature_names": NODE_FEATURE_NAMES,
        }
        torch.save(stats, Path(self.processed_dir) / "stats.pt")

    # -- access ------------------------------------------------------------
    def len(self) -> int:
        return len(self._index)

    def get(self, idx: int) -> Data:
        fname, sim_idx, t = self._index[idx]
        return torch.load(
            Path(self.processed_dir) / f"data_{sim_idx}_{t}.pt",
            weights_only=False,
        )

    # -- normalization helpers --------------------------------------------
    def get_norm_stats(self) -> dict:
        """Load (and cache) the normalization statistics."""
        if self._stats_cache is None:
            self._stats_cache = torch.load(
                Path(self.processed_dir) / "stats.pt", weights_only=False
            )
        return self._stats_cache

    def make_normalizer(self) -> NormalizeTransform:
        """Construct a :class:`NormalizeTransform` from this dataset's stats."""
        return NormalizeTransform(self.get_norm_stats())
