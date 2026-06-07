"""Training utilities: simulation-level splitting and noise injection.

Two SciML-critical pieces live here:

1. **Simulation-level data splitting** (:func:`split_by_simulation`). Splits the
   dataset so that *every graph snapshot of a given simulation* lands in the
   same fold. Random snapshot-level splits leak temporally-adjacent states of
   the same weld across train/val/test and inflate scores; splitting whole
   simulations measures true geometric/parameter generalization.

2. **MeshGraphNet training-noise injection** (:class:`TemperatureNoiseInjection`).
   Autoregressive rollouts accumulate error; injecting zero-mean Gaussian noise
   into the *input* temperature during training, and correcting the target so the
   network learns to undo the drift, makes rollouts stable (Pfaff et al., 2021):

       T̃_i^t      = T_i^t + η_i,   η_i ~ N(0, σ²)
       ΔT_target  = T_i^{t+1} − T̃_i^t = (T_i^{t+1} − T_i^t) − η_i = y_i − η_i

   Noise is applied **only** to training data (never val/test), and σ is
   configurable via :class:`TrainingConfig`.

Integration notes
-----------------
The :class:`~data.graph_builder.WeldingGraphDataset` stores *raw physical*
features and applies normalization as a separate transform. Noise is defined in
physical kelvin, so it must be injected on the raw temperature **before**
normalization. :func:`make_split_datasets` wires this up: the train fold gets
``Compose([noise, normalizer])`` and val/test get ``normalizer`` only.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform, Compose

from data.graph_builder import NODE_FEATURE_NAMES

#: Index of the temperature column in the node-feature vector (robust to layout
#: changes in the graph builder).
TEMPERATURE_INDEX = NODE_FEATURE_NAMES.index("T")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """Hyperparameters consumed by the training utilities.

    (The full training loop, added later, may extend this with optimizer/loop
    settings; only the fields needed for splitting and noise live here.)
    """

    # Simulation-level split.
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    split_seed: int = 0

    # Training-noise injection (physical kelvin, applied to raw temperature).
    noise_std: float = 0.0
    temperature_index: int = TEMPERATURE_INDEX


# ---------------------------------------------------------------------------
# 1. Simulation-level data splitting
# ---------------------------------------------------------------------------
@dataclass
class DataSplit:
    """Graph indices and simulation ids assigned to each fold."""

    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]
    train_sims: List[int]
    val_sims: List[int]
    test_sims: List[int]

    def summary(self) -> str:
        return (
            f"DataSplit(train: {len(self.train_sims)} sims / "
            f"{len(self.train_indices)} graphs, "
            f"val: {len(self.val_sims)} sims / {len(self.val_indices)} graphs, "
            f"test: {len(self.test_sims)} sims / {len(self.test_indices)} graphs)"
        )


def simulation_ids_from_dataset(dataset) -> np.ndarray:
    """Per-graph simulation id, aligned with ``range(len(dataset))``.

    Prefers the cheap deterministic ``_index`` maintained by
    :class:`WeldingGraphDataset`; otherwise falls back to reading each graph's
    ``sim_id`` attribute (works for any PyG dataset, but loads every graph).
    """
    index = getattr(dataset, "_index", None)
    if index is not None:
        return np.asarray([sim_idx for (_, sim_idx, _) in index], dtype=np.int64)

    ids = []
    for i in range(len(dataset)):
        data = dataset[i]
        sid = getattr(data, "sim_id", None)
        ids.append(int(sid) if sid is not None else 0)
    return np.asarray(ids, dtype=np.int64)


def split_by_simulation(
    dataset,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
    sim_ids: Optional[Sequence[int]] = None,
) -> DataSplit:
    """Split a graph dataset into train/val/test **at the simulation level**.

    Unique simulations are shuffled (seeded) and partitioned by the requested
    fractions; then every graph belonging to a simulation inherits that
    simulation's fold. This guarantees no simulation is split across folds.

    Parameters
    ----------
    dataset:
        A graph dataset whose graphs carry a simulation id (see
        :func:`simulation_ids_from_dataset`).
    val_fraction, test_fraction:
        Fractions of *simulations* (not graphs) assigned to val and test.
    seed:
        RNG seed for the simulation shuffle (reproducible splits).
    sim_ids:
        Optional explicit per-graph simulation ids (overrides auto-detection).
    """
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1.0:
        raise ValueError(
            "val_fraction and test_fraction must be non-negative and sum to < 1."
        )

    ids = np.asarray(sim_ids) if sim_ids is not None else simulation_ids_from_dataset(dataset)
    unique = np.unique(ids)
    n_sims = len(unique)

    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)

    # Partition the *simulation* list. Rounding keeps the requested proportions;
    # we then guarantee a non-empty training set.
    n_test = int(round(test_fraction * n_sims))
    n_val = int(round(val_fraction * n_sims))
    # Nudge to at least one simulation when the fraction is positive and the
    # simulation budget allows it (otherwise small corpora silently lose a fold).
    if test_fraction > 0 and n_test == 0 and n_sims - n_val > 1:
        n_test = 1
    if val_fraction > 0 and n_val == 0 and n_sims - n_test > 1:
        n_val = 1
    if n_val + n_test >= n_sims:  # never starve training
        n_val = min(n_val, max(n_sims - 1, 0))
        n_test = max(n_sims - n_val - 1, 0)

    test_sims = set(shuffled[:n_test].tolist())
    val_sims = set(shuffled[n_test:n_test + n_val].tolist())
    train_sims = set(shuffled[n_test + n_val:].tolist())

    for name, fold in (("train", train_sims), ("val", val_sims), ("test", test_sims)):
        if not fold and n_sims > 0:
            warnings.warn(
                f"Simulation split produced an empty '{name}' fold "
                f"({n_sims} simulations total). Consider adjusting fractions "
                f"or generating more simulations.",
                stacklevel=2,
            )

    train_idx, val_idx, test_idx = [], [], []
    for graph_idx, sim in enumerate(ids):
        if sim in train_sims:
            train_idx.append(graph_idx)
        elif sim in val_sims:
            val_idx.append(graph_idx)
        else:
            test_idx.append(graph_idx)

    return DataSplit(
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        train_sims=sorted(train_sims),
        val_sims=sorted(val_sims),
        test_sims=sorted(test_sims),
    )


# ---------------------------------------------------------------------------
# 2. Training-noise injection
# ---------------------------------------------------------------------------
class TemperatureNoiseInjection(BaseTransform):
    """Inject N(0, σ²) noise into the input temperature and correct the target.

    Operates on the **raw** node temperature column (kelvin), so it must run
    *before* any normalization transform. Fresh noise is sampled on every call
    (i.e. every ``__getitem__``), giving new perturbations each epoch.

    The same per-node noise ``η`` is added to the input temperature and
    subtracted from the target so the model learns to drive a perturbed state
    back onto the ground-truth trajectory::

        x[:, T] += η
        y       -= η          # ΔT_target = (T^{t+1} − T^t) − η

    Parameters
    ----------
    sigma:
        Noise standard deviation (kelvin). ``sigma <= 0`` is a no-op.
    temperature_index:
        Column of ``data.x`` holding the temperature.
    enabled:
        Safety switch; set ``False`` to disable (e.g. for val/test). Noise is
        also never applied when ``sigma <= 0``.
    generator:
        Optional :class:`torch.Generator` for reproducible sampling.
    """

    def __init__(
        self,
        sigma: float,
        temperature_index: int = TEMPERATURE_INDEX,
        enabled: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        self.sigma = float(sigma)
        self.temperature_index = int(temperature_index)
        self.enabled = bool(enabled)
        self.generator = generator

    def forward(self, data: Data) -> Data:
        if not self.enabled or self.sigma <= 0.0:
            return data

        ti = self.temperature_index
        # Per-node noise, shape (N, 1) to match a single feature column.
        eta = torch.randn(
            data.num_nodes, 1, generator=self.generator,
            dtype=data.x.dtype, device=data.x.device,
        ) * self.sigma

        x = data.x.clone()
        x[:, ti:ti + 1] = x[:, ti:ti + 1] + eta
        data.x = x

        if data.y is not None:
            y = data.y.clone()
            # Target correction: subtract the same noise so the model learns to
            # recover the true next-step temperature from the perturbed input.
            y[:, 0:1] = y[:, 0:1] - eta
            data.y = y

        return data

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.__class__.__name__}(sigma={self.sigma}, "
            f"temperature_index={self.temperature_index}, enabled={self.enabled})"
        )


# ---------------------------------------------------------------------------
# Transform-aware subset (lets each fold use a different transform)
# ---------------------------------------------------------------------------
class TransformedSubset(TorchDataset):
    """A subset of a graph dataset with its own per-fold transform.

    The base dataset is expected to be **transform-free**: raw graphs are
    fetched via ``base_dataset.get`` (bypassing any dataset-level transform) and
    this subset's ``transform`` is applied instead. This is what allows the
    training fold to inject noise while validation/test do not.
    """

    def __init__(
        self,
        base_dataset,
        indices: Sequence[int],
        transform: Optional[Callable[[Data], Data]] = None,
    ):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def _get_raw(self, idx: int) -> Data:
        # Prefer PyG's raw loader (no dataset-level transform); fall back to
        # indexing for plain datasets.
        get = getattr(self.base_dataset, "get", None)
        return get(idx) if callable(get) else self.base_dataset[idx]

    def __getitem__(self, i: int) -> Data:
        data = self._get_raw(self.indices[i])
        if self.transform is not None:
            data = self.transform(data)
        return data


def make_split_datasets(
    dataset,
    config: TrainingConfig,
    normalizer: Optional[Callable[[Data], Data]] = None,
    split: Optional[DataSplit] = None,
) -> Dict[str, TransformedSubset]:
    """Build train/val/test subsets with the correct per-fold transforms.

    The train fold receives noise injection (``config.noise_std``) **then**
    normalization; val/test receive normalization only. Pass the dataset
    *without* a transform; supply the normalizer explicitly (e.g. from
    ``WeldingGraphDataset.make_normalizer()``).

    Returns a dict with keys ``"train"``, ``"val"``, ``"test"``.
    """
    if split is None:
        split = split_by_simulation(
            dataset,
            val_fraction=config.val_fraction,
            test_fraction=config.test_fraction,
            seed=config.split_seed,
        )

    noise = TemperatureNoiseInjection(
        sigma=config.noise_std, temperature_index=config.temperature_index
    )

    def compose(*transforms) -> Optional[Callable[[Data], Data]]:
        active = [t for t in transforms if t is not None]
        if not active:
            return None
        return active[0] if len(active) == 1 else Compose(active)

    train_transform = compose(noise, normalizer)   # noise BEFORE normalization
    eval_transform = compose(normalizer)            # no noise on val/test

    return {
        "train": TransformedSubset(dataset, split.train_indices, train_transform),
        "val": TransformedSubset(dataset, split.val_indices, eval_transform),
        "test": TransformedSubset(dataset, split.test_indices, eval_transform),
    }
