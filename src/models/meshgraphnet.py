"""MeshGraphNet (Encoder-Processor-Decoder) for the welding thermal surrogate.

Implementation of the MeshGraphNets architecture (Pfaff et al., DeepMind, 2021)
engineered for configurability: every structural hyperparameter — latent width,
MLP depth, number of message-passing hops, activation, normalization and
aggregation — is driven by :class:`MeshGraphNetConfig`.

The forward pass maps per-node / per-edge input features to a per-node output
(here the temperature increment ``ΔT``), following the three classic stages:

* **Encoder**   — independently lifts raw node and edge features into a shared
  latent space of width ``hidden_dim`` via MLPs.
* **Processor** — ``num_processing_steps`` GraphNet blocks. Each block updates
  edges from their endpoints, aggregates them at the receiver nodes, and updates
  the nodes. Both updates use **residual connections** so gradients survive deep
  rollouts.
* **Decoder**   — projects the final node latents down to ``out_dim``.

The data contract from ``src/data/graph_builder.py`` is the intended input:
``node_in_dim=12``, ``edge_in_dim=3``, ``out_dim=1`` (ΔT).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
#: Supported activations, by name.
_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
}


@dataclass
class MeshGraphNetConfig:
    """All MeshGraphNet hyperparameters in one place.

    Parameters
    ----------
    node_in_dim, edge_in_dim, out_dim:
        Raw input node-feature width, raw input edge-feature width, and the
        per-node output width (``1`` for scalar ΔT regression).
    hidden_dim:
        Width of all latent node/edge representations and MLP hidden layers.
    num_mlp_layers:
        Number of *hidden* layers in every MLP (encoder, processor blocks,
        decoder). Each MLP is ``num_mlp_layers`` hidden layers of width
        ``hidden_dim`` followed by a linear projection to its output width.
    num_processing_steps:
        Number of message-passing hops (GraphNet blocks) in the Processor.
    activation:
        Activation name; see :data:`_ACTIVATIONS`.
    use_layer_norm:
        Apply LayerNorm at the output of encoder and processor MLPs (the
        MeshGraphNets default). The decoder never uses LayerNorm so it can emit
        unbounded physical values.
    aggregation:
        Receiver-node aggregation for messages: ``"sum"`` (MeshGraphNets
        default), ``"mean"``, ``"max"``, ``"min"``.
    """

    node_in_dim: int = 12
    edge_in_dim: int = 3
    out_dim: int = 1
    hidden_dim: int = 128
    num_mlp_layers: int = 2
    num_processing_steps: int = 15
    activation: str = "relu"
    use_layer_norm: bool = True
    aggregation: str = "sum"

    # --- Optional GENERIC structure-preserving thermodynamic head ---
    use_generic: bool = False
    #: GENERIC dissipation operator:
    #:   "energy"   — degeneracy-only (energy-conserving projection of the decoder
    #:                increment). Guarantees M·∇E = 0 but NOT entropy production.
    #:   "full"     — second-law GENERIC on TEMPERATURE: SPSD graph-Laplacian of
    #:                learned conductances acting on ∇S = 1/T. Guarantees M SPSD,
    #:                M·∇E = 0, dS/dt ≥ 0. Assumes E ∝ T (constant c_p) — so its
    #:                "energy conservation" is really temperature-sum conservation,
    #:                inconsistent under latent heat.
    #:   "enthalpy" — thermodynamically consistent: state = volumetric enthalpy h,
    #:                evolve Δh = L_w·(1/T), recover T = h⁻¹(h) via the latent-heat
    #:                curve. Genuine energy conservation + 2nd law WITH phase change.
    generic_mode: str = "energy"
    #: Enriched external source (enthalpy mode only): replace the single scalar
    #: source/cooling gains with a per-node learned modulation
    #: ``softplus(gain + MLP(latents, q, T))``. The dissipative operator stays
    #: structure-preserving; only the (non-conservative) external exchange — which
    #: alone heats the melt-pool peak — gains the flexibility to inject enough
    #: energy there. Zero-initialized, so it starts identical to the scalar head.
    enriched_source: bool = False
    #: Feature-column indices the GENERIC head reads (graph_builder 12-d layout).
    temperature_index: int = 0
    goldak_index: int = 1
    robin_onehot_index: int = 8
    t_inf_index: int = 11

    def activation_factory(self) -> Callable[[], nn.Module]:
        try:
            return _ACTIVATIONS[self.activation]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"Unknown activation '{self.activation}'. "
                f"Choose from {sorted(_ACTIVATIONS)}."
            ) from exc


# ---------------------------------------------------------------------------
# MLP builder
# ---------------------------------------------------------------------------
def build_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_hidden_layers: int,
    activation: Callable[[], nn.Module],
    layer_norm: bool,
) -> nn.Sequential:
    """Construct an MLP: ``num_hidden_layers`` hidden layers + output projection.

    With ``num_hidden_layers == 0`` this reduces to a single linear layer. A
    trailing :class:`~torch.nn.LayerNorm` is appended when ``layer_norm`` is set
    (standard in MeshGraphNets to stabilize the latent distributions).
    """
    layers: List[nn.Module] = []
    dim = in_dim
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(activation())
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    if layer_norm:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    """Independently embeds node and edge features into the latent space."""

    def __init__(self, cfg: MeshGraphNetConfig):
        super().__init__()
        act = cfg.activation_factory()
        self.node_mlp = build_mlp(
            cfg.node_in_dim, cfg.hidden_dim, cfg.hidden_dim,
            cfg.num_mlp_layers, act, cfg.use_layer_norm,
        )
        self.edge_mlp = build_mlp(
            cfg.edge_in_dim, cfg.hidden_dim, cfg.hidden_dim,
            cfg.num_mlp_layers, act, cfg.use_layer_norm,
        )

    def forward(
        self, x: torch.Tensor, edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.node_mlp(x), self.edge_mlp(edge_attr)


# ---------------------------------------------------------------------------
# Processor block (one message-passing hop)
# ---------------------------------------------------------------------------
class GraphNetBlock(nn.Module):
    """A single MeshGraphNets interaction step with residual node/edge updates.

    Convention: ``edge_index[0]`` are sender nodes ``i``, ``edge_index[1]`` are
    receiver nodes ``j``; messages are aggregated at receivers.

    Edge update:  ``e_ij <- e_ij + f_e([e_ij, x_i, x_j])``
    Node update:  ``x_j  <- x_j  + f_x([x_j, agg_i e_ij])``
    """

    def __init__(self, cfg: MeshGraphNetConfig):
        super().__init__()
        act = cfg.activation_factory()
        h = cfg.hidden_dim
        # Edge MLP consumes [edge, src_node, dst_node]; node MLP consumes
        # [node, aggregated_edges] — both produce a latent-width residual.
        self.edge_mlp = build_mlp(
            3 * h, h, h, cfg.num_mlp_layers, act, cfg.use_layer_norm
        )
        self.node_mlp = build_mlp(
            2 * h, h, h, cfg.num_mlp_layers, act, cfg.use_layer_norm
        )
        self.aggregation = cfg.aggregation

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index[0], edge_index[1]

        # --- Edge update (residual) ---
        edge_inputs = torch.cat([edge_attr, x[src], x[dst]], dim=-1)
        edge_attr = edge_attr + self.edge_mlp(edge_inputs)

        # --- Aggregate messages at receiver nodes (vectorized scatter) ---
        aggregated = scatter(
            edge_attr, dst, dim=0, dim_size=x.size(0), reduce=self.aggregation
        )

        # --- Node update (residual) ---
        node_inputs = torch.cat([x, aggregated], dim=-1)
        x = x + self.node_mlp(node_inputs)

        return x, edge_attr


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
class Decoder(nn.Module):
    """Maps final node latents to the per-node output (no LayerNorm)."""

    def __init__(self, cfg: MeshGraphNetConfig):
        super().__init__()
        act = cfg.activation_factory()
        self.node_mlp = build_mlp(
            cfg.hidden_dim, cfg.hidden_dim, cfg.out_dim,
            cfg.num_mlp_layers, act, layer_norm=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.node_mlp(x)


# ---------------------------------------------------------------------------
# GENERIC structure-preserving thermal head (optional)
# ---------------------------------------------------------------------------
def energy_conserving_projection(
    v: torch.Tensor,
    batch: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Project ``v`` onto the complement of the energy gradient grad(E).

    Applies the GENERIC degeneracy projector ``P = I - gE gE^T / ||gE||^2``
    (gE = grad E) so that ``gE . (P v) = 0`` exactly: the dissipative
    increment neither creates nor destroys internal energy (the condition
    ``M grad E = 0``). With uniform nodal tributary volumes the energy
    gradient ``gE_i = rho Cp V_i`` is constant, so ``P`` reduces to
    (weighted) mean-removal. When ``batch`` is supplied the projection is
    applied independently per graph.
    """
    w = torch.ones_like(v) if weights is None else weights
    if batch is None:
        num = (w * v).sum(dim=0, keepdim=True)
        den = (w * w).sum(dim=0, keepdim=True).clamp_min(eps)
        return v - (num / den) * w
    num = scatter(w * v, batch, dim=0, reduce="sum")
    den = scatter(w * w, batch, dim=0, reduce="sum").clamp_min(eps)
    return v - (num / den)[batch] * w


class _GenericHeadBase(nn.Module):
    """Shared machinery for the GENERIC thermal heads.

    Holds the normalization buffers (so the head can operate in PHYSICAL kelvin
    while the network is fed normalized features and emits normalized ΔT), the
    analytical external exchange (Goldak source + Newton boundary cooling), and
    the normalized↔physical conversions. Subclasses supply the *dissipative*
    operator; the external term and the energy/entropy bookkeeping are common.

    Uniform tributary volumes are assumed (``ρ Cp V_i`` constant), so ``∇E`` is
    the constant vector and energy is ``E = Σ ρCp T_i V_i``.
    """

    #: Whether the head consumes the processor node latents (full GENERIC) rather
    #: than the decoder increment (energy-only). Set by subclasses.
    uses_node_latents: bool = False

    def __init__(self, cfg: MeshGraphNetConfig):
        super().__init__()
        self.t_idx = cfg.temperature_index
        self.q_idx = cfg.goldak_index
        self.robin_idx = cfg.robin_onehot_index
        self.tinf_idx = cfg.t_inf_index
        self.source_gain = nn.Parameter(torch.zeros(1))
        self.cool_gain = nn.Parameter(torch.zeros(1))

        # Normalization constants (populated by `set_normalization`). Buffers
        # (not params) -> they ride in state_dict and move with `.to(device)`,
        # but don't count as parameters.
        f = cfg.node_in_dim
        self.register_buffer("x_mean", torch.zeros(f))
        self.register_buffer("x_std", torch.ones(f))
        self.register_buffer("norm_mask", torch.ones(f, dtype=torch.bool))
        self.register_buffer("y_mean", torch.zeros(1))
        self.register_buffer("y_std", torch.ones(1))
        self.register_buffer("_initialized", torch.zeros(1))

        # Most-recent components, cached (detached) for inspection / tests.
        self.last_dissipative: Optional[torch.Tensor] = None
        self.last_external: Optional[torch.Tensor] = None

    def set_normalization(self, stats: dict) -> None:
        """Load z-score constants so the head can de/re-normalize internally.

        Call once on a fresh model before training. On the first call it also
        initializes the source/cooling gains so ``softplus(gain)·feature_scale``
        starts at the physical ΔT scale (``q_goldak`` is ~1e10, so a zero gain
        would blow up). Subsequent calls only refresh the buffers and leave
        trained gains untouched (``_initialized`` rides in state_dict).
        """
        self.x_mean.copy_(stats["x_mean"].reshape(-1).to(self.x_mean))
        self.x_std.copy_(stats["x_std"].reshape(-1).clamp_min(1e-8).to(self.x_std))
        self.norm_mask.copy_(stats["normalize_mask"].reshape(-1).bool().to(self.norm_mask.device))
        self.y_mean.copy_(stats["y_mean"].reshape(1).to(self.y_mean))
        self.y_std.copy_(stats["y_std"].reshape(1).clamp_min(1e-8).to(self.y_std))
        if float(self._initialized) < 0.5:
            with torch.no_grad():
                # softplus(gain) ~ ΔT_scale / driver_scale  (absorbs dt/(rho Cp)).
                tgt_src = (self.y_std / self.x_std[self.q_idx]).clamp_min(1e-12)
                tgt_cool = (self.y_std / self.x_std[self.t_idx]).clamp_min(1e-12)
                self.source_gain.copy_(torch.log(torch.expm1(tgt_src)))
                self.cool_gain.copy_(torch.log(torch.expm1(tgt_cool)))
            self._initialized.fill_(1.0)

    def _phys_col(self, x_raw: torch.Tensor, idx: int) -> torch.Tensor:
        """De-normalize column ``idx`` to physical units (raw if un-normalized)."""
        c = x_raw[:, idx:idx + 1]
        if bool(self.norm_mask[idx]):
            c = c * self.x_std[idx] + self.x_mean[idx]
        return c

    def _external(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Analytical external energy exchange (physical units).

        Goldak heating (sign-positive) + Newton boundary cooling toward
        ``T_inf`` on Robin nodes. Each gain is non-negative (``softplus``) so the
        physical sign structure is preserved: sources add energy, cooling removes
        it as ``T`` exceeds ``T_inf``.
        """
        q = self._phys_col(x_raw, self.q_idx)
        t = self._phys_col(x_raw, self.t_idx)
        t_inf = self._phys_col(x_raw, self.tinf_idx)
        robin = self._phys_col(x_raw, self.robin_idx)          # one-hot, used raw
        return F.softplus(self.source_gain) * q + \
            F.softplus(self.cool_gain) * robin * (t_inf - t)

    def _to_normalized_increment(self, dT_phys: torch.Tensor) -> torch.Tensor:
        """Physical ΔT -> normalized units (the model's output contract)."""
        return (dT_phys - self.y_mean) / self.y_std


class GenericThermalHead(_GenericHeadBase):
    """Energy-only (degeneracy) GENERIC head: ``dT = P·dT_tilde + q_ext``.

    The dissipative part projects the decoder's unconstrained physical increment
    onto the complement of ``∇E`` (:func:`energy_conserving_projection`),
    guaranteeing the degeneracy ``M·∇E = 0`` — the learned redistribution
    conserves internal energy per graph, removing spurious global heating/cooling
    drift. It does NOT enforce entropy production (no SPSD operator); for that,
    use :class:`FullGenericThermalHead`.
    """

    uses_node_latents = False

    def forward(
        self,
        delta_t_tilde: torch.Tensor,
        x_raw: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # De-normalize the decoder increment to physical kelvin, project to
        # conserve energy, add the analytical external exchange, re-normalize.
        dT_tilde = delta_t_tilde * self.y_std + self.y_mean
        dissipative = energy_conserving_projection(dT_tilde, batch=batch)
        external = self._external(x_raw)

        self.last_dissipative = dissipative.detach()
        self.last_external = external.detach()
        return self._to_normalized_increment(dissipative + external)


class FullGenericThermalHead(_GenericHeadBase):
    """Full (second-law) GENERIC head: SPSD graph-Laplacian dissipation.

    The dissipative increment is ``dT_diss = L_w · μ`` where ``μ_i = ∂S/∂e_i =
    1/T_i`` is the entropy gradient and ``L_w = D − W`` is the graph Laplacian of
    **learned non-negative, symmetric** edge conductances
    ``w_ij = exp(scale)·softplus(MLP(symmetric features)) ≥ 0``. By construction
    this guarantees the full GENERIC dissipative structure:

    * **SPSD operator** — ``w ≥ 0`` and symmetric ⇒ ``L_w`` is positive
      semi-definite.
    * **Degeneracy ``M·∇E = 0``** — the Laplacian has zero column sums, so
      ``Σ_i dT_diss_i = 0`` per graph (internal energy conserved exactly).
    * **Entropy production ``dS/dt = μᵀ L_w μ = ½ Σ_ij w_ij (μ_i−μ_j)² ≥ 0``** —
      the second law, absent from the energy-only head.
    * **Heat flows hot → cold** — a hot node (small ``μ``) next to cold neighbors
      (large ``μ``) gets ``dT_diss < 0``.

    Symmetry ``w_ij = w_ji`` is obtained for free by feeding the conductance MLP
    only **symmetric** edge features (``h_i+h_j``, ``(h_i−h_j)²``, ``T_i+T_j``,
    ``|T_i−T_j|``); the bidirectional edge set then realizes both Laplacian rows.
    ``μ`` uses ``T`` clamped to a physical floor so it stays bounded even if a
    rollout briefly drives a node toward 0 K. The decoder is bypassed entirely
    in this mode (the dissipation comes from the conductances, not a free
    per-node increment, which would violate the degeneracy). The same analytical
    external exchange (Goldak source + Newton cooling) is added.
    """

    uses_node_latents = True

    #: Temperature floor [K] for the entropy gradient μ = 1/T (keeps μ bounded).
    T_FLOOR: float = 200.0

    def __init__(self, cfg: MeshGraphNetConfig):
        super().__init__(cfg)
        act = cfg.activation_factory()
        h = cfg.hidden_dim
        # Conductance MLP input: [h_i+h_j, (h_i-h_j)^2, T_sum, |T_diff|] (all
        # symmetric under i<->j) -> a single non-negative scalar per edge.
        self.cond_mlp = build_mlp(
            2 * h + 2, h, 1, cfg.num_mlp_layers, act, layer_norm=False
        )
        # Global conductance scale (absorbs the awkward 1/T² magnitude of working
        # in μ=1/T space). Init ~500 so dT_diss starts at O(1 K) for strong
        # gradients (validated by the smoke test); training tunes it.
        self.log_cond_scale = nn.Parameter(torch.tensor([math.log(500.0)]))

    def forward(
        self,
        delta_t_tilde: Optional[torch.Tensor],
        x_raw: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        node_latents: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if node_latents is None or edge_index is None:
            raise ValueError(
                "FullGenericThermalHead requires node_latents and edge_index."
            )
        h = node_latents
        src, dst = edge_index[0], edge_index[1]

        t = self._phys_col(x_raw, self.t_idx)                  # (N,1) physical T
        tscale = self.x_std[self.t_idx].clamp_min(1e-6)

        # Symmetric per-edge conductance features -> w_ij = w_ji by construction.
        hi, hj = h[src], h[dst]
        t_src, t_dst = t[src], t[dst]
        feats = torch.cat([
            hi + hj,
            (hi - hj) ** 2,
            (t_src + t_dst) / (2.0 * tscale),
            (t_src - t_dst).abs() / tscale,
        ], dim=-1)
        w = torch.exp(self.log_cond_scale) * F.softplus(self.cond_mlp(feats))  # (E,1) ≥ 0

        # Entropy gradient μ = ∂S/∂e = 1/T (clamped for a bounded, finite μ).
        mu = 1.0 / t.clamp_min(self.T_FLOOR)                   # (N,1)

        # dT_diss = (L_w μ): for directed edge (s->r) add w·(μ_r − μ_s) to r.
        # Summed over bidirectional edges this is exactly the SPSD Laplacian
        # action; its per-graph sum is 0 (energy conserved).
        edge_term = w * (mu[dst] - mu[src])
        dissipative = scatter(edge_term, dst, dim=0, dim_size=h.size(0), reduce="sum")

        external = self._external(x_raw)
        self.last_dissipative = dissipative.detach()
        self.last_external = external.detach()
        return self._to_normalized_increment(dissipative + external)


# ---------------------------------------------------------------------------
# Enthalpy-state (thermodynamically consistent) GENERIC head
# ---------------------------------------------------------------------------
def _interp1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Differentiable piecewise-linear interpolation of ``fp(xp)`` at ``x``.

    ``xp`` is a 1-D, strictly increasing grid; ``x`` has shape ``(N, 1)``. Values
    outside ``[xp[0], xp[-1]]`` are clamped (flat extrapolation). Differentiable
    w.r.t. ``x`` (the gradient is the local slope), which is what lets the
    enthalpy map sit inside the autograd graph.
    """
    xq = x.clamp(xp[0], xp[-1]).squeeze(-1)                       # (N,)
    idx = torch.searchsorted(xp, xq.contiguous(), right=True).clamp(1, xp.numel() - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    f0, f1 = fp[idx - 1], fp[idx]
    wgt = (xq - x0) / (x1 - x0).clamp_min(1e-12)
    return (f0 + wgt * (f1 - f0)).unsqueeze(-1)                   # (N, 1)


class EnthalpyGenericThermalHead(FullGenericThermalHead):
    r"""Thermodynamically consistent GENERIC head: energy (enthalpy) is the state.

    This fixes the constant-``c_p`` flaw of :class:`FullGenericThermalHead`. The
    SPSD graph-Laplacian now evolves the **volumetric enthalpy** ``h`` (energy),
    not temperature, and temperature is recovered through the nonlinear,
    latent-heat-aware enthalpy curve ``h(T) = ρ∫ c_p^{app}(τ)\,dτ`` (the classical
    *enthalpy method*). Per node:

        Δh_i = (L_w μ)_i + Δh^{ext}_i,   μ_i = 1/T_i,
        T_i^{new} = h^{-1}(h(T_i) + Δh_i),
        ΔT_i = T_i^{new} − T_i.

    Because the Laplacian conserves the quantity it acts on and that quantity is
    now **energy**, ``Σ_i (L_w μ)_i = 0`` is *genuine* energy conservation (the
    1st law) even with latent heat; entropy production ``μᵀ L_w μ ≥ 0`` (2nd law)
    and hot→cold flow are inherited unchanged. At the melt pool a large Δh
    produces almost no ΔT (the enthalpy plateau), exactly as in the FEM.

    The conductances ``w_ij`` (learned, symmetric, ≥0) and the source/cooling
    gains are inherited from :class:`FullGenericThermalHead`; only the *variable
    being conserved* changes (T → h). The enthalpy curve is a fixed physical
    function (no learning), tabulated once from :class:`MaterialProperties`.
    """

    #: Upper temperature of the enthalpy table [K] (covers the hottest sims).
    T_TABLE_MAX: float = 8000.0
    TABLE_POINTS: int = 4096

    def __init__(self, cfg: MeshGraphNetConfig):
        super().__init__(cfg)
        # Tabulate the enthalpy curve in TEMPERATURE-EQUIVALENT units,
        #   H(T) = ∫ c_p^app(τ)/c_p^sens dτ   [K]   ( = h(T)/(ρ c_p^sens) ),
        # built from the (fixed) material model used by the FEM solver, so the
        # surrogate's T<->H map is identical to the ground truth. Working in K
        # (not J/m^3) keeps the increments O(K) and the gradients O(1): the raw
        # enthalpy h~1e9 J/m^3 makes ∂T/∂h ~ 1/(ρc_p) ~ 2e-7, which pushes the
        # parameter gradients below Adam's epsilon and stalls training. The
        # rescale is a constant factor, so energy conservation is unchanged
        # (Σ ΔH_diss = 0 ⟺ Σ Δh_diss = 0). Local import keeps `models` decoupled
        # from the heavy `simulation` package unless this mode is used.
        from simulation.thermal_solver import MaterialProperties

        mat = MaterialProperties()
        import numpy as _np
        T_grid = _np.linspace(self.T_FLOOR, self.T_TABLE_MAX, self.TABLE_POINTS)
        cp_app = mat.cp_apparent(T_grid)                          # [J/(kg K)]
        dT = _np.diff(T_grid, prepend=T_grid[0])
        H_grid = _np.cumsum((cp_app / mat.cp) * dT)               # [K], monotone
        self.register_buffer("T_grid", torch.tensor(T_grid, dtype=torch.float32))
        self.register_buffer("H_grid", torch.tensor(H_grid, dtype=torch.float32))
        # Gains/conductances now produce ΔH in kelvin, so the temperature-scale
        # init of the base `set_normalization` is the right one — keep it (do NOT
        # override or mark initialized).

        # Optional per-node modulation of the source/cooling gains. Input is the
        # processor latents plus the local (normalized) drivers; the output is a
        # log-gain correction added to the global scalar gain. Zero-initialized
        # (last layer) so the head starts identical to the scalar-gain version,
        # then learns where to inject more energy (the peak).
        self.enriched_source = cfg.enriched_source
        if self.enriched_source:
            act = cfg.activation_factory()
            hdim = cfg.hidden_dim
            self.source_mlp = build_mlp(hdim + 2, hdim, 1, cfg.num_mlp_layers,
                                        act, layer_norm=False)
            self.cool_mlp = build_mlp(hdim + 2, hdim, 1, cfg.num_mlp_layers,
                                      act, layer_norm=False)
            for m in (self.source_mlp, self.cool_mlp):
                nn.init.zeros_(m[-1].weight)
                nn.init.zeros_(m[-1].bias)

    def forward(
        self,
        delta_t_tilde: Optional[torch.Tensor],
        x_raw: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        node_latents: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if node_latents is None or edge_index is None:
            raise ValueError(
                "EnthalpyGenericThermalHead requires node_latents and edge_index."
            )
        hlat = node_latents
        src, dst = edge_index[0], edge_index[1]

        T = self._phys_col(x_raw, self.t_idx).clamp_min(self.T_FLOOR)   # (N,1) [K]
        tscale = self.x_std[self.t_idx].clamp_min(1e-6)

        # Learned symmetric conductances (same construction as the full head).
        hi, hj = hlat[src], hlat[dst]
        Tsrc, Tdst = T[src], T[dst]
        feats = torch.cat([
            hi + hj,
            (hi - hj) ** 2,
            (Tsrc + Tdst) / (2.0 * tscale),
            (Tsrc - Tdst).abs() / tscale,
        ], dim=-1)
        w = torch.exp(self.log_cond_scale) * F.softplus(self.cond_mlp(feats))   # (E,1) ≥ 0

        mu = 1.0 / T                                              # entropy gradient ∂S/∂e

        # Dissipative ENERGY change (in temperature-equivalent units H [K]):
        # ΔH_diss = (L_w μ). Σ_i ΔH_diss_i = 0 ⇒ the 1st law (energy is conserved;
        # H is a constant rescale of the enthalpy, so this IS energy conservation).
        dH_diss = scatter(w * (mu[dst] - mu[src]), dst, dim=0,
                          dim_size=hlat.size(0), reduce="sum")

        # External ENERGY exchange (Goldak source adds energy; Newton boundary
        # cooling removes it). Either the inherited scalar gains, or — when
        # enriched — a per-node learned modulation that can inject more energy at
        # the peak (still ≥0 heating / sign-correct cooling, so physically sound).
        if self.enriched_source:
            q = self._phys_col(x_raw, self.q_idx)
            t_inf = self._phys_col(x_raw, self.tinf_idx)
            robin = self._phys_col(x_raw, self.robin_idx)
            q_n = x_raw[:, self.q_idx:self.q_idx + 1]
            T_n = x_raw[:, self.t_idx:self.t_idx + 1]
            tinf_n = x_raw[:, self.tinf_idx:self.tinf_idx + 1]
            src_gain = F.softplus(self.source_gain
                                  + self.source_mlp(torch.cat([hlat, q_n, T_n], dim=-1)))
            cool_gain = F.softplus(self.cool_gain
                                   + self.cool_mlp(torch.cat([hlat, T_n, tinf_n], dim=-1)))
            dH_ext = src_gain * q + cool_gain * robin * (t_inf - T)
        else:
            dH_ext = self._external(x_raw)

        # Enthalpy update, then recover temperature through the latent-heat curve.
        H_cur = _interp1d(T, self.T_grid, self.H_grid)
        T_new = _interp1d(H_cur + dH_diss + dH_ext, self.H_grid, self.T_grid)
        dT_phys = T_new - T

        self.last_dissipative = dH_diss.detach()                 # energy (H, [K])
        self.last_external = dH_ext.detach()
        return self._to_normalized_increment(dT_phys)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class MeshGraphNet(nn.Module):
    """Encoder-Processor-Decoder MeshGraphNet.

    Example
    -------
    >>> cfg = MeshGraphNetConfig(node_in_dim=12, edge_in_dim=3, out_dim=1)
    >>> model = MeshGraphNet(cfg)
    >>> dT = model(x, edge_index, edge_attr)  # (num_nodes, 1)
    """

    def __init__(self, cfg: MeshGraphNetConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.processor = nn.ModuleList(
            GraphNetBlock(cfg) for _ in range(cfg.num_processing_steps)
        )
        if cfg.use_generic and cfg.out_dim != 1:
            raise ValueError(
                "use_generic=True requires out_dim == 1 (scalar temperature state)."
            )
        # Conductance-based heads ("full", "enthalpy") derive the increment from
        # the processor latents, not the decoder, so the decoder is omitted there.
        conductance_head = cfg.use_generic and cfg.generic_mode in ("full", "enthalpy")
        self.decoder = None if conductance_head else Decoder(cfg)
        # Instantiated only when enabled -> exactly zero overhead when disabled.
        if not cfg.use_generic:
            self.generic_head = None
        elif cfg.generic_mode == "enthalpy":
            self.generic_head = EnthalpyGenericThermalHead(cfg)
        elif cfg.generic_mode == "full":
            self.generic_head = FullGenericThermalHead(cfg)
        else:
            self.generic_head = GenericThermalHead(cfg)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run a single forward pass over one graph.

        Parameters
        ----------
        x:
            Node features ``(num_nodes, node_in_dim)``.
        edge_index:
            Connectivity ``(2, num_edges)`` (long); row 0 = senders, row 1 =
            receivers. Bidirectional edges should be provided explicitly.
        edge_attr:
            Edge features ``(num_edges, edge_in_dim)``.
        batch:
            Optional ``(num_nodes,)`` PyG graph-assignment vector. Used only by
            the GENERIC head to project per graph; ignored otherwise.

        Returns
        -------
        torch.Tensor
            Per-node output ``(num_nodes, out_dim)``.
        """
        # Keep a reference to the raw inputs for the optional GENERIC head.
        x_raw = x

        # Encode raw features into latent space.
        x, edge_attr = self.encoder(x, edge_attr)

        # Processor: sequence of residual message-passing hops.
        for block in self.processor:
            x, edge_attr = block(x, edge_index, edge_attr)

        # Full GENERIC: the dissipative increment is built from the processor
        # node latents (learned conductances) — the decoder is bypassed.
        if self.generic_head is not None and self.generic_head.uses_node_latents:
            return self.generic_head(
                None, x_raw, batch=batch, node_latents=x, edge_index=edge_index
            )

        # Decode node latents to the output quantity.
        out = self.decoder(x)

        # Optional energy-only GENERIC routing. Skipped entirely when disabled
        # (generic_head is None) -> no extra cost or parameters.
        if self.generic_head is not None:
            out = self.generic_head(out, x_raw, batch)
        return out

    def set_normalization(self, stats: dict) -> None:
        """Give the GENERIC head its z-score constants (no-op when disabled).

        Call once on a fresh model before training; the constants are saved in
        ``state_dict`` so checkpoint loads (rollout / spiral) restore them.
        """
        if self.generic_head is not None:
            self.generic_head.set_normalization(stats)

    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
def _sanity_check() -> None:
    """Verify tensor shapes and gradient flow through a random graph pass."""
    torch.manual_seed(0)

    num_nodes, num_undirected = 200, 600
    cfg = MeshGraphNetConfig(
        node_in_dim=16, edge_in_dim=3, out_dim=1,
        hidden_dim=64, num_mlp_layers=2, num_processing_steps=10,
    )

    # Random bidirectional connectivity (mirrors the mesh-graph convention).
    e = torch.randint(0, num_nodes, (2, num_undirected))
    e = e[:, e[0] != e[1]]  # drop self-loops
    edge_index = torch.cat([e, e.flip(0)], dim=1)
    num_edges = edge_index.size(1)

    x = torch.randn(num_nodes, cfg.node_in_dim)
    edge_attr = torch.randn(num_edges, cfg.edge_in_dim)

    model = MeshGraphNet(cfg)
    out = model(x, edge_index, edge_attr)

    assert out.shape == (num_nodes, cfg.out_dim), out.shape
    print(f"forward OK: x{tuple(x.shape)} edge_attr{tuple(edge_attr.shape)} "
          f"-> out{tuple(out.shape)}")
    print(f"parameters: {model.num_parameters():,}")

    # Gradient flow through the deep residual stack.
    loss = out.pow(2).mean()
    loss.backward()
    grad_norm = torch.cat(
        [p.grad.flatten() for p in model.parameters() if p.grad is not None]
    ).norm()
    assert torch.isfinite(grad_norm) and grad_norm > 0, grad_norm
    print(f"backward OK: grad-norm = {grad_norm:.4e}")

    # Configurability smoke test: vary the key structural knobs.
    for act in ("relu", "silu", "gelu"):
        for steps in (1, 5):
            for ln in (True, False):
                c = MeshGraphNetConfig(
                    hidden_dim=32, num_mlp_layers=1,
                    num_processing_steps=steps, activation=act, use_layer_norm=ln,
                )
                y = MeshGraphNet(c)(x, edge_index, edge_attr)
                assert y.shape == (num_nodes, c.out_dim)
    print("configurability OK: activations x hops x layer-norm all pass")


if __name__ == "__main__":
    _sanity_check()
