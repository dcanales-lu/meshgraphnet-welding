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


class GenericThermalHead(nn.Module):
    """Structure-preserving head for the pure-thermal dissipative GENERIC system.

    The evolution ``T_dot = M(T) grad S(T) + Q_ext`` (the reversible part
    ``L grad E`` vanishes for a pure thermal problem) is realised at the
    increment level as ``dT = P . dT_tilde + q_ext``, where, using global
    energy ``E = sum rho Cp T_i V_i`` and entropy
    ``S = sum rho Cp ln(T_i) V_i``:

    * ``P . dT_tilde`` projects the decoder's unconstrained increment onto
      the complement of grad(E) (:func:`energy_conserving_projection`),
      guaranteeing the degeneracy ``M grad E = 0`` -- the learned dissipative
      redistribution conserves internal energy, removing the spurious global
      heating/cooling drift behind unphysical temperature collapse.
    * ``q_ext`` is the external energy exchange built from the analytical
      physical features: a heating term following the Goldak source field and
      a Newton boundary-cooling term relaxing Robin nodes toward ``T_inf``.
      Each carries a learnable non-negative ``softplus`` magnitude that
      absorbs the ``dt / rho Cp`` scale while preserving the physical sign
      structure (sources add energy; boundary cooling removes it).

    Uniform tributary volumes are used for grad(E) (``rho Cp V_i`` constant),
    as permitted for simplicity.
    """

    def __init__(self, cfg: MeshGraphNetConfig):
        super().__init__()
        self.t_idx = cfg.temperature_index
        self.q_idx = cfg.goldak_index
        self.robin_idx = cfg.robin_onehot_index
        self.tinf_idx = cfg.t_inf_index
        self.source_gain = nn.Parameter(torch.zeros(1))
        self.cool_gain = nn.Parameter(torch.zeros(1))
        # Most-recent components, cached (detached) for inspection / tests.
        self.last_dissipative: Optional[torch.Tensor] = None
        self.last_external: Optional[torch.Tensor] = None

    def forward(
        self,
        delta_t_tilde: torch.Tensor,
        x_raw: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Energy-conserving dissipative redistribution (M grad E = 0 by design).
        dissipative = energy_conserving_projection(delta_t_tilde, batch=batch)

        # Analytical external exchange from physical node features.
        source = F.softplus(self.source_gain) * x_raw[:, self.q_idx:self.q_idx + 1]
        robin = x_raw[:, self.robin_idx:self.robin_idx + 1]
        cooling = F.softplus(self.cool_gain) * robin * (
            x_raw[:, self.tinf_idx:self.tinf_idx + 1]
            - x_raw[:, self.t_idx:self.t_idx + 1]
        )
        external = source + cooling

        self.last_dissipative = dissipative.detach()
        self.last_external = external.detach()
        return dissipative + external


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
        self.decoder = Decoder(cfg)
        if cfg.use_generic and cfg.out_dim != 1:
            raise ValueError(
                "use_generic=True requires out_dim == 1 (scalar temperature state)."
            )
        # Instantiated only when enabled -> exactly zero overhead when disabled.
        self.generic_head = GenericThermalHead(cfg) if cfg.use_generic else None

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

        # Decode node latents to the output quantity.
        out = self.decoder(x)

        # Optional thermodynamic (GENERIC) routing. Skipped entirely when
        # disabled (generic_head is None) -> no extra cost or parameters.
        if self.generic_head is not None:
            out = self.generic_head(out, x_raw, batch)
        return out

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
