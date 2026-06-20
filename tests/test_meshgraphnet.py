"""Tests for the MeshGraphNet architecture (``models.meshgraphnet``)."""

from __future__ import annotations

import pytest
import torch

from models.meshgraphnet import (
    GraphNetBlock,
    MeshGraphNet,
    MeshGraphNetConfig,
    build_mlp,
)


def _random_graph(num_nodes=120, num_undirected=300, node_dim=12, edge_dim=3):
    e = torch.randint(0, num_nodes, (2, num_undirected))
    e = e[:, e[0] != e[1]]
    edge_index = torch.cat([e, e.flip(0)], dim=1)
    x = torch.randn(num_nodes, node_dim)
    edge_attr = torch.randn(edge_index.size(1), edge_dim)
    return x, edge_index, edge_attr


def test_forward_output_shape():
    cfg = MeshGraphNetConfig(node_in_dim=12, edge_in_dim=3, out_dim=1,
                             hidden_dim=32, num_processing_steps=4)
    x, ei, ea = _random_graph()
    out = MeshGraphNet(cfg)(x, ei, ea)
    assert out.shape == (x.size(0), cfg.out_dim)
    assert torch.all(torch.isfinite(out))


def test_gradients_flow_through_deep_stack():
    cfg = MeshGraphNetConfig(hidden_dim=32, num_processing_steps=20)
    x, ei, ea = _random_graph()
    model = MeshGraphNet(cfg)
    model(x, ei, ea).pow(2).mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) == len(list(model.parameters()))  # every param got a grad
    total = torch.cat([g.flatten() for g in grads]).norm()
    assert torch.isfinite(total) and total > 0


def test_residual_connections_preserve_dims():
    """A GraphNet block returns node/edge tensors of unchanged latent width."""
    cfg = MeshGraphNetConfig(hidden_dim=24)
    block = GraphNetBlock(cfg)
    n, m = 50, 200
    x = torch.randn(n, cfg.hidden_dim)
    edge_index = torch.randint(0, n, (2, m))
    edge_attr = torch.randn(m, cfg.hidden_dim)
    x2, e2 = block(x, edge_index, edge_attr)
    assert x2.shape == x.shape and e2.shape == edge_attr.shape


@pytest.mark.parametrize("activation", ["relu", "silu", "gelu", "tanh", "elu"])
def test_activation_configurable(activation):
    cfg = MeshGraphNetConfig(hidden_dim=16, num_processing_steps=2,
                             activation=activation)
    x, ei, ea = _random_graph(num_nodes=40, num_undirected=80)
    assert MeshGraphNet(cfg)(x, ei, ea).shape == (40, 1)


@pytest.mark.parametrize("num_mlp_layers", [0, 1, 3])
@pytest.mark.parametrize("use_layer_norm", [True, False])
def test_mlp_depth_and_layernorm_toggle(num_mlp_layers, use_layer_norm):
    cfg = MeshGraphNetConfig(hidden_dim=16, num_mlp_layers=num_mlp_layers,
                             num_processing_steps=2, use_layer_norm=use_layer_norm)
    x, ei, ea = _random_graph(num_nodes=40, num_undirected=80)
    assert MeshGraphNet(cfg)(x, ei, ea).shape == (40, 1)


def test_aggregation_variants():
    x, ei, ea = _random_graph(num_nodes=40, num_undirected=80)
    for aggr in ("sum", "mean", "max"):
        cfg = MeshGraphNetConfig(hidden_dim=16, num_processing_steps=2,
                                 aggregation=aggr)
        assert MeshGraphNet(cfg)(x, ei, ea).shape == (40, 1)


def test_build_mlp_structure():
    # num_hidden_layers=0 -> single Linear (+ optional LayerNorm).
    mlp0 = build_mlp(8, 16, 4, 0, torch.nn.ReLU, layer_norm=False)
    assert sum(isinstance(m, torch.nn.Linear) for m in mlp0) == 1
    mlp2 = build_mlp(8, 16, 4, 2, torch.nn.ReLU, layer_norm=True)
    assert sum(isinstance(m, torch.nn.Linear) for m in mlp2) == 3
    assert isinstance(mlp2[-1], torch.nn.LayerNorm)


def test_invalid_activation_raises():
    with pytest.raises(ValueError):
        MeshGraphNet(MeshGraphNetConfig(activation="not_an_activation"))


# ---------------------------------------------------------------------------
# GENERIC structure-preserving thermal head
# ---------------------------------------------------------------------------
from models.meshgraphnet import energy_conserving_projection  # noqa: E402


def _two_graph_batch(n1=40, n2=55, node_dim=12, edge_dim=3):
    """Two disjoint random graphs concatenated into one PyG-style batch."""
    x1, e1, a1 = _random_graph(n1, 120, node_dim, edge_dim)
    x2, e2, a2 = _random_graph(n2, 150, node_dim, edge_dim)
    x = torch.cat([x1, x2], dim=0)
    edge_index = torch.cat([e1, e2 + n1], dim=1)
    edge_attr = torch.cat([a1, a2], dim=0)
    batch = torch.cat([torch.zeros(n1, dtype=torch.long), torch.ones(n2, dtype=torch.long)])
    return x, edge_index, edge_attr, batch


def test_energy_conserving_projection_zeroes_weighted_sum():
    """P v is orthogonal to grad(E): with uniform volumes the per-graph sum is 0."""
    torch.manual_seed(0)
    v = torch.randn(95, 1)
    batch = torch.cat([torch.zeros(40, dtype=torch.long), torch.ones(55, dtype=torch.long)])
    proj = energy_conserving_projection(v, batch=batch)  # grad(E) = ones
    for g in batch.unique():
        assert proj[batch == g].sum().abs() < 1e-4  # grad(E) . (P v) == 0
    assert energy_conserving_projection(v).sum().abs() < 1e-4  # single-graph path


def test_generic_model_conserves_energy_per_graph():
    """use_generic=True: the dissipative increment satisfies M grad(E) = 0 per graph."""
    torch.manual_seed(0)
    cfg = MeshGraphNetConfig(hidden_dim=32, num_processing_steps=3, use_generic=True)
    model = MeshGraphNet(cfg)
    x, ei, ea, batch = _two_graph_batch()
    out = model(x, ei, ea, batch=batch)

    assert out.shape == (x.size(0), 1)
    assert torch.all(torch.isfinite(out))
    # Uniform grad(E) -> energy conservation is a zero weighted (plain) sum per graph.
    diss = model.generic_head.last_dissipative
    for g in batch.unique():
        assert diss[batch == g].sum().abs() < 1e-3


def test_generic_disabled_has_zero_overhead():
    """Disabled: no head, no extra parameters (head adds exactly 2 scalar gains)."""
    base = MeshGraphNetConfig(hidden_dim=32, num_processing_steps=3)
    gen = MeshGraphNetConfig(hidden_dim=32, num_processing_steps=3, use_generic=True)
    m_off, m_on = MeshGraphNet(base), MeshGraphNet(gen)
    assert m_off.generic_head is None
    assert m_on.generic_head is not None
    assert m_on.num_parameters() == m_off.num_parameters() + 2


def test_generic_requires_scalar_output():
    with pytest.raises(ValueError):
        MeshGraphNet(MeshGraphNetConfig(out_dim=2, use_generic=True))
