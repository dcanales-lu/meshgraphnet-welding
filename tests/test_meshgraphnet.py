"""Tests for the MeshGraphNet architecture (``models.meshgraphnet``)."""

from __future__ import annotations

import pytest
import torch

from models.meshgraphnet import (
    EnthalpyGenericThermalHead,
    FullGenericThermalHead,
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


def _realistic_stats(f=12):
    """z-score constants with a melt-pool-scale q_goldak std (~1e9)."""
    x_mean = torch.zeros(f)
    x_std = torch.ones(f)
    x_std[0] = 300.0       # temperature std [K]
    x_std[1] = 1.0e9       # q_goldak std (huge physical source)
    x_std[11] = 10.0       # T_inf std [K]
    mask = torch.ones(f, dtype=torch.bool)
    mask[[6, 7, 8]] = False  # BC one-hot columns are not normalized
    return {
        "x_mean": x_mean, "x_std": x_std,
        "y_mean": torch.tensor(0.0), "y_std": torch.tensor(30.0),
        "normalize_mask": mask,
    }


def test_generic_physical_head_scale_stable_and_conserving():
    """After set_normalization the head runs in physical units without blow-up."""
    torch.manual_seed(0)
    model = MeshGraphNet(MeshGraphNetConfig(hidden_dim=16, num_processing_steps=2,
                                            use_generic=True))
    model.set_normalization(_realistic_stats())

    # softplus(gain) must absorb dt/(rho Cp) ~ y_std / q_std (~3e-8), not ~softplus(0).
    sp_src = torch.nn.functional.softplus(model.generic_head.source_gain).item()
    assert abs(sp_src - 30.0 / 1.0e9) < 1e-8

    x, ei, ea, batch = _two_graph_batch()          # normalized-scale inputs O(1)
    out = model(x, ei, ea, batch=batch)
    assert torch.all(torch.isfinite(out))
    assert out.abs().max() < 1e3                    # no 1e9 explosion

    # Dissipative part conserves PHYSICAL energy per graph (Σ ≈ 0).
    diss = model.generic_head.last_dissipative
    for g in batch.unique():
        assert diss[batch == g].sum().abs() < 1e-2


def test_generic_set_normalization_preserves_trained_gains():
    """A second set_normalization refreshes buffers but must not re-init gains."""
    model = MeshGraphNet(MeshGraphNetConfig(hidden_dim=8, num_processing_steps=1,
                                            use_generic=True))
    model.set_normalization(_realistic_stats())
    with torch.no_grad():
        model.generic_head.source_gain.add_(1.234)   # pretend training moved it
    g = model.generic_head.source_gain.clone()
    model.set_normalization(_realistic_stats())       # refresh
    assert torch.allclose(model.generic_head.source_gain, g)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_generic_cuda_parity():
    """No device mismatch: cpu and cuda give matching output."""
    torch.manual_seed(0)
    model = MeshGraphNet(MeshGraphNetConfig(hidden_dim=16, num_processing_steps=2,
                                            use_generic=True))
    model.set_normalization(_realistic_stats())
    x, ei, ea, batch = _two_graph_batch()
    out_cpu = model(x, ei, ea, batch=batch)
    m = model.cuda()
    out_cuda = m(x.cuda(), ei.cuda(), ea.cuda(), batch=batch.cuda()).cpu()
    assert torch.allclose(out_cpu, out_cuda, atol=1e-4)


# ---------------------------------------------------------------------------
# Full (second-law) GENERIC head
# ---------------------------------------------------------------------------
def _chain_graph(temps_phys, hidden=16, t_std=300.0):
    """A bidirectional chain over nodes with the given physical temperatures.

    x_raw is all zeros except the (normalized) temperature column 0, so the
    external source/cooling terms vanish and we can probe the dissipative part.
    """
    n = len(temps_phys)
    f = 12
    x = torch.zeros(n, f)
    x[:, 0] = torch.tensor(temps_phys) / t_std        # normalized T (x_mean=0)
    und = torch.tensor([[i, i + 1] for i in range(n - 1)], dtype=torch.long).t()
    edge_index = torch.cat([und, und.flip(0)], dim=1)  # bidirectional
    h = torch.randn(n, hidden)
    return x, edge_index, h


def test_full_generic_builds_and_bypasses_decoder():
    cfg = MeshGraphNetConfig(hidden_dim=16, num_processing_steps=2,
                             use_generic=True, generic_mode="full")
    model = MeshGraphNet(cfg)
    assert model.decoder is None                       # decoder omitted in full mode
    assert isinstance(model.generic_head, FullGenericThermalHead)
    assert model.generic_head.uses_node_latents is True
    # More than +2 params (cond MLP + scale), vs energy head's +2.
    base = MeshGraphNet(MeshGraphNetConfig(hidden_dim=16, num_processing_steps=2))
    assert model.num_parameters() > base.num_parameters() + 2

    model.set_normalization(_realistic_stats())
    x, ei, ea, batch = _two_graph_batch()
    out = model(x, ei, ea, batch=batch)
    assert out.shape == (x.size(0), 1) and torch.all(torch.isfinite(out))


def test_full_generic_structure_preserving():
    """SPSD Laplacian dissipation: energy conserved, entropy produced, hot→cold."""
    torch.manual_seed(0)
    cfg = MeshGraphNetConfig(hidden_dim=16, num_processing_steps=2,
                             use_generic=True, generic_mode="full")
    head = FullGenericThermalHead(cfg)
    head.set_normalization(_realistic_stats())

    temps = [2000.0, 500.0, 400.0, 350.0, 300.0]       # node 0 hot, rest cooler
    x, edge_index, h = _chain_graph(temps)
    head(None, x, batch=None, node_latents=h, edge_index=edge_index)
    diss = head.last_dissipative                        # = L_w μ  (N,1)

    # 1) Degeneracy M·∇E = 0  -> energy conserved: Σ_i dT_diss_i ≈ 0.
    assert diss.sum().abs() < 1e-3 * diss.abs().sum().clamp_min(1e-9)
    # 2) Entropy production dS/dt = μᵀ L_w μ ≥ 0  (μ = 1/T).
    mu = 1.0 / torch.tensor(temps).clamp_min(head.T_FLOOR).reshape(-1, 1)
    assert (mu * diss).sum().item() >= -1e-4
    # 3) Heat flows hot→cold: the hottest node's dissipative increment cools it.
    assert diss[0, 0].item() < 0.0


def test_enthalpy_generic_structure_and_latent_heat():
    """Enthalpy-state GENERIC: energy (not T) conserved; melt-pool plateau."""
    torch.manual_seed(0)
    cfg = MeshGraphNetConfig(hidden_dim=16, num_processing_steps=2,
                             use_generic=True, generic_mode="enthalpy")
    model = MeshGraphNet(cfg)
    assert model.decoder is None                        # decoder bypassed
    assert isinstance(model.generic_head, EnthalpyGenericThermalHead)
    head = model.generic_head
    head.set_normalization(_realistic_stats())

    # Enthalpy curve must be monotonically increasing (invertible) and span the
    # melting range where the slope (apparent heat capacity) spikes.
    H_grid, T_grid = head.H_grid, head.T_grid
    assert torch.all(torch.diff(H_grid) > 0)
    slope = torch.diff(H_grid) / torch.diff(T_grid)     # ~ c_p^app / c_p^sens
    i_melt = int(torch.argmin((T_grid - 1748.0).abs()))
    i_cold = int(torch.argmin((T_grid - 800.0).abs()))
    assert slope[i_melt] > 5.0 * slope[i_cold]          # latent-heat spike at melt

    temps = [2000.0, 500.0, 400.0, 350.0, 300.0]
    x, edge_index, hlat = _chain_graph(temps)
    out = head(None, x, batch=None, node_latents=hlat, edge_index=edge_index)
    dh = head.last_dissipative                          # Δh_diss = L_w μ (energy)

    # 1) GENUINE energy conservation: Σ_i Δh_diss_i ≈ 0 (Laplacian on ENERGY).
    assert dh.sum().abs() < 1e-3 * dh.abs().sum().clamp_min(1e-9)
    # 2) Entropy production μᵀ L_w μ ≥ 0.
    mu = 1.0 / torch.tensor(temps).clamp_min(head.T_FLOOR).reshape(-1, 1)
    assert (mu * dh).sum().item() >= -1e-4
    # 3) Output is a finite normalized ΔT.
    assert out.shape == (5, 1) and torch.all(torch.isfinite(out))


def test_enthalpy_enriched_source_conserves_and_starts_as_scalar():
    """Enriched source: adds source/cool MLPs; zero-initialized (so it starts as
    the scalar head); dissipation still conserves energy per graph."""
    enr = MeshGraphNetConfig(hidden_dim=16, num_processing_steps=2,
                             use_generic=True, generic_mode="enthalpy",
                             enriched_source=True)
    base = MeshGraphNetConfig(hidden_dim=16, num_processing_steps=2,
                              use_generic=True, generic_mode="enthalpy")
    assert MeshGraphNet(enr).num_parameters() > MeshGraphNet(base).num_parameters()

    torch.manual_seed(0)
    m = MeshGraphNet(enr)
    head = m.generic_head
    head.set_normalization(_realistic_stats())
    # Zero-init last layer ⇒ the modulation is 0 at init ⇒ external term equals the
    # scalar-gain head (softplus(gain + 0) = softplus(gain)).
    assert torch.all(head.source_mlp[-1].weight == 0) and torch.all(head.source_mlp[-1].bias == 0)
    assert torch.all(head.cool_mlp[-1].weight == 0) and torch.all(head.cool_mlp[-1].bias == 0)

    x, ei, ea, batch = _two_graph_batch()
    out = m(x, ei, ea, batch=batch)
    assert out.shape == (x.size(0), 1) and torch.all(torch.isfinite(out))
    # Dissipative energy change still sums to ~0 per graph (1st law intact).
    dh = head.last_dissipative
    for g in batch.unique():
        assert dh[batch == g].sum().abs() < 1e-2 * dh.abs().sum().clamp_min(1e-9)


def test_enthalpy_latent_heat_damps_temperature_rise():
    """The same energy input raises T far less at the melt pool than in cold metal."""
    cfg = MeshGraphNetConfig(hidden_dim=8, num_processing_steps=1,
                             use_generic=True, generic_mode="enthalpy")
    head = MeshGraphNet(cfg).generic_head
    head.set_normalization(_realistic_stats())
    from models.meshgraphnet import _interp1d
    # Inject the same enthalpy increment ΔH (temperature-equivalent, K) at a cold
    # node (800 K) and a melting-range node (~1748 K); compare the resulting ΔT.
    dH = torch.full((2, 1), 100.0)                      # K (energy-equivalent)
    T = torch.tensor([[800.0], [1748.0]])
    H0 = _interp1d(T, head.T_grid, head.H_grid)
    T1 = _interp1d(H0 + dH, head.H_grid, head.T_grid)
    dT = (T1 - T).squeeze(-1)
    assert dT[0] > 3.0 * dT[1]                          # latent heat absorbs energy
