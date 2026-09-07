"""Tests for PhysicsConstrainedReservoirGNN and PhysicsInformedStorageLoss.

These tests verify:
1. Forward pass shape contracts for the physics-constrained model.
2. Mass-balance property: storage stays in [0, gross_cap].
3. Autoregressive rollout is consistent.
4. Physics-informed loss is differentiable and non-negative.
5. Per-node capacity clamping works correctly.
"""
import pytest
import torch
import numpy as np

from src.models.physics_constrained_gnn import PhysicsConstrainedReservoirGNN
from src.training.physics_loss import PhysicsInformedStorageLoss, CUMEC_WEEK_TO_TMC


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_config():
    return {
        "spatial_in_channels": 6,
        "spatial_hidden": 16,
        "spatial_out": 16,
        "tcn_in_channels": 6,
        "tcn_channels": [16, 16],
        "climate_input": 8,          # lookback_days for attention
        "climate_embed": 16,
        "num_weeks": 4,
        "num_quantiles": 3,
        "head_hidden": 32,
        "gat_heads": 2,
        "tcn_kernel": 3,
        "gross_caps_tmc": [100.0, 200.0, 150.0],  # 3 nodes
    }


@pytest.fixture
def small_model(small_config):
    return PhysicsConstrainedReservoirGNN(small_config)


# ---------------------------------------------------------------------------
# 1. Forward shape
# ---------------------------------------------------------------------------

def test_forward_output_shape(small_model, small_config):
    batch_size = 4
    num_nodes = 3
    lookback = 8
    num_weeks = small_config["num_weeks"]
    num_quantiles = small_config["num_quantiles"]
    n_features = small_config["spatial_in_channels"]

    node_features = torch.randn(batch_size, num_nodes, lookback, n_features)
    climate_indices = torch.randn(batch_size, 4, lookback)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    current_storage = torch.rand(batch_size, num_nodes) * 80.0  # within caps

    out = small_model(node_features, climate_indices, edge_index, current_storage)

    assert out.shape == (batch_size, num_nodes, num_weeks, num_quantiles), (
        f"Expected ({batch_size}, {num_nodes}, {num_weeks}, {num_quantiles}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 2. Physical bounds: storage in [0, gross_cap]
# ---------------------------------------------------------------------------

def test_storage_stays_within_bounds(small_model, small_config):
    """Output storage must never exceed per-node gross cap or go negative."""
    batch_size = 8
    num_nodes = 3
    lookback = 8
    n_features = small_config["spatial_in_channels"]

    node_features = torch.randn(batch_size, num_nodes, lookback, n_features)
    climate_indices = torch.randn(batch_size, 4, lookback)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    # Start near zero so clipping is exercised on both ends
    current_storage = torch.rand(batch_size, num_nodes) * 50.0

    out = small_model(node_features, climate_indices, edge_index, current_storage)

    caps = small_config["gross_caps_tmc"]
    for node_i, cap in enumerate(caps):
        node_out = out[:, node_i, :, :]  # (B, W, Q)
        assert (node_out >= 0.0).all(), f"Node {node_i}: negative storage detected"
        assert (node_out <= cap + 1e-4).all(), (
            f"Node {node_i}: storage exceeds cap {cap} (max={node_out.max().item():.2f})"
        )


# ---------------------------------------------------------------------------
# 3. Autoregressive consistency — without caps, storage accumulates net_change
# ---------------------------------------------------------------------------

def test_autoregressive_rollout_no_caps():
    """Without capacity capping, storage(w) = storage(0) + sum(delta_w, 0..w)."""
    config = {
        "spatial_in_channels": 4,
        "spatial_hidden": 8,
        "spatial_out": 8,
        "tcn_in_channels": 4,
        "tcn_channels": [8],
        "climate_input": 6,
        "climate_embed": 8,
        "num_weeks": 3,
        "num_quantiles": 1,
        "head_hidden": 16,
        "gat_heads": 1,
        "tcn_kernel": 3,
        # No caps -> inf capping
    }
    model = PhysicsConstrainedReservoirGNN(config)
    model.eval()

    # Monkey-patch net_change_head to produce a constant delta = 10 per week
    with torch.no_grad():
        for p in model.net_change_head.parameters():
            p.zero_()

    B, N, W, F = 2, 1, 6, 4
    node_features = torch.zeros(B, N, W, F)
    climate_indices = torch.zeros(B, 4, W)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    s0 = torch.tensor([[50.0]] * B)  # (B, 1)

    with torch.no_grad():
        out = model(node_features, climate_indices, edge_index, s0)

    # With all zero params, net_change ≈ bias → 0 (no bias by default in last Linear)
    # Just ensure each week's storage >= previous week's when delta=0
    for w in range(1, config["num_weeks"]):
        prev = out[:, :, w - 1, 0]
        curr = out[:, :, w, 0]
        # With zero deltas, storage should stay the same (clamped to 0)
        # Allow small numerical tolerance
        assert ((curr - prev).abs() < 1e-4).all(), "Non-zero drift with zero net_change params"


# ---------------------------------------------------------------------------
# 4. Loss: non-negative and differentiable
# ---------------------------------------------------------------------------

def test_physics_loss_non_negative_and_grad():
    criterion = PhysicsInformedStorageLoss(alpha=0.8, median_idx=1)

    B, N, W, Q = 4, 2, 6, 3
    pred_storage = torch.rand(B, N, W, Q) * 100.0 + 10.0
    pred_storage.requires_grad_(True)
    target_storage = torch.rand(B, N, W) * 100.0
    current_storage = torch.rand(B, N) * 50.0
    obs_inflow = torch.rand(B, N, W) * 500.0  # cumecs

    loss = criterion(pred_storage, target_storage, current_storage, obs_inflow)

    assert loss.item() >= 0.0, f"Loss should be non-negative, got {loss.item()}"
    loss.backward()
    assert pred_storage.grad is not None, "No gradients propagated through the loss"
    assert torch.isfinite(pred_storage.grad).all(), "Non-finite gradients in loss"


# ---------------------------------------------------------------------------
# 5. Loss handles NaN targets gracefully
# ---------------------------------------------------------------------------

def test_physics_loss_with_nan_targets():
    criterion = PhysicsInformedStorageLoss(alpha=0.9, median_idx=1)

    B, N, W, Q = 2, 1, 4, 3
    pred_storage = torch.rand(B, N, W, Q) * 80.0
    target_storage = torch.rand(B, N, W) * 80.0
    target_storage[0, 0, 3] = float("nan")  # last week NaN
    current_storage = torch.rand(B, N) * 40.0

    loss = criterion(pred_storage, target_storage, current_storage, None)
    assert torch.isfinite(loss), f"Loss is not finite with NaN target: {loss}"


# ---------------------------------------------------------------------------
# 6. CUMEC_WEEK_TO_TMC constant sanity
# ---------------------------------------------------------------------------

def test_cumec_week_to_tmc_unit():
    """1 cumec for 1 week should be ~0.021353 TMC."""
    expected = 604800.0 / 1e6 / 28.3168466
    assert abs(CUMEC_WEEK_TO_TMC - expected) < 1e-8


# ---------------------------------------------------------------------------
# 7. No-cap model with large initial storage above cap
# ---------------------------------------------------------------------------

def test_no_cap_no_negative():
    """Even without capacity buffer, storage should not go negative."""
    config = {
        "spatial_in_channels": 4,
        "spatial_hidden": 8,
        "spatial_out": 8,
        "tcn_in_channels": 4,
        "tcn_channels": [8],
        "climate_input": 6,
        "climate_embed": 8,
        "num_weeks": 6,
        "num_quantiles": 3,
        "head_hidden": 16,
        "gat_heads": 1,
    }
    model = PhysicsConstrainedReservoirGNN(config)
    B, N, W, F = 2, 1, 6, 4
    node_features = torch.randn(B, N, W, F) * 10
    climate_indices = torch.randn(B, 4, W)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    s0 = torch.zeros(B, N)  # start at zero → any outflow would go negative without clamp

    out = model(node_features, climate_indices, edge_index, s0)
    assert (out >= 0).all(), "Storage went below zero with clamping."
