"""Tests for EMAUpdater in core/ema.py.

Tests cover cosine-scheduled momentum, parameter update behavior,
and gradient tracking requirements.
"""
import copy
import math

import pytest
import torch
import torch.nn as nn

from core.ema import EMAUpdater


def make_online_and_target():
    """Create a simple online network and a target (momentum) copy."""
    online = nn.Linear(10, 5)
    target = copy.deepcopy(online)
    target.requires_grad_(False)
    return online, target


def test_ema_step_momentum_at_step_0():
    """Test 1: current_momentum at step 0 equals base_momentum."""
    ema = EMAUpdater(base_momentum=0.996, end_momentum=1.0, total_steps=1000)
    assert abs(ema.current_momentum - 0.996) < 1e-9, (
        f"Expected 0.996 at step 0, got {ema.current_momentum}"
    )


def test_ema_step_momentum_at_total_steps():
    """Test 2: current_momentum at step total_steps approaches end_momentum."""
    total_steps = 1000
    ema = EMAUpdater(base_momentum=0.996, end_momentum=1.0, total_steps=total_steps)
    # Advance to total_steps
    ema._step = total_steps
    assert abs(ema.current_momentum - 1.0) < 1e-6, (
        f"Expected ~1.0 at step {total_steps}, got {ema.current_momentum}"
    )


def test_ema_step_moves_target_toward_online():
    """Test 3: step() moves target parameters closer to online parameters."""
    online, target = make_online_and_target()
    # Corrupt target so it differs from online
    with torch.no_grad():
        for p in target.parameters():
            p.data.fill_(0.0)
    # Compute initial distance
    online_params = list(online.parameters())
    target_params = list(target.parameters())
    initial_dist = sum(
        (p_o.data - p_t.data).norm().item()
        for p_o, p_t in zip(online_params, target_params)
    )

    ema = EMAUpdater(base_momentum=0.9, end_momentum=1.0, total_steps=1000)
    ema.step(online.parameters(), target.parameters())

    final_dist = sum(
        (p_o.data - p_t.data).norm().item()
        for p_o, p_t in zip(online_params, target_params)
    )
    assert final_dist < initial_dist, (
        f"Target should move closer to online after step. "
        f"Initial dist: {initial_dist:.4f}, Final dist: {final_dist:.4f}"
    )


def test_ema_step_target_remains_no_grad():
    """Test 4: Target parameters with requires_grad=False remain requires_grad=False after step."""
    online, target = make_online_and_target()
    # Verify target has no grad before
    for p in target.parameters():
        assert not p.requires_grad, "Target should start with requires_grad=False"

    ema = EMAUpdater(base_momentum=0.996, end_momentum=1.0, total_steps=1000)
    ema.step(online.parameters(), target.parameters())

    for p in target.parameters():
        assert not p.requires_grad, (
            "Target requires_grad should remain False after EMA step"
        )


def test_ema_step_online_not_modified():
    """Test 5: Online parameters are NOT modified by step() — only target is updated."""
    online, target = make_online_and_target()
    # Save a deep copy of online params before step
    online_before = [p.data.clone() for p in online.parameters()]

    ema = EMAUpdater(base_momentum=0.996, end_momentum=1.0, total_steps=1000)
    ema.step(online.parameters(), target.parameters())

    for p_before, p_after in zip(online_before, online.parameters()):
        assert torch.allclose(p_before, p_after.data), (
            "Online parameters should not be modified by EMA step"
        )


def test_ema_step_momentum_one_leaves_target_unchanged():
    """Test 6: Multiple steps with momentum=1.0 leave target unchanged.

    When momentum=1.0: target = 1.0 * target + 0.0 * online = target (no change).
    """
    online, target = make_online_and_target()
    # Corrupt target to differ from online
    with torch.no_grad():
        for p in target.parameters():
            p.data.fill_(99.0)

    target_before = [p.data.clone() for p in target.parameters()]

    # Use momentum=1.0 by setting base_momentum=end_momentum=1.0
    ema = EMAUpdater(base_momentum=1.0, end_momentum=1.0, total_steps=1000)
    for _ in range(5):
        ema.step(online.parameters(), target.parameters())

    for p_before, p_after in zip(target_before, target.parameters()):
        assert torch.allclose(p_before, p_after.data), (
            "Target should remain unchanged when momentum=1.0"
        )


def test_ema_step_momentum_schedule_monotonically_nondecreasing():
    """Test 7: Momentum schedule is monotonically non-decreasing over steps."""
    total_steps = 100
    ema = EMAUpdater(base_momentum=0.99, end_momentum=1.0, total_steps=total_steps)

    momenta = []
    for step in range(total_steps + 1):
        ema._step = step
        momenta.append(ema.current_momentum)

    for i in range(1, len(momenta)):
        assert momenta[i] >= momenta[i - 1] - 1e-12, (
            f"Momentum should be non-decreasing, but momenta[{i}]={momenta[i]:.8f} "
            f"< momenta[{i-1}]={momenta[i-1]:.8f}"
        )
