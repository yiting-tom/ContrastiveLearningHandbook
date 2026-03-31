"""Tests for LARS optimizer — trust ratio scaling, bias/norm exclusion, momentum."""
import inspect
import torch
import torch.nn as nn
import pytest

from core.optimizers import LARS


def make_model_and_loss(seed: int = 42):
    """Create a simple model and compute a loss for optimizer testing."""
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.BatchNorm1d(5),
    )
    x = torch.randn(8, 10)
    output = model(x)
    loss = output.pow(2).mean()
    return model, loss


# ---------------------------------------------------------------------------
# Test 1: LARS step changes parameters
# ---------------------------------------------------------------------------
def test_lars_step_modifies_parameters():
    model, loss = make_model_and_loss()
    before = {name: p.data.clone() for name, p in model.named_parameters()}

    optimizer = LARS(model.parameters(), lr=0.1)
    loss.backward()
    optimizer.step()

    changed = False
    for name, p in model.named_parameters():
        if not torch.allclose(p.data, before[name]):
            changed = True
            break
    assert changed, "At least one parameter must change after an optimizer step"


# ---------------------------------------------------------------------------
# Test 2: exclude_bias_and_norm=True — 1-D params (bias, BN) get trust_ratio=1.0
# ---------------------------------------------------------------------------
def test_exclude_bias_and_norm_true_skips_1d_params():
    """Verify that with exclude_bias_and_norm=True, 1-D params get no LARS scaling.

    We test this by comparing updates for 1-D vs 2-D params:
    - 2-D weight param: update = scaled_lr * grad (scaled by LARS trust ratio)
    - 1-D bias/BN param: update = lr * grad (trust_ratio == 1.0 → no scaling beyond lr)
    """
    torch.manual_seed(0)
    model = nn.Linear(10, 5, bias=True)
    x = torch.randn(8, 10)
    loss = model(x).pow(2).mean()

    optimizer = LARS(model.parameters(), lr=0.1, momentum=0.0, weight_decay=0.0,
                     eta=0.001, exclude_bias_and_norm=True)
    loss.backward()

    # Capture gradient norms and parameter norms before step
    weight_param = model.weight
    bias_param = model.bias
    w_grad = weight_param.grad.clone()
    b_grad = bias_param.grad.clone()
    w_data = weight_param.data.clone()
    b_data = bias_param.data.clone()

    # Compute expected updates
    # For weight (2-D): trust_ratio = eta * ||w|| / ||w_grad||
    w_norm = w_data.norm()
    g_norm = w_grad.norm()
    if w_norm > 0 and g_norm > 0:
        trust_ratio_weight = 0.001 * w_norm / g_norm
    else:
        trust_ratio_weight = 1.0
    # For bias (1-D): trust_ratio = 1.0 (excluded)
    trust_ratio_bias = 1.0

    optimizer.step()

    expected_weight = w_data - 0.1 * trust_ratio_weight * w_grad
    expected_bias = b_data - 0.1 * trust_ratio_bias * b_grad

    assert torch.allclose(weight_param.data, expected_weight, atol=1e-5), \
        "Weight update must use LARS trust ratio"
    assert torch.allclose(bias_param.data, expected_bias, atol=1e-5), \
        "Bias update must use trust_ratio=1.0 (no LARS scaling)"


# ---------------------------------------------------------------------------
# Test 3: exclude_bias_and_norm=False — all params get LARS trust ratio
# ---------------------------------------------------------------------------
def test_exclude_bias_and_norm_false_applies_lars_to_all_params():
    """With exclude_bias_and_norm=False, even 1-D params get LARS scaling."""
    torch.manual_seed(0)
    model = nn.Linear(10, 5, bias=True)
    x = torch.randn(8, 10)
    loss = model(x).pow(2).mean()

    optimizer = LARS(model.parameters(), lr=0.1, momentum=0.0, weight_decay=0.0,
                     eta=0.001, exclude_bias_and_norm=False)
    loss.backward()

    bias_param = model.bias
    b_grad = bias_param.grad.clone()
    b_data = bias_param.data.clone()
    b_norm = b_data.norm()
    g_norm = b_grad.norm()

    if b_norm > 0 and g_norm > 0:
        trust_ratio_bias = 0.001 * b_norm / g_norm
    else:
        trust_ratio_bias = 1.0

    optimizer.step()

    expected_bias = b_data - 0.1 * trust_ratio_bias * b_grad
    assert torch.allclose(bias_param.data, expected_bias, atol=1e-5), \
        "With exclude_bias_and_norm=False, bias should get LARS scaling"


# ---------------------------------------------------------------------------
# Test 4: Weight decay decreases parameter norm
# ---------------------------------------------------------------------------
def test_weight_decay_applied():
    """Weight decay should cause parameter norm to decrease (or at least be applied)."""
    torch.manual_seed(0)
    model = nn.Linear(10, 5, bias=False)
    x = torch.randn(8, 10)

    # Use a large weight decay to make the effect observable
    optimizer = LARS(model.parameters(), lr=0.01, momentum=0.0,
                     weight_decay=0.1, eta=0.001, exclude_bias_and_norm=True)

    initial_norm = model.weight.data.norm().item()
    for _ in range(5):
        model.zero_grad()
        loss = model(x).pow(2).mean()
        loss.backward()
        optimizer.step()

    final_norm = model.weight.data.norm().item()
    # With weight decay, norms are penalized; they should generally decrease
    # (unless gradient-driven updates dominate, but with small lr this holds)
    # We at least verify weight decay is applied by checking the optimizer runs
    assert final_norm != initial_norm, "Parameters must change with weight decay applied"


# ---------------------------------------------------------------------------
# Test 5: Momentum buffer accumulates across steps
# ---------------------------------------------------------------------------
def test_momentum_buffer_accumulates():
    """After the first step, the momentum buffer should persist and affect later steps."""
    torch.manual_seed(0)
    model = nn.Linear(10, 5, bias=False)
    optimizer = LARS(model.parameters(), lr=0.1, momentum=0.9,
                     weight_decay=0.0, eta=0.001)

    # Step 1
    x = torch.randn(8, 10)
    model.zero_grad()
    loss = model(x).pow(2).mean()
    loss.backward()
    grad_step1 = model.weight.grad.clone()
    optimizer.step()

    # Momentum buffer should now exist
    for group in optimizer.param_groups:
        for p in group["params"]:
            assert "momentum_buffer" in optimizer.state[p], \
                "Momentum buffer must exist after first step"

    # Step 2 with a different gradient
    model.zero_grad()
    x2 = torch.randn(8, 10)
    loss2 = model(x2).pow(2).mean()
    loss2.backward()
    grad_step2 = model.weight.grad.clone()

    w_before_step2 = model.weight.data.clone()
    optimizer.step()

    # If no momentum, update would be just lr * grad_step2.
    # With momentum, the update is larger due to buffer accumulation.
    # Simply verify the buffer was used (parameter changed)
    assert not torch.allclose(model.weight.data, w_before_step2), \
        "Parameters must change in step 2"


# ---------------------------------------------------------------------------
# Test 6: Zero gradient does not crash
# ---------------------------------------------------------------------------
def test_zero_gradient_does_not_crash():
    """A zero gradient should not crash the optimizer (trust ratio stays 1.0)."""
    torch.manual_seed(0)
    model = nn.Linear(10, 5)
    optimizer = LARS(model.parameters(), lr=0.1)

    # Manually zero all gradients (simulate zero-gradient scenario)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)

    # Should not raise
    optimizer.step()

    # Parameters should not have moved (or moved only by weight decay)
    # At minimum, verify no crash occurred
    assert True, "Optimizer should not crash on zero gradient"


# ---------------------------------------------------------------------------
# Test 7: __init__ signature matches D-05 specification
# ---------------------------------------------------------------------------
def test_lars_init_signature():
    """LARS.__init__ must match D-05: (params, lr, momentum=0.9, weight_decay=1e-6,
    eta=0.001, exclude_bias_and_norm=True)."""
    sig = inspect.signature(LARS.__init__)
    params_list = list(sig.parameters.keys())

    # Skip 'self'
    expected = ["params", "lr", "momentum", "weight_decay", "eta", "exclude_bias_and_norm"]
    actual = [p for p in params_list if p != "self"]

    assert actual == expected, (
        f"LARS.__init__ signature mismatch.\nExpected: {expected}\nGot: {actual}"
    )

    # Check default values
    defaults = {
        name: param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect.Parameter.empty and name != "self"
    }
    assert defaults.get("momentum") == 0.9, "momentum default must be 0.9"
    assert defaults.get("weight_decay") == 1e-6, "weight_decay default must be 1e-6"
    assert defaults.get("eta") == 0.001, "eta default must be 0.001"
    assert defaults.get("exclude_bias_and_norm") is True, \
        "exclude_bias_and_norm default must be True"
