from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from moe_event.routing import (
    centered_router,
    first_exit_time,
    raw_order_logits,
    stable_topk_sets,
    validate_bias_free,
)
from moe_event.toy import ToyMoE


def test_t01_finite_empirical_and_population_derivatives_are_different_objects() -> None:
    # Independent scalar counterexample, not a call into the event package.
    samples = np.linspace(-0.999, 0.999, 1000)
    theta, epsilon = 0.20337, 1.0e-7

    def empirical_loss(value: float) -> float:
        return float(np.mean(((samples > value).astype(float) - (samples > 0.0)) ** 2))

    finite_difference = (empirical_loss(theta + epsilon) - empirical_loss(theta - epsilon)) / (2.0 * epsilon)
    population_derivative = 0.5
    assert finite_difference == 0.0
    assert population_derivative == 0.5


@pytest.mark.parametrize("gate_mode", ["full_softmax_selected", "topk_renormalized"])
def test_t02_rmsnorm_order_equivalence_and_both_gate_modes(gate_mode: str) -> None:
    dtype = torch.float64
    router = torch.tensor([[1.2, -0.5], [-0.2, 1.1], [0.7, 0.3]], dtype=dtype)
    gamma = torch.tensor([1.3, 0.8], dtype=dtype)
    hidden = torch.tensor([[1.2, -0.4], [0.3, 1.7]], dtype=dtype)
    raw = raw_order_logits(hidden, router, gamma)
    normalized = hidden * gamma / torch.sqrt(hidden.square().mean(dim=1, keepdim=True) + 1.0e-5)
    normalized_logits = normalized @ router.T
    assert torch.equal(stable_topk_sets(raw, 2), stable_topk_sets(normalized_logits, 2))

    model = ToyMoE(
        router,
        gamma,
        torch.ones(3, 2, 2, dtype=dtype),
        torch.ones(3, 2, 2, dtype=dtype),
        top_k=2,
    )
    result = model(hidden, torch.zeros_like(hidden), torch.eye(2, dtype=dtype), gate_mode=gate_mode)
    manual = torch.softmax(result.route.logits, dim=1).gather(1, result.route.sets)
    if gate_mode == "full_softmax_selected":
        assert torch.allclose(result.selected_weights, manual)
        assert bool(torch.all(result.selected_weights.sum(dim=1) < 1.0))
    else:
        assert torch.allclose(result.selected_weights, manual / manual.sum(dim=1, keepdim=True))
        assert torch.allclose(result.selected_weights.sum(dim=1), torch.ones(2, dtype=dtype))


def test_t03_first_exit_checks_all_active_inactive_pairs_and_boundary() -> None:
    dtype = torch.float64
    router = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=dtype)
    gamma = torch.ones(2, dtype=dtype)
    h0 = torch.tensor([[3.0, 2.0]], dtype=dtype)
    delta = torch.tensor([[-4.0, 0.0]], dtype=dtype)
    time = first_exit_time(h0, delta, router, gamma, k=2)
    # Expert 2 is initially below expert 0, but catches it at .75.  Looking
    # only at the current kth (expert 1) versus k+1 would miss this crossing.
    assert torch.allclose(time, torch.tensor([0.75], dtype=dtype))
    start = stable_topk_sets(raw_order_logits(h0, router, gamma), 2)
    endpoint = stable_topk_sets(raw_order_logits(h0 + delta, router, gamma), 2)
    assert not torch.equal(start, endpoint)
    assert math.isinf(float(first_exit_time(h0, torch.tensor([[0.1, 0.0]], dtype=dtype), router, gamma, 2)[0]))

    boundary_h = torch.tensor([[2.0, 1.0]], dtype=dtype)
    boundary_delta = torch.tensor([[-2.0, 0.0]], dtype=dtype)
    assert torch.allclose(first_exit_time(boundary_h, boundary_delta, router, gamma, 2), torch.ones(1, dtype=dtype))
    # A tie at t=1 is recorded as boundary; lower-id stable tie can preserve
    # the endpoint set rather than silently treating it as an ordinary event.
    assert torch.equal(
        stable_topk_sets(raw_order_logits(boundary_h, router, gamma), 2),
        stable_topk_sets(raw_order_logits(boundary_h + boundary_delta, router, gamma), 2),
    )


def test_t04_centered_nullspace_keeps_set_but_not_gate_or_function() -> None:
    dtype = torch.float64
    router = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, -1.0, 0.0]], dtype=dtype)
    gamma = torch.ones(3, dtype=dtype)
    centered = centered_router(router, gamma)
    delta = torch.tensor([[0.0, 0.0, 2.0]], dtype=dtype)
    assert torch.allclose(delta @ centered.T, torch.zeros(1, 3, dtype=dtype))
    model = ToyMoE(
        router,
        gamma,
        torch.tensor([[[0.0, 0.0, 1.0]], [[0.0, 0.0, -1.0]], [[0.0, 0.0, 0.5]]], dtype=dtype),
        torch.ones(3, 3, 1, dtype=dtype),
        top_k=2,
    )
    q = torch.eye(3, dtype=dtype)
    h0 = torch.tensor([[1.0, 0.6, 0.0]], dtype=dtype)
    h1 = h0 + delta
    first = model(h0, torch.zeros_like(h0), q, gate_mode="full_softmax_selected")
    second = model(h1, torch.zeros_like(h1), q, gate_mode="full_softmax_selected")
    assert torch.equal(first.route.sets, second.route.sets)
    assert not torch.allclose(first.selected_weights, second.selected_weights)
    assert not torch.allclose(first.output, second.output)


def test_t14_stable_ties_near_ties_k_extremes_and_bias_rejection() -> None:
    logits = torch.zeros(2, 4, dtype=torch.float64)
    assert torch.equal(stable_topk_sets(logits, 1), torch.tensor([[0], [0]]))
    assert torch.equal(stable_topk_sets(logits, 4), torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]))
    with pytest.raises(ValueError):
        stable_topk_sets(logits, 0)
    with pytest.raises(ValueError):
        stable_topk_sets(logits, 5)
    with pytest.raises(NotImplementedError):
        validate_bias_free(torch.zeros(4, dtype=torch.float64))
    model = ToyMoE(
        torch.tensor([[1.0], [1.0 - 1.0e-13], [-1.0]], dtype=torch.float64),
        torch.ones(1, dtype=torch.float64),
        torch.ones(3, 1, 1, dtype=torch.float64),
        torch.ones(3, 1, 1, dtype=torch.float64),
        top_k=1,
    )
    flagged = model(
        torch.ones(1, 1, dtype=torch.float64),
        torch.zeros(1, 1, dtype=torch.float64),
        torch.ones(1, 1, dtype=torch.float64),
    )
    assert bool(flagged.route.near_tie.item())
