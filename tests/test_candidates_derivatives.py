from __future__ import annotations

import inspect

import pytest
import torch

from moe_event.candidates import build_candidate_set, generate_fixed_scale_candidates
from moe_event.decomposition import finite_step_decomposition, per_token_loss
from moe_event.derivatives import fixed_branch_derivatives, fixed_branch_hvp, fixed_branch_output
from moe_event.scoring import score_candidate
from moe_event.types import QuantizationSpec

from .helpers import make_derivative_case, make_event_case


def test_t09_fixed_scale_legal_shared_candidates_noop_and_duplicates() -> None:
    source = torch.tensor([[0.31, -0.52], [0.44, -0.27]], dtype=torch.float64)
    spec = QuantizationSpec(bits=3, qmin=-3, qmax=3, scale=0.2)
    candidates = generate_fixed_scale_candidates(source, spec, p=2)
    assert torch.equal(candidates.z_values[0], torch.zeros(2, dtype=torch.float64))
    assert torch.allclose(candidates.q_for_index(0), candidates.base_q)
    for index in range(candidates.count):
        spec.validate_codes(candidates.codes_for_index(index))
    assert "target" not in inspect.signature(generate_fixed_scale_candidates).parameters

    duplicate = build_candidate_set(
        torch.tensor([[0.0]], dtype=torch.float64),
        torch.tensor([[[1.0]], [[1.0]]], dtype=torch.float64),
        spec,
        torch.tensor([[0, 0], [0, 0]], dtype=torch.long),
    )
    assert duplicate.duplicates
    assert torch.equal(duplicate.z_values[0], torch.zeros(2, dtype=torch.float64))
    with pytest.raises(ValueError):
        candidates.q_for_z(torch.tensor([0.25, 0.0], dtype=torch.float64))
    # 2 bits encode at most four distinct code values; zero point must also
    # be a representable code rather than a free floating metadata field.
    with pytest.raises(ValueError, match="codebook"):
        QuantizationSpec(bits=2, qmin=-3, qmax=3, scale=0.2)
    with pytest.raises(ValueError, match="zero_point"):
        QuantizationSpec(bits=3, qmin=-3, qmax=3, zero_point=4, scale=0.2)
    exact_two_bit = QuantizationSpec(bits=2, qmin=-2, qmax=1, zero_point=0, scale=0.2)
    exact_two_bit.validate_codes(torch.tensor([-2.0, -1.0, 0.0, 1.0], dtype=torch.float64))


def test_t10_shared_weight_reachability_blocks_tokenwise_oracle() -> None:
    # One shared scalar u controls both margins; no legal shared update flips
    # token 2 while preserving token 1, even though a tokenwise oracle could.
    def routed(margin: float) -> bool:
        return margin < 0.0

    u = -0.21
    assert routed(0.1 + u)
    assert routed(0.2 + u)
    candidates = [-0.19, -0.20, -0.21, -0.4]
    assert not any((not routed(0.1 + value)) and routed(0.2 + value) for value in candidates)


def test_t11_fixed_branch_gradient_hessian_hvp_and_gn_are_distinct() -> None:
    case = make_derivative_case()
    derivative = fixed_branch_derivatives(
        case["model"], case["inputs"], case["bias"], case["candidates"], case["target"], "full_softmax_selected"
    )
    z0 = torch.zeros(case["candidates"].p, dtype=torch.float64)

    def loss_at(z: torch.Tensor) -> torch.Tensor:
        output = fixed_branch_output(
            case["model"], case["inputs"], case["bias"], case["candidates"], z, derivative.old_sets, "full_softmax_selected"
        )
        return per_token_loss(output, case["target"]).mean()

    def central_gradient(epsilon: float) -> torch.Tensor:
        estimates = []
        for coordinate in range(case["candidates"].p):
            direction = torch.zeros_like(z0)
            direction[coordinate] = epsilon
            estimates.append(((loss_at(z0 + direction) - loss_at(z0 - direction)) / (2.0 * epsilon)).item())
        return torch.tensor(estimates, dtype=torch.float64)

    # Sweep five orders of magnitude.  The two middle steps must form a
    # stable finite-difference window and improve over the coarse endpoint;
    # this prevents a single hand-picked epsilon from passing by accident.
    steps = (1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5)
    estimates = {step: central_gradient(step) for step in steps}
    errors = {step: torch.linalg.vector_norm(estimate - derivative.gradient).item() for step, estimate in estimates.items()}
    assert errors[1.0e-3] < errors[1.0e-1]
    assert errors[1.0e-4] < errors[1.0e-1]
    assert torch.allclose(estimates[1.0e-3], estimates[1.0e-4], atol=1.0e-7, rtol=1.0e-5)
    assert torch.allclose(estimates[1.0e-4], derivative.gradient, atol=1.0e-7, rtol=1.0e-5)
    vector = torch.tensor([0.6, -0.8], dtype=torch.float64)
    hvp = fixed_branch_hvp(
        case["model"],
        case["inputs"],
        case["bias"],
        case["candidates"],
        case["target"],
        "full_softmax_selected",
        vector,
        old_sets=derivative.old_sets,
    )
    assert torch.allclose(hvp, derivative.hessian @ vector, atol=1.0e-10, rtol=1.0e-8)
    # Nonzero residual and SiLU curvature make exact Hessian and GN different.
    assert not torch.allclose(derivative.hessian, derivative.gauss_newton, atol=1.0e-10, rtol=1.0e-8)


def test_t12_event_corrected_error_equals_the_remaining_smooth_taylor_error() -> None:
    case = make_derivative_case()
    candidates = case["candidates"]
    derivative = fixed_branch_derivatives(
        case["model"], case["inputs"], case["bias"], candidates, case["target"], "topk_renormalized"
    )
    index = candidates.count - 1
    z = candidates.z_values[index]
    decomposition = finite_step_decomposition(
        case["model"],
        case["inputs"],
        case["bias"],
        candidates.base_q,
        candidates.q_for_index(index),
        case["target"],
        "topk_renormalized",
        evaluator="dense",
    )
    score = score_candidate(candidates.candidate_ids[index], z, derivative, decomposition)
    second = torch.dot(derivative.gradient, z) + 0.5 * torch.einsum("p,pq,q->", z, derivative.hessian, z)
    actual_minus_score = decomposition.actual_delta - score.event_corrected
    fixed_minus_second = decomposition.fixed_delta - second
    assert torch.allclose(actual_minus_score, fixed_minus_second, atol=1.0e-12, rtol=1.0e-10)
    # This equality measures a realized smooth remainder; it is not a beta certificate.
    assert not hasattr(derivative, "beta_certificate")
