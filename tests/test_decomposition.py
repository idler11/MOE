from __future__ import annotations

import pytest
import torch

from moe_event.decomposition import finite_step_decomposition, signed_event_from_residual_jump

from .helpers import make_event_case


@pytest.mark.parametrize("gate_mode", ["full_softmax_selected", "topk_renormalized"])
def test_t05_finite_step_identity_for_zero_no_event_and_multi_expert_replacement(gate_mode: str) -> None:
    case = make_event_case()
    model = case["model"]
    zero = finite_step_decomposition(
        model,
        case["inputs"],
        case["bias"],
        case["q0"],
        case["q0"],
        case["target"],
        gate_mode,
        evaluator="dense",
    )
    assert torch.allclose(zero.actual_delta, torch.zeros((), dtype=torch.float64))
    assert not bool(zero.event_mask.any())
    assert torch.allclose(zero.signed_event, torch.zeros((), dtype=torch.float64))

    result = finite_step_decomposition(
        model,
        case["inputs"],
        case["bias"],
        case["q0"],
        case["q1"],
        case["target"],
        gate_mode,
        evaluator="dense",
    )
    assert bool(result.event_mask.all())
    # Each route goes from {0,1} to {2,3}; it is not a single-expert shortcut.
    assert torch.equal(result.old_route.sets, torch.tensor([[0, 1], [0, 1], [0, 1]]))
    assert torch.equal(result.new_route.sets, torch.tensor([[2, 3], [2, 3], [2, 3]]))
    assert torch.allclose(result.actual_delta, result.fixed_delta + result.signed_event, atol=1.0e-12, rtol=1.0e-10)
    assert torch.allclose(result.event_per_token, result.event_formula_per_token, atol=1.0e-12, rtol=1.0e-10)


def test_t06_signed_event_has_positive_and_negative_examples() -> None:
    dtype = torch.float64
    residual = torch.tensor([[1.0], [1.0]], dtype=dtype)
    jump = torch.tensor([[-0.8], [1.0]], dtype=dtype)
    terms = signed_event_from_residual_jump(residual, jump, torch.ones(1, 1, dtype=dtype))
    assert torch.allclose(terms, torch.tensor([-0.96, 3.0], dtype=dtype))
    direct = (residual + jump).square().squeeze(1) - residual.square().squeeze(1)
    assert torch.allclose(terms, direct)
    assert terms[0] < 0 < terms[1]


def test_t07_old_input_or_dropping_cross_term_is_wrong() -> None:
    # h0=-.01, h1=.01, target=h0, old forced branch is identity and the new
    # branch adds one.  This is a hand-computed counterexample.
    target = -0.01
    fixed_new = 0.01
    actual_new = 1.01
    correct_event = (actual_new - target) ** 2 - (fixed_new - target) ** 2
    old_input_jump = 1.0
    old_input_residual = 0.0
    wrong_old_input = 2.0 * old_input_residual * old_input_jump + old_input_jump**2
    wrong_drop_cross = (actual_new - fixed_new) ** 2
    assert correct_event == pytest.approx(1.04)
    assert wrong_old_input == pytest.approx(1.0)
    assert wrong_drop_cross == pytest.approx(1.0)
    assert wrong_old_input != pytest.approx(correct_event)
    assert wrong_drop_cross != pytest.approx(correct_event)


@pytest.mark.parametrize("gate_mode", ["full_softmax_selected", "topk_renormalized"])
def test_t08_forced_route_recomputes_candidate_gate_values(gate_mode: str) -> None:
    case = make_event_case()
    model = case["model"]
    old = model(case["inputs"], case["bias"], case["q0"], gate_mode=gate_mode)
    forced_new = model(
        case["inputs"],
        case["bias"],
        case["q1"],
        forced_sets=old.route.sets,
        gate_mode=gate_mode,
    )
    manual = torch.softmax(forced_new.route.logits, dim=1).gather(1, old.route.sets)
    if gate_mode == "topk_renormalized":
        manual = manual / manual.sum(dim=1, keepdim=True)
    assert torch.allclose(forced_new.selected_weights, manual)
    assert not torch.allclose(old.selected_weights, forced_new.selected_weights)
    full = model(
        case["inputs"],
        case["bias"],
        case["q1"],
        forced_sets=old.route.sets,
        gate_mode="full_softmax_selected",
    )
    renorm = model(
        case["inputs"],
        case["bias"],
        case["q1"],
        forced_sets=old.route.sets,
        gate_mode="topk_renormalized",
    )
    assert bool(torch.all(full.selected_weights.sum(dim=1) < 1.0))
    assert torch.allclose(renorm.selected_weights.sum(dim=1), torch.ones(3, dtype=torch.float64))


@pytest.mark.parametrize("gate_mode", ["full_softmax_selected", "topk_renormalized"])
def test_t13_dense_and_event_only_parity_with_real_expert_call_reduction(gate_mode: str) -> None:
    case = make_event_case()
    dense = finite_step_decomposition(
        case["model"], case["inputs"], case["bias"], case["q0"], case["q1"], case["target"], gate_mode, evaluator="dense"
    )
    sparse = finite_step_decomposition(
        case["model"], case["inputs"], case["bias"], case["q0"], case["q1"], case["target"], gate_mode, evaluator="event_only"
    )
    assert torch.equal(dense.event_mask, sparse.event_mask)
    assert torch.allclose(dense.actual_output, sparse.actual_output, atol=1.0e-12, rtol=1.0e-10)
    assert torch.allclose(dense.fixed_output, sparse.fixed_output, atol=1.0e-12, rtol=1.0e-10)
    assert torch.allclose(dense.event_per_token, sparse.event_per_token, atol=1.0e-12, rtol=1.0e-10)
    assert sparse.expert_calls < dense.expert_calls

    q = case["q1"].detach().clone().requires_grad_(True)
    old_sets = dense.old_route.sets
    dense_fixed = case["model"](
        case["inputs"], case["bias"], q, forced_sets=old_sets, gate_mode=gate_mode, sparse=False
    ).output.square().mean()
    dense_grad = torch.autograd.grad(dense_fixed, q)[0]
    q_sparse = case["q1"].detach().clone().requires_grad_(True)
    sparse_fixed = case["model"](
        case["inputs"], case["bias"], q_sparse, forced_sets=old_sets, gate_mode=gate_mode, sparse=True
    ).output.square().mean()
    sparse_grad = torch.autograd.grad(sparse_fixed, q_sparse)[0]
    assert torch.allclose(dense_grad, sparse_grad, atol=1.0e-12, rtol=1.0e-10)
