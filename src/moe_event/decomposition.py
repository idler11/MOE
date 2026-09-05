"""Exact finite-step fixed-branch plus signed routing-event decomposition."""
from __future__ import annotations

from typing import Optional

import torch

from .routing import endpoint_events
from .toy import ToyMoE
from .types import LossDecomposition


def default_metric(hidden_dimension: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """M=I/d, so loss is mean over hidden dimension then mean over tokens."""
    return torch.eye(hidden_dimension, dtype=dtype, device=device) / float(hidden_dimension)


def validate_metric(metric: Optional[torch.Tensor], hidden_dimension: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if metric is None:
        return default_metric(hidden_dimension, dtype, device)
    if metric.shape != (hidden_dimension, hidden_dimension):
        raise ValueError("metric must have shape [hidden, hidden]")
    metric = metric.to(dtype=dtype, device=device)
    if not torch.allclose(metric, metric.transpose(0, 1), atol=1.0e-12, rtol=1.0e-10):
        raise ValueError("metric must be symmetric")
    eigenvalues = torch.linalg.eigvalsh(metric)
    if bool((eigenvalues < -1.0e-12).any()):
        raise ValueError("metric must be positive semidefinite")
    return metric


def per_token_loss(output: torch.Tensor, target: torch.Tensor, metric: Optional[torch.Tensor] = None) -> torch.Tensor:
    if output.shape != target.shape or output.ndim != 2:
        raise ValueError("output and target must share shape [tokens, hidden]")
    matrix = validate_metric(metric, output.shape[1], output.dtype, output.device)
    residual = output - target
    return torch.einsum("nd,de,ne->n", residual, matrix, residual)


def signed_event_from_residual_jump(
    residual: torch.Tensor,
    jump: torch.Tensor,
    metric: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if residual.shape != jump.shape or residual.ndim != 2:
        raise ValueError("residual and jump must share shape [tokens, hidden]")
    matrix = validate_metric(metric, residual.shape[1], residual.dtype, residual.device)
    cross = 2.0 * torch.einsum("nd,de,ne->n", residual, matrix, jump)
    quadratic = torch.einsum("nd,de,ne->n", jump, matrix, jump)
    return cross + quadratic


def jump_quadratic(
    jump: torch.Tensor,
    metric: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    matrix = validate_metric(metric, jump.shape[1], jump.dtype, jump.device)
    return torch.einsum("nd,de,ne->n", jump, matrix, jump)


def finite_step_decomposition(
    model: ToyMoE,
    inputs: torch.Tensor,
    bias: torch.Tensor,
    q0: torch.Tensor,
    q1: torch.Tensor,
    target: torch.Tensor,
    gate_mode: str,
    metric: Optional[torch.Tensor] = None,
    evaluator: str = "dense",
) -> LossDecomposition:
    """Compute Delta L=Delta L_fixed+J_event at the candidate endpoint.

    ``event_only`` never materializes all expert outputs.  It calculates the
    old and new selected paths for all tokens, then evaluates the old forced
    path at q1 only for event tokens.  Non-events reuse the new path because
    the forced old set equals the new set there; gate values are still those
    recomputed at q1.
    """
    if evaluator not in ("dense", "event_only"):
        raise ValueError("evaluator must be 'dense' or 'event_only'")
    use_sparse = evaluator == "event_only"
    old = model(inputs, bias, q0, gate_mode=gate_mode, sparse=use_sparse)
    actual = model(inputs, bias, q1, gate_mode=gate_mode, sparse=use_sparse)
    event_mask = endpoint_events(old.route.sets, actual.route.sets)
    calls = old.expert_calls + actual.expert_calls
    if evaluator == "dense":
        fixed = model(
            inputs,
            bias,
            q1,
            forced_sets=old.route.sets,
            gate_mode=gate_mode,
            sparse=False,
        )
        fixed_output = fixed.output
        calls += fixed.expert_calls
    elif bool(event_mask.any()):
        event_rows = torch.nonzero(event_mask, as_tuple=False).squeeze(1)
        forced_event = model(
            inputs.index_select(0, event_rows),
            bias.index_select(0, event_rows),
            q1,
            forced_sets=old.route.sets.index_select(0, event_rows),
            gate_mode=gate_mode,
            sparse=True,
        )
        fixed_output = actual.output.clone()
        fixed_output[event_rows] = forced_event.output
        calls += forced_event.expert_calls
    else:
        fixed_output = actual.output
    old_losses = per_token_loss(old.output, target, metric)
    fixed_losses = per_token_loss(fixed_output, target, metric)
    actual_losses = per_token_loss(actual.output, target, metric)
    residual = fixed_output - target
    jump = actual.output - fixed_output
    formula = signed_event_from_residual_jump(residual, jump, metric)
    event = actual_losses - fixed_losses
    return LossDecomposition(
        actual_delta=(actual_losses - old_losses).mean(),
        fixed_delta=(fixed_losses - old_losses).mean(),
        signed_event=event.mean(),
        actual_per_token=actual_losses - old_losses,
        fixed_per_token=fixed_losses - old_losses,
        event_per_token=event,
        event_formula_per_token=formula,
        jump_quadratic_per_token=jump_quadratic(jump, metric),
        event_mask=event_mask,
        old_route=old.route,
        new_route=actual.route,
        old_output=old.output,
        fixed_output=fixed_output,
        actual_output=actual.output,
        expert_calls=calls,
        evaluator=evaluator,
    )
