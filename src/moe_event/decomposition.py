"""Exact finite-step fixed-branch plus signed routing-event decomposition."""
from __future__ import annotations

from typing import Optional, Union

import torch

from .routing import endpoint_events
from .toy import ToyMoE
from .types import EventOnlyEventResult, EventOnlyRouteCache, LossDecomposition


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


def build_event_only_route_cache(
    model: ToyMoE,
    inputs: torch.Tensor,
    bias: torch.Tensor,
    q0: torch.Tensor,
) -> EventOnlyRouteCache:
    """Route q0 once without experts, for reuse by every candidate endpoint."""
    old_endpoint = model.route_from_q(inputs, bias, q0)
    return EventOnlyRouteCache(
        old_endpoint=old_endpoint,
        token_count=int(inputs.shape[0]),
        hidden_dimension=int(q0.shape[0]),
    )


def _event_only_endpoint_event(
    model: ToyMoE,
    inputs: torch.Tensor,
    bias: torch.Tensor,
    q0: torch.Tensor,
    q1: torch.Tensor,
    target: torch.Tensor,
    gate_mode: str,
    metric: Optional[torch.Tensor],
    route_cache: Optional[EventOnlyRouteCache],
) -> EventOnlyEventResult:
    """Evaluate only exact endpoint route events, never non-event experts.

    The dense finite-step loss is intentionally unavailable here: calculating
    it would require evaluating changed fixed-branch outputs for non-event
    tokens.  This path is the event correction used by the scorer, not a
    disguised sparse implementation of the dense oracle.
    """
    cache_was_supplied = route_cache is not None
    cache = route_cache or build_event_only_route_cache(model, inputs, bias, q0)
    if cache.token_count != inputs.shape[0] or cache.hidden_dimension != q1.shape[0]:
        raise ValueError("event-only route cache does not match these token or hidden shapes")
    old_endpoint = cache.old_endpoint
    new_endpoint = model.route_from_q(inputs, bias, q1)
    event_mask = endpoint_events(old_endpoint.route.sets, new_endpoint.route.sets)
    event_indices = torch.nonzero(event_mask, as_tuple=False).squeeze(1)
    per_token = torch.zeros(inputs.shape[0], dtype=target.dtype, device=target.device)
    formula_per_token = torch.zeros_like(per_token)
    quadratic_per_token = torch.zeros_like(per_token)
    if event_indices.numel() == 0:
        empty = target.new_empty((0, target.shape[1]))
        return EventOnlyEventResult(
            signed_event=per_token.mean(),
            event_per_token=per_token,
            event_formula_per_token=formula_per_token,
            jump_quadratic_per_token=quadratic_per_token,
            event_mask=event_mask,
            old_route=old_endpoint.route,
            new_route=new_endpoint.route,
            event_indices=event_indices,
            fixed_event_output=empty,
            actual_event_output=empty,
            expert_calls=0,
            q0_route_reused=cache_was_supplied,
            evaluator="event_only",
        )
    old_sets = old_endpoint.route.sets.index_select(0, event_indices)
    new_sets = new_endpoint.route.sets.index_select(0, event_indices)
    old_weights = model.gate_weights(new_endpoint.route, old_endpoint.route.sets, gate_mode).index_select(0, event_indices)
    new_weights = model.gate_weights(new_endpoint.route, new_endpoint.route.sets, gate_mode).index_select(0, event_indices)
    fixed_experts, actual_experts, expert_calls = model.endpoint_union_outputs(
        new_endpoint.normalized.index_select(0, event_indices),
        old_sets,
        new_sets,
        old_weights,
        new_weights,
    )
    hidden = new_endpoint.hidden.index_select(0, event_indices)
    fixed_event_output = hidden + fixed_experts
    actual_event_output = hidden + actual_experts
    target_events = target.index_select(0, event_indices)
    event_values = per_token_loss(actual_event_output, target_events, metric) - per_token_loss(fixed_event_output, target_events, metric)
    residual = fixed_event_output - target_events
    jump = actual_event_output - fixed_event_output
    formula_values = signed_event_from_residual_jump(residual, jump, metric)
    quadratic_values = jump_quadratic(jump, metric)
    per_token = per_token.index_copy(0, event_indices, event_values)
    formula_per_token = formula_per_token.index_copy(0, event_indices, formula_values)
    quadratic_per_token = quadratic_per_token.index_copy(0, event_indices, quadratic_values)
    return EventOnlyEventResult(
        signed_event=per_token.mean(),
        event_per_token=per_token,
        event_formula_per_token=formula_per_token,
        jump_quadratic_per_token=quadratic_per_token,
        event_mask=event_mask,
        old_route=old_endpoint.route,
        new_route=new_endpoint.route,
        event_indices=event_indices,
        fixed_event_output=fixed_event_output,
        actual_event_output=actual_event_output,
        expert_calls=expert_calls,
        q0_route_reused=cache_was_supplied,
        evaluator="event_only",
    )


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
    route_cache: Optional[EventOnlyRouteCache] = None,
) -> Union[LossDecomposition, EventOnlyEventResult]:
    """Compute Delta L=Delta L_fixed+J_event at the candidate endpoint.

    ``dense`` is the complete reference oracle.  ``event_only`` computes only
    J_event: it routes all tokens, then runs one de-duplicated old/new expert
    union at q1 for each event token.  It deliberately does not return dense
    outputs or total candidate loss for non-events.
    """
    if evaluator not in ("dense", "event_only"):
        raise ValueError("evaluator must be 'dense' or 'event_only'")
    if evaluator == "event_only":
        return _event_only_endpoint_event(model, inputs, bias, q0, q1, target, gate_mode, metric, route_cache)
    if route_cache is not None:
        raise ValueError("route_cache is only valid for event_only")
    old = model(inputs, bias, q0, gate_mode=gate_mode, sparse=False)
    actual = model(inputs, bias, q1, gate_mode=gate_mode, sparse=False)
    event_mask = endpoint_events(old.route.sets, actual.route.sets)
    calls = old.expert_calls + actual.expert_calls
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
