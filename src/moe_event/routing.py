"""Bias-free Top-k routing geometry with a deterministic tie policy."""
from __future__ import annotations

from typing import Optional

import torch


def validate_bias_free(router_bias: Optional[torch.Tensor]) -> None:
    """This exact geometry intentionally has no router-bias extension."""
    if router_bias is not None:
        raise NotImplementedError("router bias is outside the P1 exact routing geometry")


def canonicalize_sets(sets: torch.Tensor, num_experts: Optional[int] = None) -> torch.Tensor:
    if sets.ndim != 2:
        raise ValueError("route sets must have shape [tokens, top_k]")
    if sets.numel() and (bool((sets < 0).any()) or (num_experts is not None and bool((sets >= num_experts).any()))):
        raise ValueError("route set contains an invalid expert index")
    canonical, _ = torch.sort(sets.to(dtype=torch.long), dim=1)
    if canonical.shape[1] > 1 and bool((canonical[:, 1:] == canonical[:, :-1]).any()):
        raise ValueError("route set contains a duplicate expert")
    return canonical


def stable_topk_sets(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k with descending score and lower expert id winning exact ties."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [tokens, experts]")
    experts = logits.shape[1]
    if k < 1 or k > experts:
        raise ValueError("k must be in [1, number of experts]")
    # Expert ids are initially ascending.  Stable descending sort therefore
    # preserves lower ids first when two equal scores compare equal.
    ranking = torch.argsort(logits, dim=1, descending=True, stable=True)
    return canonicalize_sets(ranking[:, :k], experts)


def centered_router(router: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Return A=C_E R D_gamma, whose nullspace preserves Top-k membership."""
    if router.ndim != 2 or gamma.ndim != 1 or router.shape[1] != gamma.shape[0]:
        raise ValueError("router must be [experts, hidden] and gamma [hidden]")
    raw = router * gamma.unsqueeze(0)
    return raw - raw.mean(dim=0, keepdim=True)


def raw_order_logits(hidden: torch.Tensor, router: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Unnormalized logits with the same strict ordering as RMS-normalized logits."""
    if hidden.ndim != 2:
        raise ValueError("hidden must be [tokens, hidden]")
    return hidden @ (router * gamma.unsqueeze(0)).transpose(0, 1)


def boundary_margin(logits: torch.Tensor, sets: torch.Tensor) -> torch.Tensor:
    """Minimum selected-vs-unselected margin; +inf if all experts are selected."""
    sets = canonicalize_sets(sets, logits.shape[1])
    selected_mask = torch.zeros_like(logits, dtype=torch.bool)
    selected_mask.scatter_(1, sets, True)
    selected_min = logits.masked_fill(~selected_mask, float("inf")).min(dim=1).values
    inactive_max = logits.masked_fill(selected_mask, float("-inf")).max(dim=1).values
    margin = selected_min - inactive_max
    return torch.where(torch.isinf(inactive_max), torch.full_like(margin, float("inf")), margin)


def near_tie_flags(logits: torch.Tensor, sets: torch.Tensor, tolerance: float) -> torch.Tensor:
    if tolerance < 0:
        raise ValueError("tie tolerance must be nonnegative")
    return boundary_margin(logits, sets) <= tolerance


def endpoint_events(old_sets: torch.Tensor, new_sets: torch.Tensor) -> torch.Tensor:
    old_sets = canonicalize_sets(old_sets)
    new_sets = canonicalize_sets(new_sets)
    if old_sets.shape != new_sets.shape:
        raise ValueError("old and new route sets must have the same shape")
    return torch.any(old_sets != new_sets, dim=1)


def first_exit_time(
    hidden: torch.Tensor,
    delta: torch.Tensor,
    router: torch.Tensor,
    gamma: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Earliest nonpositive active/inactive margin along h+t*delta.

    The returned value is exactly 1 at an endpoint boundary.  Whether a
    deterministic tie policy changes the endpoint set is intentionally left
    to the caller to record separately.
    """
    if hidden.shape != delta.shape or hidden.ndim != 2:
        raise ValueError("hidden and delta must share shape [tokens, hidden]")
    logits0 = raw_order_logits(hidden, router, gamma)
    velocity = raw_order_logits(delta, router, gamma)
    sets = stable_topk_sets(logits0, k)
    experts = logits0.shape[1]
    all_times = []
    for token in range(hidden.shape[0]):
        active = sets[token]
        inactive_mask = torch.ones(experts, dtype=torch.bool, device=hidden.device)
        inactive_mask[active] = False
        inactive = torch.arange(experts, device=hidden.device)[inactive_mask]
        if inactive.numel() == 0:
            all_times.append(torch.full((), float("inf"), dtype=hidden.dtype, device=hidden.device))
            continue
        margins = logits0[token, active].unsqueeze(1) - logits0[token, inactive].unsqueeze(0)
        velocities = velocity[token, active].unsqueeze(1) - velocity[token, inactive].unsqueeze(0)
        roots = torch.full_like(margins, float("inf"))
        roots = torch.where(velocities < 0, margins / (-velocities), roots)
        all_times.append(roots.min())
    return torch.stack(all_times)
