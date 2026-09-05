"""Calibration-only proxy scoring; no holdout or oracle-loss arguments exist here."""
from __future__ import annotations

from typing import Iterable, List

import torch

from .types import DerivativeResult, LossDecomposition, ScoreResult


def score_candidate(
    candidate_id: str,
    z: torch.Tensor,
    derivative: DerivativeResult,
    decomposition: LossDecomposition,
) -> ScoreResult:
    """Score one candidate by g^Tz + .5z^THz plus endpoint event correction."""
    z = z.to(dtype=derivative.gradient.dtype, device=derivative.gradient.device)
    first = torch.dot(derivative.gradient, z)
    second = first + 0.5 * torch.einsum("p,pq,q->", z, derivative.hessian, z)
    signed = decomposition.signed_event.to(dtype=second.dtype, device=second.device)
    nonnegative = decomposition.jump_quadratic_per_token.mean().to(dtype=second.dtype, device=second.device)
    return ScoreResult(
        candidate_id=candidate_id,
        first_order=float(first.detach().item()),
        second_order=float(second.detach().item()),
        signed_event=float(signed.detach().item()),
        event_corrected=float((second + signed).detach().item()),
        nonnegative_jump_ablation=float((second + nonnegative).detach().item()),
    )


def select_from_calibration(scores: Iterable[ScoreResult]) -> ScoreResult:
    """Select by a calibration proxy only; evaluation belongs in metrics.py."""
    values: List[ScoreResult] = list(scores)
    if not values:
        raise ValueError("at least one calibration score is required")
    return min(values, key=lambda result: (result.event_corrected, result.candidate_id))
