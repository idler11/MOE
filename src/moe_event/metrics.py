"""Post-selection diagnostics, including explicitly labelled holdout hindsight."""
from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .types import SelectionMetrics


def _finite_losses(losses: Sequence[float]) -> List[float]:
    values = [float(value) for value in losses]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("losses must be a nonempty finite sequence")
    return values


def selection_metrics(candidate_losses: Sequence[float], selected_index: int, noop_index: int = 0) -> SelectionMetrics:
    """Evaluate an already selected code against a diagnostic oracle table."""
    losses = _finite_losses(candidate_losses)
    if selected_index < 0 or selected_index >= len(losses) or noop_index < 0 or noop_index >= len(losses):
        raise IndexError("candidate index is out of range")
    oracle = min(losses)
    opportunity = losses[noop_index] - oracle
    regret = losses[selected_index] - oracle
    if abs(opportunity) <= 1.0e-15:
        normalized = None
        reason = "zero_or_numerically_zero_opportunity"
    else:
        normalized = regret / opportunity
        reason = None
    return SelectionMetrics(
        opportunity=opportunity,
        regret=regret,
        normalized_regret=normalized,
        selected_loss=losses[selected_index],
        oracle_loss=oracle,
        reason_normalized_regret_is_null=reason,
    )


def holdout_diagnostic(holdout_losses: Sequence[float], selected_index: int, noop_index: int = 0) -> Dict[str, Optional[float]]:
    """Evaluate a fixed calibration choice; this function never chooses an index."""
    result = selection_metrics(holdout_losses, selected_index, noop_index)
    return {
        "selected_holdout_loss": result.selected_loss,
        "holdout_hindsight_oracle_loss": result.oracle_loss,
        "holdout_hindsight_regret": result.regret,
        "holdout_noop_loss": float(holdout_losses[noop_index]),
    }


def safe_spearman(left: Sequence[float], right: Sequence[float]) -> Tuple[Optional[float], Optional[str]]:
    """Small dependency-free Spearman calculation with explicit undefined cases."""
    x = _finite_losses(left)
    y = _finite_losses(right)
    if len(x) != len(y) or len(x) < 2:
        return None, "need_equal_length_at_least_two"
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None, "constant_input"

    def rank(values: Sequence[float]) -> List[float]:
        order = sorted(range(len(values)), key=lambda index: (values[index], index))
        ranks = [0.0] * len(values)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and values[order[end]] == values[order[start]]:
                end += 1
            average = (start + 1 + end) / 2.0
            for position in range(start, end):
                ranks[order[position]] = average
            start = end
        return ranks

    rx, ry = rank(x), rank(y)
    mean_x, mean_y = sum(rx) / len(rx), sum(ry) / len(ry)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    denominator = math.sqrt(sum((a - mean_x) ** 2 for a in rx) * sum((b - mean_y) ** 2 for b in ry))
    return numerator / denominator, None


def json_dumps_strict(value: Any) -> str:
    """Serialize public summaries while rejecting NaN and Infinity."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
