"""Typed data contracts for the P1 toy/reference implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


GATE_MODES: Tuple[str, str] = (
    "full_softmax_selected",
    "topk_renormalized",
)


@dataclass(frozen=True)
class QuantizationSpec:
    """A fixed-scale integer code format used by every candidate in a set."""

    bits: int
    qmin: int
    qmax: int
    scale: float
    zero_point: int = 0
    group_size: Optional[int] = None

    def __post_init__(self) -> None:
        if self.bits < 1:
            raise ValueError("bits must be positive")
        if self.qmin >= self.qmax:
            raise ValueError("qmin must be smaller than qmax")
        if self.qmax - self.qmin + 1 > 2**self.bits:
            raise ValueError("quantization codebook has more values than bits can encode")
        if self.zero_point < self.qmin or self.zero_point > self.qmax:
            raise ValueError("zero_point must lie inside the quantization codebook")
        if not torch.isfinite(torch.tensor(float(self.scale))) or self.scale <= 0:
            raise ValueError("scale must be finite and positive")
        if self.group_size is not None and self.group_size <= 0:
            raise ValueError("group_size must be positive when specified")

    def validate_codes(self, codes: torch.Tensor) -> None:
        """Reject nonintegral or out-of-format integer code tensors."""
        if not torch.isfinite(codes).all():
            raise ValueError("codes contain non-finite values")
        if not torch.allclose(codes, torch.round(codes), atol=0.0, rtol=0.0):
            raise ValueError("codes must be integral")
        if bool((codes < self.qmin).any()) or bool((codes > self.qmax).any()):
            raise ValueError("codes fall outside the fixed quantization range")

    def dequantize(self, codes: torch.Tensor) -> torch.Tensor:
        self.validate_codes(codes)
        return (codes - float(self.zero_point)) * float(self.scale)


@dataclass
class CandidateSet:
    """Unique shared-weight binary candidates derived without a loss oracle."""

    spec: QuantizationSpec
    base_codes: torch.Tensor
    code_directions: torch.Tensor
    coordinates: torch.Tensor
    z_values: torch.Tensor
    candidate_ids: List[str]
    duplicates: Dict[str, str] = field(default_factory=dict)
    rejected_masks: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.spec.validate_codes(self.base_codes)
        if self.code_directions.ndim != self.base_codes.ndim + 1:
            raise ValueError("code_directions must have leading coordinate dimension")
        if tuple(self.code_directions.shape[1:]) != tuple(self.base_codes.shape):
            raise ValueError("code_directions shape does not match base_codes")
        if self.z_values.ndim != 2 or self.z_values.shape[1] != self.code_directions.shape[0]:
            raise ValueError("z_values has the wrong coordinate dimension")
        if len(self.candidate_ids) != self.z_values.shape[0]:
            raise ValueError("candidate ids do not match z_values")
        if self.z_values.shape[0] == 0 or not bool(torch.all(self.z_values[0] == 0)):
            raise ValueError("the no-op candidate must be present first")

    @property
    def p(self) -> int:
        return int(self.code_directions.shape[0])

    @property
    def count(self) -> int:
        return int(self.z_values.shape[0])

    @property
    def base_q(self) -> torch.Tensor:
        return self.spec.dequantize(self.base_codes)

    def continuous_codes(self, z: torch.Tensor) -> torch.Tensor:
        """Continuous coordinate extension used only for local derivatives."""
        if z.ndim != 1 or z.shape[0] != self.p:
            raise ValueError("z must be a one-dimensional vector of length p")
        directions = self.code_directions.to(dtype=z.dtype, device=z.device)
        base = self.base_codes.to(dtype=z.dtype, device=z.device)
        return base + torch.einsum("p,p...->...", z, directions)

    def codes_for_z(self, z: torch.Tensor) -> torch.Tensor:
        if not torch.all((z == 0) | (z == 1)):
            raise ValueError("deployable candidate coordinates must be binary")
        codes = self.continuous_codes(z)
        self.spec.validate_codes(codes.detach())
        return codes

    def continuous_q(self, z: torch.Tensor) -> torch.Tensor:
        codes = self.continuous_codes(z)
        return (codes - float(self.spec.zero_point)) * float(self.spec.scale)

    def q_for_z(self, z: torch.Tensor) -> torch.Tensor:
        codes = self.codes_for_z(z)
        return (codes - float(self.spec.zero_point)) * float(self.spec.scale)

    def codes_for_index(self, index: int) -> torch.Tensor:
        return self.codes_for_z(self.z_values[index])

    def q_for_index(self, index: int) -> torch.Tensor:
        return self.q_for_z(self.z_values[index])

    def modifications_for_index(self, index: int) -> List[Dict[str, int]]:
        z = self.z_values[index]
        changes: List[Dict[str, int]] = []
        for coordinate, enabled in enumerate(z.tolist()):
            if enabled:
                location = self.coordinates[coordinate].tolist()
                delta = int(self.code_directions[coordinate][tuple(location)].item())
                changes.append({"coordinate": coordinate, "row": int(location[0]), "column": int(location[1]), "code_delta": delta})
        return changes


@dataclass
class RouteState:
    """Routing information at one hidden-state endpoint."""

    sets: torch.Tensor
    logits: torch.Tensor
    probabilities: torch.Tensor
    near_tie: torch.Tensor
    boundary_margin: torch.Tensor


@dataclass
class ForwardResult:
    output: torch.Tensor
    hidden: torch.Tensor
    normalized: torch.Tensor
    route: RouteState
    selected_weights: torch.Tensor
    expert_calls: int


@dataclass
class RoutedInputs:
    """All-token router state, intentionally computed without expert calls."""

    hidden: torch.Tensor
    normalized: torch.Tensor
    route: RouteState


@dataclass
class EventOnlyRouteCache:
    """Reusable q0 router result shared across candidate endpoint checks."""

    old_endpoint: RoutedInputs
    token_count: int
    hidden_dimension: int


@dataclass
class EventOnlyEventResult:
    """Exact J_event only; non-event expert outputs are deliberately absent."""

    signed_event: torch.Tensor
    event_per_token: torch.Tensor
    event_formula_per_token: torch.Tensor
    jump_quadratic_per_token: torch.Tensor
    event_mask: torch.Tensor
    old_route: RouteState
    new_route: RouteState
    event_indices: torch.Tensor
    fixed_event_output: torch.Tensor
    actual_event_output: torch.Tensor
    expert_calls: int
    q0_route_reused: bool
    evaluator: str


@dataclass
class LossDecomposition:
    """Exact endpoint decomposition, with both aggregate and per-token terms."""

    actual_delta: torch.Tensor
    fixed_delta: torch.Tensor
    signed_event: torch.Tensor
    actual_per_token: torch.Tensor
    fixed_per_token: torch.Tensor
    event_per_token: torch.Tensor
    event_formula_per_token: torch.Tensor
    jump_quadratic_per_token: torch.Tensor
    event_mask: torch.Tensor
    old_route: RouteState
    new_route: RouteState
    old_output: torch.Tensor
    fixed_output: torch.Tensor
    actual_output: torch.Tensor
    expert_calls: int
    evaluator: str


@dataclass
class DerivativeResult:
    gradient: torch.Tensor
    hessian: torch.Tensor
    gauss_newton: torch.Tensor
    old_sets: torch.Tensor
    fixed_loss_at_zero: torch.Tensor


@dataclass
class ScoreResult:
    candidate_id: str
    first_order: float
    second_order: float
    signed_event: float
    event_corrected: float
    nonnegative_jump_ablation: float


@dataclass
class SelectionMetrics:
    opportunity: float
    regret: float
    normalized_regret: Optional[float]
    selected_loss: float
    oracle_loss: float
    reason_normalized_regret_is_null: Optional[str]


@dataclass
class RunManifest:
    run_id: str
    command: Sequence[str]
    config_sha256: str
    code_commit: Optional[str]
    dirty: Optional[bool]
    device: str
    dtype: str
    seed: Optional[int]
    extra: Dict[str, Any] = field(default_factory=dict)
