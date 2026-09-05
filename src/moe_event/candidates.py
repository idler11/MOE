"""Fixed-scale, shared-weight candidate construction without loss inspection."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from .types import CandidateSet, QuantizationSpec


def _binary_masks(p: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if p < 0:
        raise ValueError("p must be nonnegative")
    values = torch.arange(2**p, device=device, dtype=torch.long)
    bits = torch.arange(p, device=device, dtype=torch.long)
    return ((values.unsqueeze(1) >> bits.unsqueeze(0)) & 1).to(dtype=dtype)


def build_candidate_set(
    base_codes: torch.Tensor,
    code_directions: torch.Tensor,
    spec: QuantizationSpec,
    coordinates: torch.Tensor,
) -> CandidateSet:
    """Enumerate legal unique binary combinations and record duplicate masks.

    This utility never clips a combination: an out-of-range combination is
    recorded as rejected rather than silently changing the candidate.
    """
    base_codes = base_codes.detach().clone().to(dtype=torch.float64)
    code_directions = code_directions.detach().clone().to(dtype=torch.float64)
    spec.validate_codes(base_codes)
    if code_directions.ndim != base_codes.ndim + 1:
        raise ValueError("code directions need a leading coordinate dimension")
    if tuple(code_directions.shape[1:]) != tuple(base_codes.shape):
        raise ValueError("direction and base shapes differ")
    p = int(code_directions.shape[0])
    if coordinates.shape != (p, 2):
        raise ValueError("coordinates must have shape [p, 2]")
    masks = _binary_masks(p, base_codes.device, base_codes.dtype)
    unique_z: List[torch.Tensor] = []
    candidate_ids: List[str] = []
    duplicates: Dict[str, str] = {}
    rejected_masks: List[str] = []
    seen: Dict[bytes, str] = {}
    for ordinal, z in enumerate(masks):
        codes = base_codes + torch.einsum("p,p...->...", z, code_directions)
        mask_id = "mask_{:0{}b}".format(ordinal, max(p, 1))
        try:
            spec.validate_codes(codes)
        except ValueError:
            rejected_masks.append(mask_id)
            continue
        digest = codes.to(dtype=torch.int64).cpu().contiguous().numpy().tobytes()
        if digest in seen:
            duplicates[mask_id] = seen[digest]
            continue
        candidate_id = "c{:0{}d}".format(len(unique_z), max(1, len(str(max(2**p - 1, 0)))))
        seen[digest] = candidate_id
        unique_z.append(z)
        candidate_ids.append(candidate_id)
    if not unique_z:
        raise ValueError("no legal candidates were generated")
    z_values = torch.stack(unique_z, dim=0)
    return CandidateSet(
        spec=spec,
        base_codes=base_codes,
        code_directions=code_directions,
        coordinates=coordinates.to(dtype=torch.long),
        z_values=z_values,
        candidate_ids=candidate_ids,
        duplicates=duplicates,
        rejected_masks=rejected_masks,
    )


def generate_fixed_scale_candidates(
    source_weights: torch.Tensor,
    spec: QuantizationSpec,
    p: int,
) -> CandidateSet:
    """Choose ambiguous legal floor/ceil coordinates using only source weights.

    The ranking is distance to the rounding midpoint, then flat tensor index.
    It neither accepts a loss nor examines calibration/holdout targets.
    """
    if source_weights.ndim != 2:
        raise ValueError("source_weights must be a matrix")
    if p < 0:
        raise ValueError("p must be nonnegative")
    weights = source_weights.detach().to(dtype=torch.float64)
    unscaled = weights / float(spec.scale) + float(spec.zero_point)
    floor = torch.floor(unscaled)
    ceil = torch.ceil(unscaled)
    base = torch.clamp(torch.round(unscaled), spec.qmin, spec.qmax)
    fractional = unscaled - floor
    ambiguity = torch.abs(fractional - 0.5)
    valid = (floor >= spec.qmin) & (ceil <= spec.qmax) & (ceil > floor)
    ambiguity = ambiguity.masked_fill(~valid, float("inf"))
    available = int(valid.sum().item())
    if p > available:
        raise ValueError("not enough valid floor/ceil coordinates for requested p")
    # Stable sort provides the specified flat-index tie break.
    chosen = torch.argsort(ambiguity.flatten(), stable=True)[:p]
    directions = torch.zeros((p,) + tuple(base.shape), dtype=torch.float64, device=base.device)
    coordinates = torch.empty((p, 2), dtype=torch.long, device=base.device)
    flat_base = base.flatten()
    flat_floor = floor.flatten()
    flat_ceil = ceil.flatten()
    for coordinate, flat_index in enumerate(chosen.tolist()):
        alternative = flat_ceil[flat_index] if flat_base[flat_index] == flat_floor[flat_index] else flat_floor[flat_index]
        directions[coordinate].flatten()[flat_index] = alternative - flat_base[flat_index]
        row = flat_index // base.shape[1]
        column = flat_index % base.shape[1]
        coordinates[coordinate] = torch.tensor([row, column], dtype=torch.long, device=base.device)
    return build_candidate_set(base, directions, spec, coordinates)
