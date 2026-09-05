"""Small, independently testable building blocks for finite-step MoE events.

This package is deliberately a CPU toy/reference implementation.  It does not
load a pretrained model and it does not make a deployment or speed claim.
"""

from .candidates import build_candidate_set, generate_fixed_scale_candidates
from .decomposition import finite_step_decomposition
from .derivatives import fixed_branch_derivatives
from .toy import ToyMoE
from .types import QuantizationSpec

__all__ = [
    "ToyMoE",
    "QuantizationSpec",
    "build_candidate_set",
    "generate_fixed_scale_candidates",
    "finite_step_decomposition",
    "fixed_branch_derivatives",
]
