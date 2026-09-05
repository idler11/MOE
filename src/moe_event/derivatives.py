"""Fixed-route coordinate derivatives and explicitly named Gauss--Newton."""
from __future__ import annotations

from typing import Optional

import torch

from .candidates import CandidateSet
from .decomposition import per_token_loss, validate_metric
from .toy import ToyMoE
from .types import DerivativeResult


def fixed_branch_output(
    model: ToyMoE,
    inputs: torch.Tensor,
    bias: torch.Tensor,
    candidate_set: CandidateSet,
    z: torch.Tensor,
    old_sets: torch.Tensor,
    gate_mode: str,
    sparse: bool = True,
) -> torch.Tensor:
    """Evaluate f_{S0}(h(Q(z))); indices are fixed but gates are not detached."""
    q = candidate_set.continuous_q(z)
    return model(
        inputs,
        bias,
        q,
        forced_sets=old_sets,
        gate_mode=gate_mode,
        sparse=sparse,
    ).output


def fixed_branch_derivatives(
    model: ToyMoE,
    inputs: torch.Tensor,
    bias: torch.Tensor,
    candidate_set: CandidateSet,
    target: torch.Tensor,
    gate_mode: str,
    metric: Optional[torch.Tensor] = None,
) -> DerivativeResult:
    """Exact autograd g/H of the smooth forced-old-set branch at z=0.

    ``gauss_newton`` is returned alongside the full Hessian rather than being
    substituted for it.  The old route indices are calculated once at q0 and
    detached because the branch definition fixes these discrete indices.
    """
    q0 = candidate_set.base_q.to(dtype=inputs.dtype, device=inputs.device)
    with torch.no_grad():
        old_sets = model(inputs, bias, q0, gate_mode=gate_mode, sparse=True).route.sets.detach()
    matrix = validate_metric(metric, target.shape[1], inputs.dtype, inputs.device)
    z0 = torch.zeros(candidate_set.p, dtype=inputs.dtype, device=inputs.device)

    def output_fn(z: torch.Tensor) -> torch.Tensor:
        return fixed_branch_output(model, inputs, bias, candidate_set, z, old_sets, gate_mode, sparse=True)

    def loss_fn(z: torch.Tensor) -> torch.Tensor:
        return per_token_loss(output_fn(z), target, matrix).mean()

    gradient = torch.autograd.functional.jacobian(loss_fn, z0, vectorize=False)
    hessian = torch.autograd.functional.hessian(loss_fn, z0, vectorize=False)
    output_jacobian = torch.autograd.functional.jacobian(output_fn, z0, vectorize=False)
    # output_jacobian is [tokens, hidden, p]; loss is token mean of e^T M e.
    gauss_newton = 2.0 * torch.einsum("ndp,de,neq->pq", output_jacobian, matrix, output_jacobian) / float(inputs.shape[0])
    return DerivativeResult(
        gradient=gradient.detach(),
        hessian=hessian.detach(),
        gauss_newton=gauss_newton.detach(),
        old_sets=old_sets,
        fixed_loss_at_zero=loss_fn(z0).detach(),
    )


def fixed_branch_hvp(
    model: ToyMoE,
    inputs: torch.Tensor,
    bias: torch.Tensor,
    candidate_set: CandidateSet,
    target: torch.Tensor,
    gate_mode: str,
    vector: torch.Tensor,
    old_sets: Optional[torch.Tensor] = None,
    metric: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """HVP for the same forced branch, used to cross-check a full Hessian."""
    q0 = candidate_set.base_q.to(dtype=inputs.dtype, device=inputs.device)
    if old_sets is None:
        with torch.no_grad():
            old_sets = model(inputs, bias, q0, gate_mode=gate_mode, sparse=True).route.sets.detach()
    matrix = validate_metric(metric, target.shape[1], inputs.dtype, inputs.device)
    z = torch.zeros(candidate_set.p, dtype=inputs.dtype, device=inputs.device, requires_grad=True)
    output = fixed_branch_output(model, inputs, bias, candidate_set, z, old_sets, gate_mode, sparse=True)
    loss = per_token_loss(output, target, matrix).mean()
    gradient = torch.autograd.grad(loss, z, create_graph=True)[0]
    return torch.autograd.grad((gradient * vector).sum(), z)[0].detach()
