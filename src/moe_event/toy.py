"""Deterministic bias-free toy MoE used for P1 mathematical reference checks."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .routing import boundary_margin, canonicalize_sets, near_tie_flags, stable_topk_sets
from .types import ForwardResult, GATE_MODES, RouteState


class ToyMoE(torch.nn.Module):
    """Residual RMSNorm + bias-free router + SiLU expert toy module.

    The model is not a claim about a production MoE architecture.  Its only
    purpose is to make the finite-step endpoint definitions executable with
    both dense and actually sparse expert evaluation paths.
    """

    def __init__(
        self,
        router: torch.Tensor,
        gamma: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
        top_k: int,
        rms_epsilon: float = 1.0e-5,
    ) -> None:
        super().__init__()
        if router.ndim != 2:
            raise ValueError("router must have shape [experts, hidden]")
        experts, hidden = router.shape
        if gamma.shape != (hidden,):
            raise ValueError("gamma shape must match router hidden dimension")
        if w_up.ndim != 3 or w_up.shape[0] != experts or w_up.shape[2] != hidden:
            raise ValueError("w_up must have shape [experts, width, hidden]")
        if w_down.ndim != 3 or w_down.shape[0] != experts or w_down.shape[1] != hidden or w_down.shape[2] != w_up.shape[1]:
            raise ValueError("w_down must have shape [experts, hidden, width]")
        if top_k < 1 or top_k > experts:
            raise ValueError("top_k must be between one and the number of experts")
        if rms_epsilon <= 0:
            raise ValueError("rms_epsilon must be positive")
        self.register_buffer("router", router.detach().clone())
        self.register_buffer("gamma", gamma.detach().clone())
        self.register_buffer("w_up", w_up.detach().clone())
        self.register_buffer("w_down", w_down.detach().clone())
        self.top_k = int(top_k)
        self.rms_epsilon = float(rms_epsilon)

    @property
    def num_experts(self) -> int:
        return int(self.router.shape[0])

    @property
    def hidden_dimension(self) -> int:
        return int(self.router.shape[1])

    @property
    def input_dimension(self) -> int:
        # Inputs are checked against Q at forward time; the model owns no Q.
        raise AttributeError("ToyMoE does not own Q; infer input dimension from q")

    @classmethod
    def random(
        cls,
        input_dimension: int,
        hidden_dimension: int,
        expert_hidden_dimension: int,
        num_experts: int,
        top_k: int,
        seed: int,
        dtype: torch.dtype = torch.float64,
        rms_epsilon: float = 1.0e-5,
    ) -> "ToyMoE":
        if input_dimension < 1 or hidden_dimension < 1 or expert_hidden_dimension < 1:
            raise ValueError("all toy dimensions must be positive")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        router = torch.randn(num_experts, hidden_dimension, generator=generator, dtype=dtype) / hidden_dimension**0.5
        gamma = torch.exp(0.1 * torch.randn(hidden_dimension, generator=generator, dtype=dtype))
        w_up = torch.randn(num_experts, expert_hidden_dimension, hidden_dimension, generator=generator, dtype=dtype) / hidden_dimension**0.5
        w_down = torch.randn(num_experts, hidden_dimension, expert_hidden_dimension, generator=generator, dtype=dtype) / expert_hidden_dimension**0.5
        return cls(router, gamma, w_up, w_down, top_k=top_k, rms_epsilon=rms_epsilon)

    def hidden_from_q(self, inputs: torch.Tensor, bias: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or bias.ndim != 2 or q.ndim != 2:
            raise ValueError("inputs, bias and q must be matrices")
        if inputs.shape[0] != bias.shape[0] or inputs.shape[1] != q.shape[1] or bias.shape[1] != q.shape[0]:
            raise ValueError("expected hidden = bias + inputs @ q.T")
        if q.shape[0] != self.hidden_dimension:
            raise ValueError("q output dimension does not match the toy hidden dimension")
        return bias + inputs @ q.transpose(0, 1)

    def rms_normalize(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 2 or hidden.shape[1] != self.hidden_dimension:
            raise ValueError("hidden has the wrong shape")
        denominator = torch.sqrt(hidden.square().mean(dim=1, keepdim=True) + self.rms_epsilon)
        return hidden * self.gamma.unsqueeze(0) / denominator

    def route(self, hidden: torch.Tensor, tie_tolerance: float = 1.0e-12) -> RouteState:
        normalized = self.rms_normalize(hidden)
        logits = normalized @ self.router.transpose(0, 1)
        sets = stable_topk_sets(logits, self.top_k)
        probabilities = torch.softmax(logits, dim=1)
        return RouteState(
            sets=sets,
            logits=logits,
            probabilities=probabilities,
            near_tie=near_tie_flags(logits, sets, tie_tolerance),
            boundary_margin=boundary_margin(logits, sets),
        )

    def _dense_expert_outputs(self, normalized: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(torch.einsum("nd,ewd->new", normalized, self.w_up))
        return torch.einsum("new,edw->ned", hidden, self.w_down)

    def _sparse_expert_outputs(self, normalized: torch.Tensor, sets: torch.Tensor) -> torch.Tensor:
        """Evaluate exactly the N*k requested expert paths, without dense masking."""
        token_outputs = []
        for token in range(normalized.shape[0]):
            selected_outputs = []
            for slot in range(sets.shape[1]):
                expert = int(sets[token, slot].item())
                up = F.silu(self.w_up[expert] @ normalized[token])
                selected_outputs.append(self.w_down[expert] @ up)
            token_outputs.append(torch.stack(selected_outputs, dim=0))
        return torch.stack(token_outputs, dim=0)

    def forward(
        self,
        inputs: torch.Tensor,
        bias: torch.Tensor,
        q: torch.Tensor,
        forced_sets: Optional[torch.Tensor] = None,
        gate_mode: str = "full_softmax_selected",
        sparse: bool = False,
        tie_tolerance: float = 1.0e-12,
    ) -> ForwardResult:
        """Evaluate the module; forcing fixes indices, never candidate gate values."""
        if gate_mode not in GATE_MODES:
            raise ValueError("unsupported gate mode")
        hidden = self.hidden_from_q(inputs, bias, q)
        normalized = self.rms_normalize(hidden)
        logits = normalized @ self.router.transpose(0, 1)
        probabilities = torch.softmax(logits, dim=1)
        if forced_sets is None:
            sets = stable_topk_sets(logits, self.top_k)
        else:
            sets = canonicalize_sets(forced_sets.to(device=logits.device), self.num_experts)
            if sets.shape != (inputs.shape[0], self.top_k):
                raise ValueError("forced_sets must have shape [tokens, top_k]")
        route = RouteState(
            sets=sets,
            logits=logits,
            probabilities=probabilities,
            near_tie=near_tie_flags(logits, sets, tie_tolerance),
            boundary_margin=boundary_margin(logits, sets),
        )
        selected_weights = probabilities.gather(1, sets)
        if gate_mode == "topk_renormalized":
            selected_weights = selected_weights / selected_weights.sum(dim=1, keepdim=True)
        if sparse:
            selected_outputs = self._sparse_expert_outputs(normalized, sets)
            expert_calls = int(inputs.shape[0] * self.top_k)
        else:
            dense_outputs = self._dense_expert_outputs(normalized)
            selected_outputs = dense_outputs.gather(
                1,
                sets.unsqueeze(2).expand(-1, -1, self.hidden_dimension),
            )
            expert_calls = int(inputs.shape[0] * self.num_experts)
        output = hidden + (selected_weights.unsqueeze(2) * selected_outputs).sum(dim=1)
        return ForwardResult(
            output=output,
            hidden=hidden,
            normalized=normalized,
            route=route,
            selected_weights=selected_weights,
            expert_calls=expert_calls,
        )
