"""Small deterministic fixtures for tests; none use legacy implementation code."""
from __future__ import annotations

import torch

from moe_event.candidates import generate_fixed_scale_candidates
from moe_event.toy import ToyMoE
from moe_event.types import QuantizationSpec


def make_event_case() -> dict:
    """A shared-Q case in which every token replaces both Top-2 experts."""
    dtype = torch.float64
    torch.manual_seed(73)
    router = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=dtype)
    gamma = torch.ones(2, dtype=dtype)
    w_up = 0.5 * torch.randn(4, 3, 2, dtype=dtype)
    w_down = 0.5 * torch.randn(4, 2, 3, dtype=dtype)
    model = ToyMoE(router, gamma, w_up, w_down, top_k=2)
    inputs = torch.ones(3, 1, dtype=dtype)
    bias = torch.tensor([[1.0, 0.8], [0.9, 0.7], [1.1, 0.9]], dtype=dtype)
    q0 = torch.zeros(2, 1, dtype=dtype)
    q1 = torch.tensor([[-2.0], [-1.6]], dtype=dtype)
    target = torch.tensor([[0.2, -0.1], [0.0, 0.3], [-0.2, 0.1]], dtype=dtype)
    return {"model": model, "inputs": inputs, "bias": bias, "q0": q0, "q1": q1, "target": target}


def make_derivative_case() -> dict:
    dtype = torch.float64
    torch.manual_seed(91)
    model = ToyMoE.random(3, 4, 5, 4, 2, seed=92, dtype=dtype)
    source = torch.tensor(
        [[0.31, -0.52, 0.44], [-0.37, 0.63, -0.28], [0.57, 0.16, -0.48], [0.22, -0.41, 0.69]],
        dtype=dtype,
    )
    spec = QuantizationSpec(bits=3, qmin=-3, qmax=3, scale=float(source.abs().max().item()) / 3.0)
    candidates = generate_fixed_scale_candidates(source, spec, p=2)
    inputs = 0.45 * torch.randn(5, 3, dtype=dtype)
    bias = 0.2 * torch.randn(5, 4, dtype=dtype)
    with torch.no_grad():
        teacher = model(inputs, bias, source, gate_mode="full_softmax_selected", sparse=False).output
    target = teacher + 0.11
    return {"model": model, "candidates": candidates, "inputs": inputs, "bias": bias, "target": target}
