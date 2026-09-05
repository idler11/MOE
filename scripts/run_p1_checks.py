"""Run fixed-seed P1 mathematical smoke checks without training or model downloads."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_event.audit import fresh_artifact_directory, git_metadata, verify_legacy_hashes, write_json
from moe_event.candidates import generate_fixed_scale_candidates
from moe_event.decomposition import finite_step_decomposition
from moe_event.derivatives import fixed_branch_derivatives
from moe_event.scoring import score_candidate
from moe_event.toy import ToyMoE
from moe_event.types import QuantizationSpec


def load_config(path: Path) -> Dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("phase") != "P1" or config.get("device") != "cpu":
        raise ValueError("this entry only accepts the CPU P1 smoke configuration")
    if config.get("reference_dtype") != "float64" or config.get("secondary_dtype") != "float32":
        raise ValueError("P1 needs float64 reference and float32 secondary check")
    if config.get("require_positive_improvement") is not False:
        raise ValueError("P1 must not make improvement a pass criterion")
    if not config.get("seeds") or not config.get("gate_modes"):
        raise ValueError("P1 smoke configuration needs nonempty seeds and gate_modes")
    return config


def build_case(config: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Generate a self-contained toy case; no external data or holdout lookup."""
    dimensions = {
        "input": int(config["input_dimension"]),
        "hidden": int(config["hidden_dimension"]),
        "expert_hidden": int(config["expert_hidden_dimension"]),
        "experts": int(config["num_experts"]),
        "top_k": int(config["top_k"]),
    }
    dtype = torch.float64
    generator = torch.Generator(device="cpu")
    generator.manual_seed(10000 + int(seed))
    model = ToyMoE.random(
        dimensions["input"],
        dimensions["hidden"],
        dimensions["expert_hidden"],
        dimensions["experts"],
        dimensions["top_k"],
        seed=20000 + int(seed),
        dtype=dtype,
        rms_epsilon=float(config["rms_epsilon"]),
    )
    teacher_q = 0.45 * torch.randn(dimensions["hidden"], dimensions["input"], generator=generator, dtype=dtype)
    quantization = config["quantization"]
    scale = float(teacher_q.abs().max().item()) / float(quantization["qmax"])
    if scale <= 0:
        scale = float(quantization["zero_tensor_scale"])
    spec = QuantizationSpec(
        bits=int(quantization["bits"]),
        qmin=int(quantization["qmin"]),
        qmax=int(quantization["qmax"]),
        zero_point=int(quantization["zero_point"]),
        scale=scale,
    )
    candidates = generate_fixed_scale_candidates(
        teacher_q,
        spec,
        p=int(quantization["candidate_coordinates"]),
    )
    inputs = torch.randn(int(config["calibration_tokens"]), dimensions["input"], generator=generator, dtype=dtype)
    bias = 0.3 * torch.randn(int(config["calibration_tokens"]), dimensions["hidden"], generator=generator, dtype=dtype)
    return {
        "model": model,
        "teacher_q": teacher_q,
        "candidates": candidates,
        "inputs": inputs,
        "bias": bias,
    }


def maximum_abs(value: torch.Tensor) -> float:
    return float(value.detach().abs().max().item()) if value.numel() else 0.0


def run_case(config: Dict[str, Any], seed: int, gate_mode: str) -> Dict[str, Any]:
    case = build_case(config, seed)
    model: ToyMoE = case["model"]
    candidates = case["candidates"]
    inputs = case["inputs"]
    bias = case["bias"]
    with torch.no_grad():
        target = model(inputs, bias, case["teacher_q"], gate_mode=gate_mode, sparse=False).output.detach()
    q0 = candidates.base_q
    # This is a fixed, source-derived candidate.  It is not selected by a
    # reference loss or a holdout value.
    candidate_index = candidates.count - 1
    z = candidates.z_values[candidate_index]
    q1 = candidates.q_for_index(candidate_index)
    dense = finite_step_decomposition(model, inputs, bias, q0, q1, target, gate_mode, evaluator="dense")
    event_only = finite_step_decomposition(model, inputs, bias, q0, q1, target, gate_mode, evaluator="event_only")
    derivatives = fixed_branch_derivatives(model, inputs, bias, candidates, target, gate_mode)
    score = score_candidate(candidates.candidate_ids[candidate_index], z, derivatives, dense)
    model32 = copy.deepcopy(model).to(dtype=torch.float32)
    with torch.no_grad():
        output32 = model32(
            inputs.to(dtype=torch.float32),
            bias.to(dtype=torch.float32),
            q1.to(dtype=torch.float32),
            gate_mode=gate_mode,
            sparse=False,
        )
    identity_error = abs(float((dense.actual_delta - dense.fixed_delta - dense.signed_event).item()))
    return {
        "seed": int(seed),
        "gate_mode": gate_mode,
        "candidate_id": candidates.candidate_ids[candidate_index],
        "candidate_modifications": candidates.modifications_for_index(candidate_index),
        "candidate_count": candidates.count,
        "duplicates_recorded": candidates.duplicates,
        "rejected_masks": candidates.rejected_masks,
        "event_token_count": int(dense.event_mask.sum().item()),
        "near_tie_old_count": int(dense.old_route.near_tie.sum().item()),
        "near_tie_new_count": int(dense.new_route.near_tie.sum().item()),
        "identity_error": identity_error,
        "signed_formula_max_abs_error": maximum_abs(dense.event_per_token - dense.event_formula_per_token),
        "dense_event_only_actual_output_max_abs_error": maximum_abs(dense.actual_output - event_only.actual_output),
        "dense_event_only_fixed_output_max_abs_error": maximum_abs(dense.fixed_output - event_only.fixed_output),
        "dense_event_only_event_max_abs_error": maximum_abs(dense.event_per_token - event_only.event_per_token),
        "dense_expert_calls": dense.expert_calls,
        "event_only_expert_calls": event_only.expert_calls,
        "full_hessian_vs_gauss_newton_max_abs_difference": maximum_abs(derivatives.hessian - derivatives.gauss_newton),
        "float32_output_max_abs_error": maximum_abs(dense.actual_output - output32.output.to(dtype=torch.float64)),
        "float32_route_sets_match_float64": bool(torch.equal(dense.new_route.sets, output32.route.sets)),
        "score": asdict(score),
        "holdout_used_for_selection": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "p1_smoke.json")
    parser.add_argument("--output", required=True, type=Path, help="new directory under artifacts/")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    output = fresh_artifact_directory(ROOT, args.output)
    started = time.perf_counter()
    legacy_before = verify_legacy_hashes(ROOT)
    rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    reference = config["reference_tolerance"]
    tolerance = max(float(reference["atol"]), float(reference["rtol"]))
    for seed in config["seeds"]:
        for gate_mode in config["gate_modes"]:
            row = run_case(config, int(seed), str(gate_mode))
            rows.append(row)
            for key in (
                "identity_error",
                "signed_formula_max_abs_error",
                "dense_event_only_actual_output_max_abs_error",
                "dense_event_only_fixed_output_max_abs_error",
                "dense_event_only_event_max_abs_error",
            ):
                if row[key] > tolerance:
                    failures.append("seed={} gate_mode={} {}={} exceeds {}".format(seed, gate_mode, key, row[key], tolerance))
    legacy_after = verify_legacy_hashes(ROOT)
    elapsed = time.perf_counter() - started
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    summary: Dict[str, Any] = {
        "status": "passed" if not failures else "failed",
        "phase": "P1",
        "scope": "CPU toy mathematical checks only; no pretrained model, training, P2 sweep, speed, or efficacy claim.",
        "config": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": config_hash,
        "device": "CPU",
        "reference_dtype": "float64",
        "secondary_dtype": "float32",
        "python": sys.version,
        "platform": platform.platform(),
        "git": git_metadata(ROOT),
        "legacy_hashes_unchanged": legacy_before == legacy_after,
        "elapsed_seconds": elapsed,
        "case_count": len(rows),
        "failures": failures,
        "cases": rows,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
