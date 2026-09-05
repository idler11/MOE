from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from moe_event.audit import fresh_artifact_directory, verify_legacy_hashes, write_json
from moe_event.metrics import holdout_diagnostic, json_dumps_strict, safe_spearman, selection_metrics
from moe_event.scoring import select_from_calibration
from moe_event.types import ScoreResult


ROOT = Path(__file__).resolve().parents[1]


def test_t15_metrics_keep_zero_opportunity_constant_rank_negative_improvement_and_json_finite() -> None:
    zero = selection_metrics([1.0, 1.0], selected_index=0)
    assert zero.opportunity == 0.0
    assert zero.normalized_regret is None
    assert zero.reason_normalized_regret_is_null == "zero_or_numerically_zero_opportunity"
    correlation, reason = safe_spearman([1.0, 1.0], [2.0, 3.0])
    assert correlation is None and reason == "constant_input"
    baseline = selection_metrics([0.0, 1.0], selected_index=0)
    event = selection_metrics([0.0, 1.0], selected_index=1)
    assert baseline.regret - event.regret < 0.0
    same = selection_metrics([0.0, 1.0], selected_index=0)
    assert same.regret == baseline.regret
    payload = {"normalized": zero.normalized_regret, "reason": zero.reason_normalized_regret_is_null}
    assert json.loads(json_dumps_strict(payload))["normalized"] is None
    with pytest.raises(ValueError):
        json_dumps_strict({"bad": float("nan")})


def test_t16_calibration_selection_isolated_from_holdout_and_leakage_is_rejected() -> None:
    scores = [
        ScoreResult("c0", 0.0, 0.0, 0.0, 0.2, 0.2),
        ScoreResult("c1", 0.0, 0.0, 0.0, -0.1, -0.1),
    ]
    selected = select_from_calibration(scores)
    assert selected.candidate_id == "c1"
    diagnostic = holdout_diagnostic([0.0, 1.0], selected_index=1)
    assert diagnostic["selected_holdout_loss"] == 1.0
    assert diagnostic["holdout_hindsight_oracle_loss"] == 0.0
    assert "holdout" not in inspect.signature(select_from_calibration).parameters
    with pytest.raises(TypeError):
        select_from_calibration(scores, holdout_losses=[0.0, 1.0])


def test_t17_legacy_hashes_immutable_and_run_id_overwrite_rejected(tmp_path: Path) -> None:
    hashes = verify_legacy_hashes(ROOT)
    manifest = json.loads((ROOT / "legacy" / "MANIFEST.json").read_text(encoding="utf-8"))
    assert hashes == manifest["files"]
    # Recalculate one digest independently, rather than trusting a shared tool.
    legacy_result = ROOT / "legacy" / "moe_theory_audit" / "results.json"
    assert hashlib.sha256(legacy_result.read_bytes()).hexdigest() == manifest["files"]["legacy/moe_theory_audit/results.json"]
    repo = tmp_path / "repo"
    (repo / "artifacts").mkdir(parents=True)
    run = fresh_artifact_directory(repo, Path("artifacts") / "run_001")
    write_json(run / "summary.json", {"status": "passed", "finite": 1.0})
    with pytest.raises(FileExistsError):
        fresh_artifact_directory(repo, Path("artifacts") / "run_001")
