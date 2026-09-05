"""Small audit helpers for safe P1 artifacts and immutable legacy verification."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from .metrics import json_dumps_strict


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_legacy_hashes(repository_root: Path) -> Dict[str, str]:
    manifest_path = repository_root / "legacy" / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual: Dict[str, str] = {}
    for relative, expected in manifest["files"].items():
        path = repository_root / relative
        digest = sha256_file(path)
        if digest != expected:
            raise RuntimeError("immutable legacy hash mismatch: " + relative)
        actual[relative] = digest
    return actual


def fresh_artifact_directory(repository_root: Path, output: Path) -> Path:
    root = repository_root.resolve()
    artifact_root = (root / "artifacts").resolve()
    resolved = output.resolve() if output.is_absolute() else (root / output).resolve()
    if not resolved.is_relative_to(artifact_root) or resolved == artifact_root:
        raise ValueError("output must be a new subdirectory of repository artifacts/")
    if resolved.exists():
        raise FileExistsError("refusing to overwrite existing run directory: " + str(resolved))
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def write_json(path: Path, value: Any) -> None:
    path.write_text(json_dumps_strict(value) + "\n", encoding="utf-8")


def git_metadata(repository_root: Path) -> Dict[str, Optional[Any]]:
    def run(args: list[str]) -> Optional[str]:
        try:
            result = subprocess.run(args, cwd=repository_root, capture_output=True, text=True, check=False, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run(["git", "rev-parse", "HEAD"])
    dirty_text = run(["git", "status", "--porcelain"])
    return {"code_commit": commit, "dirty": None if dirty_text is None else bool(dirty_text)}
