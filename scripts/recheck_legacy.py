"""Run the immutable legacy audit in a new CPU-only directory; stdlib wrapper."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_sources() -> dict[str, str]:
    manifest = json.loads((ROOT / 'legacy/MANIFEST.json').read_text(encoding='utf-8'))
    actual = {name: sha256(ROOT / name) for name in manifest['files']}
    bad = [name for name, expected in manifest['files'].items() if actual[name] != expected]
    if bad:
        raise RuntimeError('Immutable legacy source hash mismatch: ' + ', '.join(bad))
    return actual


def compare(expected: object, actual: object, rtol: float, atol: float, path: str = '$') -> list[dict]:
    differences: list[dict] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            differences.append({'path': path, 'reason': 'keys', 'expected': sorted(expected), 'actual': sorted(actual)})
        for key in expected.keys() & actual.keys():
            differences.extend(compare(expected[key], actual[key], rtol, atol, path + '.' + key))
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            differences.append({'path': path, 'reason': 'length', 'expected': len(expected), 'actual': len(actual)})
        for index, (left, right) in enumerate(zip(expected, actual)):
            differences.extend(compare(left, right, rtol, atol, f'{path}[{index}]'))
    elif type(expected) is bool or type(actual) is bool:
        if type(expected) is not type(actual) or expected != actual:
            differences.append({'path': path, 'reason': 'bool', 'expected': expected, 'actual': actual})
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        valid = math.isfinite(expected) and math.isfinite(actual)
        equal = expected == actual if type(expected) is int and type(actual) is int else math.isclose(expected, actual, rel_tol=rtol, abs_tol=atol)
        if not valid or not equal:
            differences.append({'path': path, 'reason': 'number', 'expected': expected, 'actual': actual})
    elif type(expected) is not type(actual) or expected != actual:
        differences.append({'path': path, 'reason': 'value', 'expected': expected, 'actual': actual})
    return differences


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New directory under this repository artifacts/')
    parser.add_argument('--timeout', type=float, default=180.0)
    parser.add_argument('--rtol', type=float, default=1e-6)
    parser.add_argument('--atol', type=float, default=1e-9)
    args = parser.parse_args()
    for name in ('timeout', 'rtol', 'atol'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f'{name} must be finite and nonnegative')
    if args.timeout == 0:
        parser.error('timeout must be positive')
    out = Path(args.output)
    out = (ROOT / out).resolve() if not out.is_absolute() else out.resolve()
    artifact_root = (ROOT / 'artifacts').resolve()
    if not out.is_relative_to(artifact_root) or out == artifact_root:
        parser.error('output must be a new subdirectory of repository artifacts/')
    sources_before = check_sources()
    out.mkdir(parents=True, exist_ok=False)
    versions: dict[str, str | None] = {}
    for name in ('numpy', 'scipy', 'torch', 'pytest'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    metadata = {
        'run_id': out.name,
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version,
        'platform': platform.platform(),
        'packages': versions,
        'device': 'CPU',
        'dtype': 'float64 (set by legacy script)',
        'rtol': args.rtol,
        'atol': args.atol,
        'timeout_seconds': args.timeout,
        'source_sha256': sources_before,
        'scope': 'Legacy synthetic reproduction only; not pretrained or user-GPU evidence.'
    }
    for key, command in (('git_commit', ['git', 'rev-parse', 'HEAD']), ('git_dirty', ['git', 'status', '--porcelain'])):
        try:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=10, check=False)
            metadata[key] = result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            metadata[key] = None
    source_copy = out / 'source'
    shutil.copytree(ROOT / 'legacy/moe_theory_audit', source_copy)
    (source_copy / 'results.json').unlink()
    command = [sys.executable, str(source_copy / 'verify_math.py')]
    metadata['command'] = command
    write_json(out / 'environment.json', metadata)
    missing = [name for name in ('numpy', 'scipy', 'torch') if versions[name] is None]
    if missing:
        write_json(out / 'summary.json', {'status': 'blocked_missing_dependency', 'missing': missing, 'exit_code': 2})
        print('Missing dependencies; no installation attempted:', ', '.join(missing))
        return 2
    env = dict(os.environ)
    env.update({'CUDA_VISIBLE_DEVICES': '', 'PYTHONUTF8': '1', 'PYTHONHASHSEED': '0'})
    start = time.perf_counter()
    timed_out = False
    with (out / 'stdout.txt').open('w', encoding='utf-8') as stdout, (out / 'stderr.txt').open('w', encoding='utf-8') as stderr:
        try:
            proc = subprocess.run(command, cwd=source_copy, env=env, stdout=stdout, stderr=stderr, timeout=args.timeout, check=False)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            exit_code, timed_out = 124, True
        except OSError as exc:
            stderr.write(str(exc) + '\n')
            exit_code = 127
    elapsed = time.perf_counter() - start
    sources_after = check_sources()
    differences: list[dict] = []
    parse_error = None
    produced = source_copy / 'results.json'
    if exit_code == 0 and produced.exists():
        try:
            expected = json.loads((ROOT / 'legacy/moe_theory_audit/results.json').read_text(encoding='utf-8'))
            actual = json.loads(produced.read_text(encoding='utf-8'), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            differences = compare(expected, actual, args.rtol, args.atol)
            shutil.copy2(produced, out / 'results.json')
        except (ValueError, OSError) as exc:
            parse_error = str(exc)
    else:
        parse_error = 'Legacy process failed or produced no results.json'
    passed = exit_code == 0 and parse_error is None and not differences and sources_after == sources_before
    summary = {
        'status': 'passed' if passed else 'needs_review',
        'exit_code': exit_code,
        'timed_out': timed_out,
        'elapsed_seconds': elapsed,
        'source_hashes_unchanged': sources_after == sources_before,
        'comparison': {'rtol': args.rtol, 'atol': args.atol, 'difference_count': len(differences), 'differences': differences},
        'parse_error': parse_error,
        'results_sha256': sha256(out / 'results.json') if (out / 'results.json').exists() else None,
        'finished_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Reproduction only; no scientific efficacy, generalization or speed claim.'
    }
    write_json(out / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
