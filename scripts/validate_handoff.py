"""Validate handoff files and immutable evidence without third-party packages."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    'README.md', 'AGENTS.md', 'START_PROMPT.md', 'PROJECT_STATE.json',
    'DECISIONS.md', 'docs/01_THEORY.md', 'docs/02_EXPERIMENTS.md',
    'docs/03_ROADMAP.md', 'docs/04_ENGINEERING.md',
    'docs/05_RELATED_WORK.md', 'docs/06_COLLABORATION.md',
    'configs/p1_smoke.json', 'reports/REPORT_TEMPLATE.md',
    'legacy/README.md', 'legacy/MANIFEST.json',
    'legacy/moe_theory_audit/README.md', 'legacy/moe_theory_audit/verify_math.py',
    'legacy/moe_theory_audit/results.json', 'legacy/moe_theory_audit/environment.json',
    'scripts/recheck_legacy.py', '.gitignore', '.gitattributes'
)


def read_json(path: Path) -> object:
    def reject(value: str) -> None:
        raise ValueError('Nonfinite JSON value: ' + value)
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=reject)


def main() -> int:
    errors: list[str] = []
    for name in REQUIRED:
        if not (ROOT / name).is_file():
            errors.append('Missing: ' + name)
    checked_hashes = 0
    manifest_path = ROOT / 'legacy/MANIFEST.json'
    if manifest_path.is_file():
        try:
            manifest = read_json(manifest_path)
            for name, expected in manifest['files'].items():
                path = (ROOT / name).resolve()
                if not path.is_relative_to((ROOT / 'legacy/moe_theory_audit').resolve()):
                    raise ValueError('Manifest path outside immutable legacy directory')
                if path.is_file():
                    actual = hashlib.sha256(path.read_bytes()).hexdigest()
                    if actual != expected:
                        errors.append('Legacy hash mismatch: ' + name)
                    checked_hashes += 1
        except (ValueError, KeyError, TypeError, OSError) as exc:
            errors.append('Invalid manifest: ' + str(exc))
    names = list(REQUIRED) + ['scripts/validate_handoff.py']
    for name in names:
        path = ROOT / name
        if not path.is_file():
            continue
        try:
            if path.suffix == '.json':
                read_json(path)
            elif path.suffix == '.py':
                ast.parse(path.read_text(encoding='utf-8'), filename=name)
            elif path.suffix == '.md':
                text = path.read_text(encoding='utf-8')
                for target in re.findall(r'\]\(([^)\s]+)\)', text):
                    if re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:', target) or target.startswith('#'):
                        continue
                    relative = unquote(target.split('#', 1)[0])
                    if relative and not (path.parent / relative).exists():
                        errors.append(f'Broken local link in {name}: {target}')
        except (ValueError, SyntaxError, OSError) as exc:
            errors.append(f'{name}: {exc}')
    state_path = ROOT / 'PROJECT_STATE.json'
    if state_path.exists():
        try:
            state = read_json(state_path)
            if state.get('repository') != 'idler11/MOE':
                errors.append('Wrong project repository in PROJECT_STATE.json')
            if not set(state.get('authorized_stages', [])).issubset(state.get('stages', {})):
                errors.append('Authorized phase missing from stage registry')
        except (ValueError, AttributeError, TypeError) as exc:
            errors.append('Invalid project state: ' + str(exc))
    summary = {'status': 'passed' if not errors else 'failed', 'required_files': len(REQUIRED),
               'legacy_files_hashed': checked_hashes, 'errors': errors,
               'scope': 'Handoff integrity only; no scientific implementation or result is certified.'}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
