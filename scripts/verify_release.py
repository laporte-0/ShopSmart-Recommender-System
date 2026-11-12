"""Audit local preservation and prospective deliverable files without staging Git."""

import argparse
import ast
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def ignored(relative, directory=False):
    """Interpret the simple anchored/name patterns used by this .gitignore.

    This is a prospective worktree audit, not a substitute for a real Git-index
    check. Git state is reported separately and is never mutated.
    """
    relative = relative.strip('/')
    result = False
    parts = relative.split('/')
    for raw in (ROOT / '.gitignore').read_text().splitlines():
        rule = raw.strip()
        if not rule or rule.startswith('#'):
            continue
        negate = rule.startswith('!')
        rule = rule.lstrip('!')
        anchored = rule.startswith('/')
        folder = rule.endswith('/')
        rule = rule.strip('/')
        if folder:
            ancestors = ['/'.join(parts[:i]) for i in range(1, len(parts) + (1 if directory else 0))]
            match = any(fnmatch.fnmatch(path, rule) if anchored or '/' in rule
                        else fnmatch.fnmatch(path.split('/')[-1], rule) for path in ancestors)
        else:
            match = fnmatch.fnmatch(relative if anchored or '/' in rule else parts[-1], rule)
        if match:
            result = not negate
    return result


def audit():
    for name in ['.venv/lib/site.py', 'data/raw/Electronics.csv.gz',
                 'data/processed/v2/train.csv', 'artifacts/recommenders.joblib',
                 '.local_archive/draft.md', 'ShopSmart_Review.zip', '.env',
                 '.env.local', '.aws/credentials', '.cache/rag/index.joblib']:
        assert ignored(name), f'Required local exclusion missing: {name}'
    manifest = json.loads((ROOT / 'reports/results/release_preservation.json').read_text())
    checked = 0
    missing_local = []
    for name, digest in manifest.items():
        path = ROOT / name
        if not path.exists() and (name.startswith('data/') or name.startswith('artifacts/')):
            missing_local.append(name)
            continue
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f'Scientific artifact changed: {name}'
        checked += 1
    allowed_roots = {'src', 'scripts', 'tests', 'docs', 'notebooks', 'reports'}
    allowed_files = {'README.md', 'requirements.txt', '.gitignore'}
    candidates = []
    for folder, directories, filenames in os.walk(ROOT):
        parent = Path(folder)
        directories[:] = [name for name in directories if name != '.git'
                          and not ignored(str((parent / name).relative_to(ROOT)), directory=True)]
        for name in filenames:
            path = parent / name
            relative = path.relative_to(ROOT)
            if ignored(relative.as_posix()):
                continue
            assert relative.parts[0] in allowed_roots or relative.as_posix() in allowed_files, f'Unexpected deliverable: {relative}'
            assert not path.is_symlink(), f'Unexpected symlink: {relative}'
            assert path.stat().st_size <= 5 * 1024 * 1024, f'File exceeds 5 MiB review limit: {relative}'
            candidates.append(relative)
    # High-specificity credential signatures. Do not print matched values.
    credentials = re.compile(
        r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'
        r'|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'
        r'|\bgh[pousr]_[A-Za-z0-9]{30,}\b'
        r'|\bgithub_pat_[A-Za-z0-9_]{40,}\b'
        r'|\bsk-(?:proj-)?[A-Za-z0-9_-]{40,}\b'
    )
    notebooks = 0
    for relative in candidates:
        path = ROOT / relative
        if path.suffix == '.png':
            continue
        text = path.read_text(encoding='utf-8')
        assert not credentials.search(text), f'Potential credential signature: {relative}'
        if path.suffix == '.ipynb':
            notebook = json.loads(text)
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    ast.parse(''.join(cell['source']))
            notebooks += 1
        if path.suffix == '.md':
            for target in re.findall(r'\]\(([^)]+)\)', text):
                if '://' in target or target.startswith('#'):
                    continue
                target = target.split('#')[0]
                assert (path.parent / target).exists(), f'Broken documentation link: {relative} -> {target}'
    git = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT,
                         capture_output=True, text=True, check=False)
    result = {
        'preservation_hashes_checked': checked,
        'local_artifacts_absent': missing_local,
        'notebooks_syntax_checked': notebooks,
        'prospective_deliverable_files': len(candidates),
        'prospective_deliverable_bytes': sum((ROOT / path).stat().st_size for path in candidates),
        'credential_signatures_found': 0,
        'oversize_files_found': 0,
        'git_index_available': git.returncode == 0,
        'git_note': ('Read-only status inspected; no staging or publication performed.' if git.returncode == 0
                     else 'Git metadata unavailable: empty read-only .git placeholder. Tracked files, staging, branches and remotes cannot be verified.'),
        'audit_scope': 'Prospective worktree files using the current simple .gitignore patterns; not a Git-index audit.',
        'files': sorted(str(path) for path in candidates),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional machine-readable audit result')
    args = parser.parse_args()
    result = audit()
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'PASS: {result["preservation_hashes_checked"]} preservation hashes, '
          f'{result["notebooks_syntax_checked"]} notebooks, '
          f'{result["prospective_deliverable_files"]} prospective deliverable files; '
          'no credential signatures or files above 5 MiB.')
    print(result['git_note'])


if __name__ == '__main__':
    main()
