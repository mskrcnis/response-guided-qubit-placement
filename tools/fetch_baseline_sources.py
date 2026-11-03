#!/usr/bin/env python3
"""Fetch original upstream implementations at the recorded commits."""
from pathlib import Path
import argparse
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-qsyn', action='store_true')
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    registry = json.loads((ROOT / 'experiments/published_baselines_v1/baseline_registry.json').read_text())
    for entry in registry:
        if 'commit' not in entry:
            continue
        name = 'qsyn' if entry['id'] == 'duostra' else entry['id']
        target = ROOT / 'data/external/published_baselines_v1' / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git', 'clone', '--no-checkout', entry['source'], str(target)], check=True)
            subprocess.run(['git', '-C', str(target), 'checkout', '--detach', entry['commit']], check=True)
        commit = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
        assert commit == entry['commit'], (name, 'different checkout; will not overwrite')
        if name == 'qsyn':
            patch = ROOT / 'experiments/published_baselines_v1/qsyn_adapter.patch'
            applied = subprocess.run(['git', '-C', str(target), 'apply', '--reverse', '--check', str(patch)], capture_output=True).returncode == 0
            if not applied:
                subprocess.run(['git', '-C', str(target), 'apply', '--check', str(patch)], check=True)
                subprocess.run(['git', '-C', str(target), 'apply', str(patch)], check=True)
            if args.build_qsyn:
                subprocess.run(['cmake', '-S', str(target), '-B', str(target/'build'), '-DCMAKE_BUILD_TYPE=Release'], check=True)
                subprocess.run(['cmake', '--build', str(target/'build'), '-j', str(args.jobs)], check=True)
        print(name, commit, 'ready', flush=True)


if __name__ == '__main__':
    main()
