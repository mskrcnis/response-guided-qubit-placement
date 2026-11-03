#!/usr/bin/env python3
"""Portable entry point for the versioned QST research artifact."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tools'))
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'QISKIT_NUM_PROCS', 'RAYON_NUM_THREADS']:
    os.environ[key] = '1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def portable(path):
    """Resolve historical absolute result paths within this checkout only."""
    path = Path(path)
    if not path.is_absolute():
        return ROOT / path
    try:
        path.relative_to(ROOT)
        return path
    except ValueError:
        marker = '/Research Article/'
        if marker not in str(path):
            raise ValueError(f'Unrecognized historical path: {path}')
        result = (ROOT / str(path).split(marker, 1)[1]).resolve()
        result.relative_to(ROOT)
        return result


def verify(require_artifacts=True):
    source = read(ROOT / 'provenance/repository_files.json')
    for name, expected in source.items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'Repository file differs from release: {name}')
    count = 0
    if require_artifacts:
        for name, info in read(ROOT / 'provenance/artifact_files.json').items():
            path = ROOT / name
            if not path.exists():
                raise FileNotFoundError('Run python reproduce.py fetch first; missing ' + name)
            if path.stat().st_size != info['bytes'] or sha(path) != info['sha256']:
                raise ValueError('Frozen artifact mismatch: ' + name)
            count += 1
    result = dict(status='passed', repository_files=len(source), artifact_files=count)
    print(json.dumps(result, indent=2), flush=True)
    return result


def fetch(archive=None):
    info = read(ROOT / 'provenance/release_asset.json')
    if archive is None:
        directory = ROOT / '.downloads'
        directory.mkdir(exist_ok=True)
        archive = directory / info['filename']
        if not archive.exists():
            temporary = archive.with_suffix('.part')
            with urllib.request.urlopen(info['url'], timeout=120) as source, temporary.open('wb') as target:
                import shutil
                shutil.copyfileobj(source, target)
            temporary.replace(archive)
    archive = Path(archive)
    if sha(archive) != info['sha256']:
        raise ValueError('Download SHA256 mismatch')
    expected = read(ROOT / 'provenance/artifact_files.json')
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError('Unexpected archive inventory')
        for member in members:
            destination = (ROOT / member.name).resolve()
            destination.relative_to(ROOT)
            if not member.isfile() or member.size != expected[member.name]['bytes']:
                raise ValueError('Unsafe or inconsistent archive member')
            if destination.exists():
                if sha(destination) != expected[member.name]['sha256']:
                    raise ValueError('Refusing to overwrite changed file: ' + member.name)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            stream = bundle.extractfile(member)
            temporary = destination.with_suffix(destination.suffix + '.part')
            import shutil
            with temporary.open('wb') as target:
                shutil.copyfileobj(stream, target)
            if sha(temporary) != expected[member.name]['sha256']:
                temporary.unlink()
                raise ValueError('Extracted file hash mismatch')
            temporary.replace(destination)
    verify()


def included_sources(sources):
    included = {}
    excluded = {}
    for name, expected in sources.items():
        p = ROOT / name
        if p.is_file():
            if sha(p) != expected:
                raise ValueError('Frozen original source mismatch: ' + name)
            included[name] = expected
        else:
            excluded[name] = expected
    return included, excluded


def audit():
    verify()
    import audit_published_baselines as external
    original_read = external.read
    output = ROOT / 'outputs/audit/external'
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = external.OUT / 'protocol.json'
    protocol = read(protocol_path)
    included, excluded = included_sources(protocol['source_hashes'])

    def audit_read(path):
        path = portable(path)
        if path == protocol_path:
            return dict(protocol, source_hashes=included)
        if path.is_relative_to(output):
            path = ROOT / 'reports/published_baselines_v1' / path.relative_to(output)
        return original_read(path)

    external.read = audit_read
    external.sha = lambda path: sha(portable(path))
    external.REPORT = output
    external.main()
    save(ROOT / 'outputs/audit/source_coverage.json', {
        'included_original_source_hashes': included,
        'external_or_out_of_scope_original_sources': excluded,
        'scope': 'All released artifacts are verified. Historical whole-workspace pins, external dependency clones, and the machine-specific Qsyn binary are not misrepresented as shipped files.'})
    from validate_internal_release import run
    run()


def paper():
    verify(False)
    for name in ['make_tables.py', 'make_figures.py']:
        subprocess.run([sys.executable, str(ROOT / 'paper/scripts' / name)], check=True)
    print('Rebuilt five numerical tables and seven numerical figures under paper/.')


def smoke(full=False):
    verify()
    import numpy as np
    from qiskit import qasm2
    from duplicate_aware_swarm import Owner
    import run_heldout as engine
    from validate_search_loss_diagnosis import independent
    from validate_circuit_objective import independent_p1, matrix_tape
    from published_baseline_common import validate
    plan = read(ROOT / 'experiments/exploratory_oct30_v1/plan.json')
    scenarios = [s for s in read(ROOT / 'experiments/exploratory_oct30_v1/scenarios.json') if s['status'] == 'ready']
    engine.install_region(plan)
    circuits = list(plan['bank']) if full else ['ghz_4', 'confirm_ring_n5_seed20261319']
    rows = []
    for circuit in circuits:
        ci = list(plan['bank']).index(circuit)
        logical = qasm2.load(str(ROOT / plan['bank'][circuit]['path']), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        logical.name = circuit
        ideal = engine.engine.distribution(logical)
        independent_ideal = independent_p1(matrix_tape(logical))
        for method in plan['methods']:
            for replicate in (plan['replicates'] if full else [0]):
                for rule in (plan['seed_rules'] if full else ['resampled']):
                    owner = Owner('hybrid', circuit)
                    previous = read(ROOT / 'experiments/exploratory_oct30_v1/jobs' / f'{circuit}__{method}__{replicate}__{rule}.json')
                    for scenario in (scenarios if full else [scenarios[1], scenarios[-1]]):
                        seed = 50001 + 1000 * ci + 100 * replicate + (0 if rule == 'fixed' else scenario['calendar_index'])
                        params = engine.engine.parameters(scenario)
                        native, stats = owner.search(logical, params, seed, method)
                        actual = engine.engine.distribution(native, params)
                        reference = independent(native, params, independent_ideal)
                        error = float(np.max(abs(actual - reference['distribution'])))
                        assert error < 2e-10
                        stored = next(d for d in previous['days'] if d['date'] == scenario['date'])
                        signature = engine.native_signature(native)
                        tvd = float(abs(actual - ideal).sum() / 2)
                        assert signature == stored['signature']
                        assert abs(tvd - stored['tvd']) < 2e-10
                        assert abs(stats['best_proxy'] - stored['stats']['best_proxy']) < 2e-10
                        assert stats['objective_calls'] == 56
                        validation = validate(logical, native, stats['initial_layout_local'], engine.engine.EDGES, plan['basis_gates'])
                        result = dict(circuit=circuit, method=method, replicate=replicate, seed_rule=rule,
                                      date=scenario['date'], tvd=tvd, max_channel_error=error,
                                      native_signature_matches=True, **validation)
                        rows.append(result)
                        print(circuit, method, scenario['date'], 'passed', flush=True)
                        save(ROOT / 'outputs' / ('full_search_replay.json' if full else 'smoke.json'),
                             {'status': 'running', 'fresh_searches': len(rows), 'rows': rows})
    save(ROOT / 'outputs' / ('full_search_replay.json' if full else 'smoke.json'),
         {'status': 'passed', 'fresh_searches': len(rows), 'rows': rows,
          'scope': 'Fresh search, native circuit, channel and score reproduction. Recorded wall times are not replicated. Post-search policy histories are checked separately by audit.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['fetch', 'verify', 'audit', 'paper', 'smoke', 'full-search'])
    parser.add_argument('--archive', type=Path, help='Use an already downloaded release asset')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.command == 'fetch':
        fetch(args.archive)
    elif args.command == 'verify':
        verify()
    elif args.command == 'full-search':
        smoke(full=True)
    else:
        globals()[args.command]()


if __name__ == '__main__':
    main()
