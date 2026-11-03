#!/usr/bin/env python3
"""Fresh external compilations and independent output checks against frozen results."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduce import read, save, verify
sys.path.insert(0, str(ROOT / 'scripts'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', help='Recompile all 2,055 external requests')
    parser.add_argument('--modern-python', type=Path, default=ROOT / '.venv-baselines-modern/bin/python')
    parser.add_argument('--legacy-python', type=Path, default=ROOT / '.venv-baselines-legacy/bin/python')
    args = parser.parse_args()
    os.chdir(ROOT)
    verify()
    import numpy as np
    import run_heldout as engine
    from published_baseline_common import lower, read_qasm, validate
    from validate_search_loss_diagnosis import independent
    from validate_circuit_objective import independent_p1, matrix_tape
    base = ROOT / 'experiments/published_baselines_v1'
    protocol = read(base / 'protocol.json')
    engine.install_region(protocol)
    requests = read(base / 'requests.json')
    observations = read(base / 'observations.json')
    if args.all:
        selected = list(requests)
    else:
        selected = []
        for circuit in ['ghz_4', 'confirm_ring_n5_seed20261319']:
            for method in ['lightsabre', 'mapomatic', 'sqgm', 'qmap', 'duostra']:
                observation = next(o for o in observations if o['circuit_id'] == circuit and
                                   o['method'] == method and o['date'] == protocol['paired_dates'][0] and
                                   o['replicate'] == 0 and o['seed_rule'] == 'fixed')
                selected.append(observation['request'])
    scenarios = {s['date']: s for s in read(base / 'scenarios.json')}
    output = ROOT / 'outputs' / ('baseline_full' if args.all else 'baseline_smoke')
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for number, key in enumerate(selected, 1):
        request = requests[key]
        method = request['spec']['method']
        python = args.legacy_python if method == 'sqgm' else args.modern_python
        if not python.is_file():
            raise FileNotFoundError(f'Missing baseline interpreter: {python}. See docs/REPRODUCING.md.')
        spec_path = output / (key + '.input.json')
        raw_path = output / (key + '.raw.json')
        save(spec_path, request['spec'])
        subprocess.run([str(python), str(ROOT / 'scripts/published_baseline_worker.py'),
                        str(spec_path), str(raw_path)], check=True, timeout=300)
        raw = read(raw_path)
        assert raw['status'] == 'compiled', raw
        logical = read_qasm(request['spec']['qasm'])
        native = lower(raw, protocol['basis_gates'])
        check = validate(logical, native, raw['initial_layout'], protocol['edges'], protocol['basis_gates'])
        signature = engine.native_signature(native)
        job = read(base / 'jobs' / (request['circuit_id'] + '.json'))
        associated = [r for r in job['rows'] if r.get('request') == key]
        if not args.all:
            associated = [next(r for r in associated if r['date'] == protocol['paired_dates'][0])]
        ideal = engine.engine.distribution(logical)
        lp = independent_p1(matrix_tape(logical))
        date_results = []
        for date in dict.fromkeys(r['date'] for r in associated):
            stored = next(r for r in associated if r['date'] == date)
            expected = job['evaluations'][stored['evaluation_key']]
            assert signature == expected['signature'], (method, request['circuit_id'], 'native differs')
            params = engine.engine.parameters(scenarios[date])
            actual = engine.engine.distribution(native, params)
            reference = independent(native, params, lp)
            error = float(np.max(abs(actual - reference['distribution'])))
            tvd = float(abs(actual - ideal).sum() / 2)
            assert error < 2e-10 and abs(tvd - stored['tvd']) < 2e-10
            date_results.append(dict(date=date, tvd=tvd, max_channel_error=error))
        results.append(dict(request=key, method=method, circuit=request['circuit_id'],
                            native_signature_matches=True, dates=date_results, **check))
        save(output / 'validation.json', dict(status='running', fresh_compilations=len(results), results=results))
        print(number, '/', len(selected), method, request['circuit_id'], 'passed', flush=True)
    save(output / 'validation.json', dict(status='passed', fresh_compilations=len(results), results=results,
         scope='Fresh configured-compiler, arbitrary-input measurement and exact-channel reproduction; not a rerun of published authors original benchmark tables or historical wall times.'))


if __name__ == '__main__':
    main()
