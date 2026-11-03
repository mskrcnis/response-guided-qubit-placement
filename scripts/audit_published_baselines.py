"""Independent, standard-library audit of saved baseline results and timing."""
import collections
import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/published_baselines_v1'
REPORT = ROOT / 'reports/published_baselines_v1'

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def close(actual, expected):
    assert math.isfinite(actual) and math.isfinite(expected)
    assert abs(actual - expected) < 1e-12, (actual, expected)

def mean(values):
    values = list(values)
    return math.fsum(values) / len(values)

def audit_mapomatic(request, raw):
    """Enumerate every feasible injection without calling Mapomatic code."""
    spec = request['spec']
    parameters = {(r['property'],r['qubit_a'],r['qubit_b']):r['value'] for r in spec['parameters']}
    operations = [op for op in raw['operations'] if op['name'] != 'barrier']
    active = sorted({q for op in operations for q in op['qubits']})
    interactions = {tuple(op['qubits']) for op in operations if len(op['qubits'])==2}
    edges = set(map(tuple,spec['edges']))
    physical = spec['physical']
    def cost(mapping):
        success = 1.
        for op in operations:
            qs = [physical[mapping[q]] for q in op['qubits']]
            if len(qs)==2:
                success *= 1-parameters[op['name']+'_gate_error',qs[0],qs[1]]
            elif op['name'] in ['sx','x']:
                success *= 1-parameters[op['name']+'_gate_error',qs[0],None]
            elif op['name'] in ['measure','reset']:
                success *= 1-(parameters['prob_meas1_prep0',qs[0],None]+parameters['prob_meas0_prep1',qs[0],None])/2
        return 1-success
    selected_cost = cost({q:q for q in active})
    close(selected_cost,raw['details']['selected_cost'])
    costs=[]
    for permutation in itertools.permutations(range(len(physical)),len(active)):
        mapping=dict(zip(active,permutation))
        if all((mapping[a],mapping[b]) in edges for a,b in interactions):
            costs.append(cost(mapping))
    assert len(costs)==raw['details']['candidate_embeddings']
    close(selected_cost,min(costs))
    return len(costs)

def main():
    protocol = read(OUT / 'protocol.json')
    protocol_hash = sha(OUT / 'protocol.json')
    manifests = [protocol['source_hashes'], read(OUT / 'generation_manifest.json')]
    for manifest in manifests:
        for name, expected in manifest.items():
            assert sha(ROOT / name) == expected, name
    for item in protocol['bank'].values():
        assert sha(ROOT / item['path']) == item['sha256']
    requests = read(OUT / 'requests.json')
    assert len(requests) == 2055
    mapomatic_cost_checks = 0
    for key, request in requests.items():
        expected = hashlib.sha256(json.dumps(request['spec'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        assert key == request['key'] == expected
        raw = read(request['destination'])
        assert raw['status'] == 'compiled'
        assert raw['method'] == request['spec']['method']
        assert raw['seed'] == request['spec']['seed']
        if raw['method']=='mapomatic':
            mapomatic_cost_checks += audit_mapomatic(request,raw)
    rows = []
    max_channel_error = max_povm_error = 0.
    evaluation_count = correctness_count = 0
    for circuit in protocol['bank']:
        job = read(OUT / 'jobs' / (circuit + '.json'))
        assert job['circuit_id'] == circuit
        assert not job['failed'] and job['protocol_sha256'] == protocol_hash
        assert len(job['rows']) == 432
        ideal = job['ideal']
        close(math.fsum(ideal), 1.)
        for evaluation in job['evaluations'].values():
            a, b = evaluation['distribution'], evaluation['independent_distribution']
            assert len(a) == len(b) == len(ideal)
            assert min(a + b) > -1e-12
            close(math.fsum(a), 1.)
            close(math.fsum(b), 1.)
            error = max(abs(x-y) for x,y in zip(a,b))
            assert error < 2e-10 and evaluation['independent_error'] < 2e-10
            close(evaluation['tvd'], math.fsum(abs(x-y) for x,y in zip(a,ideal)) / 2)
            max_channel_error = max(max_channel_error, error, evaluation['independent_error'])
            evaluation_count += 1
        for check in job['checks'].values():
            assert check['max_povm_error'] < 2e-8
            assert len(set(check['measurement_map'].values())) == len(ideal).bit_length()-1
            max_povm_error = max(max_povm_error, check['max_povm_error'])
            correctness_count += 1
        for row in job['rows']:
            assert row['circuit_id'] == circuit
            assert row['stratum'] == protocol['bank'][circuit]['stratum']
            assert row['paired'] == (row['date'] in protocol['paired_dates'])
            evaluation = job['evaluations'][row['evaluation_key']]
            assert evaluation['date'] == row['date']
            close(row['tvd'], evaluation['tvd'])
            if row['method'] == 'swarm_unique_56':
                close(row['tvd'], row['source_tvd'])
                assert sha(ROOT / row['artifact']['path']) == row['artifact']['sha256']
            else:
                request = requests[row['request']]
                assert request['circuit_id'] == circuit
                assert request['spec']['method'] == row['method']
                assert request['spec']['seed'] == row['seed_used']
                assert sha(request['destination']) == row['raw_sha256']
            rows.append(row)
    key = lambda r: (r['circuit_id'], r['method'], r['date'], r['replicate'], r['seed_rule'])
    expected_keys = set(itertools.product(protocol['bank'], protocol['methods'], protocol['ready_dates'], protocol['replicates'], protocol['seed_rules']))
    assert len(rows) == len({key(r) for r in rows}) == len(expected_keys) == 6480
    assert {key(r) for r in rows} == expected_keys
    obs = read(OUT / 'observations.json')
    assert len(obs) == len(rows)
    index = {key(r): r for r in rows}
    for observation in obs:
        assert all(index[key(observation)][k] == v for k,v in observation.items())
    paired = [r for r in rows if r['paired']]
    assert len(paired) == 5940
    assert set(collections.Counter(r['method'] for r in paired).values()) == {990}
    casekey = lambda r: (r['circuit_id'], r['date'], r['replicate'], r['seed_rule'])
    ours = {casekey(r): r for r in paired if r['method'] == 'swarm_unique_56'}
    for result in read(REPORT / 'method_summary.json'):
        subset = [r for r in paired if r['method'] == result['method'] and r['stratum'] == result['stratum']]
        assert len(subset) == result['cases']
        for reported, field in [('mean_tvd','tvd'), ('mean_cz','native_cz'), ('mean_depth','native_depth')]:
            close(result[reported], mean(r[field] for r in subset))
    for result in read(REPORT / 'paired_comparisons.json'):
        subset = [r for r in paired if r['method'] == result['baseline'] and r['stratum'] == result['stratum']]
        baseline = mean(r['tvd'] for r in subset)
        own = mean(ours[casekey(r)]['tvd'] for r in subset)
        close(result['baseline_mean'], baseline)
        close(result['our_mean'], own)
        close(result['mean_gain'], baseline-own)
        close(result['relative_mean_error_reduction_pct'], 100*(baseline-own)/baseline)
        gains = [r['tvd']-ours[casekey(r)]['tvd'] for r in subset]
        counts = {k: 0 for k in ['our_better','little_difference','our_worse']}
        for gain in gains:
            counts['our_better' if gain>1e-4 else 'our_worse' if gain< -1e-4 else 'little_difference'] += 1
        assert counts == result['case_outcomes']
        loo = [mean(r['tvd']-ours[casekey(r)]['tvd'] for r in subset if r['date'] != d) for d in protocol['paired_dates']]
        close(result['leave_one_date_out_gain_range'][0], min(loo))
        close(result['leave_one_date_out_gain_range'][1], max(loo))
    for file, field in [('by_circuit.json','circuit_id'), ('by_date.json','date')]:
        for result in read(REPORT/file):
            subset = [r for r in paired if r['method']==result['baseline'] and r['stratum']==result['stratum'] and r[field]==result[field]]
            assert len(subset)==result['cases']
            close(result['mean_gain'], mean(r['tvd']-ours[casekey(r)]['tvd'] for r in subset))
    changed = set(read(ROOT/'reports/exploratory_oct30_v1/calibration_audit.json')['changed_value_transition_dates'])
    for result in read(REPORT/'sensitivity.json'):
        subset = [r for r in paired if r['method']==result['baseline'] and r['stratum']==result['stratum']]
        subset = [r for r in subset if r['date'] in changed] if result['scope']=='changing_calibration_only' else [r for r in subset if r['seed_rule']==result['scope']]
        assert len(subset)==result['cases']
        close(result['mean_gain'], mean(r['tvd']-ours[casekey(r)]['tvd'] for r in subset))
    timing = read(REPORT/'serial_timing.json')
    timing_protocol = read(OUT/'serial_timing_protocol.json')
    assert timing['protocol'] == timing_protocol
    assert timing_protocol['cohort_protocol_sha256'] == protocol_hash
    trials = timing['trials']
    expected_timing = set(itertools.product(protocol['bank'], protocol['methods'], range(3)))
    assert len(trials) == len(expected_timing) == 270
    assert {(r['circuit_id'],r['method'],r['repeat']) for r in trials} == expected_timing
    for r in trials:
        assert r['source_native_reproduced'] and r['cpu'] == timing_protocol['cpu']
        assert r['date'] == protocol['paired_dates'][0] and r['seconds'] > 0
        close(r['seconds'], r['engine_seconds']+r['normalization_seconds'])
        saved = read(OUT/'serial_timing'/f"{r['circuit_id']}__{r['method']}__{r['repeat']}.json")
        assert saved == r
    for result in timing['summary']:
        values = [r['seconds'] for r in trials if r['method']==result['method'] and r['stratum']==result['stratum']]
        assert result['trials']==len(values)
        close(result['median_seconds'], statistics.median(values))
        close(result['mean_seconds'], mean(values))
        close(result['min_seconds'], min(values))
        close(result['max_seconds'], max(values))
    result = dict(status='passed', updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  frozen_source_hashes=len(manifests[0]), generation_hashes=len(manifests[1]),
                  complete_observations=len(rows), paired_observations=len(paired), unique_external_compilations=len(requests),
                  serial_timing_trials=len(trials), independent_channel_evaluations=evaluation_count,
                  independently_enumerated_mapomatic_embedding_costs=mapomatic_cost_checks,
                  all_input_measurement_checks=correctness_count, max_channel_error=max_channel_error,
                  max_measurement_povm_error=max_povm_error, protocol_sha256=protocol_hash,
                  audit_scope='Saved-data provenance, completeness, probability distributions, source reproduction, independent aggregate arithmetic and serial timing records. Earlier full channel and measurement checks are cross-checked; not a new physical experiment.')
    path=REPORT/'independent_audit.json'
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(path)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
