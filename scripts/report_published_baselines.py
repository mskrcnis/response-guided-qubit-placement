"""Paired descriptive analysis; no IID inference from repeated technical cases."""
from published_baseline_common import *
from collections import defaultdict
import csv
def mean(a):return float(np.mean(a))
def outcome(a):return 'our_better' if a>1e-4 else 'our_worse' if a< -1e-4 else 'little_difference'
def main():
    p=load(OUT/'protocol.json');jobs=[load(OUT/'jobs'/(c+'.json')) for c in p['bank']]
    assert all(j['protocol_sha256']==sha(OUT/'protocol.json') for j in jobs)
    rows=[r for j in jobs for r in j['rows'] if r['paired']];failures=[r for j in jobs for r in j['failed']]
    assert not failures,failures
    assert len(rows)==p['paired_observations']==5940
    keys=lambda r:(r['circuit_id'],r['date'],r['replicate'],r['seed_rule'])
    own={keys(r):r for r in rows if r['method']=='swarm_unique_56'}
    summary=[];comparisons=[];pairs=[];by_circuit=[];by_date=[];sensitivity=[]
    for st in sorted({r['stratum'] for r in rows}):
        for m in p['methods']:
            rs=[r for r in rows if r['stratum']==st and r['method']==m]
            assert len(rs)==(330 if st=='known_instance_new_period' else 660)
            summary.append(dict(stratum=st,method=m,cases=len(rs),circuits=len({r['circuit_id'] for r in rs}),dates=len({r['date'] for r in rs}),
                                mean_tvd=mean([r['tvd'] for r in rs]),mean_cz=mean([r['native_cz'] for r in rs]),mean_depth=mean([r['native_depth'] for r in rs])))
            if m=='swarm_unique_56':continue
            delta=[dict(stratum=st,method=m,circuit_id=r['circuit_id'],date=r['date'],replicate=r['replicate'],seed_rule=r['seed_rule'],baseline_tvd=r['tvd'],our_tvd=own[keys(r)]['tvd'],gain=r['tvd']-own[keys(r)]['tvd']) for r in rs]
            pairs.extend(delta)
            counts={k:sum(outcome(r['gain'])==k for r in delta) for k in ['our_better','little_difference','our_worse']}
            base=mean([r['baseline_tvd'] for r in delta]);ours=mean([r['our_tvd'] for r in delta])
            dates=sorted({r['date'] for r in delta});loo=[mean([r['gain'] for r in delta if r['date']!=d]) for d in dates]
            comparisons.append(dict(stratum=st,baseline=m,cases=len(delta),baseline_mean=base,our_mean=ours,mean_gain=base-ours,relative_mean_error_reduction_pct=100*(base-ours)/base,
                                    case_outcomes=counts,leave_one_date_out_gain_range=[min(loo),max(loo)]))
            for field,dest in [('circuit_id',by_circuit),('date',by_date)]:
                for value in sorted({r[field] for r in delta}):
                    a=[r for r in delta if r[field]==value]
                    dest.append(dict(stratum=st,baseline=m,**{field:value},cases=len(a),mean_baseline_tvd=mean([r['baseline_tvd'] for r in a]),mean_our_tvd=mean([r['our_tvd'] for r in a]),mean_gain=mean([r['gain'] for r in a])))
            for rule in ['fixed','resampled']:
                a=[r for r in delta if r['seed_rule']==rule]
                sensitivity.append(dict(stratum=st,baseline=m,scope=rule,cases=len(a),mean_gain=mean([r['gain'] for r in a])))
            # Calendar changes are taken from the already audited source report.
            audit=load(ROOT/'reports/exploratory_oct30_v1/calibration_audit.json')
            changed=set(audit['changed_value_transition_dates'])
            a=[r for r in delta if r['date'] in changed]
            sensitivity.append(dict(stratum=st,baseline=m,scope='changing_calibration_only',cases=len(a),mean_gain=mean([r['gain'] for r in a])))
    for name,data in [('method_summary',summary),('paired_comparisons',comparisons),('paired_cases',pairs),('by_circuit',by_circuit),('by_date',by_date),('sensitivity',sensitivity)]:dump(REPORT/(name+'.json'),data)
    with (REPORT/'comparison.csv').open('w',newline='') as f:
        cols=['stratum','baseline','cases','baseline_mean','our_mean','mean_gain','relative_mean_error_reduction_pct']
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(comparisons)
    maxerr=max(v['independent_error'] for j in jobs for v in j['evaluations'].values());maxpovm=max(v['max_povm_error'] for j in jobs for v in j['checks'].values())
    manifest={str((OUT/'jobs'/(c+'.json')).relative_to(ROOT)):sha(OUT/'jobs'/(c+'.json')) for c in p['bank']}
    for r in load(OUT/'requests.json').values():manifest[str(Path(r['destination']).relative_to(ROOT))]=sha(r['destination'])
    for f in (OUT/'native').glob('*.qpy'):manifest[str(f.relative_to(ROOT))]=sha(f)
    dump(OUT/'generation_manifest.json',manifest)
    validation=dict(status='passed',paired_rows=len(rows),external_compilation_requests=p['unique_compilation_requests'],methods=len(p['methods']),circuits=len(p['bank']),dates=len(p['paired_dates']),failed=[],
                    compiled_correctness_checks=sum(len(j['checks']) for j in jobs),independent_channel_evaluations=sum(len(j['evaluations']) for j in jobs),max_channel_error=maxerr,max_measurement_povm_error=maxpovm,
                    source_swarm_reproduced_rows=len(own),protocol_sha256=sha(OUT/'protocol.json'),generation_manifest_sha256=sha(OUT/'generation_manifest.json'))
    dump(REPORT/'validation.json',validation)
    lines=['DIRECT COMPARISON WITH PUBLISHED COMPILER IMPLEMENTATIONS','Exploratory comparison on the already exposed October 19-29 period.','',
           'All methods use the same 15 circuits, six physical qubits, eleven dates, native gates and exact noisy-output evaluator.',
           'Rows are correlated technical cases, not independent hardware experiments. New instances belong to related circuit families.','',
           'Mean simulated TVD, native CZ count and depth (smaller is better):','Group | Method | Cases | Mean TVD | Mean CZ | Mean depth']
    for r in summary:lines.append(f"{r['stratum']} | {r['method']} | {r['cases']} | {r['mean_tvd']:.8f} | {r['mean_cz']:.3f} | {r['mean_depth']:.3f}")
    lines+=['','Direct paired contrasts. Positive gain favors our swarm; negative favors the baseline.','Group | Baseline | Relative mean TVD reduction (%) | Absolute TVD gain | Our better/little/worse cases']
    for r in comparisons:lines.append(f"{r['stratum']} | {r['baseline']} | {r['relative_mean_error_reduction_pct']:+.3f}% | {r['mean_gain']:+.8f} | {r['case_outcomes']}")
    lines+=['','Comparison scope and fairness:',p['fairness'],p['budgets'],p['deterministic'],p['inference'],'',
            'Published provenance:']
    for r in p['registry']:lines.append(f"{r['id']}: {r['title']}; {r['venue']} ({r['year']}). https://doi.org/{r['doi']}. {r.get('note','')}")
    lines+=['','The common circuit adapter is not an exact reproduction of each paper\'s original end-to-end benchmark. Maintained implementations, versions and all deviations are pinned in the protocol.',
            'Mapomatic sees only eligible calibration parameters and uses its published default success-product objective. No method sees evaluated noisy TVD during selection.',
            'Duostra uses abstract unit gate durations, not measured device pulse times. Rational-pi interchange avoids the upstream decimal parser\'s 1e-4-radian quantization. Its routing algorithm is unmodified; the source patch adds permutation logging and build compatibility only.',
            'The noise model omits idle relaxation, crosstalk, leakage and calibration uncertainty. Small-circuit in-model comparisons do not establish large-circuit or physical-device superiority.',
            'These are per-date compilation comparisons, not a test of an ML decision about when to recompile. No pre-search controller is qualified.','',
            f"Validation: {validation['compiled_correctness_checks']} arbitrary-input measurement checks; {validation['independent_channel_evaluations']} independent noisy-channel evaluations. Maximum discrepancies: {maxpovm:.3g} (measurement semantics), {maxerr:.3g} (noisy probabilities).",
            'Compilation timing from the main run is concurrent and diagnostic only. Consult separate serial timing results when present.']
    (REPORT/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    dump(OUT/'checkpoint.json',dict(status='cohort_analysis_validated_serial_timing_and_final_audit_pending',updated_utc=now(),validation=validation))
    print(json.dumps(comparisons,indent=2))
if __name__=='__main__':main()
