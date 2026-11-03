"""Frozen-protocol August validation / gated September-October final test.

State engine below is adapted from run_sequential.py without editing that file.
Each process installs its own region; historical experiments stay immutable.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_NUM_PROCS']:
    os.environ[key]='1'
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime as dt
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import platform
import time
import joblib
import numpy as np
import duckdb
from qiskit import qasm2,qpy
import run_sequential as seq
from prepare_fixed_budget import fixed_search

ROOT=seq.ROOT; DATA=seq.DATA; ML=seq.ML
FIXED=ROOT/'experiments/fixed_budget_v1'
PLAN=seq.load(FIXED/'validation_protocol_v1.json')
METHODS=list(PLAN['compilers']); POLICIES=PLAN['policies']; SEEDS=PLAN['seeds']
load,sha,objsha,dump=seq.load,seq.sha,seq.objsha,seq.dump
jsonlines,native_signature,feature_vector,decide=seq.jsonlines,seq.native_signature,seq.feature_vector,seq.decide
engine,comp,install_region=seq.engine,seq.comp,seq.install_region
PHASE='validation'; OUT=ROOT/'experiments/validation_fixed56_v2'

def configure(phase):
    global PHASE,OUT
    assert phase in ['validation','final_test']
    PHASE=phase; OUT=ROOT/'experiments'/('validation_fixed56_v2' if phase=='validation' else 'final_test_fixed56_v2')

def design():
    spec=PLAN[PHASE]; first,last=spec['dates']; rid=spec['region']
    regions={r['region_id']:r for r in load(DATA/'config.json')['regions'] if r['region_id']==rid}
    assert regions[rid]['physical_qubits']==spec['physical_qubits']
    scenarios={}
    for original in load(ROOT/'data/processed/step2_v1/daily_scenarios.json'):
        if first<=original['cutoff_utc'][:10]<=last:
            s={**original,'region_id':rid,'source_scenario_id':original['scenario_id'],
               'source_scenario_sha256':objsha(original),'scenario_id':rid+'_'+original['cutoff_utc'][:10].replace('-','')}
            scenarios[s['scenario_id']]=s
    known=spec['temporal_transfer_templates']; unseen=spec['unseen_family_templates']; bank={}
    for base in [DATA,ROOT/'experiments/training_expansion_v1',ROOT/'experiments/step2_pilot_v1']:
        for item in load(base/'circuit_manifest.json'):
            cid=item['circuit_id']
            if cid in known+unseen and cid not in bank:
                p=base/'circuits'/(cid+'.qasm')
                assert sha(p)==item['qasm_sha256']
                bank[cid]={'family':item['family'],'path':str(p.relative_to(ROOT)),'regions':[rid],'sha256':sha(p)}
    assert set(bank)==set(known+unseen)
    assert len(scenarios)==(dt.date.fromisoformat(last)-dt.date.fromisoformat(first)).days+1
    jobs=[{'job_id':rid+'__'+cid+'__'+str(seed),'region_id':rid,'circuit_id':cid,'family':meta['family'],
           'seed':seed,'template_status':'known' if cid in known else 'new_family','phase':PHASE}
          for cid,meta in sorted(bank.items()) for seed in SEEDS]
    assert len(jobs)==63
    return regions,scenarios,bank,jobs

def model_packages():
    packages={}
    for m in METHODS:
        packages[m]={}
        for policy,info in PLAN['models'][m].items():
            assert sha(ROOT/info['path'])==info['sha256']
            package=joblib.load(ROOT/info['path']); assert package['input_columns']==PLAN['input_columns']
            packages[m][policy]=package
    return packages

def training_ranges():
    rows={'swarm':[r for r in jsonlines(ML/'harmonized_cases.jsonl') if r['method']=='swarm'],
          'random_fixed_56':jsonlines(FIXED/'random_training_cases.jsonl')}
    assert all(len(v)==2268 for v in rows.values())
    return {m:{k:[min(r[k] for r in rows[m]),max(r[k] for r in rows[m])] for k in PLAN['input_columns']} for m in METHODS}

def measurement_keys(scenarios):
    return {objsha({k:p[k] for k in ['backend','property','qubit_a','qubit_b','calibrated_time']})
            for s in scenarios for p in s['parameters']}

def source_gate(scenarios):
    """Run before outcomes: source identity, event chronology, and exact disjointness."""
    assert sha(FIXED/'validation_protocol_v1.json')==load(FIXED/'release_manifest.json')['protocol_sha256']
    assert load(FIXED/'label_validation.json')['status']=='passed'
    assert load(FIXED/'model_validation.json')['status']=='passed'
    for file,digest in PLAN['code_hashes'].items(): assert sha(ROOT/file)==digest,file
    model_packages()
    events=ROOT/'data/processed/step2_v1/calibration_events.parquet'
    assert sha(events)==load(DATA/'config.json')['events_sha256']
    # Include all training endpoints, not merely the final training dates.
    evidence=load(ML/'scenario_evidence.json'); rows=jsonlines(ML/'harmonized_cases.jsonl')
    train_keys={key for r in rows for field in ['old_endpoint','current_endpoint'] for key in evidence[r[field]]['measurement_keys']}
    prior_keys=set(train_keys); selection_hash=None
    if PHASE=='final_test':
        validation=ROOT/'experiments/validation_fixed56_v2'
        assert load(validation/'validation.json')['status']=='passed'
        selection=load(validation/'frozen_choices.json')
        assert selection['protocol_sha256']==sha(FIXED/'validation_protocol_v1.json')
        assert selection['validation_artifact_manifest_sha256']==sha(validation/'artifact_manifest.json')
        assert selection['validation_summary_sha256']==sha(validation/'summary.json')
        selection_hash=sha(validation/'frozen_choices.json')
        prior_keys.update(load(validation/'source_gate.json')['evaluation_measurement_keys'])
    keys=measurement_keys(scenarios.values()); assert not keys&prior_keys,'Measurement leakage'
    selected=[]; ready=0
    for s in scenarios.values():
        now=dt.datetime.fromisoformat(s['cutoff_utc'])
        assert PLAN[PHASE]['dates'][0]<=now.date().isoformat()<=PLAN[PHASE]['dates'][1]
        if s['status']=='ready':ready+=1; assert len(s['parameters'])==40
        for p in s['parameters']:
            if s['status']=='ready':
                assert p['quality_reason']=='accepted'
                for key in ['observed_time','calibrated_time']:
                    assert 0<=(now-dt.datetime.fromisoformat(p[key])).total_seconds()<=72*3600
            selected.append((s['scenario_id'],s['backend'],s['cutoff_utc'],p['property'],p['qubit_a'],p['qubit_b'],p['observed_time'],p['calibrated_time'],p['value']))
    con=duckdb.connect(); con.execute("SET TimeZone='UTC'")
    con.execute('create table selected(id varchar,backend varchar,cutoff timestamptz,property varchar,a int,b int,obs timestamptz,cal timestamptz,value double)')
    con.executemany('insert into selected values (?,?,?,?,?,?,?,?,?)',selected)
    con.read_parquet(str(events)).create_view('events')
    assert con.execute('''select count(*) from selected s where not exists (select 1 from events e
        where e.backend=s.backend and e.property=s.property and e.qubit_a=s.a and e.qubit_b is not distinct from s.b
        and e.observed_time=s.obs and e.calibrated_time=s.cal and e.value=s.value)''').fetchone()[0]==0
    assert con.execute('''select count(*) from selected s where exists (select 1 from events e
        where e.backend=s.backend and e.property=s.property and e.qubit_a=s.a and e.qubit_b is not distinct from s.b
        and e.observed_time<=s.cutoff and (e.observed_time>s.obs or (e.observed_time=s.obs and e.calibrated_time>s.cal)))''').fetchone()[0]==0
    con.close()
    result={'status':'passed','phase':PHASE,'parameter_links':len(selected),'ready_dates':ready,
            'blocked_dates':[{'date':s['cutoff_utc'][:10],'reasons':s['blocking_reasons']} for s in scenarios.values() if s['status']!='ready'],
            'training_measurement_count':len(train_keys),'prior_consumed_measurement_count':len(prior_keys),
            'evaluation_measurement_keys':sorted(keys),'shared_measurements':0,'latest_source_event_verified':True,
            'frozen_choices_sha256':selection_hash,'protocol_sha256':sha(FIXED/'validation_protocol_v1.json')}
    return result

def initialize(workers):
    assert workers==2,'Frozen accounting uses two worker processes'
    regions,scenarios,bank,jobs=design()
    gate=source_gate(scenarios)
    for sub in ['jobs','compiled','compilation_runs','noise_evaluations']:(OUT/sub).mkdir(parents=True,exist_ok=True)
    sources=[ROOT/'scripts'/f for f in ['run_heldout.py','run_sequential.py','prepare_fixed_budget.py','run_step2_pilot.py','expand_training.py','run_diversity.py','prepare_step2.py']]
    sources += [ROOT/'data/processed/step2_v1'/f for f in ['config.json','daily_scenarios.json']]
    sources += [DATA/'config.json',ML/'harmonized_cases.jsonl',ML/'scenario_evidence.json',FIXED/'random_training_cases.jsonl',FIXED/'validation_protocol_v1.json',FIXED/'release_manifest.json']
    sources += [ROOT/meta['path'] for meta in bank.values()]
    sources += [ROOT/info['path'] for m in PLAN['models'].values() for info in m.values()]
    if PHASE=='final_test':sources += [ROOT/'experiments/validation_fixed56_v2/frozen_choices.json']
    identity={'phase':PHASE,'files':{str(p.relative_to(ROOT)):sha(p) for p in sources},'workers':workers,
        'packages':{p:importlib.metadata.version(p) for p in ['qiskit','qiskit-aer','numpy','scipy','scikit-learn','joblib','duckdb']},
        'python':platform.python_version(),'threads_per_worker':1}
    if (OUT/'provenance.json').exists():assert load(OUT/'provenance.json')['identity']==identity,'Changed identity; create a new version.'
    else:dump(OUT/'provenance.json',{'identity':identity,'created_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
    dump(OUT/'source_gate.json',gate); dump(OUT/'protocol.json',PLAN); dump(OUT/'job_manifest.json',jobs)
    dump(OUT/'model_training_ranges.json',training_ranges())
    dump(OUT/'scenario_manifest.json',[{'scenario_id':s['scenario_id'],'sha256':objsha(s),'status':s['status'],
        'source_scenario_id':s['source_scenario_id'],'source_scenario_sha256':s['source_scenario_sha256'],
        'blocking_reasons':s['blocking_reasons']} for s in scenarios.values()])
    return jobs

def get_bundle(q,scenario,seed):
    bid=scenario['scenario_id']+'__'+q.name+'__'+str(seed);path=OUT/'compilation_runs'/(bid+'.json')
    if path.exists():
        record=load(path);assert record['scenario_sha256']==objsha(scenario)
        qs={}
        for m,info in record['methods'].items():
            p=ROOT/info['artifact']['path'];assert sha(p)==info['artifact']['sha256']
            with p.open('rb') as f:qs[m]=qpy.load(f)[0]
        return record,qs
    p=engine.parameters(scenario);qs={};stats={}
    qs['swarm'],stats['swarm']=engine.swarm(q,p,seed)
    qs['random_fixed_56'],stats['random_fixed_56']=fixed_search(q,p,seed)
    assert all(stats[m]['objective_calls']==56 for m in METHODS)
    assert stats['random_fixed_56']['unique_placements_compiled']==56
    ideal=engine.distribution(q)
    record={'bundle_id':bid,'scenario_id':scenario['scenario_id'],'scenario_sha256':objsha(scenario),
        'circuit_id':q.name,'region_id':scenario['region_id'],'seed':seed,'methods':{}}
    for m in METHODS:
        error=float(np.max(np.abs(engine.distribution(qs[m])-ideal)));assert error<1e-9
        artifact=OUT/'compiled'/(bid+'__'+m+'.qpy')
        record['methods'][m]={'stats':stats[m],'maximum_ideal_probability_error':error,
            'native_signature':native_signature(qs[m]),'artifact':{'path':str(artifact.relative_to(ROOT)),
            'sha256':engine.save_qpy(artifact,qs[m]),'depth':qs[m].depth(),'gate_counts':dict(qs[m].count_ops())}}
    dump(path,record);return record,qs

def execute_job(job):
    configure(job['phase'])
    start=time.perf_counter();path=OUT/'jobs'/(job['job_id']+'.json')
    if path.exists():
        saved=load(path);assert saved['job']==job
        return {'job_id':job['job_id'],'resumed':True,'rows':len(saved['rows'])}
    regions,scenarios,bank,_=design();rid=job['region_id'];install_region(regions[rid])
    q=qasm2.load(str(ROOT/bank[job['circuit_id']]['path']),custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS);q.name=job['circuit_id']
    ideal=engine.distribution(q);comp.fixed_compile(q,list(range(q.num_qubits))) # warm-up excluded
    packages=model_packages()
    ranges=load(OUT/'model_training_ranges.json')
    dates=sorted([s for s in scenarios.values() if s['region_id']==rid],key=lambda s:s['cutoff_utc'])
    state={(m,p):None for m in METHODS for p in POLICIES};bundles={};noisy={};rows=[];skipped=[];episode=0
    def evaluate(bundle_id,method,current):
        eid=bundle_id+'__'+method+'__at__'+current['scenario_id']
        if eid in noisy:return noisy[eid]
        ep=OUT/'noise_evaluations'/(eid+'.json')
        artifact=bundles[bundle_id][0]['methods'][method]['artifact']
        if ep.exists():
            item=load(ep);assert item['artifact_sha256']==artifact['sha256'] and item['current_scenario_sha256']==objsha(current)
        else:
            tic=time.perf_counter();dist=engine.distribution(bundles[bundle_id][1][method],engine.parameters(current))
            item={'evaluation_id':eid,'bundle_id':bundle_id,'method':method,'current_scenario':current['scenario_id'],
                'artifact_sha256':artifact['sha256'],'current_scenario_sha256':objsha(current),
                'distribution':dist.tolist(),'tvd':float(np.abs(dist-ideal).sum()/2),'simulation_seconds':time.perf_counter()-tic}
            dump(ep,item)
        noisy[eid]=item;return item
    for current in dates:
        date=current['cutoff_utc'][:10]
        if current['status']!='ready':
            skipped.append({'date':date,'scenario_id':current['scenario_id'],'reasons':current['blocking_reasons']})
            state={k:None for k in state};continue
        initial=all(v is None for v in state.values())
        if initial:episode+=1
        pending=[]
        # Deliberate two-phase loop: decisions finish before creating this day's fresh bundle.
        for method in METHODS:
            for policy in POLICIES:
                old_id=state[(method,policy)];feat=None;pred=None;outside=[];decision_seconds=0.;feature_seconds=0.
                if initial:choose=True
                else:
                    old_s=scenarios[bundles[old_id][0]['scenario_id']]
                    t=time.perf_counter();feat=feature_vector(q,bundles[old_id][1][method],old_s,current)
                    feature_seconds=time.perf_counter()-t
                    outside=[k for k,v in feat.items() if v<ranges[method][k][0]-1e-12 or v>ranges[method][k][1]+1e-12]
                    t=time.perf_counter();choose,pred=decide(policy,feat,packages[method].get(policy));decision_seconds=time.perf_counter()-t
                pending.append({'method':method,'policy':policy,'previous_bundle':old_id,'features':feat,
                    'predicted_delta_tvd':pred,'compile':bool(choose),'outside_training_range':outside,
                    'feature_seconds':feature_seconds,'decision_seconds':decision_seconds})
        fresh,qs=get_bundle(q,current,job['seed']);bid=fresh['bundle_id'];bundles[bid]=(fresh,qs)
        for decision in pending:
            method=decision['method'];policy=decision['policy'];old_id=decision['previous_bundle']
            chosen=bid if decision['compile'] else old_id;state[(method,policy)]=chosen
            fresh_eval=evaluate(bid,method,current)
            reuse_eval=None if initial else evaluate(old_id,method,current)
            chosen_eval=fresh_eval if decision['compile'] else reuse_eval
            old_sig=None if initial else bundles[old_id][0]['methods'][method]['native_signature']
            changed=None if initial else old_sig!=bundles[chosen][0]['methods'][method]['native_signature']
            stats=fresh['methods'][method]['stats'];search=stats['total_search_seconds'] if decision['compile'] else 0.
            rows.append({'row_id':job['job_id']+'__'+date+'__'+method+'__'+policy,
                **job,'date':date,'scenario_id':current['scenario_id'],'episode':episode,'initialization':initial,
                **decision,'selected_bundle':chosen,'selected_evaluation':chosen_eval['evaluation_id'],
                'reuse_evaluation':None if initial else reuse_eval['evaluation_id'],'fresh_evaluation':fresh_eval['evaluation_id'],
                'selected_tvd':chosen_eval['tvd'],'reuse_tvd':None if initial else reuse_eval['tvd'],
                'fresh_tvd':fresh_eval['tvd'],'local_oracle_regret_tvd':None if initial else chosen_eval['tvd']-min(reuse_eval['tvd'],fresh_eval['tvd']),
                'native_circuit_changed':changed,'search_seconds_charged':search,
                'placement_compilations_charged':stats['unique_placements_compiled'] if decision['compile'] else 0,
                'proposals_charged':stats['objective_calls'] if decision['compile'] else 0,
                'search_timing_observation':bid+'__'+method if decision['compile'] else None,
                'random_budget_discovery_swarm_seconds':0.,
                'controller_seconds_charged':(decision['feature_seconds'] if policy in ['drift','ridge','forest'] else 0.)+decision['decision_seconds']})
    record={'job':job,'ideal_distribution':ideal.tolist(),'rows':rows,'skipped':skipped,
        'unique_bundles':list(bundles),'noise_evaluation_ids':[v['evaluation_id'] for v in noisy.values()],
        'elapsed_seconds':time.perf_counter()-start,'episodes':episode}
    dump(path,record);return {'job_id':job['job_id'],'rows':len(rows),'seconds':record['elapsed_seconds'],'resumed':False}

def export(name,rows):
    p=OUT/(name+'.jsonl');tmp=p.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows));tmp.replace(p)
    con=duckdb.connect();dest=OUT/(name+'.tmp.parquet');con.read_json(str(p)).write_parquet(str(dest),compression='zstd');dest.replace(OUT/(name+'.parquet'));con.close()

def aggregate(jobs):
    records=[load(OUT/'jobs'/(j['job_id']+'.json')) for j in jobs]
    rows=[r for job in records for r in job['rows']]
    export('decisions',rows)
    counts={'status':'generation_complete_validation_pending','completed_jobs':len(records),'remaining_jobs':0,
        'decision_records':len(rows),'unique_bundles':len(list((OUT/'compilation_runs').glob('*.json'))),
        'unique_compiled_artifacts':len(list((OUT/'compiled').glob('*.qpy'))),
        'noise_evaluations':len(list((OUT/'noise_evaluations').glob('*.json'))),'held_out_outcomes_generated':True,'phase':PHASE,'hardware_jobs':0}
    dump(OUT/'checkpoint.json',counts);print(json.dumps(counts),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);parser.add_argument('--max-jobs',type=int)
    parser.add_argument('--phase',choices=['validation','final_test'],default='validation')
    parser.add_argument('--gate-only',action='store_true')
    args=parser.parse_args();configure(args.phase);jobs=initialize(args.workers)
    if args.gate_only:print('Source/model gates passed; no outcomes generated',flush=True);return
    pending=[j for j in jobs if not (OUT/'jobs'/(j['job_id']+'.json')).exists()]
    if args.max_jobs:pending=pending[:args.max_jobs]
    completed=len(jobs)-sum(not (OUT/'jobs'/(j['job_id']+'.json')).exists() for j in jobs)
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(execute_job,j) for j in pending]
            for future in as_completed(futures):
                result=future.result();completed+=1
                dump(OUT/'checkpoint.json',{'status':'running','completed_jobs':completed,'remaining_jobs':len(jobs)-completed,'last_job':result})
                print(f"{completed}/{len(jobs)} {result['job_id']} {result.get('seconds',0):.2f}s",flush=True)
    if all((OUT/'jobs'/(j['job_id']+'.json')).exists() for j in jobs):aggregate(jobs)

if __name__=='__main__':main()
