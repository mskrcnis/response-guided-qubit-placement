"""Versioned sequential policy rollouts on training-period calibration history."""
import os
for variable in ['RAYON_NUM_THREADS','OMP_NUM_THREADS','QISKIT_NUM_PROCS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[variable]='1'
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime as dt
import hashlib
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
import run_step2_pilot as engine
import expand_training as comp
from run_diversity import install_region
from prepare_step2 import ROOT,dump

OUT=ROOT/'experiments/sequential_v1'
DATA=ROOT/'data/processed/diversity_v1'
ML=ROOT/'experiments/ml_preliminary_v1'
METHODS=['swarm','random_matched']
POLICIES=['always','never','periodic_3','periodic_7','drift','ridge','forest']
FIRST='2026-07-20';LAST='2026-07-31'
UNSEEN=['ghz_5','product_5','rotcx_star_n5_l6']
SEEDS=[11,22,33]

def load(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def objsha(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def jsonlines(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s]

def native_signature(q):
    return objsha({'global_phase':str(q.global_phase),'ops':[
        [x.operation.name,[str(v) for v in x.operation.params],
         [q.find_bit(b).index for b in x.qubits],[q.find_bit(b).index for b in x.clbits]]
        for x in q.data if x.operation.name!='barrier']})

def design():
    regions={r['region_id']:r for r in load(DATA/'config.json')['regions']}
    scenarios={s['scenario_id']:s for s in load(DATA/'daily_scenarios.json') if FIRST<=s['cutoff_utc'][:10]<=LAST}
    bank={}
    for r in load(DATA/'circuit_manifest.json'):
        bank[r['circuit_id']]={'family':r['family'],'path':str((DATA/'circuits'/(r['circuit_id']+'.qasm')).relative_to(ROOT)),
            'regions':list(regions),'sha256':r['qasm_sha256']}
    for r in load(ROOT/'experiments/training_expansion_v1/circuit_manifest.json'):
        p=ROOT/'experiments/training_expansion_v1/circuits'/(r['circuit_id']+'.qasm')
        bank[r['circuit_id']]={'family':r['family'],'path':str(p.relative_to(ROOT)),'regions':['ibm_fez_r0'],'sha256':r['qasm_sha256']}
    jobs=[{'job_id':rid+'__'+cid+'__'+str(seed),'region_id':rid,'circuit_id':cid,
        'family':meta['family'],'seed':seed,'template_status':'unseen' if cid in UNSEEN else 'familiar'}
        for cid,meta in sorted(bank.items()) for rid in meta['regions'] for seed in SEEDS]
    assert len(bank)==18 and len(jobs)==126 and len(scenarios)==36
    return regions,scenarios,bank,jobs

def training_ranges():
    rows={r['uid']:r for r in jsonlines(ML/'harmonized_cases.jsonl')};splits=load(ML/'split_audit.json')
    cols=load(ML/'feature_schema.json')['input_columns']
    return {m:{k:[min(rows[u][k] for u in splits[m]['july_audit']['training_uids']),
                      max(rows[u][k] for u in splits[m]['july_audit']['training_uids'])] for k in cols} for m in METHODS}

def initialize(workers):
    regions,scenarios,bank,jobs=design()
    assert load(ML/'validation.json')['status']=='passed'
    for sub in ['jobs','compiled','compilation_runs','noise_evaluations']:(OUT/sub).mkdir(parents=True,exist_ok=True)
    source_files=[ROOT/'scripts'/f for f in ['run_sequential.py','run_step2_pilot.py','expand_training.py','run_diversity.py','prepare_step2.py']]
    source_files += [DATA/f for f in ['config.json','daily_scenarios.json','circuit_manifest.json']]
    source_files += [ROOT/'data/processed/step2_v1/config.json',ROOT/'experiments/training_expansion_v1/circuit_manifest.json']
    source_files += [ML/f for f in ['validation.json','selection.json','protocol.json','provenance.json','feature_schema.json',
                                    'harmonized_cases.jsonl','split_audit.json','scenario_evidence.json']]
    source_files += [ML/'models'/m/f'audit__{family}.joblib' for m in METHODS for family in ['drift','ridge','forest']]
    source_files += [ROOT/meta['path'] for meta in bank.values()]
    identity={'files':{str(p.relative_to(ROOT)):sha(p) for p in source_files},'workers':workers,
        'packages':{p:importlib.metadata.version(p) for p in ['qiskit','qiskit-aer','numpy','scipy','scikit-learn','joblib','duckdb']},
        'python':platform.python_version(),'threads_per_worker':1}
    if (OUT/'provenance.json').exists():assert load(OUT/'provenance.json')['identity']==identity,'Changed identity; create a new version.'
    else:dump(OUT/'provenance.json',{'identity':identity,'created_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
    protocol={'version':'sequential_v1','date_start':FIRST,'date_end':LAST,'methods':METHODS,'policies':POLICIES,
        'jobs':len(jobs),'seeds':SEEDS,'models':'audit models fitted through July 8, settings selected through July 12; never final_refit models',
        'trigger':'ridge/forest predicted TVD benefit >1e-10; fixed drift thresholds from ML selection; periodic elapsed calendar days since last compilation >=3 or >=7',
        'initialization':'First ready day of every contiguous valid-data episode compiles both methods for all policies. Common initialization counted separately from triggered recompiles.',
        'missing_day':'No outcome imputation; skip blocked states, end episode, reset every policy together on next ready day. This is a data-validity reset, not a measured device outage.',
        'state':'Each policy retains its own selected native circuit and compilation-date calibration. Reuse can exceed 7 days; record model-input range violations without changing policy.',
        'comparison':'Primary observed selected TVD and triggered recompile counts. Local oracle regret is conditional on each policy state, not a shared globally optimal sequential oracle.',
        'timing':'Fresh searches measured anew once per scenario/circuit/seed/method under two workers. Alternative-policy charges sum those observed search times when called; not independently measured end-to-end deployment runtimes. Shared-library generation time separate.',
        'random_budget':'Per-date random search uses the unique-placement count of same-date swarm; budget construction is an offline matched-work convention. Its PSO cost is separately recorded, not hidden as deployable zero-cost budget discovery.',
        'decision_order':'All decisions for a date use only retained circuits and past/current calibration before that date\'s fresh search is available.',
        'seed_policy':'Fixed same search seed across dates; no search-cache reuse between fresh searches. Library reused across alternative policy evaluations only.',
        'quality':'Exact output-distribution TVD under inherited independent depolarizing + assignment-error simulator; no hardware jobs',
        'evaluation':'Internal retrospective July development audit; previous July results already inspected; no policy tuning here',
        'protection':'No August onward calibration outcomes or QFT/QAOA circuits evaluated',
        'uncertainty':'Correlated histories, circuits and seeds; descriptive metrics, no independent-trial significance claim',
        'cost_utility':None}
    if (OUT/'protocol.json').exists():assert load(OUT/'protocol.json')==protocol
    dump(OUT/'protocol.json',protocol);dump(OUT/'job_manifest.json',jobs)
    dump(OUT/'model_training_ranges.json',training_ranges())
    dump(OUT/'scenario_manifest.json',[{'scenario_id':s['scenario_id'],'sha256':objsha(s),'status':s['status'],
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
    qs['random_matched'],stats['random_matched']=comp.baseline(q,p,seed,'random_matched',stats['swarm']['unique_placements_compiled'])
    assert stats['swarm']['unique_placements_compiled']==stats['random_matched']['unique_placements_compiled']
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

def feature_vector(q,reuse,old,current):
    values=comp.features(q,reuse,engine.parameters(old),engine.parameters(current))
    now=dt.datetime.fromisoformat(current['cutoff_utc']);then=dt.datetime.fromisoformat(old['cutoff_utc'])
    values.update({'reuse_age_days':(now-then).days,
        'max_current_measurement_age_hours':max((now-dt.datetime.fromisoformat(p['calibrated_time'])).total_seconds()/3600 for p in current['parameters']),
        'physical_edge_count':len(engine.EDGES)//2,
        'physical_max_degree':max(collections.Counter(a for a,b in engine.EDGES).values())})
    return values

def decide(policy,features,package=None):
    if policy=='always':return True,None
    if policy=='never':return False,None
    if policy.startswith('periodic_'):return features['reuse_age_days']>=int(policy.split('_')[1]),None
    if policy=='drift':return features['mean_abs_parameter_change']>package['model']['threshold'],None
    x=np.array([[features[k] for k in package['input_columns']]])
    pred=float(package['model'].predict(x)[0]);return pred>1e-10,pred

def execute_job(job):
    start=time.perf_counter();path=OUT/'jobs'/(job['job_id']+'.json')
    if path.exists():
        saved=load(path);assert saved['job']==job
        return {'job_id':job['job_id'],'resumed':True,'rows':len(saved['rows'])}
    regions,scenarios,bank,_=design();rid=job['region_id'];install_region(regions[rid])
    q=qasm2.load(str(ROOT/bank[job['circuit_id']]['path']));q.name=job['circuit_id']
    ideal=engine.distribution(q);comp.fixed_compile(q,list(range(q.num_qubits))) # warm-up excluded
    packages={m:{p:joblib.load(ML/'models'/m/f'audit__{p}.joblib') for p in ['drift','ridge','forest']} for m in METHODS}
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
                'search_timing_observation':bid+'__'+method if decision['compile'] else None,
                'random_budget_discovery_swarm_seconds':fresh['methods']['swarm']['stats']['total_search_seconds'] if method=='random_matched' and decision['compile'] else 0.,
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
        'noise_evaluations':len(list((OUT/'noise_evaluations').glob('*.json'))),'held_out_outcomes_generated':False,'hardware_jobs':0}
    dump(OUT/'checkpoint.json',counts);print(json.dumps(counts),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);parser.add_argument('--max-jobs',type=int)
    args=parser.parse_args();jobs=initialize(args.workers)
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
