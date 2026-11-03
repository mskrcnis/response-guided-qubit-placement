"""Replace offline PSO-derived random budgets with a predeclared proposal count."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_NUM_PROCS']:os.environ[key]='1'
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor,as_completed
import datetime as dt
import importlib.metadata
import multiprocessing
import platform
import time
import numpy as np
from qiskit import qasm2,qpy
import run_sequential as seq

ROOT=seq.ROOT;ML=seq.ML;OUT=ROOT/'experiments/fixed_budget_v1'
BUDGET=56;METHOD='random_fixed_56'
load,sha,objsha,dump=seq.load,seq.sha,seq.objsha,seq.dump

def inputs():
    rows=[r for r in seq.jsonlines(ML/'harmonized_cases.jsonl') if r['method']=='random_matched']
    regions={r['region_id']:r for r in load(seq.DATA/'config.json')['regions']}
    scenarios={s['scenario_id']:s for s in load(seq.DATA/'daily_scenarios.json')}
    _,_,bank,jobs=seq.design()
    assert len(rows)==2268 and len(jobs)==126
    return rows,regions,scenarios,bank,jobs

def scenario_id(region,date):return region+'_'+date.replace('-','')

def initialize():
    rows,regions,scenarios,bank,jobs=inputs()
    for name in ['jobs','compilation_runs','compiled','cases','noise_evaluations','models']:(OUT/name).mkdir(parents=True,exist_ok=True)
    config=seq.engine.CFG
    assert config['swarm']['particles']*(config['swarm']['iterations']+1)==BUDGET
    files=[ROOT/'scripts'/f for f in ['prepare_fixed_budget.py','run_sequential.py','run_diversity.py','run_step2_pilot.py','expand_training.py','prepare_step2.py']]
    files += [ML/f for f in ['harmonized_cases.jsonl','provenance.json','validation.json','split_audit.json','scenario_evidence.json','feature_schema.json','selection.json']]
    files += [seq.DATA/f for f in ['config.json','daily_scenarios.json','circuit_manifest.json']]
    files += [ROOT/meta['path'] for meta in bank.values()]
    identity={'files':{str(p.relative_to(ROOT)):sha(p) for p in files},'workers':2,'native_threads':1,
        'python':platform.python_version(),'packages':{k:importlib.metadata.version(k) for k in ['qiskit','qiskit-aer','numpy','scipy','scikit-learn','duckdb','joblib']}}
    if (OUT/'provenance.json').exists():assert load(OUT/'provenance.json')['identity']==identity
    else:dump(OUT/'provenance.json',{'identity':identity,'created_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
    protocol={'version':'fixed_budget_v1','proposal_budget':BUDGET,'budget_basis':'8 swarm particles times (1 initial + 6 iterations); defined by existing algorithm, not an outcome-optimized count',
        'swarm':'Unchanged: exactly 56 proposal/objective calls; duplicate placement cache can reduce native compilation work',
        'random_fixed_56':'Identity placement plus 55 seeded unique random injective placements; no PSO run required',
        'fairness':'Same proposal count, not equal unique compilations or wall time. Report all three resources separately.',
        'labels':'Rebuild all 2268 random-search pairs and their reuse-native features. Old variable-budget random labels/models are incompatible.',
        'swarm_compatibility':'Original swarm labels/models remain compatible because swarm code/config/budget are unchanged',
        'training_only':'Same June-July pairs, circuits, regions and seeds. No August onward or QFT/QAOA outcomes.',
        'noise':'Unchanged exact density-matrix depolarizing + assignment model; no hardware jobs',
        'timing':'One search per scenario/circuit/seed, two worker processes; shared exploratory timings, no repeated runtime benchmark'}
    if (OUT/'generation_protocol.json').exists():assert load(OUT/'generation_protocol.json')==protocol
    dump(OUT/'generation_protocol.json',protocol);dump(OUT/'job_manifest.json',jobs)
    return jobs

def fixed_search(q,p,seed):
    circuit,stats=seq.comp.baseline(q,p,seed,'random_matched',BUDGET)
    stats['budget_source']='predeclared 56 proposals; independent of swarm outcomes/work'
    stats['method']=METHOD
    assert stats['unique_placements_compiled']==stats['objective_calls']==BUDGET
    assert len({tuple(x) for x in stats['candidate_layouts']})==BUDGET
    return circuit,stats

def bundle(q,s,seed):
    bid=s['scenario_id']+'__'+q.name+'__'+str(seed);path=OUT/'compilation_runs'/(bid+'.json')
    if path.exists():
        r=load(path);assert r['scenario_sha256']==objsha(s)
        p=ROOT/r['artifact']['path'];assert sha(p)==r['artifact']['sha256']
        with p.open('rb') as f:compiled=qpy.load(f)[0]
        return r,compiled
    compiled,stats=fixed_search(q,seq.engine.parameters(s),seed)
    error=float(np.max(np.abs(seq.engine.distribution(q)-seq.engine.distribution(compiled))));assert error<1e-9
    p=OUT/'compiled'/(bid+'.qpy')
    r={'bundle_id':bid,'scenario_id':s['scenario_id'],'scenario_sha256':objsha(s),'circuit_id':q.name,
        'region_id':s['region_id'],'seed':seed,'method':METHOD,'stats':stats,'maximum_ideal_output_error':error,
        'artifact':{'path':str(p.relative_to(ROOT)),'sha256':seq.engine.save_qpy(p,compiled)},'native_signature':seq.native_signature(compiled)}
    dump(path,r);return r,compiled

def execute(job):
    dest=OUT/'jobs'/(job['job_id']+'.json')
    if dest.exists():assert load(dest)['job']==job;return {'job_id':job['job_id'],'resumed':True}
    start=time.perf_counter();rows,regions,scenarios,bank,_=inputs()
    seq.install_region(regions[job['region_id']]);q=qasm2.load(str(ROOT/bank[job['circuit_id']]['path']));q.name=job['circuit_id']
    seq.comp.fixed_compile(q,list(range(q.num_qubits)))
    cases=[r for r in rows if (r['region_id'],r['circuit_id'],r['seed'])==(job['region_id'],job['circuit_id'],job['seed'])]
    cache={};evaluated={};case_ids=[];ideal=seq.engine.distribution(q)
    def evaluate(bid,s):
        eid=bid+'__at__'+s['scenario_id']
        if eid not in evaluated:
            p=OUT/'noise_evaluations'/(eid+'.json')
            if p.exists():
                ev=load(p);assert ev['artifact_sha256']==cache[bid][0]['artifact']['sha256'] and ev['current_scenario_sha256']==objsha(s)
            else:
                distribution=seq.engine.distribution(cache[bid][1],seq.engine.parameters(s))
                ev={'evaluation_id':eid,'bundle_id':bid,'current_scenario':s['scenario_id'],'current_scenario_sha256':objsha(s),
                    'artifact_sha256':cache[bid][0]['artifact']['sha256'],'distribution':distribution.tolist(),
                    'tvd':float(np.abs(distribution-ideal).sum()/2)}
                dump(p,ev)
            evaluated[eid]=ev
        return evaluated[eid]
    for source in cases:
        old=scenarios[scenario_id(source['region_id'],source['old_date'])]
        now=scenarios[scenario_id(source['region_id'],source['current_date'])]
        assert old['status']==now['status']=='ready' and now['temporal_split']=='train'
        assert now['cutoff_utc'][:10]<'2026-08-01'
        bids=[]
        for s in [old,now]:
            bid=s['scenario_id']+'__'+q.name+'__'+str(job['seed']);bids.append(bid)
            if bid not in cache:cache[bid]=bundle(q,s,job['seed'])
        reuse,fresh=bids;old_ev=evaluate(reuse,now);new_ev=evaluate(fresh,now)
        features=seq.feature_vector(q,cache[reuse][1],old,now)
        uid=source['uid'].removesuffix('__random_matched')+'__'+METHOD
        record={'uid':uid,'source_uid':source['uid'],'job_id':job['job_id'],'method':METHOD,
            'old_scenario':old['scenario_id'],'current_scenario':now['scenario_id'],
            'reuse_bundle':reuse,'fresh_bundle':fresh,'reuse_evaluation':old_ev['evaluation_id'],
            'fresh_evaluation':new_ev['evaluation_id'],'ideal_distribution':ideal.tolist(),
            'features':features,'outcomes':{'reuse_tvd':old_ev['tvd'],'fresh_tvd':new_ev['tvd'],
                'delta_tvd':old_ev['tvd']-new_ev['tvd'],'fresh_search_seconds':cache[fresh][0]['stats']['total_search_seconds']}}
        dump(OUT/'cases'/(uid+'.json'),record);case_ids.append(uid)
    dump(dest,{'job':job,'case_ids':case_ids,'bundle_ids':list(cache),'evaluation_ids':list(evaluated),'elapsed_seconds':time.perf_counter()-start})
    return {'job_id':job['job_id'],'cases':len(cases),'seconds':time.perf_counter()-start}

def export(name,rows):
    import duckdb,json
    path=OUT/(name+'.jsonl');tmp=path.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows));tmp.replace(path)
    c=duckdb.connect();tmp=OUT/(name+'.tmp.parquet');c.read_json(str(path)).write_parquet(str(tmp),compression='zstd');tmp.replace(OUT/(name+'.parquet'));c.close()

def aggregate():
    old={r['uid']:r for r in seq.jsonlines(ML/'harmonized_cases.jsonl')}
    rows=[]
    for p in sorted((OUT/'cases').glob('*.json')):
        r=load(p);source=old[r['source_uid']]
        row={**source,**r['features'],**r['outcomes'],'uid':r['uid'],'method':METHOD,'source_uid':r['source_uid'],
             'source_case_file':str(p.relative_to(ROOT)),
             'fresh_group':source['fresh_group'].removesuffix('__random_matched')+'__'+METHOD,
             'timing_observation_id':'fixed_budget_v1__'+r['fresh_bundle']+'__'+METHOD}
        rows.append(row)
    assert len(rows)==2268;export('random_training_cases',rows)
    dump(OUT/'checkpoint.json',{'status':'labels_saved_validation_pending','completed_jobs':126,'remaining_jobs':0,
        'training_pairs':len(rows),'compile_bundles':len(list((OUT/'compilation_runs').glob('*.json'))),
        'noise_evaluations':len(list((OUT/'noise_evaluations').glob('*.json'))),'models_retrained':False,'held_out_outcomes_generated':False})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--max-jobs',type=int);args=parser.parse_args()
    jobs=initialize();todo=[j for j in jobs if not (OUT/'jobs'/(j['job_id']+'.json')).exists()];done=len(jobs)-len(todo)
    if args.max_jobs:todo=todo[:args.max_jobs]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(execute,j) for j in todo]):
            r=future.result();done+=1;dump(OUT/'checkpoint.json',{'status':'generating_labels','completed_jobs':done,'remaining_jobs':126-done,'last_job':r})
            print(f"{done}/126 {r['job_id']} {r.get('seconds',0):.2f}s",flush=True)
    if all((OUT/'jobs'/(j['job_id']+'.json')).exists() for j in jobs):aggregate()

if __name__=='__main__':main()
