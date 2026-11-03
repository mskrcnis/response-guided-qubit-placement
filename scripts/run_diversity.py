"""Execute the frozen diversity design, using two isolated local worker processes.

Existing validated compiler/simulator functions are reused unchanged. Each
process installs its region context before executing a region/circuit/seed job;
there is no concurrent region mutation within a process and no source-file edit.
"""
import os
for variable in ['RAYON_NUM_THREADS','OMP_NUM_THREADS','QISKIT_NUM_PROCS','OPENBLAS_NUM_THREADS']:
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
import random
import time
import numpy as np
import duckdb
from qiskit import qasm2, qpy
import run_step2_pilot as engine
import expand_training as comparators
from prepare_step2 import ROOT, dump

DATA=ROOT/'data/processed/diversity_v1'
OUT=ROOT/'experiments/diversity_v1'
METHODS=comparators.METHODS

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def object_sha(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()

def install_region(region):
    physical=list(region['physical_qubits'])
    edges=[[physical.index(a),physical.index(b)] for a,b in region['directed_cz_edges']]
    # All referenced functions access these two modules' runtime globals.
    # A worker executes one job at a time. Original files/configs remain frozen.
    engine.PHYSICAL=physical;engine.EDGES=edges
    comparators.PHYSICAL=physical;comparators.EDGES=edges
    assert engine.PHYSICAL==comparators.PHYSICAL and engine.EDGES==comparators.EDGES

def load_design():
    config=json.loads((DATA/'config.json').read_text())
    scenarios={s['scenario_id']:s for s in json.loads((DATA/'daily_scenarios.json').read_text())}
    cases=json.loads((DATA/'case_manifest.json').read_text())
    regions={r['region_id']:r for r in config['regions']}
    return config,scenarios,cases,regions

def bundle(q,scenario,seed):
    bundle_id=scenario['scenario_id']+'__'+q.name+'__'+str(seed)
    path=OUT/'compilation_runs'/(bundle_id+'.json')
    if path.exists():
        record=json.loads(path.read_text());assert record['scenario_sha256']==object_sha(scenario)
        circuits={}
        for method,meta in record['methods'].items():
            p=ROOT/meta['artifact']['path'];assert sha(p)==meta['artifact']['sha256']
            with p.open('rb') as f:circuits[method]=qpy.load(f)[0]
        return record,circuits
    pvals=engine.parameters(scenario);compiled={};stats={}
    compiled['swarm'],stats['swarm']=engine.swarm(q,pvals,seed)
    order=METHODS[1:].copy();random.Random(int(hashlib.sha256(bundle_id.encode()).hexdigest()[:16],16)).shuffle(order)
    for method in order:
        compiled[method],stats[method]=comparators.baseline(q,pvals,seed,method,stats['swarm']['unique_placements_compiled'])
    ideal=engine.distribution(q)
    record={'bundle_id':bundle_id,'scenario_id':scenario['scenario_id'],'scenario_sha256':object_sha(scenario),
      'region_id':scenario['region_id'],'backend':scenario['backend'],'circuit_id':q.name,'seed':seed,
      'execution_order':['swarm']+order,'methods':{}}
    for method,circuit in compiled.items():
        err=float(np.max(np.abs(engine.distribution(circuit)-ideal)));assert err<1e-9
        path_qpy=OUT/'compiled'/(bundle_id+'__'+method+'.qpy')
        record['methods'][method]={'stats':stats[method],
          'artifact':{'path':str(path_qpy.relative_to(ROOT)),'sha256':engine.save_qpy(path_qpy,circuit),
            'native_depth':circuit.depth(),'native_gate_counts':dict(circuit.count_ops())},
          'maximum_ideal_probability_error':err}
    assert stats['swarm']['unique_placements_compiled']==stats['random_matched']['unique_placements_compiled']
    dump(path,record)
    return record,compiled

def execute_job(job):
    config,scenarios,all_cases,regions=load_design()
    region_id,circuit_id,seed=job;region=regions[region_id];install_region(region)
    q=qasm2.load(str(DATA/'circuits'/(circuit_id+'.qasm')));q.name=circuit_id
    cases=[x for x in all_cases if (x['region_id'],x['circuit_id'],x['seed'])==tuple(job)]
    comparators.fixed_compile(q,list(range(q.num_qubits)))  # warm-up, unmeasured
    bundle_cache={};noise_cache={};ideal=engine.distribution(q);new=0
    for case in cases:
        path=OUT/'cases'/(case['case_id']+'.json')
        if path.exists():
            record=json.loads(path.read_text());assert record['case']==case and set(record['methods'])==set(METHODS)
            continue
        old_s=scenarios[case['old_scenario']];now_s=scenarios[case['current_scenario']]
        for s in [old_s,now_s]:
            if s['scenario_id'] not in bundle_cache:bundle_cache[s['scenario_id']]=bundle(q,s,seed)
        old_bundle,old_circuits=bundle_cache[old_s['scenario_id']]
        fresh_bundle,fresh_circuits=bundle_cache[now_s['scenario_id']]
        old=engine.parameters(old_s);now=engine.parameters(now_s)
        record={'case':case,'evidence_type':'Retrospective scalar-calibration-informed simulation',
          'ideal_distribution':ideal.tolist(),'source_scenario_hashes':{s['scenario_id']:object_sha(s) for s in [old_s,now_s]},
          'compilation_bundles':{'reuse':old_bundle['bundle_id'],'fresh':fresh_bundle['bundle_id']},'methods':{}}
        for method in METHODS:
            noisy={};reuse_flags={}
            for arm,compiled,b in [('reuse',old_circuits[method],old_bundle),('fresh',fresh_circuits[method],fresh_bundle)]:
                k=(b['bundle_id'],method,now_s['scenario_id']);reuse_flags[arm]=k in noise_cache
                if k not in noise_cache:noise_cache[k]=engine.distribution(compiled,now)
                noisy[arm]=noise_cache[k]
            tvd={a:float(np.abs(v-ideal).sum()/2) for a,v in noisy.items()}
            if method in ['random_single','qiskit_sabre_layout']:
                assert old_circuits[method]==fresh_circuits[method] and abs(tvd['reuse']-tvd['fresh'])<1e-12
            features=comparators.features(q,old_circuits[method],old,now)
            features['reuse_age_days']=case['age_days']
            cutoff=dt.datetime.fromisoformat(now_s['cutoff_utc'])
            features['max_current_measurement_age_hours']=max((cutoff-dt.datetime.fromisoformat(p['calibrated_time'])).total_seconds()/3600 for p in now_s['parameters'])
            degrees=collections.Counter(a for a,b in engine.EDGES)
            features['physical_edge_count']=len(engine.EDGES)//2
            features['physical_max_degree']=max(degrees.values())
            stats=fresh_bundle['methods'][method]['stats']
            record['methods'][method]={'features':features,
              'outcomes':{'reuse_tvd':tvd['reuse'],'fresh_tvd':tvd['fresh'],'delta_tvd':tvd['reuse']-tvd['fresh'],
                'fresh_search_seconds':stats['total_search_seconds'],'binary_cost_aware_label':None},
              'distributions':{arm:v.tolist() for arm,v in noisy.items()},
              'noisy_evaluation_cached':reuse_flags,
              'artifact_refs':{arm:b['methods'][method]['artifact'] for arm,b in [('reuse',old_bundle),('fresh',fresh_bundle)]},
              'timing_observation_id':fresh_bundle['bundle_id']+'__'+method}
        dump(path,record);new+=1
    return {'job':job,'cases':len(cases),'new_cases':new}

def initialize(workers):
    for sub in ['cases','compiled','compilation_runs']:(OUT/sub).mkdir(parents=True,exist_ok=True)
    identity={'files':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/prepare_diversity.py',
      ROOT/'scripts/expand_training.py',ROOT/'scripts/run_step2_pilot.py',ROOT/'scripts/prepare_step2.py',
      DATA/'config.json',DATA/'daily_scenarios.json',DATA/'case_manifest.json',DATA/'circuit_manifest.json',
      ROOT/'data/processed/step2_v1/config.json']},
      'packages':{k:importlib.metadata.version(k) for k in ['qiskit','qiskit-aer','numpy','duckdb']},
      'python':platform.python_version(),'worker_processes':workers,'native_threads_per_worker':1}
    p=OUT/'provenance.json'
    if p.exists():assert json.loads(p.read_text())['identity']==identity,'Changed execution identity; create a new version.'
    else:dump(p,{'identity':identity,'created_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
    protocol={'worker_processes':workers,'execution':'Isolated process jobs per region/circuit/seed; one region at a time per process',
      'compiler_reuse':'Each scenario/circuit/seed compiled once per method and reused across age comparisons, with artifact/bundle links',
      'noise_cache':'Exact evaluations cached only for identical compilation bundle, method and current scenario; repeated labels are not independent',
      'timing_warning':'Shared timing observations across ages; two workers may contend. Exploratory wall times, not a controlled performance benchmark',
      'search_settings':'Inherited frozen step2_v1; identical unique-placement counts for PSO/random matched',
      'four_methods':METHODS,'ml_trained':False,'hardware_jobs':0,'held_out_outcomes_generated':False}
    dump(OUT/'protocol.json',protocol)

def aggregate(cases):
    feature_rows=[];outcome_rows=[];by_method=collections.defaultdict(list);strata=collections.defaultdict(list)
    for case in cases:
        record=json.loads((OUT/'cases'/(case['case_id']+'.json')).read_text());assert record['case']==case
        for method,r in record['methods'].items():
            ids={**case,'method':method,'row_id':case['case_id']+'__'+method,
              'timing_observation_id':r['timing_observation_id']}
            feature_rows.append({**ids,**r['features']});outcome_rows.append({**ids,**r['outcomes']})
            by_method[method].append(r['outcomes'])
            strata[(method,case['region_id'],case['age_days'],case['pattern'],case['layers'])].append(r['outcomes'])
    for name,rows in [('features',feature_rows),('outcomes',outcome_rows)]:
        path=OUT/(name+'.jsonl');tmp=path.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows));tmp.replace(path)
        c=duckdb.connect();temp=OUT/(name+'.tmp.parquet');c.read_json(str(path)).write_parquet(str(temp),compression='zstd');temp.replace(OUT/(name+'.parquet'));c.close()
    def stats(rows):
        d=np.array([r['delta_tvd'] for r in rows])
        return {'paired_cases':len(rows),'fresh_better':int((d>1e-10).sum()),'tie':int((np.abs(d)<=1e-10).sum()),
          'reuse_better':int((d < -1e-10).sum()),'mean_delta_tvd':float(d.mean()),
          'mean_reuse_tvd':float(np.mean([r['reuse_tvd'] for r in rows])),
          'mean_fresh_tvd':float(np.mean([r['fresh_tvd'] for r in rows]))}
    summary={'status':'generation_complete_validation_pending','cases':len(cases),'method_pairs':len(outcome_rows),
      'unique_compilation_bundles':len(list((OUT/'compilation_runs').glob('*.json'))),
      'unique_compiled_qpy_files':len(list((OUT/'compiled').glob('*.qpy'))),
      'methods':{k:stats(v) for k,v in by_method.items()},'ml_trained':False,'held_out_outcomes_generated':False}
    dump(OUT/'summary.json',summary)
    dump(OUT/'stratified_summary.json',[{'method':k[0],'region_id':k[1],'age_days':k[2],'pattern':k[3],'layers':k[4],**stats(v)} for k,v in strata.items()])
    schema={'join_key':'row_id','recommended_initial_method':'swarm','input_columns':list(next(iter(json.loads((OUT/'cases'/(cases[0]['case_id']+'.json')).read_text())['methods'].values()))['features']),
      'targets':['delta_tvd','fresh_search_seconds'],'timing_group_key':'timing_observation_id',
      'grouping_warning':'Shared circuits, date endpoints, region, seeds and timing observations; do not random-split rows',
      'metadata_columns':list({**cases[0],'method':'','row_id':'','timing_observation_id':''}),
      'forbidden_inputs':['fresh compilation scores/artifacts','simulated outcomes','future/held-out observations']}
    dump(OUT/'feature_schema.json',schema)
    dump(OUT/'checkpoint.json',{'status':'generation_complete_validation_pending','completed_cases':len(cases),'remaining_cases':0})
    return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);parser.add_argument('--max-jobs',type=int);args=parser.parse_args()
    initialize(args.workers);config,scenarios,cases,regions=load_design()
    jobs=sorted({(c['region_id'],c['circuit_id'],c['seed']) for c in cases})
    pending=[job for job in jobs if any(not (OUT/'cases'/(c['case_id']+'.json')).exists() for c in cases
      if (c['region_id'],c['circuit_id'],c['seed'])==job)]
    if args.max_jobs is not None:pending=pending[:args.max_jobs]
    started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(execute_job,job) for job in pending]
        for future in as_completed(futures):
            result=future.result();count=len(list((OUT/'cases').glob('*.json')))
            dump(OUT/'checkpoint.json',{'status':'running','completed_cases':count,'expected_cases':len(cases),
              'remaining_cases':len(cases)-count,'last_completed_job':result['job']})
            print(f'{count}/{len(cases)} cases saved; {time.perf_counter()-started:.1f}s this invocation',flush=True)
    if all((OUT/'cases'/(c['case_id']+'.json')).exists() for c in cases):print(json.dumps(aggregate(cases),indent=2))
    else:print('Requested job subset saved; generation incomplete.',flush=True)

if __name__=='__main__':main()
