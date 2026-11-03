"""Full 58-day-pair training expansion. Never evaluates held-out outcomes."""
import os
os.environ.setdefault('RAYON_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('QISKIT_NUM_PROCS','1')
import argparse
import collections
import datetime as dt
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import random
import time
import numpy as np
import duckdb
from qiskit import transpile, qasm2
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import SabreLayout
from run_step2_pilot import (ROOT, DATA, CFG, PHYSICAL, EDGES, BASIS, circuits,
    parameters, proxy, swarm, distribution, save_qpy)
from prepare_step2 import dump

OUT=ROOT/'experiments/training_expansion_v1'
METHODS=['swarm','random_single','random_matched','qiskit_sabre_layout']
SEEDS=[11,22,33]

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def fixed_compile(q,layout):
    return transpile(q,basis_gates=BASIS,coupling_map=EDGES,initial_layout=list(layout),
      routing_method=CFG['routing_method'],optimization_level=CFG['optimization_level'],seed_transpiler=0)

def baseline(q,p,seed,method,budget=None):
    start=time.perf_counter();rng=random.Random(seed);compile_seconds=0.;trace=[]
    if method=='qiskit_sabre_layout':
        pm=PassManager(SabreLayout(CouplingMap(EDGES),seed=seed,max_iterations=3,
          swap_trials=1,layout_trials=1,skip_routing=True))
        pm.run(q)
        layout=tuple(pm.property_set['layout'][b] for b in q.qubits)
        proposals=[layout]
    elif method=='random_single':proposals=[tuple(rng.sample(range(len(PHYSICAL)),q.num_qubits))]
    elif method=='random_matched':
        assert isinstance(budget,int) and budget>0
        identity=tuple(range(q.num_qubits))
        population=[p for p in itertools.permutations(range(len(PHYSICAL)),q.num_qubits) if p!=identity]
        assert budget<=len(population)+1
        proposals=[identity]+rng.sample(population,budget-1)
    else:raise ValueError(method)
    best=None;best_score=float('inf');winning=None
    for layout in proposals:
        t=time.perf_counter();compiled=fixed_compile(q,layout);compile_seconds+=time.perf_counter()-t
        score=proxy(compiled,p)
        if score<best_score:best_score=score;best=compiled;winning=layout
        trace.append(best_score)
    return best,{'initial_layout_local':list(winning),'initial_layout_physical':[PHYSICAL[i] for i in winning],
      'total_search_seconds':time.perf_counter()-start,'transpile_seconds_inside_search':compile_seconds,
      'unique_placements_compiled':len(proposals),'objective_calls':len(proposals),
      'best_proxy':best_score,'best_proxy_trace':trace,'seed':seed,
      'budget_source':'per-arm swarm unique placements' if method=='random_matched' else 'one final placement',
      'candidate_layouts':[list(p) for p in proposals]}

def features(q,reuse,old,now):
    keys=sorted(now,key=str);a=np.array([old[k] for k in keys]);b=np.array([now[k] for k in keys])
    result={'num_qubits':q.num_qubits,'logical_depth':q.depth(),
      'logical_two_qubit_gates':sum(len(x.qubits)==2 for x in q.data),
      'reuse_native_depth':reuse.depth(),'reuse_cz_count':int(reuse.count_ops().get('cz',0)),
      'mean_abs_parameter_change':float(np.abs(a-b).mean()),'max_abs_parameter_change':float(np.abs(a-b).max()),
      'mean_current_parameter':float(b.mean()),'reuse_proxy_old':proxy(reuse,old),'reuse_proxy_current':proxy(reuse,now)}
    for prop in CFG['properties']:
        ks=[k for k in keys if k[0]==prop];old_v=np.array([old[k] for k in ks]);new_v=np.array([now[k] for k in ks])
        result[prop+'_mean_current']=float(new_v.mean())
        result[prop+'_mean_abs_change']=float(np.abs(new_v-old_v).mean())
    return result

def inputs():
    all_s=json.loads((DATA/'daily_scenarios.json').read_text())
    scenarios={s['scenario_id']:s for s in all_s if s['temporal_split']=='train'}
    pairs=[p for p in json.loads((DATA/'pair_manifest.json').read_text()) if p['temporal_split']=='train']
    training=[(name,family,q) for name,family,split,q in circuits() if split=='train']
    assert len(pairs)==58 and len(training)==6
    cases=[{'case_id':p['pair_id']+'__'+name+'__'+str(seed),**p,'circuit_id':name,'circuit_family':family,
      'circuit_split':'train','seed':seed} for p in pairs for name,family,q in training for seed in SEEDS]
    return scenarios,training,cases

def initialize():
    OUT.mkdir(parents=True,exist_ok=True)
    for sub in ['cases','compiled','circuits']: (OUT/sub).mkdir(exist_ok=True)
    identity={'sha256':{str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'scripts/run_step2_pilot.py',
        ROOT/'scripts/prepare_step2.py',DATA/'config.json',DATA/'daily_scenarios.json',DATA/'pair_manifest.json']},
      'packages':{k:importlib.metadata.version(k) for k in ['qiskit','qiskit-aer','numpy','duckdb']},
      'python':platform.python_version(),'platform':platform.platform(),
      'thread_environment':{k:os.environ[k] for k in ['RAYON_NUM_THREADS','OMP_NUM_THREADS','QISKIT_NUM_PROCS']}}
    path=OUT/'provenance.json'
    if path.exists():assert json.loads(path.read_text())['identity']==identity,'Execution identity changed. Preserve this experiment; create a new version.'
    else:dump(path,{'identity':identity,'created_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
    scenarios,training,cases=inputs()
    dump(OUT/'case_manifest.json',cases)
    config={'scope':'All prepared training time-pairs, existing six train circuits, three seeds, fixed single Fez patch',
      'cases':1044,'methods':METHODS,'expected_method_pairs':4176,'seeds':SEEDS,
      'noise_model_reference':'data/processed/step2_v1/config.json',
      'random_matched_budget':'Same number of unique native-circuit placements as swarm, separately for each old/fresh arm; same proxy, same routing, no wall-time matching',
      'qiskit_baseline':'SABRE topology-only layout, 3 iterations, layout_trials=1, swap_trials=1, skip_routing=True; final basic routing/level 1 match other methods',
      'seed_policy':'Same search seed across old/fresh removes independent-seed variation; random_single and SABRE ignore calibration and should not change',
      'run_order':'Seeded shuffle of old/fresh swarm order; then seeded shuffle of remaining method/arm calls; sequential execution, one warm-up',
      'no_cache_between_searches':True,'held_out_outcomes_used':False,'binary_cost_label':None,
      'timing_limit':'One execution per case/arm; exploratory wall times, not a repeated controlled timing benchmark',
      'independence_limit':'Cases share circuits, dates and calibration events; do not treat seeds or method rows as independent evidence'}
    dump(OUT/'protocol.json',config)
    manifest=[]
    for name,family,q in training:
        text=qasm2.dumps(q);path=OUT/'circuits'/f'{name}.qasm';path.write_text(text)
        manifest.append({'circuit_id':name,'family':family,'split':'train','qasm_sha256':digest(path),'qubits':q.num_qubits})
    dump(OUT/'circuit_manifest.json',manifest)
    return scenarios,training,cases

def execute_case(case,q,old_s,now_s):
    old=parameters(old_s);now=parameters(now_s);ideal=distribution(q);seed=case['seed']
    rng=random.Random(int(hashlib.sha256(case['case_id'].encode()).hexdigest()[:16],16))
    compiled={};stats={};execution_order=[]
    arms=['reuse','fresh'];rng.shuffle(arms)
    for arm in arms:
        k=('swarm',arm);compiled[k],stats[k]=swarm(q,old if arm=='reuse' else now,seed);execution_order.append(k)
    jobs=[(method,arm) for method in METHODS[1:] for arm in ['reuse','fresh']];rng.shuffle(jobs)
    for method,arm in jobs:
        k=(method,arm);budget=stats[('swarm',arm)]['unique_placements_compiled']
        compiled[k],stats[k]=baseline(q,old if arm=='reuse' else now,seed,method,budget);execution_order.append(k)
    result={'case':case,'source_scenario_hashes':{s['scenario_id']:hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest() for s in [old_s,now_s]},
      'ideal_distribution':ideal.tolist(),'compilation_order':execution_order,'methods':{}}
    for method in METHODS:
        reuse=compiled[(method,'reuse')];fresh=compiled[(method,'fresh')]
        noisy={};artifacts={};semantic_errors={}
        tic=time.perf_counter()
        for arm,body in [('reuse',reuse),('fresh',fresh)]:
            semantic_errors[arm]=float(np.max(np.abs(distribution(body)-ideal)))
            assert semantic_errors[arm]<1e-9
            noisy[arm]=distribution(body,now)
            path=OUT/'compiled'/f"{case['case_id']}__{method}__{arm}.qpy"
            artifacts[arm]={'path':str(path.relative_to(ROOT)),'sha256':save_qpy(path,body),
              'native_depth':body.depth(),'native_gate_counts':dict(body.count_ops())}
        reuse_tvd=float(np.abs(noisy['reuse']-ideal).sum()/2);fresh_tvd=float(np.abs(noisy['fresh']-ideal).sum()/2)
        if method in ['random_single','qiskit_sabre_layout']:
            assert reuse==fresh,'Calibration-blind same-seed control changed circuit'
            assert abs(reuse_tvd-fresh_tvd)<1e-12
        result['methods'][method]={'features':features(q,reuse,old,now),
          'outcomes':{'reuse_tvd':reuse_tvd,'fresh_tvd':fresh_tvd,'delta_tvd':reuse_tvd-fresh_tvd,
            'fresh_search_seconds':stats[(method,'fresh')]['total_search_seconds'],
            'evaluation_and_save_seconds':time.perf_counter()-tic,'binary_cost_aware_label':None},
          'compilation':{arm:stats[(method,arm)] for arm in ['reuse','fresh']},
          'distributions':{arm:v.tolist() for arm,v in noisy.items()},'artifacts':artifacts,
          'ideal_distribution_max_error':semantic_errors}
    for arm in ['reuse','fresh']:
        assert stats[('random_matched',arm)]['unique_placements_compiled']==stats[('swarm',arm)]['unique_placements_compiled']
    return result

def aggregate(cases):
    rows=[];feature_rows=[];groups=collections.defaultdict(list);daygroups=collections.defaultdict(list)
    for case in cases:
        path=OUT/'cases'/f"{case['case_id']}.json";record=json.loads(path.read_text());assert record['case']==case
        for method,r in record['methods'].items():
            ids={**case,'method':method,'row_id':case['case_id']+'__'+method}
            rows.append({**ids,**r['outcomes']})
            feature_rows.append({**ids,**r['features']})
            groups[method].append(r);daygroups[(method,case['pair_id'])].append(r['outcomes']['delta_tvd'])
    for name,data in [('features',feature_rows),('outcomes',rows)]:
        path=OUT/(name+'.jsonl');temp=path.with_suffix('.tmp')
        temp.write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in data));temp.replace(path)
        c=duckdb.connect();table=c.read_json(str(path));temp_parquet=OUT/(name+'.tmp.parquet');table.write_parquet(str(temp_parquet),compression='zstd');temp_parquet.replace(OUT/(name+'.parquet'));c.close()
    summary={'status':'training_expansion_complete','training_cases':len(cases),'method_pair_rows':len(rows),
      'distinct_day_pairs':len({x['pair_id'] for x in cases}),'methods':{},'held_out_outcomes_generated':False,
      'ml_trained':False,'hardware_jobs':0,'evidence_type':'Exact-density-matrix simulation under explicit scalar-calibration channel assumptions'}
    for method,recs in groups.items():
        d=np.array([r['outcomes']['delta_tvd'] for r in recs]);cost=np.array([r['outcomes']['fresh_search_seconds'] for r in recs])
        summary['methods'][method]={'paired_cases':len(recs),'fresh_better':int((d>1e-10).sum()),
          'tie':int((np.abs(d)<=1e-10).sum()),'reuse_better':int((d < -1e-10).sum()),
          'mean_delta_tvd':float(d.mean()),'median_delta_tvd':float(np.median(d)),
          'mean_fresh_tvd':float(np.mean([r['outcomes']['fresh_tvd'] for r in recs])),
          'mean_reuse_tvd':float(np.mean([r['outcomes']['reuse_tvd'] for r in recs])),
          'fresh_search_seconds_median':float(np.median(cost)),
          'fresh_search_seconds_p10_p90':[float(x) for x in np.quantile(cost,[.1,.9])]}
    dump(OUT/'daily_summary.json',[{'method':k[0],'pair_id':k[1],'cases':len(v),'mean_delta_tvd':float(np.mean(v))} for k,v in daygroups.items()])
    dump(OUT/'summary.json',summary)
    dump(OUT/'checkpoint.json',{'status':'complete','completed_cases':len(cases),'remaining_cases':0,'expected_case_count':1044})
    return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--max-new-cases',type=int);args=parser.parse_args()
    scenarios,training,cases=initialize();qmap={name:q for name,family,q in training}
    # Warm-up independent of measured results; no noise or test inputs.
    fixed_compile(training[0][2],list(range(training[0][2].num_qubits)))
    completed=0;new=0;start=time.perf_counter()
    for case in cases:
        path=OUT/'cases'/f"{case['case_id']}.json"
        if path.exists():
            saved=json.loads(path.read_text());assert saved['case']==case and set(saved['methods'])==set(METHODS)
            completed+=1;continue
        if args.max_new_cases is not None and new>=args.max_new_cases:break
        result=execute_case(case,qmap[case['circuit_id']],scenarios[case['old_scenario']],scenarios[case['current_scenario']])
        dump(path,result);new+=1;completed+=1
        dump(OUT/'checkpoint.json',{'status':'running','completed_cases':completed,'expected_case_count':len(cases),
          'remaining_cases':len(cases)-completed,'last_completed':case['case_id']})
        if new%18==0:print(f'{completed}/{len(cases)} cases saved; {time.perf_counter()-start:.1f}s this invocation',flush=True)
    if all((OUT/'cases'/f"{x['case_id']}.json").exists() for x in cases):print(json.dumps(aggregate(cases),indent=2))
    else:print(f'Saved {new} new cases; expansion unfinished.',flush=True)

if __name__=='__main__':main()
