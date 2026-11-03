"""Small, training-only, exact-density-matrix pilot; not a publication benchmark."""
import collections
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import random
import time
import importlib.metadata
import platform
import numpy as np
from qiskit import QuantumCircuit, transpile, qasm2, qpy
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from prepare_step2 import ROOT, dump

DATA=ROOT/'data/processed/step2_v1'
OUT=ROOT/'experiments/step2_pilot_v1'
CFG=json.loads((DATA/'config.json').read_text())
PHYSICAL=CFG['physical_qubits']
EDGES=[[PHYSICAL.index(a),PHYSICAL.index(b)] for a,b in CFG['directed_cz_edges']]
BASIS=CFG['basis_gates']

def circuits():
    result=[]
    for family in ['ghz','product','qft','qaoa']:
        for n in [3,4,5]:
            q=QuantumCircuit(n,name=f'{family}_{n}')
            if family=='ghz':
                q.h(0)
                for i in range(n-1):q.cx(i,i+1)
            elif family=='product':
                for i in range(n):q.ry((i+1)*0.37,i);q.rz((i+1)*0.19,i)
            elif family=='qft':
                q.x(0)
                for i in range(n):
                    q.h(i)
                    for j in range(i+1,n):q.cp(math.pi/(2**(j-i)),j,i)
                for i in range(n//2):q.swap(i,n-i-1)
            else:
                q.h(range(n))
                for layer in range(2):
                    for i in range(n):q.rzz(0.43+0.17*layer,i,(i+1)%n)
                    for i in range(n):q.rx(0.29+0.13*layer,i)
            q.measure_all()
            split=next(k for k,v in CFG['circuit_family_splits'].items() if family in v)
            result.append((q.name,family,split,q))
    return result

def parameters(scenario):
    assert scenario['status']=='ready'
    return {(r['property'],r['qubit_a'],r['qubit_b']):r['value'] for r in scenario['parameters']}

def measured_qubits(q):
    mapping={q.find_bit(x.clbits[0]).index:q.find_bit(x.qubits[0]).index
             for x in q.data if x.operation.name=='measure'}
    assert set(mapping)==set(range(q.num_clbits))
    return [mapping[i] for i in range(q.num_clbits)]

def gate_error(p,name,qubits):
    physical=[PHYSICAL[i] for i in qubits]
    return p[(name+'_gate_error',physical[0],physical[1] if len(physical)==2 else None)]

def proxy(q,p):
    """Negative log product-of-success surrogate, not measured circuit fidelity."""
    cost=0.0
    for x in q.data:
        name=x.operation.name
        if name in ['barrier','measure']:continue
        assert name in BASIS
        r=gate_error(p,name,[q.find_bit(b).index for b in x.qubits])
        cost-=math.log1p(-r)
    for i in measured_qubits(q):
        a=PHYSICAL[i]
        r=(p[('prob_meas0_prep1',a,None)]+p[('prob_meas1_prep0',a,None)])/2
        cost-=math.log1p(-r)
    return cost

def swaps_toward(current,target):
    cur=list(current); ops=[]
    for i,x in enumerate(target):
        if cur[i]!=x:
            j=cur.index(x);ops.append((i,j));cur[i],cur[j]=cur[j],cur[i]
    assert cur==list(target)
    return ops

def swarm(q,p,seed):
    """Permutation PSO with swap-sequence velocities and fixed basic routing."""
    started=time.perf_counter();rng=random.Random(seed);opts=CFG['swarm']
    size=len(PHYSICAL);cache={};compile_seconds=0.0;evaluations=0
    def evaluate(position):
        nonlocal compile_seconds,evaluations
        placement=tuple(position[:q.num_qubits]);evaluations+=1
        if placement not in cache:
            tic=time.perf_counter()
            compiled=transpile(q,basis_gates=BASIS,coupling_map=EDGES,
                initial_layout=list(placement),routing_method=CFG['routing_method'],
                optimization_level=CFG['optimization_level'],seed_transpiler=0)
            compile_seconds+=time.perf_counter()-tic
            cache[placement]=(proxy(compiled,p),compiled)
        return cache[placement][0]
    positions=[list(range(size))]+[rng.sample(range(size),size) for _ in range(opts['particles']-1)]
    velocities=[[] for _ in positions];best=[x.copy() for x in positions]
    scores=[evaluate(x) for x in positions];best_scores=scores.copy()
    idx=min(range(len(scores)),key=lambda i:scores[i]);global_best=positions[idx].copy();global_score=scores[idx]
    trace=[global_score]
    for iteration in range(opts['iterations']):
        # Synchronous global best from the previous iteration.
        target=global_best.copy()
        for i,pos in enumerate(positions):
            velocity=[s for s in velocities[i] if rng.random()<opts['inertia']]
            velocity += [s for s in swaps_toward(pos,best[i]) if rng.random()<opts['cognitive']]
            velocity += [s for s in swaps_toward(pos,target) if rng.random()<opts['social']]
            if rng.random()<opts['mutation_probability']:velocity.append(tuple(rng.sample(range(size),2)))
            for a,b in velocity:pos[a],pos[b]=pos[b],pos[a]
            assert sorted(pos)==list(range(size))
            velocities[i]=velocity;scores[i]=evaluate(pos)
            if scores[i]<best_scores[i]:best_scores[i]=scores[i];best[i]=pos.copy()
        idx=min(range(len(best_scores)),key=lambda i:best_scores[i])
        if best_scores[idx]<global_score:global_score=best_scores[idx];global_best=best[idx].copy()
        trace.append(global_score)
    layout=tuple(global_best[:q.num_qubits]);compiled=cache[layout][1]
    return compiled,{'initial_layout_local':list(layout),'initial_layout_physical':[PHYSICAL[i] for i in layout],
      'total_search_seconds':time.perf_counter()-started,'transpile_seconds_inside_search':compile_seconds,
      'unique_placements_compiled':len(cache),'objective_calls':evaluations,
      'best_proxy':global_score,'best_proxy_trace':trace,'seed':seed}

def noise_model(p):
    noise=NoiseModel(basis_gates=BASIS)
    for name in ['rz','sx','x']:
        for i in range(len(PHYSICAL)):
            r=gate_error(p,name,[i])
            if r:noise.add_quantum_error(depolarizing_error(2*r,1),name,[i])
    for a,b in EDGES:
        r=gate_error(p,'cz',[a,b])
        if r:noise.add_quantum_error(depolarizing_error(4*r/3,2),'cz',[a,b])
    return noise

def apply_assignment(dist,p,measured):
    result=np.array(dist,dtype=float).copy()
    for bit,local in enumerate(measured):
        physical=PHYSICAL[local]
        p01=p[('prob_meas1_prep0',physical,None)]
        p10=p[('prob_meas0_prep1',physical,None)]
        for zero in range(len(result)):
            if (zero>>bit)&1:continue
            one=zero|(1<<bit);x,y=result[zero],result[one]
            result[zero]=(1-p01)*x+p10*y
            result[one]=p01*x+(1-p10)*y
    return result

def distribution(q,p=None):
    measured=measured_qubits(q)
    body=q.remove_final_measurements(inplace=False)
    if p is None:full=Statevector.from_instruction(body).probabilities()
    else:
        body.save_probabilities()
        simulator=AerSimulator(method='density_matrix',noise_model=noise_model(p),max_parallel_threads=1)
        run=simulator.run(body,shots=None).result();assert run.success
        full=np.asarray(run.data(0)['probabilities'])
    marginal=np.zeros(2**q.num_clbits)
    for basis,value in enumerate(full):
        bits=sum(((basis>>physical)&1)<<classical for classical,physical in enumerate(measured))
        marginal[bits]+=value
    if p is not None:marginal=apply_assignment(marginal,p,measured)
    assert np.isfinite(marginal).all() and marginal.min()>-1e-10
    assert abs(marginal.sum()-1)<1e-9
    return marginal

def save_qpy(path,q):
    buf=io.BytesIO();qpy.dump(q,buf);binary=buf.getvalue()
    tmp=path.with_suffix('.tmp');tmp.write_bytes(binary);tmp.replace(path)
    return hashlib.sha256(binary).hexdigest()

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'circuits').mkdir(exist_ok=True);(OUT/'trials').mkdir(exist_ok=True)
    provenance={'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [DATA/'config.json',DATA/'daily_scenarios.json',DATA/'pair_manifest.json',
                  Path(__file__),ROOT/'scripts/prepare_step2.py']},
        'packages':{name:importlib.metadata.version(name) for name in ['qiskit','qiskit-aer','numpy']},
        'python':platform.python_version(),'platform':platform.platform()}
    provenance_path=OUT/'run_provenance.json'
    if provenance_path.exists():
        assert json.loads(provenance_path.read_text())['execution_identity']==provenance, 'Inputs/code/environment changed; use a new experiment version.'
    else:
        dump(provenance_path,{'execution_identity':provenance,
          'recorded_after_initial_pilot':any((OUT/'trials').glob('*.json'))})
    manifest=[]
    for name,family,split,q in circuits():
        text=qasm2.dumps(q);path=OUT/'circuits'/f'{name}.qasm';path.write_text(text)
        manifest.append({'circuit_id':name,'family':family,'circuit_split':split,
          'num_qubits':q.num_qubits,'source':'Locally generated protocol fixture, not ICCAD hardware execution',
          'qasm_sha256':hashlib.sha256(text.encode()).hexdigest()})
    dump(OUT/'circuit_manifest.json',manifest)
    scenarios={s['scenario_id']:s for s in json.loads((DATA/'daily_scenarios.json').read_text())}
    pairs=[p for p in json.loads((DATA/'pair_manifest.json').read_text()) if p['temporal_split']=='train'][:3]
    expected=[]
    for pair in pairs:
        old_s=scenarios[pair['old_scenario']];now_s=scenarios[pair['current_scenario']]
        old=parameters(old_s);now=parameters(now_s)
        for name,family,split,q in circuits():
            if split!='train':continue
            ideal=distribution(q)
            for seed in [11,22,33]:
                trial_id=pair['pair_id']+'__'+name+'__'+str(seed);expected.append(trial_id)
                path=OUT/'trials'/f'{trial_id}.json'
                if path.exists():continue
                reuse,old_search=swarm(q,old,seed)
                fresh,new_search=swarm(q,now,seed)
                for compiled in [reuse,fresh]:
                    assert np.max(np.abs(distribution(compiled)-ideal))<1e-9,'Compilation changed ideal output distribution'
                tic=time.perf_counter();reuse_dist=distribution(reuse,now);fresh_dist=distribution(fresh,now)
                eval_time=time.perf_counter()-tic
                reuse_tvd=float(np.abs(reuse_dist-ideal).sum()/2)
                fresh_tvd=float(np.abs(fresh_dist-ideal).sum()/2)
                values=np.array([now[k] for k in sorted(now,key=str)])
                before=np.array([old[k] for k in sorted(now,key=str)])
                features={'num_qubits':q.num_qubits,'logical_depth':q.depth(),
                  'logical_two_qubit_gates':sum(len(x.qubits)==2 for x in q.data),
                  'reuse_native_depth':reuse.depth(),'reuse_cz_count':int(reuse.count_ops().get('cz',0)),
                  'mean_abs_parameter_change':float(np.abs(values-before).mean()),
                  'max_abs_parameter_change':float(np.abs(values-before).max()),
                  'reuse_proxy_old':proxy(reuse,old),'reuse_proxy_current':proxy(reuse,now),
                  'mean_current_parameter':float(values.mean())}
                qpy_info={}
                for arm,compiled in [('reuse',reuse),('fresh',fresh)]:
                    qp=OUT/'trials'/f'{trial_id}__{arm}.qpy';qpy_info[arm]={'file':str(qp.relative_to(ROOT)),'sha256':save_qpy(qp,compiled)}
                record={'trial_id':trial_id,**pair,'circuit_id':name,'circuit_split':split,'seed':seed,
                  'evidence_type':'Exact density-matrix simulation using measured scalar calibration with assumed noise channels',
                  'features_available_before_recompile':features,
                  'outcomes':{'reuse_tvd':reuse_tvd,'fresh_tvd':fresh_tvd,'delta_tvd':reuse_tvd-fresh_tvd,
                    'fresh_search_seconds':new_search['total_search_seconds'],
                    'evaluation_seconds':eval_time,'cost_aware_binary_label':None},
                  'ideal_distribution':ideal.tolist(),'reuse_distribution':reuse_dist.tolist(),'fresh_distribution':fresh_dist.tolist(),
                  'old_compilation':old_search,'fresh_compilation':new_search,'compiled_artifacts':qpy_info,
                  'current_scenario_sha256':hashlib.sha256(json.dumps(now_s,sort_keys=True).encode()).hexdigest()}
                dump(path,record)
                dump(OUT/'checkpoint.json',{'last_completed':trial_id,'status':'pilot_running'})
                print('Saved '+trial_id,flush=True)
    results=[json.loads((OUT/'trials'/f'{t}.json').read_text()) for t in expected]
    dump(OUT/'features.json',[{'trial_id':r['trial_id'],**r['features_available_before_recompile']} for r in results])
    dump(OUT/'outcomes.json',[{'trial_id':r['trial_id'],**r['outcomes']} for r in results])
    deltas=[r['outcomes']['delta_tvd'] for r in results]
    summary={'status':'pilot_complete_not_training_or_full_benchmark','paired_trials':len(results),
      'chronological_pairs':len(pairs),'circuit_ids':sorted({r['circuit_id'] for r in results}),
      'seeds':[11,22,33],'training_only':True,
      'fresh_better_tvd':sum(x>1e-10 for x in deltas),'tie_tvd':sum(abs(x)<=1e-10 for x in deltas),
      'reuse_better_tvd':sum(x < -1e-10 for x in deltas),
      'mean_delta_tvd':float(np.mean(deltas)),'min_delta_tvd':min(deltas),'max_delta_tvd':max(deltas),
      'interpretation':'Descriptive smoke-test counts over dependent cases; not independent samples or a performance claim',
      'validation_and_test_outcomes_generated':False}
    dump(OUT/'summary.json',summary);dump(OUT/'checkpoint.json',{'status':'pilot_complete','completed_trial_ids':expected})
    print(json.dumps(summary,indent=2))

if __name__=='__main__':run()
