"""Reconstruct search paths and independently check candidate scores/outcomes."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','QISKIT_NUM_PROCS']:os.environ[k]='1'
import collections,datetime,itertools,math,multiprocessing,random
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from qiskit import qasm2,qpy
import run_heldout as src
from validate_circuit_objective import matrix_tape,independent_p1
from run_coverage_gain import ROOT,load,dump,sha
OUT=ROOT/'experiments/directional_full_search_v1';SOURCE=ROOT/'experiments/training_coverage_labels_v1';DELTA=1e-4
def close(a,b):assert abs(a-b)<2e-10,(a,b)

def reconstruct(method,n,seed,stats):
    recorded=stats['candidates'];values={tuple(r['layout']):r['score'] for r in recorded};encounter=[];calls=0
    def evaluate(position):
        nonlocal calls
        calls+=1;layout=tuple(position[:n]);assert layout in values
        if layout not in encounter:encounter.append(layout)
        return values[layout]
    rng=random.Random(seed)
    if method=='random_fixed_56':
        initial=tuple(range(n));population=[x for x in itertools.permutations(range(6),n) if x!=initial];proposals=[initial]+rng.sample(population,55);scores=[evaluate(x) for x in proposals];best=min(range(56),key=lambda i:scores[i]);winning=proposals[best];trace=np.minimum.accumulate(scores).tolist();value=scores[best];assert stats['candidate_layouts']==[list(x) for x in proposals]
    else:
        cfg=src.engine.CFG['swarm'];positions=[list(range(6))]+[rng.sample(range(6),6) for _ in range(cfg['particles']-1)];vel=[[] for _ in positions];pbest=[x.copy() for x in positions];pvalue=[evaluate(x) for x in positions];index=min(range(len(pvalue)),key=lambda i:pvalue[i]);gbest=pbest[index].copy();value=pvalue[index];trace=[value]
        def moves(current,target):
            current=current.copy();out=[]
            for i,want in enumerate(target):
                if current[i]!=want:
                    j=current.index(want);out.append((i,j));current[i],current[j]=current[j],current[i]
            assert current==target;return out
        for _ in range(cfg['iterations']):
            target=gbest.copy()
            for i in range(len(positions)):
                v=[x for x in vel[i] if rng.random()<cfg['inertia']]
                for x in moves(positions[i],pbest[i]):
                    if rng.random()<cfg['cognitive']:v.append(x)
                for x in moves(positions[i],target):
                    if rng.random()<cfg['social']:v.append(x)
                if rng.random()<cfg['mutation_probability']:v.append(tuple(rng.sample(range(6),2)))
                for a,b in v:positions[i][a],positions[i][b]=positions[i][b],positions[i][a]
                vel[i]=v;score=evaluate(positions[i])
                if score<pvalue[i]:pvalue[i]=score;pbest[i]=positions[i].copy()
            i=min(range(len(pvalue)),key=lambda j:pvalue[j])
            if pvalue[i]<value:value=pvalue[i];gbest=pbest[i].copy()
            trace.append(value)
        winning=tuple(gbest[:n])
    assert calls==stats['objective_calls']==56 and encounter==[tuple(r['layout']) for r in recorded]
    assert len(encounter)==stats['unique_placements_compiled'];assert list(winning)==stats['initial_layout_local'];assert trace==stats['best_proxy_trace'] and value==stats['best_proxy']
    return winning

def validate_circuit(cid):
    p=load(OUT/'protocol.json');available=p['environment']['affinity_available'][::2];identity=multiprocessing.current_process()._identity;os.sched_setaffinity(0,{available[(identity[-1]-1)%len(available)]})
    src.configure('validation');regions,scenarios,bank,_=src.design();src.install_region(regions['ibm_fez_r0']);q=qasm2.load(str(ROOT/bank[cid]['path']),custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS);q.name=cid;lp=independent_p1(matrix_tape(q));ideal=src.engine.distribution(q);cache={};scores={};noise={};candidate_checks=0;searches=0
    oldindex=load(SOURCE/'fresh_index.json')
    def compiled(layout):
        if layout not in cache:
            c=src.comp.fixed_compile(q,layout);assert max(abs(src.engine.distribution(c)-ideal))<1e-9;cache[layout]=c
        return cache[layout]
    def expected_score(day,layout,objective,params):
        key=(day,layout,objective)
        if key not in scores:
            native=compiled(layout);terms=[]
            for ins in native.data:
                if ins.operation.name in ['barrier','measure']:continue
                ids=[src.engine.PHYSICAL[native.find_bit(b).index] for b in ins.qubits];rate=params[(ins.operation.name+'_gate_error',ids[0],ids[1] if len(ids)==2 else None)];terms.append(-math.log1p(-rate))
            for ins in native.data:
                if ins.operation.name!='measure':continue
                a=src.engine.PHYSICAL[native.find_bit(ins.qubits[0]).index];i=native.find_bit(ins.clbits[0]).index;p01=params[('prob_meas1_prep0',a,None)];p10=params[('prob_meas0_prep1',a,None)];rate=(p01+p10)/2 if objective=='baseline' else p01*(1-lp[i])+p10*lp[i];terms.append(-math.log1p(-rate))
            scores[key]=math.fsum(terms)
        return scores[key]
    for day in p['dates']:
        params=src.engine.parameters(scenarios['ibm_fez_r0_'+day.replace('-','')])
        for method in p['methods']:
            for seed in p['seeds']:
                key=f'{method}__{cid}__{day}__{seed}';r=load(OUT/'jobs'/(key+'.json'));assert (r['method'],r['circuit_id'],r['date'],r['seed'])==(method,cid,day,seed);assert r['historical_baseline']==oldindex[key];old=load(ROOT/oldindex[key]['path']);assert sha(ROOT/oldindex[key]['path'])==oldindex[key]['sha256'];assert max(abs(np.array(r['ideal_distribution'])-ideal))<1e-12
                assert r['execution_order']==(p['objectives'] if (seed+p['dates'].index(day)+p['methods'].index(method))%2==0 else p['objectives'][::-1])
                for objective,out in r['outputs'].items():
                    s=out['stats'];assert s['objective']==objective and s['capture'];assert s['search_wall_seconds']>0
                    if objective=='directional_flip':assert max(abs(np.array(s['logical_p1_estimate'])-lp))<1e-12
                    else:assert s['logical_p1_estimate'] is None
                    for c in s['candidates']:close(c['score'],expected_score(day,tuple(c['layout']),objective,params));candidate_checks+=1
                    # Use saved floating values only after independent arithmetic checks;
                    # strict tie branches preserve the engine's actual rounding order.
                    winning=reconstruct(method,q.num_qubits,seed,s);native=compiled(winning);sig=src.native_signature(native);assert sig==out['signature'];assert sha(ROOT/out['artifact']['path'])==out['artifact']['sha256']
                    with (ROOT/out['artifact']['path']).open('rb') as f:stored=qpy.load(f)[0]
                    assert src.native_signature(stored)==sig
                    nkey=(day,sig)
                    if nkey not in noise:noise[nkey]=src.engine.distribution(native,params)
                    distribution=noise[nkey];assert max(abs(distribution-np.array(out['distribution'])))<1e-12;close(out['tvd'],np.maximum(distribution-ideal,0).sum())
                    if objective=='baseline':
                        assert sig==old['fresh_signature'];close(out['tvd'],old['fresh_tvd'])
                        for f in ['best_proxy','best_proxy_trace','initial_layout_local','unique_placements_compiled','objective_calls']:assert s[f]==old['stats'][f]
                    searches+=1
                if method=='random_fixed_56':assert [x['layout'] for x in r['outputs']['baseline']['stats']['candidates']]==[x['layout'] for x in r['outputs']['directional_flip']['stats']['candidates']]
    return dict(circuit_id=cid,searches=searches,candidate_scores=candidate_checks,unique_layouts_compiled=len(cache),distinct_scores_recomputed=len(scores),distinct_noisy_winners_replayed=len(noise))

def main():
    p=load(OUT/'protocol.json');hashes=0
    for manifest in [p['files'],load(OUT/'generation_manifest.json'),load(OUT/'artifact_manifest.json'),load(OUT/'reuse_sources.json')]:
        for f,h in manifest.items():assert sha(ROOT/f)==h,f;hashes+=1
    old=load(SOURCE/'validation.json');assert old['status']=='passed' and old['artifact_manifest_sha256']==sha(SOURCE/'artifact_manifest.json');prior=load(ROOT/'experiments/circuit_objective_v1/validation.json');assert prior['status']=='passed' and prior['artifact_manifest_sha256']==sha(ROOT/'experiments/circuit_objective_v1/artifact_manifest.json')
    replays=[]
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(validate_circuit,cid) for cid in sorted(p['bank'])]):
            replays.append(f.result());dump(OUT/'validation_progress.json',dict(status='replaying',completed_circuits=len(replays),total_circuits=18,replays=replays));print('Independent replay',len(replays),'/18 circuits',flush=True)
    rows=load(OUT/'paired_quality.json');assert len(rows)==4608;index={r['key']:r for r in rows};jobs={};groups=collections.defaultdict(list)
    for r in rows:
        j=load(OUT/'jobs'/(r['key']+'.json'));jobs[r['key']]=j;close(r['baseline_tvd'],j['outputs']['baseline']['tvd']);close(r['directional_tvd'],j['outputs']['directional_flip']['tvd']);close(r['delta'],r['directional_tvd']-r['baseline_tvd']);assert r['changed']==(j['outputs']['baseline']['signature']!=j['outputs']['directional_flip']['signature']);assert r['baseline_unique']==j['outputs']['baseline']['stats']['unique_placements_compiled'] and r['directional_unique']==j['outputs']['directional_flip']['stats']['unique_placements_compiled'];groups[(r['method'],r['circuit_id'],r['date'])].append(j)
    vv=load(OUT/'seed_variability.json');assert len(vv)==576
    for r in vv:
        rr=groups[(r['method'],r['circuit_id'],r['date'])];assert len(rr)==r['seeds']==8
        for objective,values in r['objectives'].items():
            x=[j['outputs'][objective]['tvd'] for j in rr];mu=sum(x)/8;sd=math.sqrt(sum((v-mu)**2 for v in x)/7);close(values['mean'],mu);close(values['sd'],sd);close(values['minimum'],min(x));close(values['maximum'],max(x));assert values['unique_winners']==len({j['outputs'][objective]['signature'] for j in rr})
    original={r['uid']:r for r in load(SOURCE/'dataset.json')};primary={r['uid'] for r in load(SOURCE/'fresh_contexts.json')};reuse=load(OUT/'reuse_pairs.json');assert len(reuse)==17280
    for r in reuse:
        d=original[r['uid']];ev=load(ROOT/d['reuse_evaluation_path']);j=jobs[r['key']];close(r['retained_tvd'],sum(abs(x-y) for x,y in zip(ev['distribution'],j['ideal_distribution']))/2);assert r['primary']==(r['uid'] in primary)
        for name,obj in [('baseline','baseline'),('directional','directional_flip')]:
            gain=ev['tvd']-j['outputs'][obj]['tvd'];close(r[name+'_gain'],gain);assert r[name+'_status']==('help' if gain>DELTA else 'harm' if gain<-DELTA else 'little')
    def select(r,s):
        f,v=s['stratum'],s['value'];return r['method']==s['method'] and (f=='all' or f=='period' and r['date'] in (p['dates'][:8] if v=='early8' else p['dates'][8:]) or f not in ['all','period'] and r[f]==v)
    qs=load(OUT/'quality_summary.json');rs=load(OUT/'reuse_summary.json');assert len(qs)==44 and len(rs)==88
    for s in qs:
        rr=[r for r in rows if select(r,s)];vg=[r for r in vv if select(r,s)];assert s['search_pairs']==len(rr) and s['groups']==len(vg)
        for field,column in [('baseline_mean_tvd','baseline_tvd'),('directional_mean_tvd','directional_tvd'),('mean_delta','delta'),('baseline_mean_unique','baseline_unique'),('directional_mean_unique','directional_unique')]:close(s[field],sum(r[column] for r in rr)/len(rr))
        assert s['wins']==sum(r['delta']<-DELTA for r in rr) and s['losses']==sum(r['delta']>DELTA for r in rr) and s['little']==sum(abs(r['delta'])<=DELTA for r in rr) and s['changed']==sum(r['changed'] for r in rr);close(s['best_delta'],min(r['delta'] for r in rr));close(s['worst_delta'],max(r['delta'] for r in rr));close(s['baseline_mean_seed_sd'],sum(r['objectives']['baseline']['sd'] for r in vg)/len(vg));close(s['directional_mean_seed_sd'],sum(r['objectives']['directional_flip']['sd'] for r in vg)/len(vg))
    for s in rs:
        rr=[r for r in reuse if select(r,s) and (s['population']=='expanded' or r['primary'])];assert len(rr)==s['outcomes'] and len({r['uid'] for r in rr})==s['contexts'];t={b:{a:0 for a in ['help','little','harm']} for b in ['help','little','harm']}
        for r in rr:t[r['baseline_status']][r['directional_status']]+=1
        assert t==s['transitions']
        for objective in ['baseline','directional']:close(s[objective+'_mean_gain'],sum(r[objective+'_gain'] for r in rr)/len(rr));assert s[objective+'_harms']==sum(r[objective+'_status']=='harm' for r in rr)
    for s in load(OUT/'concentration.json'):
        rr=[r for r in rows if r['method']==s['method']];benefits={d:sum(max(-r['delta'],0) for r in rr if r['date']==d) for d in p['dates']};assert benefits==s['positive_benefit_by_date'];assert s['largest_benefit_date']==max(benefits,key=benefits.get);close(s['largest_date_share'],max(benefits.values())/sum(benefits.values()))
        for r in s['leave_date_out']:close(r['mean_delta'],np.mean([a['delta'] for a in rr if a['date']!=r['excluded_date']]))
    timings=load(OUT/'timing.json');assert len(timings)==len({t['key'] for t in timings})==48
    for t in timings:
        assert set(t['order'])==set(p['objectives'])
        for name,s in t['measurements'].items():
            original=jobs[t['key']]['outputs'][name];assert s['signature']==original['signature'] and s['unique_compiled']==original['stats']['unique_placements_compiled'] and s['objective_calls']==56 and s['best_score']==original['stats']['best_proxy'] and s['seconds']>0
    receipt=dict(status='passed',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes_checked=hashes,searches_reconstructed=sum(r['searches'] for r in replays),candidate_scores_checked=sum(r['candidate_scores'] for r in replays),unique_native_layouts_recompiled=sum(r['unique_layouts_compiled'] for r in replays),distinct_noisy_winners_replayed=sum(r['distinct_noisy_winners_replayed'] for r in replays),winner_outcomes_checked=9216,historical_baseline_matches=4608,quality_pairs_checked=4608,reuse_pairs_checked=17280,variability_groups_checked=576,quality_summaries_checked=44,reuse_summaries_checked=88,timing_outputs_checked=96,replays=sorted(replays,key=lambda r:r['circuit_id']),artifact_manifest_sha256=sha(OUT/'artifact_manifest.json'),generation_manifest_sha256=sha(OUT/'generation_manifest.json'),validator_sha256=sha(Path(__file__)))
    assert receipt['searches_reconstructed']==9216;dump(OUT/'validation.json',receipt);dump(OUT/'validation_progress.json',dict(status='passed',completed_circuits=18,total_circuits=18));print('Full matched search experiment independently validated.',flush=True)
if __name__=='__main__':main()
