"""Independent proposal, compiler, channel and own-state checks; immutable receipts."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','QISKIT_NUM_PROCS','RAYON_NUM_THREADS']:os.environ[k]='1'
import random,itertools,datetime,multiprocessing,collections
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from qiskit import qasm2,qpy
import duplicate_swarm_experiment as run
from duplicate_swarm_experiment import ROOT,OUT,load,dump,sha
from validate_directional_full_search import reconstruct
from validate_search_loss_diagnosis import independent
from validate_circuit_objective import independent_p1,matrix_tape
from validate_response_search import payload
src=run.original.src
def close(a,b):assert abs(a-b)<2e-10,(a,b)

def replay(n,seed,stats):
    values={tuple(x['layout']):x['score'] for x in stats['candidates']};seen=set();encounter=[];audit=[]
    rng=random.Random(seed);other=random.Random(seed ^ 0xD09A56);deck=list(itertools.permutations(range(6),n));other.shuffle(deck);cfg=src.engine.CFG['swarm']
    def score(pos,step,i):
        raw=pos.copy();repeat=tuple(pos[:n]) in seen
        if repeat:
            while deck[-1] in seen:deck.pop()
            prefix=deck.pop();pos[:]=list(prefix)+sorted(set(range(6))-set(prefix))
        prefix=tuple(pos[:n]);assert prefix not in seen and prefix in values;seen.add(prefix);encounter.append(list(prefix));audit.append(dict(iteration=step,particle=i,raw_position=raw,position=pos.copy(),replaced=repeat));return values[prefix],repeat
    def moves(a,b):
        a=list(a);out=[]
        for i,x in enumerate(b):
            if a[i]!=x:j=a.index(x);out.append((i,j));a[i],a[j]=a[j],a[i]
        return out
    positions=[list(range(6))]+[rng.sample(range(6),6) for _ in range(cfg['particles']-1)];vel=[[] for _ in positions];pv=[score(x,0,i)[0] for i,x in enumerate(positions)];best=[x.copy() for x in positions];i=min(range(len(pv)),key=pv.__getitem__);globalpos=best[i].copy();value=pv[i];trace=[value]
    for step in range(1,cfg['iterations']+1):
        target=globalpos.copy()
        for i,pos in enumerate(positions):
            vv=[]
            for pair in vel[i]:
                if rng.random()<cfg['inertia']:vv.append(pair)
            for pair in moves(pos,best[i]):
                if rng.random()<cfg['cognitive']:vv.append(pair)
            for pair in moves(pos,target):
                if rng.random()<cfg['social']:vv.append(pair)
            if rng.random()<cfg['mutation_probability']:vv.append(tuple(rng.sample(range(6),2)))
            for a,b in vv:pos[a],pos[b]=pos[b],pos[a]
            val,replaced=score(pos,step,i);vel[i]=[] if replaced else vv
            if val<pv[i]:pv[i]=val;best[i]=pos.copy()
        i=min(range(len(pv)),key=pv.__getitem__)
        if pv[i]<value:value=pv[i];globalpos=best[i].copy()
        trace.append(value)
    assert len(seen)==56==stats['objective_calls']==stats['unique_placements_compiled'];assert audit==stats['proposal_trace'] and encounter==[x['layout'] for x in stats['candidates']];assert trace==stats['best_proxy_trace'] and value==stats['best_proxy'] and globalpos[:n]==stats['initial_layout_local'];assert stats['replacements']==sum(x['replaced'] for x in audit)
    return tuple(globalpos[:n])

def circuit(cid):
    run.init();p=run.P;dest=OUT/'validation_jobs'/(cid+'.json')
    if dest.exists():
        r=load(dest);assert r['validator_sha256']==sha(__file__) and r['generation_manifest_sha256']==sha(OUT/'generation_manifest.json');return r
    q=qasm2.load(str(ROOT/p['bank'][cid]['path']),custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS);q.name=cid;ideal=src.engine.distribution(q);lp=independent_p1(matrix_tape(q));compiled={};natives={};reference={};reused=set();computed=set();maximum=0.;checks=searches=ownchecks=controls=0
    oldroot=ROOT/'experiments/search_loss_diagnosis_v1';oldmanifest=load(oldroot/'generation_manifest.json');oldvalidation=load(oldroot/'validation.json');assert oldvalidation['status']=='passed' and oldvalidation['generation_manifest_sha256']==sha(oldroot/'generation_manifest.json')
    # Reuse independently validated reference results only with unchanged hashes.
    for day in p['dates']:
        f=oldroot/'jobs'/f'{cid}__{day}.json';assert sha(f)==oldmanifest[str(f.relative_to(ROOT))];j=load(f);assert max(abs(np.array(j['ideal'])-ideal))<2e-10
        for r in j['states']:reference[day,r['signature']]=r
    used=set();params={d:src.engine.parameters(run.original.SCENARIOS['ibm_fez_r0_'+d.replace('-','')]) for d in [p['initial_date']]+p['dates']}
    def expected(sig,day):
        key=day,sig;used.add(key)
        if key not in reference:
            r=independent(natives[sig],params[day],lp);assert max(abs(r['ideal']-ideal))<2e-10;r['tvd']=float(abs(r['distribution']-ideal).sum()/2);reference[key]=r;computed.add(key)
        elif key not in computed:reused.add(key)
        return reference[key]
    def compile_(layout):
        key=tuple(layout)
        if key not in compiled:compiled[key]=src.comp.fixed_compile(q,list(layout))
        return compiled[key]
    for job in [x for x in run.jobs(p) if x[0]==cid]:
        jid=run.original.key(job);r=load(OUT/'jobs'/(jid+'.json'));_,method,obj,rep,rule=job;assert [r[k] for k in ['circuit_id','method','objective','replicate','seed_rule']]==list(job);assert max(abs(np.array(r['ideal_distribution'])-ideal))<2e-10
        old=load(ROOT/'experiments/response_full_search_v1/jobs'/(jid+'.json')) if method!='swarm_unique_56' else None
        for sig,a in r['artifacts'].items():
            assert sha(ROOT/a['path'])==a['sha256']
            if sig not in natives:
                with (ROOT/a['path']).open('rb') as f:n=qpy.load(f)[0]
                assert src.native_signature(n)==sig and max(abs(src.engine.distribution(n)-ideal))<1e-9;natives[sig]=n
        seen=set();hits=misses=bytes_=0;states={};ri={(x['date'],x['policy']):x for x in r['rows']};assert len(ri)==len(r['rows'])==16
        schedule={(x['replicate'],x['date']):x['seed'] for x in load(run.original.SCHEDULE) if x['circuit_id']==cid}
        assert [x['date'] for x in r['days']]==[p['initial_date']]+p['dates']
        for di,d in enumerate(r['days']):
            day=d['date'];s=d['stats'];seed=schedule[rep,p['initial_date'] if rule=='fixed' else day];assert seed==d['seed'];dh=dm=db=0
            for c in s['candidates']:
                sig=c['signature'];rebuilt=compile_(c['layout']);assert src.native_signature(rebuilt)==sig;err=abs(c['score']-expected(sig,day)['scores']['hybrid']);maximum=max(maximum,err);assert err<2e-10;checks+=1
                if sig in seen:dh+=1
                else:dm+=1;db+=payload(rebuilt);seen.add(sig)
            hits+=dh;misses+=dm;bytes_+=db
            assert s['cache_delta']['hits']==dh and s['cache_delta']['misses']==dm and s['cache_delta']['array_bytes']==db and s['cache_entries']==len(seen) and s['cache_array_bytes']==bytes_;assert s['placement_cache_hits']==56-len(s['candidates'])
            win=replay(q.num_qubits,seed,s) if method=='swarm_unique_56' else reconstruct(method,q.num_qubits,seed,s);assert src.native_signature(compile_(win))==d['signature'];close(d['tvd'],expected(d['signature'],day)['tvd']);searches+=1
            if old:
                od=old['days'][di];assert d['signature']==od['signature'] and d['seed']==od['seed'];close(d['tvd'],od['tvd'])
                for f in ['candidates','best_proxy','best_proxy_trace','initial_layout_local','unique_placements_compiled','objective_calls']:assert s[f]==od['stats'][f],(jid,day,f)
                controls+=1
            if not states:
                for policy in ['always_adopt','retain_unless_better']:states[policy]=(d['signature'],day)
            else:
                for policy in ['always_adopt','retain_unless_better']:
                    row=ri[day,policy];sig,prev=states[policy];assert row['retained_signature']==sig and row['retained_date']==prev and row['seed']==seed;close(row['incumbent_score'],expected(sig,day)['scores']['hybrid']);close(row['fresh_score'],s['best_proxy']);adopt=policy=='always_adopt' or row['fresh_score']<row['incumbent_score']-1e-12;assert adopt==row['adopt'];selected=d['signature'] if adopt else sig;assert row['selected_signature']==selected and row['fresh_signature']==d['signature'] and row['native_changed']==(selected!=sig)
                    close(row['reuse_tvd'],expected(sig,day)['tvd']);close(row['fresh_tvd'],d['tvd']);close(row['selected_tvd'],expected(selected,day)['tvd']);close(row['candidate_gain'],row['reuse_tvd']-row['fresh_tvd']);assert row['age_days']==(datetime.date.fromisoformat(day)-datetime.date.fromisoformat(prev)).days
                    delta=row['incumbent_cache_delta'];assert delta['hits']==1 and delta['misses']==delta['array_bytes']==0;hits+=1;ownchecks+=1
                    if old:
                        prior=next(x for x in old['rows'] if x['date']==day and x['policy']==policy)
                        for field in ['retained_signature','retained_date','adopt','selected_signature','fresh_signature','native_changed','age_days']:assert row[field]==prior[field]
                    if adopt:states[policy]=(selected,day)
        for e in r['evaluations']:
            ref=expected(e['signature'],e['date']);assert max(abs(np.array(e['distribution'])-ref['distribution']))<2e-10;close(e['tvd'],ref['tvd'])
        assert r['cache_entries']==len(seen) and r['cache_totals']['hits']==hits and r['cache_totals']['misses']==misses and r['cache_totals']['array_bytes']==bytes_
    out=dict(status='passed',circuit_id=cid,validator_sha256=sha(__file__),generation_manifest_sha256=sha(OUT/'generation_manifest.json'),searches=searches,candidate_checks=checks,own_state_rows=ownchecks,control_searches=controls,unique_layouts=len(compiled),independent_reference_states=len(used),reused_validated_states=len(reused),newly_recomputed_states=len(computed),max_score_error=maximum)
    dump(dest,out);return out
def main():
    p=run.verify()
    for f,h in load(OUT/'generation_manifest.json').items():assert sha(ROOT/f)==h,f
    (OUT/'validation_jobs').mkdir(exist_ok=True);rows=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(circuit,c) for c in p['bank']]):
            r=f.result();rows.append(r);dump(OUT/'validation_progress.json',dict(completed=len(rows),total=5,results=rows));print(r,flush=True)
    assert sum(r['searches'] for r in rows)==810 and sum(r['control_searches'] for r in rows)==540 and sum(r['own_state_rows'] for r in rows)==1440
    dump(OUT/'validation.json',dict(status='passed',protocol_sha256=sha(OUT/'protocol.json'),generation_manifest_sha256=sha(OUT/'generation_manifest.json'),validator_sha256=sha(__file__),results=rows,searches=810,control_searches=540,own_state_rows=1440,max_score_error=max(r['max_score_error'] for r in rows)))
if __name__=='__main__':main()
