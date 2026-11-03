"""Independent replay and direct (non-coefficient-cache) response verification."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','QISKIT_NUM_PROCS','RAYON_NUM_THREADS']:os.environ[k]='1'
import math,datetime,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from qiskit import qasm2,qpy
import response_full_search as run
from response_full_search import ROOT,OUT,load,dump,sha,src
from validate_directional_full_search import reconstruct
from validate_circuit_objective import independent_p1,matrix_tape
from distribution_response import response

def close(a,b,tol=2e-10):assert abs(a-b)<tol,(a,b)
def payload(q):
    keys=set()
    for x in q.data:
        if x.operation.name in ['measure','barrier']:continue
        ids=[q.find_bit(b).index for b in x.qubits];keys.add((x.operation.name,*ids))
    return 8*(2**q.num_clbits)*(1+len(keys)+2*q.num_clbits)

def circuit(cid):
    run.init();p=run.P;q=qasm2.load(str(ROOT/p['bank'][cid]['path']),custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS);q.name=cid;ideal=src.engine.distribution(q);lp=independent_p1(matrix_tape(q));compiled={};natives={};direct={};noise={};maxerr=0.;checks=0;searches=0;rows_checked=0;hist_checks=0
    historical_manifest=load(ROOT/'experiments/directional_full_search_v1/generation_manifest.json')
    schedule={(r['replicate'],r['date']):r['seed'] for r in load(run.SCHEDULE) if r['circuit_id']==cid}
    params={day:src.engine.parameters(run.SCENARIOS['ibm_fez_r0_'+day.replace('-','')]) for day in [p['initial_date']]+p['dates']}
    def compile_layout(layout):
        layout=tuple(layout)
        if layout not in compiled:
            n=src.comp.fixed_compile(q,list(layout));assert np.max(abs(src.engine.distribution(n)-ideal))<1e-9;compiled[layout]=n
        return compiled[layout]
    def expected(sig,day,obj):
        key=(sig,day,obj)
        if key not in direct:
            native=natives[sig];pr=params[day]
            if obj in ['first_order','hybrid']:
                out=response(native,pr,src.engine.PHYSICAL)
                for o,v in out['scores'].items():direct[sig,day,o]=v
            else:
                terms=[]
                for x in native.data:
                    name=x.operation.name
                    if name=='barrier':continue
                    ids=[src.engine.PHYSICAL[native.find_bit(b).index] for b in x.qubits]
                    if name=='measure':
                        b=native.find_bit(x.clbits[0]).index;p01=pr['prob_meas1_prep0',ids[0],None];p10=pr['prob_meas0_prep1',ids[0],None];rate=(p01+p10)/2 if obj=='baseline' else p01*(1-lp[b])+p10*lp[b]
                    else:rate=pr[name+'_gate_error',ids[0],ids[1] if len(ids)==2 else None]
                    terms.append(-math.log1p(-rate))
                direct[key]=math.fsum(terms)
        return direct[key]
    def tvd(sig,day):
        key=(sig,day)
        if key not in noise:noise[key]=src.engine.distribution(natives[sig],params[day])
        return float(np.abs(noise[key]-ideal).sum()/2)
    for job in [j for j in run.jobs(p) if j[0]==cid]:
        r=load(OUT/'jobs'/(run.key(job)+'.json'));_,method,obj,rep,rule=job
        assert [r[f] for f in ['circuit_id','method','objective','replicate','seed_rule']]==list(job)
        assert np.max(abs(np.array(r['ideal_distribution'])-ideal))<1e-12
        for sig,a in r['artifacts'].items():
            assert sha(ROOT/a['path'])==a['sha256']
            with (ROOT/a['path']).open('rb') as f:n=qpy.load(f)[0]
            assert src.native_signature(n)==sig
            if sig not in natives:assert np.max(abs(src.engine.distribution(n)-ideal))<1e-9;natives[sig]=n
        seen=set();hits=misses=bytes_=0;states={};ri={(x['date'],x['policy']):x for x in r['rows']};assert len(ri)==len(r['rows'])==16
        for d in r['days']:
            day=d['date'];s=d['stats'];seed=schedule[rep,p['initial_date']] if rule=='fixed' else schedule[rep,day];assert d['seed']==seed
            assert s['placement_cache_hits']==56-s['unique_placements_compiled'];dh=dm=db=0
            for c in s['candidates']:
                sig=c['signature'];rebuilt=compile_layout(c['layout']);assert src.native_signature(rebuilt)==sig
                e=abs(c['score']-expected(sig,day,obj));maxerr=max(maxerr,e);assert e<2e-10;checks+=1
                if obj in ['first_order','hybrid']:
                    if sig in seen:dh+=1
                    else:dm+=1;db+=payload(rebuilt);seen.add(sig)
            hits+=dh;misses+=dm;bytes_+=db
            assert s['cache_delta']['hits']==dh and s['cache_delta']['misses']==dm and s['cache_delta']['array_bytes']==db
            assert s['cache_entries']==len(seen) and s['cache_array_bytes']==bytes_
            winner=reconstruct(method,q.num_qubits,seed,s);assert src.native_signature(compile_layout(winner))==d['signature'];close(d['tvd'],tvd(d['signature'],day));searches+=1
            if obj in ['baseline','directional_flip']:
                oldpath=ROOT/'experiments/directional_full_search_v1/jobs'/(f'{method}__{cid}__{day}__{seed}.json');assert sha(oldpath)==historical_manifest[str(oldpath.relative_to(ROOT))];old=load(oldpath)['outputs'][obj];assert old['signature']==d['signature'];close(old['tvd'],d['tvd']);hist_checks+=1
            if not states:
                for pol in ['always_adopt','retain_unless_better']:states[pol]=(d['signature'],day)
            else:
                for pol in ['always_adopt','retain_unless_better']:
                    row=ri[day,pol];sig,old=states[pol];assert row['retained_signature']==sig and row['retained_date']==old and row['seed']==seed
                    close(row['incumbent_score'],expected(sig,day,obj));close(row['fresh_score'],s['best_proxy']);adopt=pol=='always_adopt' or s['best_proxy']<row['incumbent_score']-1e-12;assert row['adopt']==adopt
                    selected=d['signature'] if adopt else sig;assert row['selected_signature']==selected and row['fresh_signature']==d['signature'];assert row['native_changed']==(selected!=sig)
                    close(row['reuse_tvd'],tvd(sig,day));close(row['fresh_tvd'],d['tvd']);close(row['selected_tvd'],tvd(selected,day));close(row['candidate_gain'],row['reuse_tvd']-row['fresh_tvd']);assert row['age_days']==(datetime.date.fromisoformat(day)-datetime.date.fromisoformat(old)).days
                    delta=row['incumbent_cache_delta'];assert delta['misses']==delta['array_bytes']==0 and delta['hits']==int(obj in ['first_order','hybrid']);hits+=delta['hits'];rows_checked+=1
                    if adopt:states[pol]=(selected,day)
        assert len(r['days'])==9 and r['cache_entries']==len(seen)
        assert r['cache_totals']['hits']==hits and r['cache_totals']['misses']==misses and r['cache_totals']['array_bytes']==bytes_
        for e in r['evaluations']:
            close(e['tvd'],tvd(e['signature'],e['date']));assert np.max(abs(np.array(e['distribution'])-noise[e['signature'],e['date']]))<1e-12
    out=dict(circuit_id=cid,searches=searches,candidate_checks=checks,rows_checked=rows_checked,historical_scalar_checks=hist_checks,unique_layouts=len(compiled),unique_native_signatures=len(natives),direct_score_entries=len(direct),response_states=sum(k[2]=='hybrid' for k in direct),noise_states=len(noise),max_score_error=maxerr)
    dump(OUT/'validation_jobs'/(cid+'.json'),out);return out

def main():
    p=run.verify();assert load(OUT/'checkpoint.json')['completed_jobs']==240
    for f,h in load(OUT/'generation_manifest.json').items():assert sha(ROOT/f)==h,f
    (OUT/'validation_jobs').mkdir(exist_ok=True);out=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        for fut in as_completed([pool.submit(circuit,c) for c in p['bank']]):
            out.append(fut.result());print(out[-1],flush=True);dump(OUT/'validation_progress.json',dict(completed=len(out),total=5,results=out))
    assert sum(r['searches'] for r in out)==2160 and sum(r['rows_checked'] for r in out)==3840
    dump(OUT/'validation.json',dict(status='passed',protocol_sha256=sha(OUT/'protocol.json'),generation_manifest_sha256=sha(OUT/'generation_manifest.json'),validator_sha256=sha(__file__),reference_validators={n:sha(ROOT/'scripts'/n) for n in ['validate_directional_full_search.py','validate_circuit_objective.py']},results=out,searches=2160,own_state_rows=3840,max_score_error=max(r['max_score_error'] for r in out)))
if __name__=='__main__':main()
