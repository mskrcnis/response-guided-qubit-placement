"""Independent exact channels and tangent propagation; no Aer or cache builder."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','QISKIT_NUM_PROCS','RAYON_NUM_THREADS']:os.environ[k]='1'
import math,multiprocessing,collections
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from qiskit import qasm2,qpy
from qiskit.quantum_info import DensityMatrix,Statevector
import search_loss_diagnosis as run
from search_loss_diagnosis import ROOT,OUT,load,dump,sha,src
from validate_circuit_objective import independent_p1,matrix_tape
LOGICAL={};NATIVES={};GROUPS={}
def reset(rho,ids,n):
    key=tuple(ids),n
    if key not in GROUPS:
        other=[i for i in range(n) if i not in ids];groups=[]
        for z in range(2**len(ids)):
            groups.append(np.array([sum(((i>>k)&1)<<bit for k,bit in enumerate(other))+sum(((z>>k)&1)<<bit for k,bit in enumerate(ids)) for i in range(2**len(other))]))
        GROUPS[key]=groups
    groups=GROUPS[key];red=sum(rho[np.ix_(g,g)] for g in groups)/len(groups);out=np.zeros_like(rho)
    for g in groups:out[np.ix_(g,g)]=red
    return out

def independent(q,p,lp):
    n=q.num_qubits;dim=2**n;sv=Statevector.from_int(0,dim);rho=np.zeros((dim,dim),complex);rho[0,0]=1;tangent=np.zeros_like(rho);measured={};terms=[]
    for ins in q.data:
        name=ins.operation.name;ids=[q.find_bit(b).index for b in ins.qubits]
        if name=='barrier':continue
        if name=='measure':measured[q.find_bit(ins.clbits[0]).index]=ids[0];continue
        physical=[src.engine.PHYSICAL[i] for i in ids];rate=p[name+'_gate_error',physical[0],physical[1] if len(ids)==2 else None];terms.append(-math.log1p(-rate));lam=(2**len(ids))/(2**len(ids)-1)*rate
        sv=sv.evolve(ins.operation,qargs=ids);ideal=np.outer(sv.data,sv.data.conj());tangent=DensityMatrix(tangent).evolve(ins.operation,qargs=ids).data
        if lam:tangent+=lam*(reset(ideal,ids,n)-ideal)
        rho=DensityMatrix(rho).evolve(ins.operation,qargs=ids).data
        if lam:rho=(1-lam)*rho+lam*reset(rho,ids,n)
    bins=2**q.num_clbits;destination=np.array([sum(((i>>local)&1)<<bit for bit,local in measured.items()) for i in range(dim)]);p0=np.bincount(destination,weights=abs(sv.data)**2,minlength=bins);dg=np.bincount(destination,weights=tangent.diagonal().real,minlength=bins);full=np.bincount(destination,weights=rho.diagonal().real,minlength=bins);assignment=np.ones((1,1));dr=np.zeros(bins);base_terms=terms.copy();direction_terms=terms.copy()
    for bit in range(q.num_clbits):
        physical=src.engine.PHYSICAL[measured[bit]];p01=p['prob_meas1_prep0',physical,None];p10=p['prob_meas0_prep1',physical,None];assignment=np.kron(np.array([[1-p01,p10],[p01,1-p10]]),assignment);base_terms.append(-math.log1p(-(p01+p10)/2));direction_terms.append(-math.log1p(-(p01*(1-lp[bit])+p10*lp[bit])))
        for i in range(bins):
            if (i>>bit)&1:continue
            j=i|(1<<bit);flow=p01*p0[i]-p10*p0[j];dr[i]-=flow;dr[j]+=flow
    full=assignment@full;first=p0+dg+dr;hybrid=assignment@(p0+dg)
    return dict(distribution=full,ideal=p0,scores=dict(baseline=math.fsum(base_terms),directional_flip=math.fsum(direction_terms),first_order=float(np.abs(dg+dr).sum()/2),hybrid=float(np.abs(hybrid-p0).sum()/2)),negative_mass=dict(first_order=float(np.maximum(-first,0).sum()),hybrid=float(np.maximum(-hybrid,0).sum())))

def execute(job):
    c,d=job['circuit_id'],job['date'];dest=OUT/'validation_jobs'/f'{c}__{d}.json';path=OUT/'jobs'/f'{c}__{d}.json'
    if dest.exists():
        receipt=load(dest);assert receipt['generation_sha256']==sha(path) and receipt['validator_sha256']==sha(__file__);return receipt
    if c not in LOGICAL:
        q=qasm2.load(str(ROOT/run.P['bank'][c]['path']),custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS);LOGICAL[c]=independent_p1(matrix_tape(q))
    j=load(path);assert [r['signature'] for r in j['states']]==job['signatures'];params=src.engine.parameters(run.SCENARIOS['ibm_fez_r0_'+d.replace('-','')]);maximum=dict(distribution=0.,score=0.,negative_mass=0.,ideal=0.);si={r['signature']:r for r in j['states']}
    for r in j['states']:
        key=c,r['signature'];a=r['artifact'];assert a==run.MANIFEST['artifacts'][c][r['signature']] and sha(ROOT/a['path'])==a['sha256']
        if key not in NATIVES:
            with (ROOT/a['path']).open('rb') as f:q=qpy.load(f)[0]
            assert src.native_signature(q)==r['signature'];NATIVES[key]=q
        v=independent(NATIVES[key],params,LOGICAL[c]);maximum['distribution']=max(maximum['distribution'],float(np.max(abs(v['distribution']-r['distribution']))));maximum['ideal']=max(maximum['ideal'],float(np.max(abs(v['ideal']-j['ideal']))));tvd=float(np.abs(v['distribution']-np.array(j['ideal'])).sum()/2);assert abs(tvd-r['tvd'])<2e-10
        maximum['score']=max(maximum['score'],max(abs(v['scores'][o]-r['scores'][o]) for o in run.OBJECTIVES));maximum['negative_mass']=max(maximum['negative_mass'],max(abs(v['negative_mass'][o]-r['negative_mass'][o]) for o in ['first_order','hybrid']));assert all(x<2e-10 for x in maximum.values()),(c,d,maximum)
    candidate_checks=winner_checks=own_checks=0
    for pool in run.MANIFEST['pools'].values():
        if (pool['circuit_id'],pool['date'])!=(c,d):continue
        for r in pool['candidates']:assert abs(si[r['signature']]['scores'][pool['objective']]-r['score'])<2e-10;candidate_checks+=1
        winner=si[pool['winner']];assert abs(winner['tvd']-pool['winner_tvd'])<2e-10 and abs(winner['scores'][pool['objective']]-pool['winner_score'])<2e-10;assert pool['winner'] in {r['signature'] for r in pool['candidates']};assert pool['winner_score']<=min(r['score'] for r in pool['candidates'])+1e-12;winner_checks+=1
    for r in run.MANIFEST['own_response_rows']:
        if (r['circuit_id'],r['date'])!=(c,d):continue
        for role in ['fresh','retained']:
            s=si[r[role+'_signature']];score=r['fresh_score'] if role=='fresh' else r['incumbent_score'];tvd=r['fresh_tvd'] if role=='fresh' else r['reuse_tvd'];assert abs(s['scores'][r['objective']]-score)<2e-10 and abs(s['tvd']-tvd)<2e-10
        assert r['adopt']==(r['fresh_score']<r['incumbent_score']-1e-12);own_checks+=1
    receipt=dict(status='passed',circuit_id=c,date=d,states=len(j['states']),candidate_checks=candidate_checks,winner_checks=winner_checks,own_checks=own_checks,maximum_errors=maximum,generation_sha256=sha(path),validator_sha256=sha(__file__));dump(dest,receipt);return receipt

def main():
    p=run.verify();manifest=load(OUT/'input_manifest.json')
    for f,h in load(OUT/'generation_manifest.json').items():assert sha(ROOT/f)==h,f
    for f,h in manifest['source_jobs'].items():assert sha(ROOT/f)==h,f
    results=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),initializer=run.init) as pool:
        for f in as_completed([pool.submit(execute,j) for j in manifest['jobs']]):
            r=f.result();results.append(r);dump(OUT/'validation_progress.json',dict(completed=len(results),total=len(manifest['jobs']),states=sum(x['states'] for x in results)));print('Validated',len(results),'/',len(manifest['jobs']),r['circuit_id'],r['date'],r['states'],flush=True)
    assert sum(r['states'] for r in results)==p['counts']['native_date_states'] and sum(r['winner_checks'] for r in results)==p['counts']['unique_pools'] and sum(r['own_checks'] for r in results)==960
    dump(OUT/'validation.json',dict(status='passed',protocol_sha256=sha(OUT/'protocol.json'),generation_manifest_sha256=sha(OUT/'generation_manifest.json'),validator_sha256=sha(__file__),reference_p1_sha256=sha(ROOT/'scripts/validate_circuit_objective.py'),results=results,states=sum(r['states'] for r in results),candidate_checks=sum(r['candidate_checks'] for r in results),winner_checks=sum(r['winner_checks'] for r in results),own_checks=960,max_errors={k:max(r['maximum_errors'][k] for r in results) for k in ['distribution','ideal','score','negative_mass']}))
if __name__=='__main__':main()
