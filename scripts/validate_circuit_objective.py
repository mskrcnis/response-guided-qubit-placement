"""Independent density-matrix local approximation and exact-label replay."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','QISKIT_NUM_PROCS']:os.environ[k]='1'
import collections,datetime,math
from pathlib import Path
import numpy as np
from qiskit import qasm2,qpy
from qiskit.quantum_info import Operator
import run_heldout as src
from run_coverage_gain import ROOT,load,dump,sha
OUT=ROOT/'experiments/circuit_objective_v1';EPS=1e-12;DELTA=1e-4
def close(x,y):assert abs(x-y)<2e-10,(x,y)
def matrix_tape(q):
    out=[];measurement={}
    for ins in q.data:
        name=ins.operation.name;ids=[q.find_bit(b).index for b in ins.qubits]
        if name=='barrier':continue
        if name=='measure':measurement[q.find_bit(ins.clbits[0]).index]=ids[0];continue
        out.append((name,ids,Operator(ins.operation).data))
    return q.num_qubits,out,[measurement[i] for i in range(q.num_clbits)]
def independent_p1(t,p=None):
    n,ops,measure=t;rho=[np.diag([1.,0.]).astype(complex) for _ in range(n)];identity=np.eye(2)/2
    for name,ids,U in ops:
        a=ids[0]
        if len(ids)==1:rho[a]=U@rho[a]@U.conj().T
        else:
            b=ids[1];joint=U@np.kron(rho[b],rho[a])@U.conj().T;joint=joint.reshape(2,2,2,2);rho[a]=np.trace(joint,axis1=0,axis2=2);rho[b]=np.trace(joint,axis1=1,axis2=3)
        for i in ids:rho[i]=rho[i]/np.trace(rho[i])
        if p is not None:
            physical=[src.engine.PHYSICAL[i] for i in ids];r=p[(name+'_gate_error',physical[0],physical[1] if len(ids)==2 else None)];lam=2*r if len(ids)==1 else 4*r/3
            for i in ids:rho[i]=(1-lam)*rho[i]+lam*identity
    return np.array([rho[i][1,1].real for i in measure])
def main():
    protocol=load(OUT/'protocol.json');hashes=0
    for manifest in [protocol['files'],load(OUT/'artifact_manifest.json'),load(OUT/'label_manifest.json'),load(ROOT/'experiments/probe_budget_v1/native_manifest.json')]:
        for f,h in manifest.items():assert sha(ROOT/f)==h,f;hashes+=1
    old=load(ROOT/'experiments/probe_budget_v1/input_validation.json');assert old['status']=='passed' and old['native_manifest_sha256']==sha(ROOT/'experiments/probe_budget_v1/native_manifest.json') and old['validator_sha256']==sha(ROOT/'scripts/validate_probe_budget_inputs.py')
    seal=load(OUT/'prediction_seal.json');assert seal['no_exact_labels_generated'] and seal['predictions_sha256']==sha(OUT/'predictions.json') and seal['logical_estimates_sha256']==sha(OUT/'logical_estimates.json') and seal['protocol_sha256']==sha(OUT/'protocol.json')
    src.configure('validation');regions,scenarios,bank,_=src.design();src.install_region(regions['ibm_fez_r0']);native={};matrix={};approx={};logical={};ideal={};logicalest={r['circuit_id']:r['estimated_logical_p1'] for r in load(OUT/'logical_estimates.json')}
    for cid in protocol['bank']:
        q=qasm2.load(str(ROOT/bank[cid]['path']),custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS);logical[cid]=independent_p1(matrix_tape(q));assert max(abs(logical[cid]-np.array(logicalest[cid])))<1e-12;ideal[cid]=src.engine.distribution(q);assert max(abs(ideal[cid]-np.array(load(OUT/'ideal_labels.json')[cid])))<EPS
    for r in load(ROOT/'experiments/probe_budget_v1/compiled_index.json'):
        with (ROOT/r['path']).open('rb') as f:q=qpy.load(f)[0]
        key=(r['circuit_id'],r['probe']);native[key]=q;matrix[key]=matrix_tape(q);approx[key]=independent_p1(matrix[key]);assert max(abs(src.engine.distribution(q)-ideal[r['circuit_id']]))<1e-9
    predictions=load(OUT/'predictions.json');lookup={r['key']:r for r in predictions};assert len(lookup)==len(predictions)==4608;labels={}
    for day in protocol['dates']:
        p=src.engine.parameters(scenarios['ibm_fez_r0_'+day.replace('-','')])
        for l in load(OUT/'labels'/(day+'.json')):
            labels[l['key']]=l;r=lookup[l['key']];key=(r['circuit_id'],r['probe']);q=native[key];t=matrix[key];lp=logical[r['circuit_id']];np1=independent_p1(t,p);assert l['date']==r['date']==day
            gate_terms=[]
            for name,ids,U in t[1]:
                a=src.engine.PHYSICAL[ids[0]];b=src.engine.PHYSICAL[ids[1]] if len(ids)==2 else None;gate_terms.append(-math.log1p(-p[(name+'_gate_error',a,b)]))
            flip=[];bias=[];response=[]
            for i,qidx in enumerate(t[2]):
                a=src.engine.PHYSICAL[qidx];p01=p[('prob_meas1_prep0',a,None)];p10=p[('prob_meas0_prep1',a,None)];flip.append(-math.log1p(-(p01*(1-lp[i])+p10*lp[i])));bias.append(abs(p01*(1-lp[i])-p10*lp[i]));response.append(abs(p01*(1-np1[i])+(1-p10)*np1[i]-approx[key][i]))
            expected={'baseline':src.engine.proxy(q,p),'directional_flip':math.fsum(gate_terms+flip),'directional_bias':math.fsum(gate_terms+bias),'local_response':math.fsum(response)}
            for name,value in expected.items():close(r['scores'][name],value)
            replay=src.engine.distribution(q,p);assert max(abs(replay-np.array(l['distribution'])))<EPS;close(l['tvd'],np.maximum(replay-ideal[r['circuit_id']],0).sum())
        print('Independently checked scores and exact labels:',day,flush=True)
    assert len(labels)==4608;groups=collections.defaultdict(list)
    for r in predictions:groups[(r['date'],r['circuit_id'])].append(r)
    choices=load(OUT/'choices.json');fixed=[r for r in choices if r['method']!='past_only_selector'];assert len(fixed)==1152
    def pick(rr,name):return min(rr,key=lambda r:(round(r['scores'][name],12),r['probe']))
    for c in fixed:
        rr=groups[(c['date'],c['circuit_id'])];chosen=pick(rr,c['method']);base=pick(rr,'baseline');best=min(labels[r['key']]['tvd'] for r in rr);ts=[labels[r['key']]['tvd'] for r in rr if r['scores'][c['method']]<=min(r['scores'][c['method']] for r in rr)+EPS]
        assert c['key']==chosen['key'] and c['probe']==chosen['probe'] and c['baseline_probe']==base['probe'] and c['score_tie_count']==len(ts)
        close(c['tvd'],labels[chosen['key']]['tvd']);close(c['baseline_tvd'],labels[base['key']]['tvd']);close(c['delta'],c['tvd']-c['baseline_tvd']);close(c['best_observed_tvd'],best);close(c['regret'],c['tvd']-best);close(c['tie_best_regret'],min(ts)-best);close(c['tie_worst_regret'],max(ts)-best)
    ranks=load(OUT/'rank_groups.json');assert len(ranks)==2304
    for r in ranks:
        rr=sorted(groups[(r['date'],r['circuit_id'])],key=lambda r:r['probe'])
        if r['deduplicated']:
            unique={}
            for x in rr:unique.setdefault(x['signature'],x)
            rr=list(unique.values())
        assert r['candidates']==len(rr);counts={k:0 for k in ['concordant','discordant','both_tied','score_tied','quality_tied']}
        for i,a in enumerate(rr):
            for b in rr[i+1:]:
                x=a['scores'][r['method']]-b['scores'][r['method']];y=labels[a['key']]['tvd']-labels[b['key']]['tvd']
                k=('both_tied' if abs(y)<=EPS else 'score_tied') if abs(x)<=EPS else 'quality_tied' if abs(y)<=EPS else 'concordant' if np.sign(x)==np.sign(y) else 'discordant';counts[k]+=1
        assert counts==r['pairs']
    selectors=load(OUT/'selectors.json');assert len(selectors)==16
    for s in selectors:
        earlier=[r for r in fixed if r['date']<s['date']];days=sorted({r['date'] for r in earlier});assert days==s['prior_dates'] and all(day<s['date'] for day in days)
        expected={name:sum(r['tvd'] for r in earlier if r['method']==name)/sum(r['method']==name for r in earlier) for name in protocol['methods']} if earlier else {}
        for name,value in expected.items():close(s['prior_means'][name],value)
        name=min(protocol['methods'],key=lambda m:(round(expected[m],12),protocol['methods'].index(m))) if len(days)>=4 else 'baseline';assert s['method']==name and s['fallback']==(len(days)<4)
        actual=[r for r in choices if r['method']=='past_only_selector' and r['date']==s['date']];assert len(actual)==18
        for r in actual:
            original=next(c for c in fixed if c['date']==r['date'] and c['circuit_id']==r['circuit_id'] and c['method']==name);assert r==dict(original,selected_objective=name,method='past_only_selector',mature=not s['fallback'])
    summaries=load(OUT/'summary.json');assert len(summaries)==115
    for s in summaries:
        field,value=s['stratum'],s['value'];dates=protocol['early_dates'] if value=='early8' else protocol['later_training_dates'] if value=='later8' else protocol['dates'][4:]
        cc=[c for c in choices if c['method']==s['method'] and (field=='all' or field=='period' and c['date'] in dates or field not in ['all','period'] and c[field]==value)];assert s['groups']==len(cc)
        for metric,column in [('mean_tvd','tvd'),('mean_baseline_tvd','baseline_tvd'),('mean_delta','delta'),('mean_regret','regret'),('mean_tie_best_regret','tie_best_regret'),('mean_tie_worst_regret','tie_worst_regret')]:close(s[metric],sum(c[column] for c in cc)/len(cc))
        assert s['wins']==sum(c['delta']<-DELTA for c in cc) and s['losses']==sum(c['delta']>DELTA for c in cc) and s['little']==sum(abs(c['delta'])<=DELTA for c in cc);assert s['changed_placements']==sum(c['probe']!=c['baseline_probe'] for c in cc);close(s['worst_delta'],max(c['delta'] for c in cc));close(s['best_delta'],min(c['delta'] for c in cc))
        count=collections.Counter();keys={(c['date'],c['circuit_id']) for c in cc}
        for r in ranks:
            if r['method']==s['method'] and not r['deduplicated'] and (r['date'],r['circuit_id']) in keys:count.update(r['pairs'])
        assert dict(count)==s['pair_counts'];den=count['concordant']+count['discordant']
        if den:close(s['concordance'],count['concordant']/den)
        else:assert s['concordance'] is None
    timing=load(OUT/'timing.json');assert len(timing)==360
    assert len({(r['circuit_id'],r['date'],r['method'],r['mode'],r['rep']) for r in timing})==360
    for t in timing:
        assert t['seconds']>0 and t['rep'] in range(5);rr=sorted(groups[(t['date'],t['circuit_id'])],key=lambda r:r['probe']);expected=[r['scores'][t['method']] for r in rr];assert max(abs(np.array(expected)-t['scores']))<EPS;assert t['probe']==pick(rr,t['method'])['probe']
    dump(OUT/'validation.json',dict(status='passed',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes_checked=hashes,independent_scores=18432,exact_noisy_distributions_replayed=4608,native_ideal_distributions_checked=288,local_density_reference=True,choices_checked=1440,rank_groups_checked=2304,summaries_checked=115,past_only_selections_checked=16,timing_outputs_checked=360,artifact_manifest_sha256=sha(OUT/'artifact_manifest.json'),validator_sha256=sha(Path(__file__))));print('Full circuit-objective experiment independently validated.',flush=True)
if __name__=='__main__':main()
