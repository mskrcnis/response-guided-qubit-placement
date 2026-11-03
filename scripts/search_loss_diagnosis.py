"""Frozen offline candidate-pool diagnostic; never changes historical searches."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','QISKIT_NUM_PROCS','RAYON_NUM_THREADS']:os.environ[k]='1'
import sys,datetime,multiprocessing,time,collections,itertools
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from qiskit import qasm2,qpy
import run_heldout as src
import circuit_sensitive_scores as scalar
from response_cache_batched import build
from response_cache import evaluate
from run_coverage_gain import ROOT,load,dump,sha
SOURCE=ROOT/'experiments/response_full_search_v1';OUT=ROOT/'experiments/search_loss_diagnosis_v1';REPORT=ROOT/'reports/search_loss_diagnosis_v1'
OBJECTIVES=['baseline','directional_flip','first_order','hybrid'];EPS=1e-4
P=SCENARIOS=MANIFEST=None;CACHE={};LOGICAL={}
def pk(c,d,s,m,o):return '__'.join(map(str,[c,d,s,m,o]))
def prepare():
    assert not OUT.exists();old=load(SOURCE/'protocol.json');assert load(SOURCE/'validation.json')['status']=='passed' and load(ROOT/'reports/response_full_search_v1/delivery_validation.json')['status']=='passed'
    pools={};artifacts={};states=collections.defaultdict(set);own=[];sourcefiles={}
    original_manifest=load(SOURCE/'generation_manifest.json')
    for f in sorted((SOURCE/'jobs').glob('*.json')):
        assert sha(f)==original_manifest[str(f.relative_to(ROOT))];sourcefiles[str(f.relative_to(ROOT))]=sha(f);j=load(f);c=j['circuit_id'];artifacts.setdefault(c,{})
        for sig,a in j['artifacts'].items():
            assert a['sha256']==original_manifest[a['path']];artifacts[c].setdefault(sig,a)
        for day in j['days'][1:]:
            d=day['date'];s=day['seed'];m=j['method'];o=j['objective'];key=pk(c,d,s,m,o);v=dict(pool_id=key,circuit_id=c,date=d,seed=s,method=m,objective=o,candidates=day['stats']['candidates'],winner=day['signature'],winner_tvd=day['tvd'],winner_score=day['stats']['best_proxy'],proposal_calls=56,unique_candidates=day['stats']['unique_placements_compiled'])
            member=dict(replicate=j['replicate'],seed_rule=j['seed_rule'],job_path=str(f.relative_to(ROOT)))
            if key in pools:assert all(pools[key][k]==val for k,val in v.items());pools[key]['memberships'].append(member)
            else:pools[key]=dict(v,memberships=[member])
            states[c,d].update(x['signature'] for x in v['candidates'])
        for row in j['rows']:
            if j['objective'] in ['first_order','hybrid'] and row['policy']=='retain_unless_better':
                own.append(dict(circuit_id=c,method=j['method'],objective=j['objective'],replicate=j['replicate'],seed_rule=j['seed_rule'],job_path=str(f.relative_to(ROOT)),**row));states[c,row['date']].update([row['retained_signature'],row['fresh_signature']])
    for c,aa in artifacts.items():
        for sig,a in aa.items():assert sha(ROOT/a['path'])==a['sha256']
    manifest=dict(pools=pools,artifacts=artifacts,jobs=[dict(circuit_id=c,date=d,signatures=sorted(ss)) for (c,d),ss in sorted(states.items())],own_response_rows=own,source_jobs=sourcefiles)
    files=[ROOT/'scripts'/n for n in ['search_loss_diagnosis.py','validate_search_loss_diagnosis.py','response_cache_batched.py','response_cache.py','distribution_response.py','circuit_sensitive_scores.py','run_step2_pilot.py','run_heldout.py','run_sequential.py','run_coverage_gain.py']]+[SOURCE/n for n in ['protocol.json','generation_manifest.json','validation.json','summary.json','followup_cases.json']]+[ROOT/'reports/response_full_search_v1/delivery_validation.json']
    files += [ROOT/'data/processed/step2_v1/daily_scenarios.json',src.DATA/'config.json']+[ROOT/v['path'] for v in old['bank'].values()]
    p=dict(version='search_loss_diagnosis_v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files={str(f.relative_to(ROOT)):sha(f) for f in files},bank=old['bank'],dates=old['dates'],workers=4,score_epsilon=1e-12,quality_epsilon=EPS,numeric_validation_tolerance=2e-10,
      scope='All saved observation search pools from five-template/three-replicate/fixed-and-resampled pilot, not only losses. Deduplicate identical circuit/date/seed/method/objective searches and native/date simulations; retain memberships and report raw and deduplicated denominators. Initialization search pools excluded; retained natives included at current dates for960 protected response observations. Previously inspected development scenarios only.',
      purpose='Separate candidate availability and surrogate selection errors. No new search, added proposal, optimizer update, classifier training, noise-model change or fresh validation. Offline all-candidate noisy simulation and cross-scoring are diagnostic oracle work, not free features or a56-proposal policy.',
      labels='Unchanged exact Aer density-matrix depolarizing gates and directional independent readout assignment. Cross-score every saved native/date under all four objectives. Batched calibration-independent coefficients reused only to reduce diagnostic computation, not to claim deployment savings. Record response negative mass and predicted score range; no clipping.',
      comparisons='(A) first_order and hybrid versus directional_flip within each compiler:240 raw matched comparisons per arm,960 total. (B) swarm versus random at identical objective/circuit/date/replicate/seed-rule:240 per objective,960 total. Include all outcomes. For loss subsets, preserve81 deduplicated objective-specific response-swarm-vs-directional comparisons from123 raw rows. Harmful own-state accepted updates are a separate denominator.',
      definitions='For selected candidate a in pool A and comparator winner b in pool B, let a*=min true TVD(A), u*=min true TVD(A union B). loss=a-b; selection_regret=a-a*; coverage_regret=a*-u*; comparator_regret=b-u*. Identity loss=selection_regret+coverage_regret-comparator_regret. Loss if loss>1e-4. Within_pool_selection_gap if selection_regret>1e-4; missing_comparator_quality if a*>b+1e-4. Four categories both / pool_only / ranking_only / tolerance_boundary_neither. Both causes can coexist; this is descriptive attribution conditional on saved pools, not intervention proof.',
      additional='Report comparator-winner membership in A and its response-score margin. If missing and comparator score is lower by>1e-12, this is direct evidence of an unvisited candidate preferred by the same objective. Otherwise a better comparator can still be disfavored by the score. Cross-rank each pool under all four scores using strict min and first encountered tie; also report all candidates within1e-12 of min and their true-TVD range. Preserve original selected winner even if rounding/tie ordering differs. Cross-pool winners and pool unions do not constitute new equal-budget algorithms.',
      analysis='Report raw weighted memberships and unique-search comparisons separately. Primary losses and all-case comparisons by objective/compiler, template, date and seed rule. Compare proposal/native overlap and unique compilation counts. All-candidate response prediction MAE, negative mass, rank/counterfactual chosen quality and own-history accepted harm explanation. No IID confidence intervals, p values or unsupported universal swarm superiority/failure.',
      validation='Verify source hashes, recorded scores and winner/retained labels, pool membership and all decompositions. Independently recompute every native/date full-noise distribution and both response scores using Qiskit DensityMatrix unitary propagation plus a separately implemented depolarizing partial trace and Kronecker readout map. No Aer noise-model helper, batched builder or shared reset function in independent validator. Save per-date generation/validation atomically. Validate report/figure against machine-readable summary.',
      environment=dict(cores=sorted(os.sched_getaffinity(0)),python=sys.version),counts=dict(source_jobs=len(sourcefiles),unique_pools=len(pools),raw_searches=sum(len(v['memberships']) for v in pools.values()),native_date_states=sum(len(j['signatures']) for j in manifest['jobs']),date_jobs=len(manifest['jobs']),own_rows=len(own)))
    OUT.mkdir();dump(OUT/'input_manifest.json',manifest);p['input_manifest_sha256']=sha(OUT/'input_manifest.json');dump(OUT/'protocol.json',p);dump(OUT/'checkpoint.json',dict(status='protocol_frozen',completed_jobs=0,total_jobs=len(manifest['jobs']),next_action='Generate all saved candidate labels, independent replay, summary and report.'));print(p['counts'],flush=True)
def verify():
    p=load(OUT/'protocol.json');assert p['input_manifest_sha256']==sha(OUT/'input_manifest.json')
    for f,h in p['files'].items():assert sha(ROOT/f)==h,f
    return p

def init():
    global P,SCENARIOS,MANIFEST
    P=load(OUT/'protocol.json');MANIFEST=load(OUT/'input_manifest.json');cores=P['environment']['cores'][::2];i=multiprocessing.current_process()._identity;os.sched_setaffinity(0,{cores[((i[-1] if i else 1)-1)%len(cores)]});src.configure('validation');regions,SCENARIOS,_,_=src.design();src.install_region(regions['ibm_fez_r0'])
def logical(cid):
    if cid not in LOGICAL:
        q=qasm2.load(str(ROOT/P['bank'][cid]['path']),custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS);LOGICAL[cid]=(q,src.engine.distribution(q),scalar.probabilities(scalar.tape(q,range(q.num_qubits))))
    return LOGICAL[cid]
def native(cid,sig):
    key=cid,sig
    if key not in CACHE:
        a=MANIFEST['artifacts'][cid][sig];assert sha(ROOT/a['path'])==a['sha256']
        with (ROOT/a['path']).open('rb') as f:q=qpy.load(f)[0]
        assert src.native_signature(q)==sig;_,ideal,lp=logical(cid);assert np.max(abs(src.engine.distribution(q)-ideal))<1e-9;CACHE[key]=(q,build(q,src.engine.PHYSICAL),dict(tape=scalar.tape(q,src.engine.PHYSICAL),logical_p1=lp))
    return CACHE[key]
def execute(job):
    c,d=job['circuit_id'],job['date'];dest=OUT/'jobs'/f'{c}__{d}.json'
    if dest.exists():return load(dest)['counts']
    params=src.engine.parameters(SCENARIOS['ibm_fez_r0_'+d.replace('-','')]);_,ideal,_=logical(c);rows=[];started=time.perf_counter();partial=OUT/'partial'/f'{c}__{d}.json';saved=load(partial)['states'] if partial.exists() else [];done={r['signature']:r for r in saved}
    for i,sig in enumerate(job['signatures']):
        if sig in done:rows.append(done[sig]);continue
        q,coeff,tape=native(c,sig);r=evaluate(coeff,params,details=True);scores=dict(baseline=src.engine.proxy(q,params),directional_flip=scalar.score('directional_flip',tape,params),**r['scores']);distribution=src.engine.distribution(q,params)
        rows.append(dict(signature=sig,artifact=MANIFEST['artifacts'][c][sig],distribution=distribution.tolist(),tvd=float(abs(distribution-ideal).sum()/2),scores=scores,negative_mass={o:float(np.maximum(-np.array(r[o+'_distribution']),0).sum()) for o in ['first_order','hybrid']}))
        if (i+1)%25==0:dump(partial,dict(circuit_id=c,date=d,states=rows))
    counts=dict(circuit_id=c,date=d,states=len(rows));dump(dest,dict(circuit_id=c,date=d,ideal=ideal.tolist(),states=rows,counts=counts,generation_seconds=time.perf_counter()-started));return counts

def run():
    p=verify();manifest=load(OUT/'input_manifest.json');pending=[j for j in manifest['jobs'] if not (OUT/'jobs'/f"{j['circuit_id']}__{j['date']}.json").exists()];done=len(manifest['jobs'])-len(pending)
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),initializer=init) as pool:
        for f in as_completed([pool.submit(execute,j) for j in pending]):
            r=f.result();done+=1;dump(OUT/'checkpoint.json',dict(status='generating',completed_jobs=done,total_jobs=len(manifest['jobs']),last_job=r,next_action='Complete native/date labels, then independent validator.'));print(done,'/',len(manifest['jobs']),r,flush=True)
    dump(OUT/'generation_manifest.json',{str(f.relative_to(ROOT)):sha(f) for f in sorted((OUT/'jobs').glob('*.json'))});dump(OUT/'checkpoint.json',dict(status='generation_complete_validation_pending',completed_jobs=done,total_jobs=done,next_action='Run validate_search_loss_diagnosis.py, then analysis/report.'))

def analyse():
    p=verify();assert load(OUT/'validation.json')['status']=='passed';man=load(OUT/'input_manifest.json');pools=man['pools'];states={};poolstats={}
    for f in sorted((OUT/'jobs').glob('*.json')):
        j=load(f)
        for r in j['states']:states[j['circuit_id'],j['date'],r['signature']]=r
    def state(pool,sig):return states[pool['circuit_id'],pool['date'],sig]
    for key,pool in pools.items():
        sigs=[r['signature'] for r in pool['candidates']];rows=[state(pool,s) for s in sigs];winner=state(pool,pool['winner']);best=min(rows,key=lambda r:r['tvd']);rescored={}
        for o in OBJECTIVES:
            chosen=min(rows,key=lambda r:r['scores'][o]);minimum=chosen['scores'][o];ties=[r for r in rows if r['scores'][o]<=minimum+1e-12];rescored[o]=dict(signature=chosen['signature'],score=minimum,tvd=chosen['tvd'],near_ties=len(ties),tie_min_tvd=min(r['tvd'] for r in ties),tie_max_tvd=max(r['tvd'] for r in ties))
        poolstats[key]=dict(pool_id=key,circuit_id=pool['circuit_id'],date=pool['date'],seed=pool['seed'],method=pool['method'],objective=pool['objective'],memberships=pool['memberships'],unique_candidates=len(rows),proposals=56,winner=pool['winner'],winner_tvd=winner['tvd'],best_signature=best['signature'],best_tvd=best['tvd'],selection_regret=winner['tvd']-best['tvd'],rescored=rescored)
    def compare(a,b,kind):
        A,B=pools[a],pools[b];ps=poolstats[a];qa,qb=state(A,A['winner']),state(B,B['winner']);sa={r['signature'] for r in A['candidates']};sb={r['signature'] for r in B['candidates']};u=min(state(A,s)['tvd'] for s in sa|sb);loss=qa['tvd']-qb['tvd'];sel=qa['tvd']-ps['best_tvd'];cov=ps['best_tvd']-u;comp=qb['tvd']-u;assert abs(loss-(sel+cov-comp))<1e-12
        miss=ps['best_tvd']>qb['tvd']+EPS;rank=sel>EPS;category='both' if miss and rank else 'pool_only' if miss else 'ranking_only' if rank else 'tolerance_boundary_neither';margin=qa['scores'][A['objective']]-qb['scores'][A['objective']]
        return dict(comparison_id=a+'___VERSUS___'+b,kind=kind,pool_a=a,pool_b=b,circuit_id=A['circuit_id'],date=A['date'],seed=A['seed'],objective=A['objective'],method=A['method'],a_tvd=qa['tvd'],b_tvd=qb['tvd'],loss=loss,outcome='worse' if loss>EPS else 'better' if loss< -EPS else 'little',selection_regret=sel,coverage_regret=cov,comparator_regret=comp,within_pool_selection_gap=rank,missing_comparator_quality=miss,loss_category=category if loss>EPS else None,comparator_winner_visited=B['winner'] in sa,comparator_score_margin=margin,unvisited_comparator_preferred_by_same_objective=B['winner'] not in sa and margin>1e-12,native_overlap=len(sa&sb),union_size=len(sa|sb),jaccard=len(sa&sb)/len(sa|sb),a_candidates=len(sa),b_candidates=len(sb),rescore_b_with_a=poolstats[b]['rescored'][A['objective']],rescore_a_with_b=ps['rescored'][B['objective']],memberships=A['memberships'])
    comparisons=[]
    for key,A in pools.items():
        c,d,se,m,o=[A[x] for x in ['circuit_id','date','seed','method','objective']]
        if o in ['first_order','hybrid']:comparisons.append(compare(key,pk(c,d,se,m,'directional_flip'),'response_vs_directional'))
        if m=='swarm':comparisons.append(compare(key,pk(c,d,se,'random_fixed_56',o),'swarm_vs_random'))
    summaries=[]
    def stats(rr):
        losses=[r for r in rr if r['outcome']=='worse'];return dict(n=len(rr),better=sum(r['outcome']=='better' for r in rr),little=sum(r['outcome']=='little' for r in rr),worse=len(losses),mean_a_tvd=float(np.mean([r['a_tvd'] for r in rr])),mean_b_tvd=float(np.mean([r['b_tvd'] for r in rr])),mean_loss=float(np.mean([r['loss'] for r in rr])),loss_categories=dict(collections.Counter(r['loss_category'] for r in losses)),loss_unvisited_preferred=sum(r['unvisited_comparator_preferred_by_same_objective'] for r in losses),loss_comparator_visited=sum(r['comparator_winner_visited'] for r in losses),mean_a_candidates=float(np.mean([r['a_candidates'] for r in rr])),mean_b_candidates=float(np.mean([r['b_candidates'] for r in rr])),mean_jaccard=float(np.mean([r['jaccard'] for r in rr])),loss_mean_selection_regret=float(np.mean([r['selection_regret'] for r in losses])) if losses else None,loss_mean_coverage_regret=float(np.mean([r['coverage_regret'] for r in losses])) if losses else None)
    for kind,m,o in sorted({(r['kind'],r['method'],r['objective']) for r in comparisons}):
        base=[r for r in comparisons if (r['kind'],r['method'],r['objective'])==(kind,m,o)]
        for weighting in ['unique_search','raw_memberships']:
            rr=base if weighting=='unique_search' else [dict(r,replicate=mem['replicate'],seed_rule=mem['seed_rule']) for r in base for mem in r['memberships']]
            strata=[('all','all')]+[('circuit_id',c) for c in p['bank']]+[('date',d) for d in p['dates']]+([('seed_rule',r) for r in ['fixed','resampled']] if weighting=='raw_memberships' else [])
            for field,value in strata:
                group=[r for r in rr if field=='all' or r[field]==value]
                if group:summaries.append(dict(kind=kind,method=m,objective=o,weighting=weighting,stratum=field,value=value,**stats(group)))
    own=[]
    for r in man['own_response_rows']:
        a=states[r['circuit_id'],r['date'],r['fresh_signature']];b=states[r['circuit_id'],r['date'],r['retained_signature']];pool=poolstats[pk(r['circuit_id'],r['date'],r['seed'],r['method'],r['objective'])];own.append(dict(r,verified_gain=b['tvd']-a['tvd'],fresh_selection_regret=a['tvd']-pool['best_tvd'],pool_can_avoid_harm=pool['best_tvd']<=b['tvd']+EPS,score_margin=b['scores'][r['objective']]-a['scores'][r['objective']]))
    prediction={o:dict(n=len(states),mae=float(np.mean([abs(r['scores'][o]-r['tvd']) for r in states.values()])),max_negative_mass=max(r['negative_mass'][o] for r in states.values()),negative_mass_over_1e12=sum(r['negative_mass'][o]>1e-12 for r in states.values()),scores_over_one=sum(r['scores'][o]>1+1e-12 for r in states.values())) for o in ['first_order','hybrid']}
    for name,x in [('pool_diagnostics',poolstats),('comparisons',comparisons),('own_state_diagnostics',own)]:dump(OUT/(name+'.json'),x)
    summary=dict(status='analysis_complete',counts=p['counts'],summaries=summaries,prediction=prediction,harmful_accepted=[r for r in own if r['adopt'] and r['verified_gain']< -EPS],validation=load(OUT/'validation.json'),qualified_controller=False,research_complete=False,independent_confirmation=False)
    dump(OUT/'summary.json',summary);dump(OUT/'analysis_manifest.json',{str(f.relative_to(ROOT)):sha(f) for f in [OUT/n for n in ['protocol.json','input_manifest.json','generation_manifest.json','validation.json','pool_diagnostics.json','comparisons.json','own_state_diagnostics.json','summary.json']]});print('Analysis complete',p['counts'],flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
