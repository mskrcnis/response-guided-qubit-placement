"""Frozen five-template response full-search development pilot, atomic trajectory jobs."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','QISKIT_NUM_PROCS','RAYON_NUM_THREADS']:os.environ[k]='1'
import sys,time,datetime,multiprocessing,resource,importlib.metadata
from concurrent.futures import ProcessPoolExecutor,as_completed
from qiskit import qasm2,qpy
import numpy as np
import run_heldout as src
from response_search_engine import Owner
from run_coverage_gain import ROOT,load,dump,sha
OUT=ROOT/'experiments/response_full_search_v1'
REPORT=ROOT/'reports/response_full_search_v1'
SCHEDULE=ROOT/'experiments/own_state_predictor_v1/seed_schedule.json'
CIRCUITS=['ghz_4','product_4','rotcx_line_n4_l6','rotcx_ring_n4_l6','rotcx_star_n5_l6']
OBJECTIVES=['baseline','directional_flip','first_order','hybrid'];METHODS=['swarm','random_fixed_56'];RULES=['fixed','resampled']
P=SCENARIOS=None

def prepare():
    assert not OUT.exists();old=load(ROOT/'experiments/directional_full_search_v1/protocol.json')
    files=[ROOT/'scripts'/n for n in ['response_search_engine.py','response_full_search.py','response_cache.py','response_cache_batched.py','distribution_response.py','circuit_sensitive_scores.py','run_heldout.py','run_sequential.py','run_step2_pilot.py','prepare_fixed_budget.py','run_diversity.py','run_coverage_gain.py']]+[SCHEDULE,ROOT/'data/processed/step2_v1/daily_scenarios.json',src.DATA/'config.json',ROOT/'data/processed/step2_v1/config.json']
    for ex in ['directional_full_search_v1','response_cache_batched_v1']:
        files += [ROOT/'experiments'/ex/n for n in ['protocol.json','validation.json','generation_manifest.json']]
    files += [ROOT/old['bank'][c]['path'] for c in CIRCUITS]
    p=dict(version='response_full_search_v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files={str(f.relative_to(ROOT)):sha(f) for f in files},bank={c:old['bank'][c] for c in CIRCUITS},initial_date='2026-08-15',dates=[f'2026-08-{i:02d}' for i in range(16,24)],methods=METHODS,objectives=OBJECTIVES,replicates=[0,1,2],seed_rules=RULES,workers=4,proposal_budget=56,score_epsilon=1e-12,gain_epsilon=1e-4,total_trajectories=240,total_searches=2160,
      scope='Development pilot: five preselected representatives covering GHZ, product, line, ring, star and the prior timing panel; three first scheduled replicates. Previously inspected original-training dates. Not the full18-template/eight-replicate expansion or independent confirmation.',
      fairness='Four actual objective-specific searches. Unchanged original PSO and random56 implementations, tie behavior and seed schedule. Equal56 proposals, not equal unique compilations or runtime. Own protected and always-adopt histories share the identical objective-specific search stream; their cache/search costs are separately interpretable because both search every date. Incumbent remains an external final candidate.',
      cache='One empty persistent response-coefficient cache per circuit/compiler/objective/replicate/seed-rule trajectory. Key includes logical identity, physical order, native signature including measurement map, n and m. No sharing of coefficients across arms, replicates, seed rules or previous experiments. Every miss builds batched coefficients. Current calibration is applied afresh; no calibration-dependent score caching across dates. Repeated placement proposals use the original within-search cache. Scalar controls remain unchanged and receive no response cache.',
      ordering='Select fresh winner and both adoption choices before noisy labels. Protected accepts iff fresh_score < current incumbent_score -1e-12. Always adopts. Each policy retains its own actual native and compilation date. Seeds follow frozen schedule independent of adoption. Fixed rule uses initialization seed.',
      labels='Exact density-matrix depolarizing gates plus independent directional assignment readout. Check every new native ideal distribution against logical ideal. No shots, idle/crosstalk/leakage or hardware. Noise labels excluded from search and adoption. Dense ideal propagation in response objective is oracle-like and exponential.',
      metrics='Fresh TVD paired by method/circuit/date/replicate/rule; improved/worse/little at1e-4. Own-history accepted help/harm/little, action coverage, dates, selectedTVD versus own always-fresh, age and rejection errors. Per method/objective/rule screens retain zero mean excessTVD, <=10% action harm, >=30 actions, >=10% coverage, >=4 dates. Screens are UPDATE diagnostics, never controller qualification because every day runs search. Family/date strata and leave-one-template/date-out descriptive sensitivity; no IID tests.',
      validation='Historical scalar winners/traces must match exact original library. Independently replay proposal logic using recorded candidate scores; recompile each distinct recorded layout, check signature. Direct uncached response calculation independently checks each native/current-date score used, including incumbents. Independently recompute exact winner and retained distributions; validate own-state transitions and cache hit/miss/payload accounting.',
      timing='Generation four pinned single-thread workers is descriptive only. After generation and validation stop: clean pinned serial complete nine-date trajectories for all five templates, two compilers, four objectives, both seed rules, replicate0 (80 trajectories,720 searches), no capture/artifact writes or noisy labels inside measured search/adoption. Empty caches initially, retained persist. Include parameter conversion, approximate logical p1 where used, compilation, native-key formation, all cold/miss builds, score kernels and protected incumbent scoring. Resident input loading and offline simulation excluded. Rotate objective order by template/method/rule parity; compare complete cold-to-warm totals, first date and subsequent dates; timing executes exactly once with no selected repeats. Audit save happens between trajectories. Persist cache array payload/entries and separate fresh-process tracemalloc probes for four method/objective arms on star n5 l6 resampled rep0, excluding imports and noisy labels. Python-tracked peak is not complete process memory.',
      boundaries='No ML retraining or promotion; no thresholds altered; no search saved. Cache reuse and speed are measured in actual proposals. Full population expansion, lower-cost score alternatives, presearch controller and genuinely fresh confirmation remain open.',
      environment=dict(python=sys.version,packages={n:importlib.metadata.version(n) for n in ['qiskit','qiskit-aer','numpy']},cores=sorted(os.sched_getaffinity(0))))
    OUT.mkdir();dump(OUT/'protocol.json',p);dump(OUT/'checkpoint.json',dict(status='protocol_frozen',completed_jobs=0,total_jobs=240,next_action='Run240 full-search trajectories, validate, time and report.'))
def verify():
    p=load(OUT/'protocol.json')
    for f,h in p['files'].items():assert sha(ROOT/f)==h,f
    return p

def init():
    global P,SCENARIOS
    P=load(OUT/'protocol.json');cores=P['environment']['cores'][::2];i=multiprocessing.current_process()._identity;os.sched_setaffinity(0,{cores[((i[-1] if i else 1)-1)%len(cores)]});src.configure('validation');regions,SCENARIOS,_,_=src.design();src.install_region(regions['ibm_fez_r0'])
def jobs(p):return [(c,m,o,rep,rule) for c in p['bank'] for m in METHODS for rep in p['replicates'] for rule in RULES for o in OBJECTIVES]
def key(job):return '__'.join(map(str,job))
def execute(job):
    cid,method,obj,rep,rule=job;jid=key(job);dest=OUT/'jobs'/(jid+'.json')
    if dest.exists():return jid
    q=qasm2.load(str(ROOT/P['bank'][cid]['path']),custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS);q.name=cid;ideal=src.engine.distribution(q);artifacts={};natives={};evals={};days=[];rows=[];state={};schedule={(r['replicate'],r['date']):r['seed'] for r in load(SCHEDULE) if r['circuit_id']==cid}
    def save(sig,native):
        if sig not in artifacts:
            path=OUT/'native'/jid/(sig+'.qpy');path.parent.mkdir(parents=True,exist_ok=True);h=src.engine.save_qpy(path,native);artifacts[sig]=dict(path=str(path.relative_to(ROOT)),sha256=h);natives[sig]=native
    owner=Owner(obj,cid,save)
    def label(sig,day,p):
        k=day+'__'+sig
        if k not in evals:
            native=natives[sig];err=float(np.max(abs(src.engine.distribution(native)-ideal)));assert err<1e-9
            d=src.engine.distribution(native,p);evals[k]=dict(date=day,signature=sig,distribution=d.tolist(),tvd=float(abs(d-ideal).sum()/2),ideal_error=err)
        return evals[k]['tvd']
    for day in [P['initial_date']]+P['dates']:
        p=src.engine.parameters(SCENARIOS['ibm_fez_r0_'+day.replace('-','')]);seed=schedule[rep,P['initial_date']] if rule=='fixed' else schedule[rep,day]
        native,stats=owner.search(q,p,seed,method);sig=src.native_signature(native);save(sig,native)
        if obj in ['baseline','directional_flip']:
            old=load(ROOT/'experiments/directional_full_search_v1/jobs'/('__'.join(map(str,[method,cid,day,seed]))+'.json'))['outputs'][obj]
            assert sig==old['signature']
            for f in ['best_proxy','best_proxy_trace','initial_layout_local','unique_placements_compiled','objective_calls']:assert stats[f]==old['stats'][f],(jid,day,f)
        pending=[]
        if state:
            for policy in ['always_adopt','retain_unless_better']:
                oldsig,oldday=state[policy];before=owner.totals.copy();t=time.perf_counter();inc=owner.evaluate(natives[oldsig],p);seconds=time.perf_counter()-t;adopt=policy=='always_adopt' or stats['best_proxy']<inc-P['score_epsilon']
                pending.append(dict(policy=policy,retained_signature=oldsig,retained_date=oldday,incumbent_score=inc,incumbent_seconds=seconds,incumbent_cache_delta={k:owner.totals[k]-before[k] for k in before},adopt=adopt,selected_signature=sig if adopt else oldsig))
        else:
            for policy in ['always_adopt','retain_unless_better']:state[policy]=(sig,day)
        # Offline noisy outcomes occur only after both score-based choices are fixed.
        ft=label(sig,day,p)
        for r in pending:
            reuse=label(r['retained_signature'],day,p);gain=reuse-ft;r.update(date=day,seed=seed,fresh_signature=sig,fresh_tvd=ft,reuse_tvd=reuse,selected_tvd=ft if r['adopt'] else reuse,candidate_gain=gain,fresh_score=stats['best_proxy'],native_changed=r['selected_signature']!=r['retained_signature'],age_days=(datetime.date.fromisoformat(day)-datetime.date.fromisoformat(r['retained_date'])).days);rows.append(r)
            if r['adopt']:state[r['policy']]=(sig,day)
        days.append(dict(date=day,seed=seed,signature=sig,stats=stats,tvd=ft))
    dump(dest,dict(job_id=jid,circuit_id=cid,method=method,objective=obj,replicate=rep,seed_rule=rule,ideal_distribution=ideal.tolist(),days=days,rows=rows,artifacts=artifacts,evaluations=list(evals.values()),cache_totals=owner.totals,cache_entries=len(owner.cache),process_hwm_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));return jid

def run():
    p=verify();(OUT/'jobs').mkdir(exist_ok=True);alljobs=jobs(p);pending=[j for j in alljobs if not (OUT/'jobs'/(key(j)+'.json')).exists()];done=len(alljobs)-len(pending)
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),initializer=init) as pool:
        for f in as_completed([pool.submit(execute,j) for j in pending]):
            jid=f.result();done+=1;dump(OUT/'checkpoint.json',dict(status='generating',completed_jobs=done,total_jobs=len(alljobs),last_job=jid,next_action='Complete remaining trajectories, independently validate, benchmark serially and report.'));print(done,'/',len(alljobs),jid,flush=True)
    dump(OUT/'generation_manifest.json',{str(f.relative_to(ROOT)):sha(f) for sub in ['jobs','native'] for f in sorted((OUT/sub).rglob('*')) if f.is_file()});dump(OUT/'checkpoint.json',dict(status='generation_complete_validation_pending',completed_jobs=done,total_jobs=done,next_action='Validate all scores/search paths/labels then clean timing and report.'))
if __name__=='__main__':globals()[sys.argv[1]]()
