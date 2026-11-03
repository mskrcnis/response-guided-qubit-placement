"""Frozen-input, user-authorized exploratory replay of the available calendar.

Original algorithms, collection rules, raw evidence and prior analyses are preserved.
Each trajectory has its own cold-to-warm owner and its own policy histories.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','QISKIT_NUM_PROCS','RAYON_NUM_THREADS']:
    os.environ[k] = '1'
import datetime, hashlib, json, multiprocessing, shutil, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from qiskit import qasm2, qpy
from duplicate_aware_swarm import Owner
import run_heldout as src
from validate_duplicate_swarm import replay
from validate_directional_full_search import reconstruct
from validate_search_loss_diagnosis import independent
from validate_circuit_objective import independent_p1, matrix_tape
from confirmation_readiness import assess, parse
from run_coverage_gain import ROOT, load, dump, sha

BASE = ROOT/'experiments/swarm_confirmation_v1'
OUT = ROOT/'experiments/exploratory_oct30_v1'
REPORT = ROOT/'reports/exploratory_oct30_v1'
POLICIES = ['always_adopt', 'retain_unless_better', 'never_recompile']

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(x): return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',',':')).encode()).hexdigest()
def key(job): return '__'.join(map(str, job[1:]))
def jobs(p):
    return [(ci,c,m,r,rule) for ci,c in enumerate(p['bank']) for r in p['replicates'] for rule in p['seed_rules'] for m in p['methods']]

def prepare():
    assert not OUT.exists(), 'Preserve prior run; use run to resume.'
    p=load(BASE/'protocol.json'); frozen_at=now()
    for f,h in p['files'].items(): assert sha(ROOT/f)==h,f
    receipt_files=sorted((BASE/'collection/receipts').glob('*.json'))
    receipts=[load(f) for f in receipt_files]
    assert receipts and all(parse(r['received_at_utc'])<=parse(frozen_at) for r in receipts)
    rows=[]; previous_ready=False
    for di,day in enumerate([p['initial_date']]+p['dates']):
        cutoff=parse(day+'T00:00:00+00:00')
        if cutoff>parse(frozen_at): break
        s=assess(p,receipts,cutoff); ready=s['status']=='ready'
        rows.append(dict(date=day,cutoff_utc=cutoff.isoformat(),calendar_index=di,
                         episode_initialization=ready and not previous_ready,
                         paired_observation=ready and previous_ready and day in p['dates'],**s))
        previous_ready=ready
    assert any(s['paired_observation'] for s in rows)
    OUT.mkdir(); REPORT.mkdir(parents=True)
    dump(OUT/'scenarios.json',rows)
    dump(OUT/'receipts_snapshot.json',receipts)
    sources=dict(p['files'])
    # Pin the executed dependency closure, including non-protocol helpers.
    for f in sorted((ROOT/'scripts').glob('*.py')):
        if f.name not in ['report_exploratory_oct30.py','plot_exploratory_oct30.py','finalize_exploratory_oct30.py','refresh_exploratory_oct30_progress.py','refresh_confirmation_progress.py']:
            sources[str(f.relative_to(ROOT))]=sha(f)
    for f in receipt_files: sources[str(f.relative_to(ROOT))]=sha(f)
    for r in receipts:
        assert r['protocol_sha256'] in [sha(BASE/'protocol.json')]+p.get('compatible_receipt_protocols',[])
        for a in [r['regional']]+r['sources']:
            assert sha(ROOT/a['path'])==a['sha256'],a['path']
            sources[a['path']]=a['sha256']
    for f in ['scenarios.json','receipts_snapshot.json']: sources[str((OUT/f).relative_to(ROOT))]=sha(OUT/f)
    plan={k:p[k] for k in ['bank','methods','replicates','seed_rules','physical_qubits','directed_cz_edges','basis_gates','objective','proposal_budget','seed_formula']}
    plan.update(created_utc=frozen_at,status='frozen_exploratory_before_new_outcomes',
        original_protocol_sha256=sha(BASE/'protocol.json'),source_hashes=sources,workers=8,
        receipt_count=len(receipts),last_receipt_utc=max(r['received_at_utc'] for r in receipts),
        dates=[s['date'] for s in rows],ready_dates=[s['date'] for s in rows if s['status']=='ready'],
        paired_dates=[s['date'] for s in rows if s['paired_observation']],
        blocked_dates=[s['date'] for s in rows if s['status']!='ready'],
        trajectories=270,searches=270*sum(s['status']=='ready' for s in rows),
        paired_cases=270*sum(s['paired_observation'] for s in rows),
        policies=POLICIES,
        design='Complete available fixed daily calendar, initialized October18. All methods use identical eligible dates, circuit population, replicate seeds and 56 proposal budgets. Each method/seed/rule has an isolated cold-to-warm owner. Always-adopt, score-protected and never-recompile histories remain distinct. Blocked dates reset every history and cache together; the next ready date is initialization only. No outcome-driven selection or tuning.',
        outcome='Exact simulated output TVD from logical ideal; lower is better. Gain=reuse-current TVD minus fresh-current TVD; help/little/harm thresholds +/-1e-4. Score gate accepts iff fresh score<own retained score-1e-12, determined before outcome labels. Never-recompile means keep each episode initialization circuit. It incurs no later search operationally; counterfactual search is for this evaluation only.',
        aggregation='Equal date, then equal circuit, replicate and seed-rule weights within each of the two predefined strata. Report all dates, circuits, seed rules and leave-one-date-out sensitivity. No IID p-values. Scalar vector changes are not independent hardware trials. Do not combine these outcomes with the earlier Oct21 pilot as extra observations.',
        validation='Reconstruct every seeded proposal sequence; independently recompile each winner; check QPY round trips. Independently evaluate exact noise channels and tangent hybrid scores for every winner and each policy incumbent, and verify saved arithmetic and history. Nonwinner candidate scores use the unchanged previously validated engine; this is not a new exhaustive independent all-candidate audit.',
        timing='Capture candidate evaluations, unique placements, coefficient-cache builds/bytes, and observed search/incumbent times throughout the concurrent generation. These times are workload diagnostics, not an isolated runtime comparison or speedup claim. The original final protocol requires three pinned serial technical repeats of all replicate0 trajectories; that final-confirmation timing remains outstanding and is not replaced by this exploratory run.',
        scope='User-authorized exploratory extension, not independent confirmation, not physical-device validation, not a qualified pre-search ML policy. Earlier Oct21 outcomes already exposed part of this period. The 28-date criterion and original calendar are not satisfied or silently changed. Algorithms and collection continue unchanged.',
        prior_exposure='experiments/exploratory_oct21_v1/protocol_deviation.json',
        qualified_controller=False,confirmation_complete=False)
    dump(OUT/'plan.json',plan)
    deviation=dict(at_utc=frozen_at,user_request='Then proceed',authorized_scope='Analyse all currently available valid dates while collection continues, as recommended in the preceding assistant message.',
        original_protocol_sha256=plan['original_protocol_sha256'],plan_sha256=sha(OUT/'plan.json'),
        exposed_daily_cutoffs=plan['ready_dates'],prior_exposure=plan['prior_exposure'],
        change='Second explicitly authorized exploratory interim analysis supersedes no-interim-outcome rule for this available-data replay only.',
        algorithms_changed=False,filters_changed=False,collection_changed=False,
        consequence='These observed outcomes cannot be counted as an untouched confirmation set; preserve disclosure and freeze any future independent evaluation before its outcomes are inspected.')
    dump(OUT/'protocol_deviation.json',deviation)
    dump(BASE/'protocol_deviations/user_authorized_interim_20261030.json',deviation)
    dump(OUT/'checkpoint.json',dict(status='prepared_before_outcomes',completed_trajectories=0,total_trajectories=270,completed_searches=0,total_searches=plan['searches'],confirmation_complete=False))
    backup=ROOT/'checkpoints/before_exploratory_oct30';backup.mkdir(exist_ok=True)
    for name in ['RESUME.txt','README.txt','NEXT_EXPERIMENT_PLAN.txt','checkpoints/overnight_research_progress.json','scripts/refresh_confirmation_progress.py']:
        if (ROOT/name).exists(): shutil.copy2(ROOT/name,backup/(name.replace('/','__')))
    print(json.dumps({k:plan[k] for k in ['created_utc','receipt_count','ready_dates','paired_dates','blocked_dates','trajectories','searches','paired_cases']},indent=2),flush=True)

def init():
    p=load(OUT/'plan.json'); src.install_region(p)
    cores=sorted(os.sched_getaffinity(0)); ident=multiprocessing.current_process()._identity
    os.sched_setaffinity(0,{cores[((ident[-1] if ident else 1)-1)%min(8,len(cores))]})

def execute(job):
    ci,c,method,rep,rule=job; jid=key(job); dest=OUT/'jobs'/(jid+'.json')
    ph=sha(OUT/'plan.json'); code=sha(__file__)
    if dest.exists():
        r=load(dest); assert r['plan_sha256']==ph and r['script_sha256']==code
        return jid
    p=load(OUT/'plan.json'); scenarios=load(OUT/'scenarios.json')
    q=qasm2.load(str(ROOT/p['bank'][c]['path']),custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS); q.name=c
    ideal=src.engine.distribution(q); lp=independent_p1(matrix_tape(q))
    owner=None; states={}; natives={}; artifacts={}; days=[]; maximum=0.; episode=-1
    for s in scenarios:
        if s['status']!='ready':
            owner=None; states={}
            days.append(dict(date=s['date'],status='blocked',blocking_reasons=s['blocking_reasons']))
            continue
        if owner is None:
            owner=Owner('hybrid',c); episode+=1
        params=src.engine.parameters(s); seed=50001+1000*ci+100*rep+(0 if rule=='fixed' else s['calendar_index'])
        native,stats=owner.search(q,params,seed,method)
        layout=replay(q.num_qubits,seed,stats) if method=='swarm_unique_56' else reconstruct(method,q.num_qubits,seed,stats)
        sig=src.native_signature(native)
        assert src.native_signature(src.comp.fixed_compile(q,list(layout)))==sig
        assert stats['objective_calls']==56
        path=OUT/'native'/jid/(sig+'.qpy'); path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists(): src.engine.save_qpy(path,native)
        with path.open('rb') as f: stored=qpy.load(f)[0]
        assert src.native_signature(stored)==sig
        natives[sig]=stored; artifacts[sig]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path))
        decisions=[]
        for policy,(old_sig,old_date) in states.items():
            started=time.perf_counter(); inc=owner.evaluate(natives[old_sig],params); elapsed=time.perf_counter()-started
            adopt=policy=='always_adopt' or (policy=='retain_unless_better' and stats['best_proxy']<inc-1e-12)
            decisions.append(dict(policy=policy,retained_signature=old_sig,retained_date=old_date,
                incumbent_score=inc,incumbent_seconds=elapsed,adopt=adopt,selected_signature=sig if adopt else old_sig,
                fresh_signature=sig,fresh_score=stats['best_proxy'],age_days=(parse(s['date']+'T00:00:00+00:00')-parse(old_date+'T00:00:00+00:00')).days))
        # Decisions for every policy are fixed before any current noisy outputs.
        evaluations={}
        for ns in dict.fromkeys([sig]+[d['retained_signature'] for d in decisions]):
            actual=src.engine.distribution(natives[ns],params); ref=independent(natives[ns],params,lp)
            error=max(float(abs(ref['distribution']-actual).max()),float(abs(ref['ideal']-ideal).max()))
            scores=[stats['best_proxy']] if ns==sig else []
            scores += [d['incumbent_score'] for d in decisions if d['retained_signature']==ns]
            error=max([error]+[abs(ref['scores']['hybrid']-v) for v in scores])
            assert error<2e-10,(jid,s['date'],error); maximum=max(maximum,error)
            assert abs(actual.sum()-1)<1e-10 and actual.min()>-1e-10
            evaluations[ns]=dict(distribution=actual.tolist(),reference_distribution=ref['distribution'].tolist(),
                independent_score=ref['scores']['hybrid'],tvd=float(abs(actual-ideal).sum()/2),max_error=error)
        fresh=evaluations[sig]['tvd']
        for d in decisions:
            reuse=evaluations[d['retained_signature']]['tvd']
            d.update(reuse_tvd=reuse,fresh_tvd=fresh,gain=reuse-fresh,selected_tvd=fresh if d['adopt'] else reuse,
                     native_changed=d['selected_signature']!=d['retained_signature'])
        if not states:
            states={policy:(sig,s['date']) for policy in POLICIES}
        else:
            for d in decisions:
                if d['adopt']: states[d['policy']]=(sig,s['date'])
        day=dict(date=s['date'],cutoff_utc=s['cutoff_utc'],status='ready',episode=episode,
            episode_initialization=s['episode_initialization'],paired_observation=s['paired_observation'],
            seed=seed,signature=sig,stats=stats,tvd=fresh,decisions=decisions,evaluations=evaluations)
        days.append(day)
        dump(OUT/'progress'/(jid+'.json'),dict(job_id=jid,last_date=s['date'],completed_searches=sum(d['status']=='ready' for d in days),total_searches=len(p['ready_dates']),updated_utc=now(),plan_sha256=ph))
    dump(dest,dict(status='validated_exploratory_trajectory',circuit_id=c,method=method,replicate=rep,seed_rule=rule,
        stratum=p['bank'][c]['stratum'],ideal_distribution=ideal.tolist(),days=days,artifacts=artifacts,
        max_independent_error=maximum,final_owner_cache_totals=None if owner is None else owner.totals,
        plan_sha256=ph,script_sha256=code))
    return jid

def verify_sources():
    p=load(OUT/'plan.json')
    for f,h in p['source_hashes'].items(): assert sha(ROOT/f)==h,f
    assert sha(BASE/'protocol.json')==p['original_protocol_sha256']
    return p

def run():
    p=verify_sources(); work=jobs(p); assert len(work)==270
    if not (OUT/'exposure_started.json').exists():
        dump(OUT/'exposure_started.json',dict(at_utc=now(),scope='Exploratory available-calendar replay; not independent confirmation',plan_sha256=sha(OUT/'plan.json')))
    with ProcessPoolExecutor(max_workers=p['workers'],mp_context=multiprocessing.get_context('spawn'),initializer=init) as pool:
        for n,f in enumerate(as_completed([pool.submit(execute,j) for j in work]),1):
            jid=f.result()
            dump(OUT/'checkpoint.json',dict(status='running_exploratory_trajectories',completed_trajectories=n,total_trajectories=270,last_job=jid,updated_utc=now(),confirmation_complete=False))
            print(n,'/270',jid,flush=True)
    manifest={str(f.relative_to(ROOT)):sha(f) for sub in ['jobs','native'] for f in sorted((OUT/sub).rglob('*')) if f.is_file()}
    dump(OUT/'generation_manifest.json',manifest)
    dump(OUT/'checkpoint.json',dict(status='generated_trajectory_checks_passed_reporting_pending',completed_trajectories=270,total_trajectories=270,completed_searches=p['searches'],total_searches=p['searches'],confirmation_complete=False))

if __name__=='__main__': {'prepare':prepare,'run':run}[sys.argv[1]]()
