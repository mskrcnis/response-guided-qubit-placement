"""Portable paper-cohort validator, adapted from the preserved original report.

The earlier October 21 pilot-overlap regression is outside this release; no
pilot result is counted as extra evidence. All current-cohort distributions,
proposal counts, histories, artifact signatures and summaries remain checked.
"""
from report_exploratory_oct30 import *
import report_exploratory_oct30 as original
REPORT = ROOT / 'outputs/audit/internal'

def validate(p):
    src.install_region(p)
    manifest=load(OUT/'generation_manifest.json')
    for f,h in manifest.items(): assert sha(ROOT/f)==h,f
    for f,h in p['source_hashes'].items(): assert sha(ROOT/f)==h,f
    assert sha(BASE/'protocol.json')==p['original_protocol_sha256']
    records=[load(f) for f in sorted((OUT/'jobs').glob('*.json'))]
    expected={(c,m,r,rule) for c in p['bank'] for m in p['methods'] for r in p['replicates'] for rule in p['seed_rules']}
    actual=[(r['circuit_id'],r['method'],r['replicate'],r['seed_rule']) for r in records]
    assert len(records)==len(set(actual))==270 and set(actual)==expected
    overlap_checks=0
    scenarios=load(OUT/'scenarios.json');circuits=list(p['bank']);flat=[];search_rows=[];maxerror=0.;candidate_count=0;eval_count=0;unchanged_checks=0
    for r in records:
        assert r['status']=='validated_exploratory_trajectory'
        assert r['plan_sha256']==sha(OUT/'plan.json') and r['script_sha256']==sha(ROOT/'scripts/run_exploratory_oct30.py')
        assert [d['date'] for d in r['days']]==p['dates']
        assert r['stratum']==p['bank'][r['circuit_id']]['stratum']
        ideal=np.array(r['ideal_distribution']);close(ideal.sum(),1.);assert ideal.min()>-1e-10
        for sig,a in r['artifacts'].items():
            assert sha(ROOT/a['path'])==a['sha256']
            with (ROOT/a['path']).open('rb') as f: native=qpy.load(f)[0]
            assert src.native_signature(native)==sig
        states={}
        for d,s in zip(r['days'],scenarios):
            assert d['status']==s['status']
            if s['status']!='ready':
                assert d['blocking_reasons']==s['blocking_reasons'];states={};continue
            assert d['episode_initialization']==(not states)==s['episode_initialization']
            assert d['paired_observation']==s['paired_observation']
            assert d['seed']==50001+1000*circuits.index(r['circuit_id'])+100*r['replicate']+(0 if r['seed_rule']=='fixed' else s['calendar_index'])
            st=d['stats']; assert st['objective_calls']==56
            assert len(st['candidates'])==st['unique_placements_compiled']==len({tuple(c['layout']) for c in st['candidates']})
            if r['method']!='swarm': assert st['unique_placements_compiled']==56
            assert st['best_proxy']==min(c['score'] for c in st['candidates'])
            candidate_count+=len(st['candidates'])
            sig=d['signature'];assert sig in r['artifacts'];close(st['best_proxy'],d['evaluations'][sig]['independent_score'])
            assert set(d['evaluations'])=={sig}|{v[0] for v in states.values()}
            for ns,e in d['evaluations'].items():
                assert ns in r['artifacts']
                for field in ['distribution','reference_distribution']:
                    dist=np.array(e[field]);close(dist.sum(),1.);assert dist.min()>-1e-10
                    close(float(abs(dist-ideal).sum()/2),e['tvd'])
                err=float(abs(np.array(e['distribution'])-e['reference_distribution']).max())
                assert err<2e-10 and e['max_error']<2e-10
                maxerror=max(maxerror,err,e['max_error']);eval_count+=1
            close(d['tvd'],d['evaluations'][sig]['tvd'])
            meta={k:r[k] for k in ['circuit_id','stratum','method','replicate','seed_rule']}
            search_rows.append(dict(**meta,date=d['date'],paired_observation=d['paired_observation'],signature=sig,seed=d['seed'],tvd=d['tvd'],
                objective_calls=st['objective_calls'],unique_placements=st['unique_placements_compiled'],
                cache_builds=st['cache_delta']['misses'],cache_hits=st['cache_delta']['hits'],cache_array_bytes=st['cache_array_bytes'],
                search_seconds=st['wall_seconds'],coefficient_build_seconds=st['cache_delta']['build_seconds'],
                coefficient_kernel_seconds=st['cache_delta']['kernel_seconds'],audit_seconds=st['audit_seconds'],
                protected_incumbent_seconds=sum(x['incumbent_seconds'] for x in d['decisions'] if x['policy']=='retain_unless_better')))
            if not states:
                assert not d['decisions'];states={policy:(sig,d['date']) for policy in POLICIES}
                continue
            assert len(d['decisions'])==3 and {x['policy'] for x in d['decisions']}==set(POLICIES)
            for x in d['decisions']:
                old,old_date=states[x['policy']]
                assert (x['retained_signature'],x['retained_date'])==(old,old_date)
                close(x['incumbent_score'],d['evaluations'][old]['independent_score'])
                close(x['fresh_score'],st['best_proxy']);assert x['fresh_signature']==sig
                accept=x['policy']=='always_adopt' or (x['policy']=='retain_unless_better' and st['best_proxy']<x['incumbent_score']-1e-12)
                assert x['adopt']==accept and x['selected_signature']==(sig if accept else old)
                close(x['reuse_tvd'],d['evaluations'][old]['tvd']);close(x['fresh_tvd'],d['tvd'])
                close(x['gain'],x['reuse_tvd']-x['fresh_tvd']);close(x['selected_tvd'],d['evaluations'][x['selected_signature']]['tvd'])
                assert x['native_changed']==(x['selected_signature']!=old)
                assert x['age_days']==(datetime.date.fromisoformat(d['date'])-datetime.date.fromisoformat(old_date)).days
                flat.append(dict(**meta,date=d['date'],**x))
                if accept:states[x['policy']]=(sig,d['date'])
        assert r['max_independent_error']<2e-10;maxerror=max(maxerror,r['max_independent_error'])
        if r['seed_rule']=='fixed':
            for i in range(1,len(scenarios)):
                before,after=scenarios[i-1:i+1]
                if before['status']!='ready' or after['status']!='ready':continue
                av=[x['value'] for x in before['parameters']];bv=[x['value'] for x in after['parameters']]
                if av!=bv:continue
                a,b=r['days'][i-1:i+1]
                assert a['signature']==b['signature'] and a['stats']['best_proxy']==b['stats']['best_proxy']
                close(a['tvd'],b['tvd'])
                daily=next(x for x in b['decisions'] if x['policy']=='always_adopt')
                protected=next(x for x in b['decisions'] if x['policy']=='retain_unless_better')
                close(daily['gain'],0.);assert not protected['adopt'];unchanged_checks+=1
    assert len(search_rows)==p['searches'] and len(flat)==p['paired_cases']*3
    validation=dict(status='passed_exploratory_validation',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        validator_sha256=sha(__file__),plan_sha256=sha(OUT/'plan.json'),generation_manifest_sha256=sha(OUT/'generation_manifest.json'),
        trajectories=len(records),searches=len(search_rows),scored_proposals=len(search_rows)*56,
        distinct_layout_candidate_records=candidate_count,policy_update_rows=len(flat),independent_native_date_evaluations=eval_count,
        unchanged_calibration_fixed_seed_checks=unchanged_checks,
        prior_pilot_overlap_regression_checks=overlap_checks,prior_pilot_generation_manifest_sha256=None,
        max_independent_error=maxerror,tolerance=2e-10,scope=p['validation'],confirmation_complete=False)
    dump(REPORT/'validation.json',validation); dump(REPORT/'policy_cases.json',flat);dump(REPORT/'search_cases.json',search_rows)
    return flat,search_rows,validation

def compare(a, b, path='root'):
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a)==set(b), path
        for k in a: compare(a[k], b[k], path+'/'+k)
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a)==len(b), path
        for i,(x,y) in enumerate(zip(a,b)): compare(x,y,path+'/'+str(i))
    elif isinstance(a, (int,float)) and not isinstance(a,bool):
        assert abs(a-b)<=max(1e-11,abs(a)*1e-13), (path,a,b)
    else: assert a==b, (path,a,b)


def run():
    from reproduce import included_sources
    REPORT.mkdir(parents=True,exist_ok=True)
    original.REPORT=REPORT
    p=load(OUT/'plan.json')
    included,excluded=included_sources(p['source_hashes'])
    p=dict(p,source_hashes=included)
    cal=original.calibration(p)
    flat,search_rows,validation=validate(p)
    original.summarize(p,flat,search_rows,cal,validation)
    compared=[]
    for name in ['method_summary','by_date','by_circuit','by_seed_rule','by_date_seed_rule',
                 'paired_comparisons','comparison_summary','leave_one_date_out',
                 'calibration_change_sensitivity','policy_screens','seed_variability',
                 'computational_work','accepted_harmful_updates','policy_cases','search_cases']:
        compare(load(REPORT/(name+'.json')),load(ROOT/'reports/exploratory_oct30_v1'/(name+'.json')),name)
        compared.append(name)
    result=dict(status='passed',trajectories=validation['trajectories'],searches=validation['searches'],
                policy_update_rows=validation['policy_update_rows'],reproduced_tables=compared,
                prior_pilot_overlap_regression='Not rerun; earlier pilot is outside the paper-cohort release',
                included_original_source_hashes=len(included),external_or_out_of_scope_original_sources=excluded)
    dump(REPORT/'release_validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='external_or_out_of_scope_original_sources'},indent=2))
