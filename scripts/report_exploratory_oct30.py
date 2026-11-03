"""Independent saved-output accounting and descriptive available-period synthesis."""
import collections, datetime, importlib.metadata, json, sys
import numpy as np
from qiskit import qpy
from run_exploratory_oct30 import ROOT, BASE, OUT, REPORT, POLICIES, load, dump, sha, digest, src, parse

STRATA=['known_instance_new_period','new_instance_new_period']
LABELS={'swarm':'Original swarm','swarm_unique_56':'Modified swarm','random_fixed_56':'Random search'}

def counts(values):
    return dict(help=sum(v>1e-4 for v in values),little=sum(abs(v)<=1e-4 for v in values),harm=sum(v< -1e-4 for v in values))
def mean(values): return float(np.mean(values)) if len(values) else None
def close(a,b): assert abs(a-b)<2e-10,(a,b)
def describe(rows):
    a=[x for x in rows if x['adopt']]
    return dict(cases=len(rows),dates=len({x['date'] for x in rows}),circuits=len({x['circuit_id'] for x in rows}),
        mean_reuse_tvd=mean([x['reuse_tvd'] for x in rows]),mean_fresh_tvd=mean([x['fresh_tvd'] for x in rows]),
        mean_selected_tvd=mean([x['selected_tvd'] for x in rows]),mean_gain=mean([x['gain'] for x in rows]),
        recompile_outcomes=counts([x['gain'] for x in rows]),actions=len(a),action_dates=len({x['date'] for x in a}),
        coverage=len(a)/len(rows),accepted_outcomes=counts([x['gain'] for x in a]),
        accepted_harm_fraction=sum(x['gain']< -1e-4 for x in a)/len(a) if a else None,
        mean_selected_minus_fresh=mean([x['selected_tvd']-x['fresh_tvd'] for x in rows]),
        beneficial_rejections=sum(not x['adopt'] and x['gain']>1e-4 for x in rows),
        harmful_rejections=sum(not x['adopt'] and x['gain']< -1e-4 for x in rows),
        worst_accepted_harm=max([max(0.,-x['gain']) for x in a],default=0.),
        accepted_total_harm=sum(-x['gain'] for x in a if x['gain']< -1e-4),
        native_changes=sum(x['native_changed'] for x in rows))

def calibration(p):
    scenarios=load(OUT/'scenarios.json'); seen_values=set(); seen_events=set(); rows=[]; previous=None
    previous_event=None; all_events=set(); changed_dates=[]; unchanged_dates=[]
    for s in scenarios:
        if s['status']!='ready':
            rows.append({k:s[k] for k in ['date','status','blocking_reasons']});previous=None;previous_event=None;continue
        values=[(r['property'],r['qubit_a'],r['qubit_b'],r['value']) for r in s['parameters']]
        events=[(r['property'],r['qubit_a'],r['qubit_b'],r['calibrated_time']) for r in s['parameters']]
        vh=digest(values); eh=digest(events)
        changes=None if previous is None else sum(a!=b for a,b in zip(values,previous))
        event_changes=None if previous_event is None else sum(a!=b for a,b in zip(events,previous_event))
        if s['paired_observation']:(changed_dates if changes else unchanged_dates).append(s['date'])
        ages=[(parse(s['cutoff_utc'])-parse(r['calibrated_time'])).total_seconds()/3600 for r in s['parameters']]
        rows.append(dict(date=s['date'],status='ready',episode_initialization=s['episode_initialization'],paired_observation=s['paired_observation'],
            value_vector_sha256=vh,measurement_vector_sha256=eh,new_value_vector=vh not in seen_values,
            new_measurement_vector=eh not in seen_events,changed_parameter_values=changes,
            changed_measurement_identities=event_changes,max_measurement_age_hours=max(ages),
            parameters=[dict(property=x['property'],qubit_a=x['qubit_a'],qubit_b=x['qubit_b'],value=x['value'],calibrated_time=x['calibrated_time'],observed_time=x['observed_time'],first_received_at_utc=x['first_received_at_utc']) for x in s['parameters']]))
        seen_values.add(vh);seen_events.add(eh);all_events.update(events);previous=values;previous_event=events
    receipts=load(OUT/'receipts_snapshot.json');scheduled=parse('2026-10-14T12:15:00+00:00');slots=[]
    while scheduled+datetime.timedelta(minutes=15)<=parse(p['created_utc']):
        matches=[r['received_at_utc'] for r in receipts if 0<=(parse(r['received_at_utc'])-scheduled).total_seconds()<=900]
        slots.append(dict(scheduled_utc=scheduled.isoformat(),receipts_within_15_minutes=matches))
        scheduled+=datetime.timedelta(hours=6)
    metadata={}
    for r in receipts:
        assert r['hardware_jobs']==0 and not r['credentials_used']
        for name in ['metadata','tree']:
            f=BASE/'collection/revisions'/r['revision']/(name+'.json')
            relative=str(f.relative_to(ROOT))
            if relative not in metadata:metadata[relative]=sha(f)
            assert metadata[relative]==r[name+'_sha256']
    receipt_qa=dict(receipts=len(receipts),distinct_source_revisions=len({r['revision'] for r in receipts}),
        distinct_regional_files=len({r['regional']['sha256'] for r in receipts}),scheduled_slots_due=len(slots),
        slots_without_receipt_within_15_minutes=[x['scheduled_utc'] for x in slots if not x['receipts_within_15_minutes']],
        latest_receipt_utc=receipts[-1]['received_at_utc'],latest_archive_observation_utc=receipts[-1]['source_max_observed'],
        metadata_hashes=metadata,hardware_jobs=0,credentials_used=False,
        note='This audit is frozen at the analysis input snapshot. Missing on-time receipts do not establish their cause. Source archive freshness is distinct from required-patch parameter freshness.')
    dump(REPORT/'receipt_audit.json',receipt_qa)
    result=dict(ready_dates=len(p['ready_dates']),paired_dates=len(p['paired_dates']),
        distinct_value_vectors=len(seen_values),distinct_measurement_vectors=len(seen_events),distinct_scalar_measurement_identities=len(all_events),
        changed_value_transition_dates=changed_dates,unchanged_value_transition_dates=unchanged_dates,
        blocked_dates=p['blocked_dates'],receipt_qa_summary={k:v for k,v in receipt_qa.items() if k!='metadata_hashes'},rows=rows,
        interpretation='Temporal sample sizes are not independent trials. Values and timestamps are per-parameter latest eligible observations, not a guaranteed simultaneous backend snapshot. Unchanged dates remain in the primary analysis; changed-only summaries are sensitivity analyses.')
    dump(REPORT/'calibration_audit.json',result)
    return result

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
    oldroot=ROOT/'experiments/exploratory_oct21_v1'
    oldmanifest=load(oldroot/'generation_manifest.json')
    oldvalidation=load(ROOT/'reports/exploratory_oct21_v1/validation.json')
    assert oldvalidation['generation_manifest_sha256']==sha(oldroot/'generation_manifest.json')
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
        # Regression only: the earlier pilot is never counted as extra evidence.
        oldfile=oldroot/'jobs'/('__'.join(map(str,[r['circuit_id'],r['method'],r['replicate'],r['seed_rule']]))+'.json')
        assert sha(oldfile)==oldmanifest[str(oldfile.relative_to(ROOT))]
        oldday=load(oldfile)['days'][0]
        newday=next(d for d in r['days'] if d['date']=='2026-10-20')
        assert oldday['cutoff_utc']==newday['cutoff_utc']
        assert oldday['seed']==newday['seed'] and oldday['signature']==newday['signature']
        close(oldday['tvd'],newday['tvd'])
        for field in ['candidates','best_proxy','best_proxy_trace','initial_layout_local','unique_placements_compiled','objective_calls']:
            assert oldday['stats'][field]==newday['stats'][field],field
        overlap_checks+=1
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
        prior_pilot_overlap_regression_checks=overlap_checks,prior_pilot_generation_manifest_sha256=sha(oldroot/'generation_manifest.json'),
        max_independent_error=maxerror,tolerance=2e-10,scope=p['validation'],confirmation_complete=False)
    dump(REPORT/'validation.json',validation); dump(REPORT/'policy_cases.json',flat);dump(REPORT/'search_cases.json',search_rows)
    return flat,search_rows,validation

def groups(rows,keys):
    grouped=collections.defaultdict(list)
    for r in rows:grouped[tuple(r[k] for k in keys)].append(r)
    return [dict(zip(keys,k),**describe(v)) for k,v in sorted(grouped.items())]

def summarize(p,flat,search_rows,cal,validation):
    summary=groups(flat,['stratum','method','policy'])
    per_date=groups(flat,['date','stratum','method','policy'])
    per_circuit=groups(flat,['circuit_id','stratum','method','policy'])
    per_rule=groups(flat,['stratum','seed_rule','method','policy'])
    per_date_rule=groups(flat,['date','stratum','seed_rule','method','policy'])
    for name,rows in [('method_summary',summary),('by_date',per_date),('by_circuit',per_circuit),('by_seed_rule',per_rule),('by_date_seed_rule',per_date_rule)]:dump(REPORT/(name+'.json'),rows)
    lookup={(x['date'],x['circuit_id'],x['replicate'],x['seed_rule'],x['method'],x['policy']):x for x in flat}
    pairs=[]
    for x in flat:
        if x['method']!='swarm_unique_56':continue
        for control in ['swarm','random_fixed_56']:
            y=lookup[x['date'],x['circuit_id'],x['replicate'],x['seed_rule'],control,x['policy']]
            pairs.append(dict(date=x['date'],circuit_id=x['circuit_id'],stratum=x['stratum'],replicate=x['replicate'],seed_rule=x['seed_rule'],
                policy=x['policy'],control=control,gain=y['selected_tvd']-x['selected_tvd']))
    comparisons=[]; leave_out=[]
    for stratum in STRATA:
        for policy in POLICIES:
            for control in ['swarm','random_fixed_56']:
                rr=[x for x in pairs if (x['stratum'],x['policy'],x['control'])==(stratum,policy,control)]
                date_means=[dict(date=d,gain=mean([x['gain'] for x in rr if x['date']==d])) for d in p['paired_dates']]
                circuit_means=[dict(circuit_id=c,gain=mean([x['gain'] for x in rr if x['circuit_id']==c])) for c in p['bank'] if p['bank'][c]['stratum']==stratum]
                loo=[dict(omitted_date=d,mean_gain=mean([x['gain'] for x in rr if x['date']!=d])) for d in p['paired_dates']]
                comparisons.append(dict(stratum=stratum,policy=policy,control=control,paired_cases=len(rr),mean_gain=mean([x['gain'] for x in rr]),
                    case_outcomes=counts([x['gain'] for x in rr]),date_mean_outcomes=counts([x['gain'] for x in date_means]),
                    circuit_mean_outcomes=counts([x['gain'] for x in circuit_means]),date_means=date_means,circuit_means=circuit_means,
                    leave_one_date_out_min=min(x['mean_gain'] for x in loo),leave_one_date_out_max=max(x['mean_gain'] for x in loo)))
                leave_out.extend([dict(stratum=stratum,policy=policy,control=control,**x) for x in loo])
    dump(REPORT/'paired_comparisons.json',pairs);dump(REPORT/'comparison_summary.json',comparisons);dump(REPORT/'leave_one_date_out.json',leave_out)
    sensitivity=[]
    for label,dates in [('changed_calibration',cal['changed_value_transition_dates']),('unchanged_calibration',cal['unchanged_value_transition_dates'])]:
        rr=[x for x in flat if x['date'] in dates]
        sensitivity.extend([dict(calibration_group=label,**x) for x in groups(rr,['stratum','method','seed_rule','policy'])])
    dump(REPORT/'calibration_change_sensitivity.json',sensitivity)
    screens=[]
    for r in per_rule:
        if r['policy']!='retain_unless_better':continue
        tests=dict(nonpositive_mean_selected_minus_fresh=r['mean_selected_minus_fresh']<=0,
            harm_fraction_at_most_10_percent=r['accepted_harm_fraction'] is not None and r['accepted_harm_fraction']<=.10,
            at_least_30_actions=r['actions']>=30,coverage_at_least_10_percent=r['coverage']>=.1,at_least_4_action_dates=r['action_dates']>=4)
        screens.append(dict(**r,criteria=tests,passes_descriptive_screen=all(tests.values()),qualified_controller=False))
    dump(REPORT/'policy_screens.json',screens)
    variability=[]
    for stratum in STRATA:
        for method in p['methods']:
            for rule in p['seed_rules']:
                subset=[x for x in flat if (x['stratum'],x['method'],x['seed_rule'],x['policy'])==(stratum,method,rule,'always_adopt')]
                cvs=[]
                for c in p['bank']:
                    for date in p['paired_dates']:
                        v=[x['fresh_tvd'] for x in subset if x['circuit_id']==c and x['date']==date]
                        if not v:continue
                        assert len(v)==3
                        cvs.append(dict(circuit_id=c,date=date,min_tvd=min(v),max_tvd=max(v),sample_sd=float(np.std(v,ddof=1))))
                variability.append(dict(stratum=stratum,method=method,seed_rule=rule,cell_count=len(cvs),mean_within_cell_seed_sd=mean([x['sample_sd'] for x in cvs]),
                    maximum_within_cell_range=max(x['max_tvd']-x['min_tvd'] for x in cvs),cells=cvs))
    dump(REPORT/'seed_variability.json',variability)
    overhead=[]
    for stratum in STRATA:
        for method in p['methods']:
            for rule in p['seed_rules']:
                rr=[x for x in search_rows if (x['stratum'],x['method'],x['seed_rule'])==(stratum,method,rule)]
                overhead.append(dict(stratum=stratum,method=method,seed_rule=rule,searches=len(rr),
                    scored_proposals=sum(x['objective_calls'] for x in rr),unique_layout_evaluations=sum(x['unique_placements'] for x in rr),
                    coefficient_builds=sum(x['cache_builds'] for x in rr),mean_unique_layouts=mean([x['unique_placements'] for x in rr]),
                    max_cache_array_bytes=max(x['cache_array_bytes'] for x in rr),observed_total_search_seconds=sum(x['search_seconds'] for x in rr),
                    observed_median_search_seconds=float(np.median([x['search_seconds'] for x in rr])),
                    observed_protected_incumbent_seconds=sum(x['protected_incumbent_seconds'] for x in rr),
                    limitation='Concurrent eight-worker measurements. No isolated runtime ratio, walltime superiority, or scaling claim. Cache-array bytes exclude Python and simulator allocations.'))
    dump(REPORT/'computational_work.json',overhead)
    failures=sorted([x for x in flat if x['policy']=='retain_unless_better' and x['adopt'] and x['gain']< -1e-4],key=lambda x:x['gain'])
    dump(REPORT/'accepted_harmful_updates.json',failures)
    primary=[]
    for st in STRATA:
        rr=[x for x in comparisons if x['stratum']==st and x['policy']=='always_adopt']
        primary.append(dict(stratum=st,mean_gain_positive_vs_both_controls=all(x['mean_gain']>1e-10 for x in rr),
                            controls={x['control']:x['mean_gain'] for x in rr},independent_confirmation=False))
    result=dict(status='validated_exploratory_available_period',receipt_count=p['receipt_count'],ready_dates=p['ready_dates'],paired_dates=p['paired_dates'],
        blocked_dates=p['blocked_dates'],trajectories=p['trajectories'],searches=p['searches'],paired_cases=p['paired_cases'],
        distinct_calibration_value_vectors=cal['distinct_value_vectors'],changed_transitions=len(cal['changed_value_transition_dates']),
        unchanged_transitions=len(cal['unchanged_value_transition_dates']),groups=summary,primary_descriptive_direction=primary,
        comparison_summary=comparisons,policy_screen_passes=sum(x['passes_descriptive_screen'] for x in screens),policy_screens=len(screens),
        max_independent_error=validation['max_independent_error'],confirmation_complete=False,qualified_controller=False,
        runtime_status='Computational work and concurrent diagnostic times reported; final serial timing remains outstanding.')
    dump(REPORT/'summary.json',result)
    return result,per_circuit,per_rule,variability,failures,overhead

def write_report(p,result,cal,per_circuit,per_rule,variability,failures,overhead):
    s=result['groups']; ix={(x['stratum'],x['method'],x['policy']):x for x in s}
    lines=['AVAILABLE-CALIBRATION EXPLORATORY STUDY -- INPUTS FROZEN 30 OCTOBER 2026','',
        'Status: all available-period trajectories generated and independently checked. This is exploratory simulation evidence, not completed independent confirmation.',
        f"Input snapshot: {p['receipt_count']} receipts; {len(p['ready_dates'])} usable cutoffs ({p['ready_dates'][0]} through {p['ready_dates'][-1]}).",
        f"Initialization is excluded from paired results: {len(p['paired_dates'])} daily comparisons; {p['trajectories']} trajectories; {p['searches']} searches; {p['paired_cases']} method/circuit/seed/rule/date paired cases.",
        'Each search has 56 scored proposals. Original swarm can revisit placements; modified swarm and random search each evaluate 56 distinct placements per search.',
        'Population: 5 known development anchors and 10 new exact instances of related families; all 4 or 5 logical qubits on the fixed six-qubit physical patch.',
        'Three search seeds and both fixed/resampled seed rules. Each method has separate always-adopt, score-protected and never-recompile histories.',
        f"Calibration: {cal['distinct_value_vectors']} distinct numeric vectors; {len(cal['changed_value_transition_dates'])} changing-value transitions; {len(cal['unchanged_value_transition_dates'])} unchanged transitions ({', '.join(cal['unchanged_value_transition_dates'])}).",
        'These dates and seeds are correlated repeated evaluations. No claim of statistical independence, p-values, or formal superiority is made.',
        'Excluded cutoff 2026-10-30: 12 X/SX gate-error parameters exceeded the existing 72-hour freshness limit. No filter was relaxed and no invalid value was clipped.',
        f"Collection coverage in the frozen snapshot: {cal['receipt_qa_summary']['scheduled_slots_due']} slots due; {len(cal['receipt_qa_summary']['slots_without_receipt_within_15_minutes'])} lack a receipt within 15 minutes. See receipt_audit.json for exact slots; their cause is not inferred.",
        '', 'MEAN SIMULATED OUTPUT TVD (lower is better)',
        'Equal date/circuit/replicate/seed-rule weighting within each stratum. Never-recompile retains the episode initialization output circuit.',
        'Stratum | Method | Daily recompile | Score-protected | Never recompile | Paired cases per policy']
    for st in STRATA:
        for m in p['methods']:
            rr=[ix[st,m,pol] for pol in POLICIES]
            lines.append(f"{st} | {LABELS[m]} | {rr[0]['mean_selected_tvd']:.8f} | {rr[1]['mean_selected_tvd']:.8f} | {rr[2]['mean_selected_tvd']:.8f} | {rr[0]['cases']}")
    lines+=['','RECOMPILATION BENEFIT VERSUS EACH METHOD PREVIOUS DAILY WINNER','Threshold: gain>0.0001 helpful; gain<-0.0001 harmful; otherwise little difference.',
        'Stratum | Method | Helpful/little/harmful | Mean reuse-minus-fresh TVD']
    for st in STRATA:
        for m in p['methods']:
            r=ix[st,m,'always_adopt'];c=r['recompile_outcomes']
            lines.append(f"{st} | {LABELS[m]} | {c['help']}/{c['little']}/{c['harm']} | {r['mean_gain']:+.8f}")
    lines+=['','MODIFIED SWARM VERSUS CONTROLS','Positive differences mean lower TVD for modified swarm. Differences are absolute TVD units, not percentage improvements.']
    for r in result['comparison_summary']:
        if r['policy']=='never_recompile':continue
        lines.append(f"{r['stratum']} | {r['policy']} | vs {LABELS[r['control']]}: mean gain {r['mean_gain']:+.8f}; date means help/little/harm {r['date_mean_outcomes']}; circuit means {r['circuit_mean_outcomes']}; leave-one-date-out range [{r['leave_one_date_out_min']:+.8f}, {r['leave_one_date_out_max']:+.8f}].")
    lines+=['','SCORE-PROTECTED ADOPTION','This is a post-search decision. It can reject a harmful compiled result, but it does not avoid the search cost.','Stratum | Method | Accepted actions | Harmful accepted | Rejected harmful | Rejected helpful | Mean protected-minus-fresh TVD']
    for st in STRATA:
        for m in p['methods']:
            r=ix[st,m,'retain_unless_better']
            risk='no accepted actions' if r['accepted_harm_fraction'] is None else f"{100*r['accepted_harm_fraction']:.2f}%"
            lines.append(f"{st} | {LABELS[m]} | {r['actions']}/{r['cases']} | {r['accepted_outcomes']['harm']} ({risk}) | {r['harmful_rejections']} | {r['beneficial_rejections']} | {r['mean_selected_minus_fresh']:+.8f}")
    lines+=['','SEARCH VARIABILITY','Stratum | Method | Seed rule | Mean within-date/circuit sample SD across 3 seeds | Largest seed range']
    for v in variability:lines.append(f"{v['stratum']} | {LABELS[v['method']]} | {v['seed_rule']} | {v['mean_within_cell_seed_sd']:.8f} | {v['maximum_within_cell_range']:.8f}")
    lines+=['','COMPUTATIONAL WORK AND COST LIMIT','Searches run concurrently with 8 single-thread workers. Observed elapsed times are diagnostic only. Final 3-repeat serial full-trajectory timing remains outstanding.',
        'Stratum | Method | Seed rule | Mean distinct placements/search | Coefficient builds | Observed median search seconds | Maximum cache array MiB']
    for r in overhead:lines.append(f"{r['stratum']} | {LABELS[r['method']]} | {r['seed_rule']} | {r['mean_unique_layouts']:.2f} | {r['coefficient_builds']} | {r['observed_median_search_seconds']:.4f} | {r['max_cache_array_bytes']/2**20:.2f}")
    lines+=['','LIMITATIONS AND OUTSTANDING EVIDENCE',
        '- Independent gate depolarization and asymmetric measurement assignment are simulated; no physical-device runs, idle noise, leakage, crosstalk or calibration-estimation uncertainty are included.',
        '- Values at a cutoff are the latest eligible per-parameter archive observations; they need not represent a simultaneous device calibration. Historical topology/native basis remain simulator assumptions.',
        '- Exact density-matrix probabilities eliminate shot sampling noise but do not validate the physical noise model. Search seeds are technical stochastic repetitions, not hardware trials.',
        '- Small 4/5-qubit circuits and one fixed six-qubit patch do not establish scaling. The hybrid objective uses ideal response calculations with exponential state representation.',
        '- Independent numerical checks cover every winning and retained circuit; nonwinner scores reuse the frozen previously validated engine. This is not an exhaustive new independent all-candidate score audit.',
        '- The user authorized both the 21 October and 30 October interim analyses. The period is exposed and is not untouched confirmation. Earlier pilot observations are not pooled as extra evidence.',
        '- No ML training, threshold tuning, outcome-dependent date filtering, or compiler modification occurred. No pre-search ML controller is qualified.',
        '- Minimum 28 valid paired dates and final calendar closure remain unmet. The original protocol is preserved, with the interim deviations documented separately.',
        '- Collection continues through 22 November. More dates can assess temporal consistency, but cannot erase interim exposure; an independent confirmation claim requires a suitable frozen unexamined evaluation.',
        '- Computational work is reported; clean serial timing, larger circuits and physical-device validation remain outstanding.',
        '',f"Validation: max independent numerical discrepancy {result['max_independent_error']:.3g}, tolerance 2e-10. All planned trajectories and paired rows are accounted for.",
        'Supporting JSON files contain complete policy cases, calibration provenance, per-date/circuit/seed-rule results, sensitivity analyses, harmful updates, workload and validation.']
    (REPORT/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    brief=['EXPLORATORY AVAILABLE-PERIOD FINDINGS','',
        f"Completed {p['searches']} searches across {len(p['paired_dates'])} valid paired dates. {cal['distinct_value_vectors']} distinct calibration-value states; {len(cal['changed_value_transition_dates'])} changing transitions.",
        'Noisy simulation only. Independent confirmation and a pre-search ML decision rule remain unqualified.','',
        'Fresh daily recompilation, mean TVD (lower better):','Group | Original swarm | Modified swarm | Random search']
    for st in STRATA:
        brief.append(st+' | '+' | '.join(f"{ix[st,m,'always_adopt']['mean_selected_tvd']:.8f}" for m in p['methods']))
    brief+=['','Modified swarm versus original and random:']
    for r in result['comparison_summary']:
        if r['policy']=='never_recompile':continue
        brief.append(f"{r['stratum']}, {r['policy']}, vs {LABELS[r['control']]}: {r['mean_gain']:+.8f} absolute TVD gain; date outcomes {r['date_mean_outcomes']}.")
    variant=[x for x in s if x['method']=='swarm_unique_56' and x['policy']=='always_adopt']
    totals={k:sum(x['recompile_outcomes'][k] for x in variant) for k in ['help','little','harm']}
    protected=[x for x in s if x['method']=='swarm_unique_56' and x['policy']=='retain_unless_better']
    brief+=['',f"Modified recompilation vs its previous daily winner: {totals} across {sum(x['cases'] for x in variant)} correlated cases.",
        f"Modified score-protected history: {sum(x['actions'] for x in protected)} accepted actions, {sum(x['accepted_outcomes']['harm'] for x in protected)} harmful; {sum(x['harmful_rejections'] for x in protected)} harmful and {sum(x['beneficial_rejections'] for x in protected)} beneficial candidates rejected.",
        'The protected counts use its own retained history; do not subtract them from always-adopt harm counts because the incumbent states differ.',
        '', 'Weak circuits versus either control (fresh mean across all dates/seeds/rules):']
    ci={(x['circuit_id'],x['method'],x['policy']):x for x in per_circuit}
    for c in p['bank']:
        v=ci[c,'swarm_unique_56','always_adopt']['mean_selected_tvd']
        gains={m:ci[c,m,'always_adopt']['mean_selected_tvd']-v for m in ['swarm','random_fixed_56']}
        if min(gains.values())< -1e-4:brief.append(c+': '+', '.join(f"vs {LABELS[m]} {v:+.8f}" for m,v in gains.items()))
    brief+=['','Largest accepted harmful modified-swarm updates:']
    for x in [x for x in failures if x['method']=='swarm_unique_56'][:10]:
        brief.append(f"{x['date']} | {x['circuit_id']} | {x['seed_rule']} seed-replicate{x['replicate']} | TVD worsening {-x['gain']:.8f}")
    brief+=['','Interpretation boundary: these short-period averages and failures inform development; they do not establish temporal superiority, physical-device advantage, runtime superiority, or ML-controller readiness. Keep collection running; report per-date and per-circuit variation and preserve the unchanged-data transitions.']
    (REPORT/'FINDINGS.txt').write_text('\n'.join(brief)+'\n')

def main():
    p=load(OUT/'plan.json');cal=calibration(p);flat,search_rows,validation=validate(p)
    result,per_circuit,per_rule,variability,failures,overhead=summarize(p,flat,search_rows,cal,validation)
    write_report(p,result,cal,per_circuit,per_rule,variability,failures,overhead)
    dump(REPORT/'environment.json',dict(python=sys.version,packages={x:importlib.metadata.version(x) for x in ['qiskit','qiskit-aer','numpy','duckdb','matplotlib']},generation_workers=p['workers'],threads_per_worker=1))
    dump(OUT/'checkpoint.json',dict(status='validated_results_plots_delivery_pending',completed_trajectories=270,total_trajectories=270,completed_searches=p['searches'],total_searches=p['searches'],report=str((REPORT/'RESULTS.txt').relative_to(ROOT)),confirmation_complete=False,next_action='Render and inspect plots; validate delivery manifest and update recovery files. Collection continues.'))
    print(json.dumps({k:result[k] for k in ['status','searches','paired_cases','distinct_calibration_value_vectors','changed_transitions','unchanged_transitions','primary_descriptive_direction','max_independent_error']},indent=2))

if __name__=='__main__':main()
