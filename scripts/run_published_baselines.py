"""Frozen available-data comparison with five external compiler implementations."""
from published_baseline_common import *
import sys,subprocess,time,shutil,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
from qiskit import qpy

def prepare():
    assert not (OUT/'protocol.json').exists(),'Protocol already frozen; resume instead.'
    prior=load(SOURCE/'plan.json');preflight=[]
    for m in METHODS:
        for item in load(OUT/f'preflight_batch_{m}.json'):
            r=load(item['destination']);assert r['status']=='compiled',(m,item['circuit_id'],r)
            s=item['spec'];v=validate(read_qasm(s['qasm']),lower(r,s['basis']),r['initial_layout'],s['edges'],s['basis'])
            preflight.append(dict(method=m,circuit=item['circuit_id'],**v))
    dump(OUT/'preflight_correctness.json',preflight)
    p={k:prior[k] for k in ['bank','physical_qubits','directed_cz_edges','basis_gates','replicates','seed_rules','paired_dates','ready_dates']}
    p['edges']=[[p['physical_qubits'].index(a),p['physical_qubits'].index(b)] for a,b in p['directed_cz_edges']]
    scenarios=[s for s in load(SOURCE/'scenarios.json') if s['status']=='ready']
    requests={};obs=[];own_sources={}
    for ci,(c,meta) in enumerate(p['bank'].items()):
        assert sha(ROOT/meta['path'])==meta['sha256']
        qasm=(ROOT/meta['path']).read_text()
        for rep in p['replicates']:
            for rule in p['seed_rules']:
                source=SOURCE/'jobs'/f'{c}__swarm_unique_56__{rep}__{rule}.json'
                own_sources[str(source.relative_to(ROOT))]=sha(source)
                own=load(source);days={d['date']:d for d in own['days'] if d['status']=='ready'}
                for s in scenarios:
                    seed=50001+1000*ci+100*rep+(0 if rule=='fixed' else s['calendar_index'])
                    base=dict(circuit_id=c,stratum=meta['stratum'],replicate=rep,seed_rule=rule,date=s['date'],paired=s['paired_observation'],seed=seed)
                    d=days[s['date']];assert d['seed']==seed
                    art=own['artifacts'][d['signature']];assert sha(ROOT/art['path'])==art['sha256']
                    obs.append(dict(**base,method='swarm_unique_56',source=str(source.relative_to(ROOT)),artifact=art,
                                    initial_layout=d['stats']['initial_layout_local'],source_tvd=d['tvd'],source_signature=d['signature']))
                    for m in METHODS:
                        spec=dict(method=m,seed=0 if m in ['qmap','duostra'] else seed,qasm=qasm,physical=p['physical_qubits'],basis=p['basis_gates'],edges=p['edges'],
                                  parameters=[{k:r[k] for k in ['property','qubit_a','qubit_b','value']} for r in s['parameters']] if m=='mapomatic' else [])
                        key=digest(spec);requests[key]=dict(spec=spec,destination=str(OUT/'raw'/(key+'.json')),key=key,circuit_id=c)
                        obs.append(dict(**base,method=m,request=key,seed_used=spec['seed']))
    for m in METHODS:dump(OUT/f'compile_{m}.json',[r for r in requests.values() if r['spec']['method']==m])
    dump(OUT/'requests.json',requests);dump(OUT/'observations.json',obs);dump(OUT/'scenarios.json',scenarios)
    sources={str(SOURCE.relative_to(ROOT))+'/'+f:sha(SOURCE/f) for f in ['plan.json','scenarios.json']}
    sources.update(own_sources)
    for f in ['published_baseline_worker.py','published_baseline_common.py','run_published_baselines.py','run_step2_pilot.py','validate_search_loss_diagnosis.py','validate_circuit_objective.py','duplicate_aware_swarm.py','response_search_engine.py']:
        sources['scripts/'+f]=sha(ROOT/'scripts'/f)
    for d in ['sqgm','mapomatic','qsyn']:
        base=ROOT/'data/external/published_baselines_v1'/d
        tracked=subprocess.check_output(['git','-C',str(base),'ls-files'],text=True).splitlines()
        for f in tracked:
            if f.endswith(('.py','.cpp','.hpp','.h','.txt','.bib')) or f in ['CMakeLists.txt','README.md']:
                path=base/f
                if path.is_file():sources[str(path.relative_to(ROOT))]=sha(path)
    sources['data/external/published_baselines_v1/qsyn/build/qsyn']=sha(ROOT/'data/external/published_baselines_v1/qsyn/build/qsyn')
    for f in ['baseline_registry.json','qsyn_adapter.patch','requests.json','observations.json','scenarios.json','preflight_correctness.json']:
        sources[str((OUT/f).relative_to(ROOT))]=sha(OUT/f)
    env={}
    for name in ['.venv-data','.venv-baselines-modern','.venv-baselines-legacy']:
        cmd=[str(ROOT/name/'bin/python'),'-c','import importlib.metadata as m,json,sys; print(json.dumps(dict(python=sys.version,packages={x.metadata["Name"]:x.version for x in m.distributions()})))']
        env[name]=json.loads(subprocess.check_output(cmd,text=True))
    dump(OUT/'environment.json',env)
    p.update(created_utc=now(),methods=METHODS+['swarm_unique_56'],source_hashes=sources,
        registry=load(OUT/'baseline_registry.json'),unique_compilation_requests=len(requests),observations=len(obs),paired_observations=sum(x['paired'] for x in obs),workers=4,
        objective='Primary mean fresh-circuit TVD under the identical calibrated depolarizing plus asymmetric assignment model. Also report native CZ count, depth, compile work and serial cost. Lower is better.',
        fairness='Same raw circuits, 6-qubit patch, directed CZ edges, CZ/RZ/SX/X native basis, eligible dates and output definitions. Baselines retain their own objectives. Default heuristic QMAP, default depth-4 DFS Duostra, author SQGM decay with enhanced heuristic and 3-iteration SabreLayout; LightSABRE 20 layout and 20 swap trials, 3 iterations; Mapomatic unmodified default success-product scoring after that LightSABRE pipeline. Common zero-optimization CX/1Q lowering before baselines, terminal-measurement barrier, common native CZ lowering optimization_level=1 after routing. Existing swarm native outputs unchanged.',
        budgets='Native-configuration comparison, not equal runtime or equal candidate work. Swarm has 56 unique placements. LightSABRE has 20 layout and 20 routing trials per configured layout. Mapomatic enumerates valid post-routing embeddings; QMAP/Duostra native search. No winner chosen with noisy evaluation TVD.',
        timing='Separate three serial cold-input technical repeats on all 15 circuits at the first paired date, replicate0 fixed seed, each method. Same pinned CPU, native threads1. Includes common preprocessing/native lowering and method work, excludes package startup and input disk reads where possible. Duostra external executable/temporary IO included. Separate from concurrent cohort generation, no matched-walltime superiority claim.',
        deterministic='QMAP and Duostra ignore external seeds; compile once per circuit. Reused deterministic outputs and repeated static calibration states are not independent trials.',
        inference='Equal date/circuit/replicate/seed-rule weight within each predefined stratum. Paired differences, per-circuit and per-date means, seed-rule and changing-calibration sensitivity, leave-one-date-out range. No IID p-values or statistically significant superiority claim from correlated technical cases.',
        correctness='Every compiled output checked for native topology and arbitrary-input measurement POVM equivalence (ancillas initialized zero). Aer and independent exact-channel calculations must agree within2e-10 for every unique native/date evaluation. Interchange uses full-precision operation JSON, not rounded legacy QASM. Threshold2e-8 for POVM error (actual errors retained).',
        scope='User-authorized post-hoc exploratory external-baseline comparison on already exposed Oct19-29 data. Four peer-reviewed 2023-2024 tool/method papers and one 2024 preprint. Maintained implementations pinned, not reproduction of original paper tables. One small patch and 4/5-qubit circuit instances, no real-device test, no ML controller, no independent confirmation.',
        not_changed=['original collection schedule','existing algorithms','source calibration files','source exploratory results','freshness thresholds'])
    dump(OUT/'protocol.json',p)
    dump(OUT/'checkpoint.json',dict(status='frozen_before_cohort_outcomes',updated_utc=now(),requests=len(requests),observations=len(obs),paired=sum(x['paired'] for x in obs)))
    print(json.dumps({k:p[k] for k in ['unique_compilation_requests','observations','paired_observations']},indent=2),flush=True)

def verify():
    p=load(OUT/'protocol.json')
    for f,h in p['source_hashes'].items():assert sha(ROOT/f)==h,f
    return p

def compile_all():
    p=verify()
    for m in METHODS:
        env='.venv-baselines-legacy' if m=='sqgm' else '.venv-baselines-modern'
        run=subprocess.run([str(ROOT/env/'bin/python'),str(ROOT/'scripts/published_baseline_worker.py'),str(OUT/f'compile_{m}.json')])
        assert run.returncode==0,m
        items=load(OUT/f'compile_{m}.json');failed=[x['key'] for x in items if load(x['destination'])['status']!='compiled']
        dump(OUT/f'compile_status_{m}.json',dict(method=m,total=len(items),compiled=len(items)-len(failed),failed=failed))
        print(m,'compiled',len(items)-len(failed),'failed',len(failed),flush=True)
    dump(OUT/'checkpoint.json',dict(status='compiled_evaluation_pending',updated_utc=now(),methods={m:load(OUT/f'compile_status_{m}.json') for m in METHODS}))

def evaluate_circuit(c):
    import run_heldout as src
    from validate_search_loss_diagnosis import independent
    from validate_circuit_objective import independent_p1,matrix_tape
    p=load(OUT/'protocol.json');src.install_region(p)
    logical=read_qasm((ROOT/p['bank'][c]['path']).read_text());ideal=src.engine.distribution(logical);lp=independent_p1(matrix_tape(logical))
    obs=[r for r in load(OUT/'observations.json') if r['circuit_id']==c]
    scenarios={s['date']:s for s in load(OUT/'scenarios.json')};cache={};checks={};evaluations={};rows=[];failures=[]
    for i,o in enumerate(obs):
        if o['method']=='swarm_unique_56':
            key=o['artifact']['sha256']
            if key not in cache:
                with (ROOT/o['artifact']['path']).open('rb') as f:native=qpy.load(f)[0]
                cache[key]=native;checks[key]=validate(logical,native,o['initial_layout'],p['edges'],p['basis_gates'])
            raw_hash=o['artifact']['sha256'];compile_time=None
        else:
            key=o['request'];raw_path=OUT/'raw'/(key+'.json');r=load(raw_path)
            if r['status']!='compiled':failures.append(dict(**o,reason='compile_failure',details=r));continue
            if key not in cache:
                native=lower(r,p['basis_gates']);checks[key]=validate(logical,native,r['initial_layout'],p['edges'],p['basis_gates']);cache[key]=native
                path=OUT/'native'/(key+'.qpy');path.parent.mkdir(exist_ok=True);src.engine.save_qpy(path,native)
            raw_hash=sha(raw_path);compile_time=r['compile_seconds']
        native=cache[key];sig=src.native_signature(native);evkey=sig+'__'+o['date']
        if evkey not in evaluations:
            params=src.engine.parameters(scenarios[o['date']]);a=src.engine.distribution(native,params);b=independent(native,params,lp)
            error=max(float(np.max(abs(a-b['distribution']))),float(np.max(abs(ideal-b['ideal']))))
            assert error<2e-10,(c,o['method'],error)
            evaluations[evkey]=dict(signature=sig,date=o['date'],distribution=a.tolist(),independent_distribution=b['distribution'].tolist(),tvd=float(np.abs(a-ideal).sum()/2),independent_error=error)
        ev=evaluations[evkey]
        if o['method']=='swarm_unique_56':assert abs(ev['tvd']-o['source_tvd'])<2e-10
        rows.append(dict(**o,tvd=ev['tvd'],evaluation_key=evkey,artifact_key=key,raw_sha256=raw_hash,compile_seconds=compile_time,**checks[key]))
        if (i+1)%72==0:dump(OUT/'progress'/(c+'.json'),dict(circuit=c,rows=len(rows),total=len(obs),updated_utc=now()))
    result=dict(circuit_id=c,rows=rows,failed=failures,checks=checks,evaluations=evaluations,ideal=ideal.tolist(),protocol_sha256=sha(OUT/'protocol.json'))
    dump(OUT/'jobs'/(c+'.json'),result)
    return dict(circuit=c,rows=len(rows),failed=len(failures),checks=len(checks),independent_evaluations=len(evaluations),max_error=max(v['independent_error'] for v in evaluations.values()))

def evaluate_all():
    p=verify();results=[]
    pending=[c for c in p['bank'] if not (OUT/'jobs'/(c+'.json')).exists()]
    with ProcessPoolExecutor(max_workers=p['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(evaluate_circuit,c) for c in pending]):
            r=f.result();results.append(r);dump(OUT/'checkpoint.json',dict(status='evaluating',completed=len(list((OUT/'jobs').glob('*.json'))),total=len(p['bank']),last=r,updated_utc=now()));print(r,flush=True)
    assert all((OUT/'jobs'/(c+'.json')).exists() for c in p['bank'])
    dump(OUT/'checkpoint.json',dict(status='cohort_evaluated_analysis_pending',completed_circuits=len(p['bank']),updated_utc=now()))

if __name__=='__main__':
    {'prepare':prepare,'compile':compile_all,'evaluate':evaluate_all}[sys.argv[1]]()
