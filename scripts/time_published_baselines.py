"""Serial cold-object timing on a declared all-circuit subset, three repeats."""
from published_baseline_common import *
import sys,time,subprocess,random
from duplicate_aware_swarm import Owner
import run_heldout as src
def main():
    p=load(OUT/'protocol.json');src.install_region(p)
    assert all((OUT/'jobs'/(c+'.json')).exists() for c in p['bank']),'Finish concurrent evaluation before serial timing.'
    cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    first=p['paired_dates'][0];scenario=next(s for s in load(OUT/'scenarios.json') if s['date']==first);params=src.engine.parameters(scenario)
    obs=load(OUT/'observations.json');reqs=load(OUT/'requests.json')
    todo=[(c,m,r) for c in p['bank'] for m in p['methods'] for r in range(3)]
    random.Random(20261031).shuffle(todo)
    protocol=dict(created_utc=now(),cohort_protocol_sha256=sha(OUT/'protocol.json'),cpu=cpu,native_threads=1,date=first,replicate=0,seed_rule='fixed',repeats=3,tasks=todo,
                  definition='Cold per-call compiler/search objects, resident inputs, serial process, pinned logical CPU. Ours includes coefficient-cache construction. Baseline engine seconds plus common CZ lowering. Package startup/input disk IO excluded; lazy method imports inside the baseline function remain included. Duostra includes external executable startup and temporary file IO. Different language/framework versions; descriptive pipeline cost, not equal-time algorithm-quality experiment.')
    dest=OUT/'serial_timing_protocol.json'
    if dest.exists():old=load(dest);assert old['tasks']==[list(x) for x in todo] and old['cohort_protocol_sha256']==protocol['cohort_protocol_sha256']
    else:dump(dest,protocol)
    results=[]
    for i,(c,m,repeat) in enumerate(todo):
        dest=OUT/'serial_timing'/f'{c}__{m}__{repeat}.json'
        if dest.exists():results.append(load(dest));continue
        q=read_qasm((ROOT/p['bank'][c]['path']).read_text())
        o=next(o for o in obs if o['circuit_id']==c and o['method']==m and o['date']==first and o['replicate']==0 and o['seed_rule']=='fixed')
        if m=='swarm_unique_56':
            start=time.perf_counter();owner=Owner('hybrid',c);native,stats=owner.search(q,params,o['seed'],m);total=time.perf_counter()-start
            assert src.native_signature(native)==o['source_signature']
            result=dict(seconds=total,engine_seconds=total,normalization_seconds=0.,source_native_reproduced=True,cache_build_seconds=owner.totals['build_seconds'],search_work=stats['objective_calls'])
        else:
            request=reqs[o['request']];env='.venv-baselines-legacy' if m=='sqgm' else '.venv-baselines-modern'
            specpath=OUT/'serial_timing_inputs'/(o['request']+'.json');dump(specpath,request['spec'])
            rawpath=OUT/'serial_timing_raw'/f'{c}__{m}__{repeat}.json'
            run=subprocess.run([str(ROOT/env/'bin/python'),str(ROOT/'scripts/published_baseline_worker.py'),str(specpath),str(rawpath)],capture_output=True,text=True,timeout=300)
            assert run.returncode==0,(c,m,run.stderr);raw=load(rawpath);assert raw['status']=='compiled',raw
            start=time.perf_counter();native=lower(raw,p['basis_gates']);normalize=time.perf_counter()-start
            source_native=lower(load(request['destination']),p['basis_gates'])
            assert src.native_signature(native)==src.native_signature(source_native),(c,m)
            result=dict(seconds=raw['compile_seconds']+normalize,engine_seconds=raw['compile_seconds'],normalization_seconds=normalize,source_native_reproduced=True)
        result.update(circuit_id=c,stratum=p['bank'][c]['stratum'],method=m,repeat=repeat,cpu=cpu,date=first,seed=o['seed'],created_utc=now())
        dump(dest,result);results.append(result)
        if (i+1)%15==0:print('Serial timing',i+1,'/',len(todo),flush=True)
    summary=[]
    for st in sorted({r['stratum'] for r in results}):
        for m in p['methods']:
            rs=[r for r in results if r['stratum']==st and r['method']==m]
            summary.append(dict(stratum=st,method=m,trials=len(rs),median_seconds=float(np.median([r['seconds'] for r in rs])),mean_seconds=float(np.mean([r['seconds'] for r in rs])),min_seconds=min(r['seconds'] for r in rs),max_seconds=max(r['seconds'] for r in rs)))
    dump(REPORT/'serial_timing.json',dict(protocol=load(OUT/'serial_timing_protocol.json'),summary=summary,trials=results))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
