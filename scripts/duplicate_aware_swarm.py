"""Budget-preserving PSO intervention; frozen before output-quality labels."""
import itertools, random, time
import run_heldout as src
from response_search_engine import Owner as OriginalOwner

def swarm(q,p,seed):
    start=time.perf_counter();e=src.engine;cfg=e.CFG['swarm'];size=len(e.PHYSICAL);n=q.num_qubits
    rng=random.Random(seed);replacement_rng=random.Random(seed ^ 0xD09A56)
    # Independent stream: shuffling does not consume the original PSO RNG.
    unseen=list(itertools.permutations(range(size),n));replacement_rng.shuffle(unseen)
    cache={};compile_seconds=0.;proposal_seconds=time.perf_counter()-start;proposals=[]
    def evaluate(pos,iteration,particle):
        nonlocal compile_seconds,proposal_seconds
        t=time.perf_counter();raw=pos.copy();replaced=tuple(pos[:n]) in cache
        if replaced:
            while unseen and unseen[-1] in cache:unseen.pop()
            assert unseen,'Frozen budget exceeds placement population'
            layout=unseen.pop();pos[:]=list(layout)+[i for i in range(size) if i not in layout]
        layout=tuple(pos[:n]);assert layout not in cache
        proposal_seconds+=time.perf_counter()-t
        t=time.perf_counter();native=e.transpile(q,basis_gates=e.BASIS,coupling_map=e.EDGES,initial_layout=list(layout),routing_method=e.CFG['routing_method'],optimization_level=e.CFG['optimization_level'],seed_transpiler=0);compile_seconds+=time.perf_counter()-t
        value=e.proxy(native,p);cache[layout]=(value,native)
        proposals.append(dict(iteration=iteration,particle=particle,raw_position=raw,position=pos.copy(),replaced=replaced))
        return value,replaced
    t=time.perf_counter();positions=[list(range(size))]+[rng.sample(range(size),size) for _ in range(cfg['particles']-1)];proposal_seconds+=time.perf_counter()-t
    velocities=[[] for _ in positions];scores=[]
    for i,pos in enumerate(positions):scores.append(evaluate(pos,0,i)[0])
    best=[x.copy() for x in positions];best_scores=scores.copy();idx=min(range(len(scores)),key=lambda i:scores[i]);global_best=best[idx].copy();global_score=scores[idx];trace=[global_score]
    for iteration in range(cfg['iterations']):
        target=global_best.copy()
        for i,pos in enumerate(positions):
            t=time.perf_counter();v=[s for s in velocities[i] if rng.random()<cfg['inertia']]
            v += [s for s in e.swaps_toward(pos,best[i]) if rng.random()<cfg['cognitive']]
            v += [s for s in e.swaps_toward(pos,target) if rng.random()<cfg['social']]
            if rng.random()<cfg['mutation_probability']:v.append(tuple(rng.sample(range(size),2)))
            for a,b in v:pos[a],pos[b]=pos[b],pos[a]
            proposal_seconds+=time.perf_counter()-t
            score,replaced=evaluate(pos,iteration+1,i);velocities[i]=[] if replaced else v
            if score<best_scores[i]:best_scores[i]=score;best[i]=pos.copy()
        idx=min(range(len(best_scores)),key=lambda i:best_scores[i])
        if best_scores[idx]<global_score:global_score=best_scores[idx];global_best=best[idx].copy()
        trace.append(global_score)
    layout=tuple(global_best[:n]);assert len(cache)==len(proposals)==56
    return cache[layout][1],dict(initial_layout_local=list(layout),initial_layout_physical=[e.PHYSICAL[i] for i in layout],total_search_seconds=time.perf_counter()-start,transpile_seconds_inside_search=compile_seconds,proposal_generation_seconds=proposal_seconds,unique_placements_compiled=len(cache),objective_calls=len(proposals),best_proxy=global_score,best_proxy_trace=trace,seed=seed,replacements=sum(x['replaced'] for x in proposals),proposal_trace=proposals)

class Owner(OriginalOwner):
    def search(self,q,p,seed,method,capture=True):
        if method!='swarm_unique_56':return super().search(q,p,seed,method,capture)
        original=src.engine.swarm
        try:
            src.engine.swarm=swarm
            return super().search(q,p,seed,'swarm',capture)
        finally:src.engine.swarm=original
