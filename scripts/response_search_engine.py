"""Full-search response pilot. Each owner holds its own cold-start native cache."""
import time
import numpy as np
import run_heldout as src
import circuit_sensitive_scores as scalar
import response_cache as rc
from response_cache_batched import build

class Owner:
    def __init__(self, objective, circuit_id, on_native=None):
        self.objective=objective;self.circuit_id=circuit_id;self.cache={};self.on_native=on_native
        self.totals=dict(hits=0,misses=0,build_seconds=0.,kernel_seconds=0.,key_seconds=0.,array_bytes=0)
        self.lp=None
    def evaluate(self,q,p):
        if self.objective=='baseline':return src.engine_original_proxy(q,p)
        if self.objective=='directional_flip':return scalar.score('directional_flip',dict(tape=scalar.tape(q,src.engine.PHYSICAL),logical_p1=self.lp),p)
        t=time.perf_counter();sig=src.native_signature(q);key=(self.circuit_id,tuple(src.engine.PHYSICAL),q.num_qubits,q.num_clbits,sig);self.totals['key_seconds']+=time.perf_counter()-t
        if key not in self.cache:
            t=time.perf_counter();c=build(q,src.engine.PHYSICAL);self.totals['build_seconds']+=time.perf_counter()-t;self.totals['misses']+=1
            self.cache[key]=c;self.totals['array_bytes']+=sum(a.nbytes for a in rc.arrays(c).values())
        else:self.totals['hits']+=1
        t=time.perf_counter();v=rc.evaluate(self.cache[key],p)[self.objective];self.totals['kernel_seconds']+=time.perf_counter()-t
        return v
    def search(self,q,p,seed,method,capture=True):
        started=time.perf_counter();original_proxy=src.engine.proxy;original_comp=src.comp.proxy;ot=src.engine.transpile;of=src.comp.fixed_compile
        # Saved original pointer avoids recursive callback for baseline.
        src.engine_original_proxy=original_proxy
        self.lp=scalar.probabilities(scalar.tape(q,range(q.num_qubits))) if self.objective=='directional_flip' else None
        before=self.totals.copy();last=[];candidates=[];overhead=0.
        def tr(*a,**kw):
            nonlocal last
            last=list(kw['initial_layout']);return ot(*a,**kw)
        def fx(circuit,layout):
            nonlocal last
            last=list(layout);return of(circuit,layout)
        def ev(native,params):
            nonlocal overhead
            v=self.evaluate(native,params)
            if capture:
                t=time.perf_counter();sig=src.native_signature(native)
                if self.on_native:self.on_native(sig,native)
                candidates.append(dict(layout=last.copy(),signature=sig,score=v));overhead+=time.perf_counter()-t
            return v
        try:
            src.engine.transpile=tr;src.comp.fixed_compile=fx;src.engine.proxy=ev;src.comp.proxy=ev
            native,stats=src.engine.swarm(q,p,seed) if method=='swarm' else src.fixed_search(q,p,seed)
        finally:src.engine.proxy=original_proxy;src.comp.proxy=original_comp;src.engine.transpile=ot;src.comp.fixed_compile=of
        elapsed=time.perf_counter()-started
        assert stats['objective_calls']==56
        if capture:assert len(candidates)==stats['unique_placements_compiled']==len({tuple(c['layout']) for c in candidates})
        stats.update(objective=self.objective,candidates=candidates,logical_p1=None if self.lp is None else self.lp.tolist(),wall_seconds=elapsed,audit_seconds=overhead,cache_delta={k:self.totals[k]-before[k] for k in before},cache_entries=len(self.cache),cache_array_bytes=self.totals['array_bytes'],placement_cache_hits=56-stats['unique_placements_compiled'])
        return native,stats
