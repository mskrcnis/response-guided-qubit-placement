"""Isolated adapters for published implementations; no output-quality selection.

SQGM runs in its own Terra 0.22.3 environment. Other Python methods use Qiskit
2.5.2. Common CZ lowering and independent outcome evaluation happen downstream.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_NUM_PROCS']:
    os.environ[key] = '1'
import json, sys, time, subprocess, tempfile, re, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT/'data/external/published_baselines_v1'

def compile_case(spec):
    from qiskit import QuantumCircuit, transpile
    from qiskit.transpiler import CouplingMap, PassManager
    from qiskit.transpiler.passes import SabreLayout
    seed=spec['seed']; edges=spec['edges']; method=spec['method']
    try:
        from qiskit import qasm2
        q=qasm2.loads(spec['qasm'],custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    except ImportError:
        q=QuantumCircuit.from_qasm_str(spec['qasm'])
    n=q.num_qubits; begin=time.perf_counter()
    # Lower unsupported gates, without connectivity mapping or optimizations.
    logical=transpile(q,basis_gates=['rz','sx','x','cx'],optimization_level=0)
    # The evaluator models terminal measurement. This standard pass prevents a
    # router from measuring a finished logical qubit before routing other ones.
    from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements
    logical=PassManager(BarrierBeforeFinalMeasurements()).run(logical)
    cm=CouplingMap(edges); details={}; initial=None
    if method in ['lightsabre','mapomatic']:
        mapper=SabreLayout(cm,seed=seed,max_iterations=3,layout_trials=20,swap_trials=20)
        routed=PassManager(mapper).run(logical)
        initial=routed.layout.initial_index_layout(filter_ancillas=True)[:n]
        details['layout_trials']=20; details['swap_trials']=20; details['max_iterations']=3
        if method=='mapomatic':
            sys.path.insert(0,str(EXTERNAL/'mapomatic'))
            import mapomatic as mm
            from types import SimpleNamespace
            routed=transpile(routed,basis_gates=spec['basis'],optimization_level=1)
            active=sorted(routed.find_bit(x).index for x in mm.active_bits(routed)[0])
            small=mm.deflate_circuit(routed)
            layouts=mm.matching_layouts(small,edges,strict_direction=True)
            assert layouts
            params={(r['property'],r['qubit_a'],r['qubit_b']):r['value'] for r in spec['parameters']}
            physical=spec['physical']
            class Properties:
                def gate_error(self,name,qubits):
                    if isinstance(qubits,int):qubits=[qubits]
                    return params[name+'_gate_error',physical[qubits[0]],physical[qubits[1]] if len(qubits)==2 else None]
                def readout_error(self,index):
                    p=physical[index]
                    return (params['prob_meas1_prep0',p,None]+params['prob_meas0_prep1',p,None])/2
            backend=SimpleNamespace(properties=lambda:Properties(),configuration=lambda:SimpleNamespace(basis_gates=spec['basis']))
            ranked=mm.evaluate_layouts(small,layouts,backend)
            layout=ranked[0][0]
            initial=[layout[active.index(i)] for i in initial]
            routed=mm.inflate_circuit(small,layout,len(physical))
            details.update(candidate_embeddings=len(layouts),selected_embedding=layout,selected_cost=ranked[0][1],cost='Unmodified Mapomatic default product of gate and mean-readout success; no TVD labels')
    elif method=='qmap':
        from mqt.qmap.plugins.qiskit.sc import compile_
        from mqt.qmap.sc import Architecture
        routed,results=compile_(logical,Architecture(len(spec['physical']),set(map(tuple,edges))))
        initial=routed.layout.initial_index_layout(filter_ancillas=True)[:n]
        details['qmap_results']=results.json()
    elif method=='sqgm':
        sys.path.insert(0,str(EXTERNAL/'sqgm'))
        from router.sqgm_swap import SQGMSwap
        from qiskit.transpiler.passes import FullAncillaAllocation,EnlargeWithAncilla,ApplyLayout,Decompose
        from qiskit.circuit.library import SwapGate
        mapper=SabreLayout(cm,seed=seed,max_iterations=3)
        pm=PassManager([mapper,FullAncillaAllocation(cm),EnlargeWithAncilla(),ApplyLayout()])
        mapped=pm.run(logical)
        initial=[pm.property_set['layout'][bit] for bit in logical.qubits]
        router=SQGMSwap(cm,heuristic='decay',enhance_h=True,seed=seed)
        routed=PassManager([router,Decompose(SwapGate)]).run(mapped)
        details.update(heuristic='decay',enhance_h=True,layout='Terra 0.22.3 SabreLayout, max_iterations=3',routing='Author SQGMSwap unmodified')
    elif method=='duostra':
        from qiskit import qasm2
        bare=logical.remove_final_measurements(inplace=False)
        bare.data=[x for x in bare.data if x.operation.name!='barrier']
        with tempfile.TemporaryDirectory(prefix='duostra_') as td:
            from fractions import Fraction
            import math
            # Qsyn's decimal parser quantizes at 1e-4 radians. Its documented
            # rational-pi syntax avoids that approximation without changing
            # the router. Keep denominators within its signed-int range.
            lines=['OPENQASM 2.0;','include "qelib1.inc";',f'qreg q[{n}];'];angle_errors=[]
            for ins in bare.data:
                name=ins.operation.name
                if ins.operation.params:
                    assert name=='rz'
                    theta=float(ins.operation.params[0]);fraction=Fraction(theta/math.pi).limit_denominator(10_000_000)
                    angle_errors.append(abs(float(fraction)*math.pi-theta))
                    name+=f'({fraction.numerator}*pi/{fraction.denominator})'
                lines.append(name+' '+','.join(f'q[{bare.find_bit(b).index}]' for b in ins.qubits)+';')
            assert max(angle_errors,default=0)<1e-10
            td=Path(td);(td/'input.qasm').write_text('\n'.join(lines)+'\n')
            adjacency=[[b for a,b in edges if a==i] for i in range(len(spec['physical']))]
            topology='NAME: fez_patch\nQUBITNUM: 6\nGATESET: {x, rz, h, id, sx, cnot}\nCOUPLINGMAP: '+json.dumps(adjacency)+'\n'
            # Qsyn's layout parser requires all four arrays, even though this
            # topology-only baseline uses the paper's abstract gate-time model.
            topology+='SGERROR: '+json.dumps([0.0]*6)+'\nSGTIME: '+json.dumps([1.0]*6)+'\n'
            topology+='CNOTERROR: '+json.dumps([[0.0]*len(a) for a in adjacency])+'\n'
            topology+='CNOTTIME: '+json.dumps([[2.0]*len(a) for a in adjacency])+'\n'
            (td/'device.layout').write_text(topology)
            (td/'run.dof').write_text('qc read input.qasm\ndevice read device.layout\nduostra --check --mute-tqdm\nqc write output.qasm\nquit -f\n')
            exe=EXTERNAL/'qsyn/build/qsyn'
            run=subprocess.run([str(exe),'-v','run.dof'],cwd=td,capture_output=True,text=True,timeout=300)
            if run.returncode or not (td/'output.qasm').exists():raise RuntimeError(str(run.returncode)+'\n'+run.stdout+'\n'+run.stderr)
            start_map={int(a):int(b) for a,b in re.findall(r'BENCHMARK_INITIAL (\d+) (\d+)',run.stdout)}
            end_map={int(a):int(b) for a,b in re.findall(r'BENCHMARK_FINAL (\d+) (\d+)',run.stdout)}
            initial=[start_map[i] for i in range(n)]
            routed=qasm2.load(str(td/'output.qasm'),custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
            from qiskit import ClassicalRegister
            routed.add_register(ClassicalRegister(n,'c'))
            for i in range(n):routed.measure(end_map[i],i)
            details.update(qsyn_stdout=run.stdout,qsyn_stderr=run.stderr,deterministic=True,
                           scheduler='search',depth=4,placer='dfs',router='duostra',
                           max_angle_interchange_error=max(angle_errors,default=0),
                           timing_note='Includes external executable startup and temporary file IO.')
    else:
        raise ValueError(method)
    elapsed=time.perf_counter()-begin
    try:
        from qiskit import qasm2
        output=qasm2.dumps(routed)
    except ImportError:
        output=routed.qasm()
    return dict(qasm=output,initial_layout=initial,method=method,seed=seed,compile_seconds=elapsed,details=details,
                operations=[dict(name=x.operation.name,params=[float(v) for v in x.operation.params],
                                 qubits=[routed.find_bit(b).index for b in x.qubits],
                                 clbits=[routed.find_bit(b).index for b in x.clbits]) for x in routed.data],
                global_phase=float(routed.global_phase),classical_width=routed.num_clbits,
                raw_counts=dict(routed.count_ops()),raw_depth=routed.depth(),logical_qubits=n,physical_width=routed.num_qubits)

if __name__=='__main__':
    spec=json.loads(Path(sys.argv[1]).read_text())
    batch=spec if isinstance(spec,list) else [dict(spec=spec,destination=sys.argv[2])]
    for i,item in enumerate(batch):
        dest=Path(item['destination'])
        if dest.exists():continue
        try:
            out=compile_case(item['spec']);out['status']='compiled'
        except Exception:
            out=dict(status='failed',traceback=traceback.format_exc(),method=item['spec']['method'])
        dest.parent.mkdir(parents=True,exist_ok=True)
        tmp=dest.with_suffix('.tmp'); tmp.write_text(json.dumps(out,indent=2)); tmp.replace(dest)
        if len(batch)>1 and (i+1)%25==0:print(f'{i+1}/{len(batch)} {item["spec"]["method"]}',flush=True)
        if out['status']=='failed':print(out['traceback'],file=sys.stderr,flush=True)
