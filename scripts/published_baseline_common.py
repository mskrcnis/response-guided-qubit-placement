"""Common native lowering, correctness checks and provenance utilities."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_NUM_PROCS']:
    os.environ[key]='1'
from pathlib import Path
import hashlib,json,datetime
import numpy as np
from qiskit import qasm2,transpile,QuantumCircuit
from qiskit.quantum_info import Operator
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/published_baselines_v1'
REPORT=ROOT/'reports/published_baselines_v1'
SOURCE=ROOT/'experiments/exploratory_oct30_v1'
METHODS=['lightsabre','mapomatic','sqgm','qmap','duostra']
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def read_qasm(s):return qasm2.loads(s,custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
def lower(raw,basis):
    if isinstance(raw,dict):
        q=QuantumCircuit(raw['physical_width'],raw['classical_width'])
        for x in raw['operations']:
            getattr(q,x['name'])(*x['params'],*x['qubits'],*x['clbits'])
        q.global_phase=raw['global_phase']
    else:q=read_qasm(raw)
    # Barriers have no channel. Removing them also permits terminal measurements
    # on separate wires to be recognized after an upstream pass reorders them.
    q.data=[x for x in q.data if x.operation.name!='barrier']
    return transpile(q,basis_gates=basis,optimization_level=1,seed_transpiler=0)
def validate(logical,native,initial,edges,basis):
    """Compare all-input logical measurement POVMs, including ancilla leakage.

    Native columns inject arbitrary logical inputs into the declared initial
    layout with all ancillas zero. Rows are grouped by the actual classical
    measurement destinations. This checks more than zero-state probabilities.
    """
    n=logical.num_qubits; assert len(initial)==n and len(set(initial))==n
    assert native.num_clbits==logical.num_clbits==n
    assert all(x.operation.name in basis+['measure','barrier'] for x in native.data)
    for x in native.data:
        if x.operation.num_qubits==2 and x.operation.name!='barrier':
            assert tuple(native.find_bit(b).index for b in x.qubits) in set(map(tuple,edges))
    measured={native.find_bit(x.clbits[0]).index:native.find_bit(x.qubits[0]).index for x in native.data if x.operation.name=='measure'}
    assert set(measured)==set(range(n)) and len(set(measured.values()))==n
    body=native.remove_final_measurements(inplace=False)
    assert all(x.operation.name!='measure' for x in body.data)
    ref=Operator(logical.remove_final_measurements(inplace=False)).data
    u=Operator(body).data
    columns=[sum(((i>>k)&1)<<j for k,j in enumerate(initial)) for i in range(2**n)]
    v=u[:,columns]
    bins=np.array([sum(((i>>measured[k])&1)<<k for k in range(n)) for i in range(2**native.num_qubits)])
    error=max(float(np.max(abs(v[bins==y].conj().T@v[bins==y]-np.outer(ref[y].conj(),ref[y])))) for y in range(2**n))
    assert error<2e-8,('measurement semantics',error)
    return dict(max_povm_error=error,logical_input_dimension=2**n,measurement_map=measured,
                native_cz=int(native.count_ops().get('cz',0)),native_depth=native.depth(),
                native_1q=sum(x.operation.num_qubits==1 and x.operation.name!='measure' for x in native.data))
