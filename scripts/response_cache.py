"""Calibration-independent dense response coefficients; still exponential in output size."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','QISKIT_NUM_PROCS','RAYON_NUM_THREADS']:os.environ[k]='1'
import numpy as np
from distribution_response import response

def build(q,physical):
    keys=set();measured={}
    for ins in q.data:
        name=ins.operation.name;ids=[q.find_bit(b).index for b in ins.qubits]
        if name=='barrier':continue
        if name=='measure':measured[q.find_bit(ins.clbits[0]).index]=physical[ids[0]];continue
        keys.add((name+'_gate_error',physical[ids[0]],physical[ids[1]] if len(ids)==2 else None))
    gate_keys=sorted(keys,key=lambda k:(k[0],k[1],-1 if k[2] is None else k[2]));read_keys=[(name,measured[bit],None) for bit in range(q.num_clbits) for name in ['prob_meas1_prep0','prob_meas0_prep1']];zero={k:0. for k in gate_keys+read_keys};coeff=[];ideal=None
    for key in gate_keys:
        params=dict(zero);params[key]=1.;r=response(q,params,physical);coeff.append(r['gate_response'])
        if ideal is None:ideal=np.array(r['ideal_distribution'])
    assert ideal is not None
    rc=np.zeros((len(read_keys),len(ideal)))
    for bit in range(q.num_clbits):
        for zero_index in range(len(ideal)):
            if zero_index&(1<<bit):continue
            one=zero_index|(1<<bit);rc[2*bit,zero_index]=-ideal[zero_index];rc[2*bit,one]=ideal[zero_index];rc[2*bit+1,zero_index]=ideal[one];rc[2*bit+1,one]=-ideal[one]
    return dict(gate_keys=gate_keys,readout_keys=read_keys,ideal=ideal,gate_coefficients=np.array(coeff),readout_coefficients=rc,qubits=q.num_qubits,classical_bits=q.num_clbits)
def evaluate(cache,params,details=False):
    rates=np.array([params[tuple(k)] for k in cache['gate_keys']]);reads=np.array([params[tuple(k)] for k in cache['readout_keys']]);dg=rates@cache['gate_coefficients'];dr=reads@cache['readout_coefficients'];ideal=cache['ideal'];delta=dg+dr;hybrid=(ideal+dg).copy()
    for bit in range(cache['classical_bits']):
        view=hybrid.reshape(-1,2,2**bit);x=view[:,0,:].copy();y=view[:,1,:].copy();p01,p10=reads[2*bit:2*bit+2];view[:,0,:]=(1-p01)*x+p10*y;view[:,1,:]=p01*x+(1-p10)*y
    scores=dict(first_order=float(abs(delta).sum()/2),hybrid=float(abs(hybrid-ideal).sum()/2))
    if not details:return scores
    return dict(scores=scores,ideal_distribution=ideal.tolist(),gate_response=dg.tolist(),readout_response=dr.tolist(),first_order_distribution=(ideal+delta).tolist(),hybrid_distribution=hybrid.tolist())
def arrays(cache):return {k:cache[k] for k in ['ideal','gate_coefficients','readout_coefficients']}
def metadata(cache):return {k:v for k,v in cache.items() if k not in ['ideal','gate_coefficients','readout_coefficients']}
def load_cache(meta,npz):
    with np.load(npz,allow_pickle=False) as a:return dict(meta,**{k:a[k].copy() for k in a.files})
