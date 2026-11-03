"""Exact-density first-order noise response. Offline diagnostic, exponential cost."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','QISKIT_NUM_PROCS','RAYON_NUM_THREADS']:os.environ[k]='1'
import numpy as np
from qiskit.quantum_info import DensityMatrix

def reset_subsystem(rho,ids,n):
    selected=[n-1-i for i in ids];rest=[i for i in range(n) if i not in selected];order=rest+selected+[n+i for i in rest]+[n+i for i in selected];dr=2**len(rest);ds=2**len(selected)
    x=rho.reshape([2]*(2*n)).transpose(order).reshape(dr,ds,dr,ds);reduced=np.trace(x,axis1=1,axis2=3)
    return np.einsum('ab,ij->aibj',reduced,np.eye(ds)/ds).reshape([2]*(2*n)).transpose(np.argsort(order)).reshape(2**n,2**n)
def response(q,params,physical):
    n=q.num_qubits;rho=np.zeros((2**n,2**n),complex);rho[0,0]=1;derivative=np.zeros_like(rho);measured={};opcount=0
    for ins in q.data:
        name=ins.operation.name;ids=[q.find_bit(b).index for b in ins.qubits]
        if name=='barrier':continue
        if name=='measure':measured[q.find_bit(ins.clbits[0]).index]=ids[0];continue
        assert name in ['rz','sx','x','cz'];opcount+=1
        rho=DensityMatrix(rho).evolve(ins.operation,qargs=ids).data
        derivative=DensityMatrix(derivative).evolve(ins.operation,qargs=ids).data
        rate=params[name+'_gate_error',physical[ids[0]],physical[ids[1]] if len(ids)==2 else None];lam=(2 if len(ids)==1 else 4/3)*rate
        if lam:derivative+=lam*(reset_subsystem(rho,ids,n)-rho)
    size=2**q.num_clbits;p0=np.zeros(size);dg=np.zeros(size)
    for index in range(2**n):
        bitstring=sum(((index>>local)&1)<<bit for bit,local in measured.items());p0[bitstring]+=rho[index,index].real;dg[bitstring]+=derivative[index,index].real
    dr=np.zeros(size)
    for bit,local in measured.items():
        a=physical[local];p01=params['prob_meas1_prep0',a,None];p10=params['prob_meas0_prep1',a,None]
        for zero in range(size):
            if zero&(1<<bit):continue
            one=zero|(1<<bit);flow=p01*p0[zero]-p10*p0[one];dr[zero]-=flow;dr[one]+=flow
    hybrid=p0+dg
    for bit,local in measured.items():
        a=physical[local];p01=params['prob_meas1_prep0',a,None];p10=params['prob_meas0_prep1',a,None]
        for zero in range(size):
            if zero&(1<<bit):continue
            one=zero|(1<<bit);x,y=hybrid[zero],hybrid[one];hybrid[zero]=(1-p01)*x+p10*y;hybrid[one]=p01*x+(1-p10)*y
    delta=dg+dr;first=p0+delta
    assert abs(p0.sum()-1)<1e-10 and abs(dg.sum())<1e-10 and abs(dr.sum())<1e-10
    return dict(ideal_distribution=p0.tolist(),gate_response=dg.tolist(),readout_response=dr.tolist(),first_order_distribution=first.tolist(),hybrid_distribution=hybrid.tolist(),scores=dict(first_order=float(abs(delta).sum()/2),hybrid=float(abs(hybrid-p0).sum()/2)),negative_mass=dict(first_order=float(np.maximum(-first,0).sum()),hybrid=float(np.maximum(-hybrid,0).sum())),native_operations=opcount)
