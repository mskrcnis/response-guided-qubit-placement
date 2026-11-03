"""Exact shared ideal/batched tangent propagation for the existing cache schema."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','QISKIT_NUM_PROCS','RAYON_NUM_THREADS']:os.environ[k]='1'
import numpy as np
from distribution_response import reset_subsystem

def evolve_batch(stack,name,ids,angle=0.):
    dim=stack.shape[1];indices=np.arange(dim)
    if name=='x':
        perm=indices^(1<<ids[0]);return stack[:,perm,:][:,:,perm]
    if name=='sx':
        perm=indices^(1<<ids[0]);a=(1+1j)/2;b=(1-1j)/2
        temp=a*stack+b*stack[:,perm,:]
        return a.conjugate()*temp+b.conjugate()*temp[:,:,perm]
    if name=='rz':phase=np.exp(1j*(angle/2)*(2*((indices>>ids[0])&1)-1))
    elif name=='cz':phase=1-2*(((indices>>ids[0])&1)*((indices>>ids[1])&1))
    else:raise ValueError(name)
    stack*=phase[None,:,None];stack*=phase.conj()[None,None,:];return stack

def build(q,physical):
    ops=[];keys=set();measured={}
    for ins in q.data:
        name=ins.operation.name;ids=[q.find_bit(b).index for b in ins.qubits]
        if name=='barrier':continue
        if name=='measure':measured[q.find_bit(ins.clbits[0]).index]=ids[0];continue
        assert name in ['rz','sx','x','cz'];key=(name+'_gate_error',physical[ids[0]],physical[ids[1]] if len(ids)==2 else None);keys.add(key);ops.append((name,ids,float(ins.operation.params[0]) if name=='rz' else 0.,key))
    gate_keys=sorted(keys,key=lambda k:(k[0],k[1],-1 if k[2] is None else k[2]));index={k:i+1 for i,k in enumerate(gate_keys)};n=q.num_qubits;dim=2**n;stack=np.zeros((len(gate_keys)+1,dim,dim),dtype=complex);stack[0,0,0]=1
    for name,ids,angle,key in ops:
        stack=evolve_batch(stack,name,ids,angle);ideal=stack[0]
        stack[index[key]]+=(2 if len(ids)==1 else 4/3)*(reset_subsystem(ideal,ids,n)-ideal)
    bins=2**q.num_clbits;destination=np.array([sum(((i>>local)&1)<<bit for bit,local in measured.items()) for i in range(dim)]);diagonal=stack.diagonal(axis1=1,axis2=2).real;outputs=np.array([np.bincount(destination,weights=row,minlength=bins) for row in diagonal]);p0=outputs[0];rc=np.zeros((2*q.num_clbits,bins))
    for bit in range(q.num_clbits):
        zero=np.array([i for i in range(bins) if not i&(1<<bit)]);one=zero|(1<<bit);rc[2*bit,zero]=-p0[zero];rc[2*bit,one]=p0[zero];rc[2*bit+1,zero]=p0[one];rc[2*bit+1,one]=-p0[one]
    return dict(gate_keys=gate_keys,readout_keys=[(name,physical[measured[bit]],None) for bit in range(q.num_clbits) for name in ['prob_meas1_prep0','prob_meas0_prep1']],ideal=p0.copy(),gate_coefficients=outputs[1:].copy(),readout_coefficients=rc,qubits=n,classical_bits=q.num_clbits)
