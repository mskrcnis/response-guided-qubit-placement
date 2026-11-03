"""Linear-size local Bloch approximations; no exact simulator or outcome labels."""
import math
import numpy as np

NAMES=['baseline','directional_flip','directional_bias','local_response']

def tape(q,physical):
    ops=[];measured={}
    for inst in q.data:
        name=inst.operation.name;ids=tuple(q.find_bit(b).index for b in inst.qubits)
        if name=='barrier':continue
        if name=='measure':measured[q.find_bit(inst.clbits[0]).index]=ids[0];continue
        assert name in ['h','x','sx','rx','ry','rz','cx','cz','swap'],name
        ops.append((name,ids,float(inst.operation.params[0]) if inst.operation.params else 0.))
    assert set(measured)==set(range(q.num_clbits))
    return dict(n=q.num_qubits,physical=list(physical),ops=ops,measured=[measured[i] for i in range(q.num_clbits)])

def local_bloch(t,params=None):
    """After each entangler, keep only single-qubit reduced Bloch vectors.

    This explicitly discards inter-qubit correlations. It is an approximation,
    not an ideal/noisy statevector or density-matrix calculation.
    """
    v=np.zeros((t['n'],3));v[:,2]=1
    for name,ids,angle in t['ops']:
        a=ids[0];x,y,z=v[a]
        if name=='h':v[a]=[z,-y,x]
        elif name=='x':v[a]=[x,-y,-z]
        elif name=='sx':v[a]=[x,-z,y]
        elif name in ['rx','ry','rz']:
            c,s=math.cos(angle),math.sin(angle)
            if name=='rx':v[a]=[x,c*y-s*z,s*y+c*z]
            elif name=='ry':v[a]=[c*x+s*z,y,-s*x+c*z]
            else:v[a]=[c*x-s*y,s*x+c*y,z]
        elif name=='swap':v[a],v[ids[1]]=v[ids[1]].copy(),v[a].copy()
        else:
            b=ids[1];u,w,k=v[b]
            if name=='cz':v[a]=[x*k,y*k,z];v[b]=[u*z,w*z,k]
            else:v[a]=[x*u,y*u,z];v[b]=[u,w*z,k*z]
        if params is not None:
            physical=[t['physical'][i] for i in ids]
            rate=params[(name+'_gate_error',physical[0],physical[1] if len(physical)==2 else None)]
            shrink=1-(2*rate if len(ids)==1 else 4*rate/3)
            for i in ids:v[i]*=shrink
    return v

def probabilities(t,params=None):
    v=local_bloch(t,params);return np.clip((1-v[t['measured'],2])/2,0,1)

def cache(native,logical_p1,physical):
    t=tape(native,physical)
    return dict(tape=t,logical_p1=np.array(logical_p1),native_p1=probabilities(t))

def gate_cost(t,p):
    value=0.
    for name,ids,angle in t['ops']:
        a=t['physical'][ids[0]];b=t['physical'][ids[1]] if len(ids)==2 else None
        value-=math.log1p(-p[(name+'_gate_error',a,b)])
    return value

def score(name,c,p):
    assert name in NAMES;t=c['tape'];logical=c['logical_p1']
    if name=='local_response':
        noisy=probabilities(t,p);out=0.
        for i,q in enumerate(t['measured']):
            a=t['physical'][q];p01=p[('prob_meas1_prep0',a,None)];p10=p[('prob_meas0_prep1',a,None)]
            observed=p01+(1-p01-p10)*noisy[i];out+=abs(observed-c['native_p1'][i])
        return float(out)
    value=gate_cost(t,p)
    for i,q in enumerate(t['measured']):
        a=t['physical'][q];p01=p[('prob_meas1_prep0',a,None)];p10=p[('prob_meas0_prep1',a,None)]
        if name=='baseline':value-=math.log1p(-(p01+p10)/2)
        elif name=='directional_flip':value-=math.log1p(-(p01*(1-logical[i])+p10*logical[i]))
        else:value+=abs(p01*(1-logical[i])-p10*logical[i])
    return float(value)
