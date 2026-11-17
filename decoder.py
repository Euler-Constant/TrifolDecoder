"""Raw-syndrome D1/D2 agreement gate with a selectively invoked specialist."""
import json
from pathlib import Path
import numpy as np
from numba import njit, prange
from circuit_model import problem
from kernels import search

@njit(cache=True)
def paired_bp(s,prior,cols,ptr,fast_iterations,strong_iterations):
    """Share an identical raw-input BP prefix; D1 never corrects D2's input."""
    n=len(prior);ne=len(cols);m=len(ptr)-1
    v=prior[cols].copy();msg=np.zeros(ne);post=prior.copy()
    hard=np.zeros(n,np.uint8);post1=prior.copy();hard1=hard.copy();valid1=False
    valid=False;steps=0
    for it in range(strong_iterations):
        # Each tanh is used in both prefix/suffix scans; evaluate it once.
        tv=np.empty(ne)
        for k in range(ne): tv[k]=np.tanh(v[k]*.5)
        for r in range(m):
            start=ptr[r];end=ptr[r+1];prefix=np.ones(end-start+1);suffix=1.
            for k in range(start,end): prefix[k-start+1]=prefix[k-start]*tv[k]
            for k in range(end-1,start-1,-1):
                prod=prefix[k-start]*suffix
                if s[r]: prod=-prod
                prod=max(-.999999999999,min(.999999999999,prod))
                msg[k]=2*np.arctanh(prod);suffix*=tv[k]
        post=prior.copy()
        for k in range(ne): post[cols[k]]+=msg[k]
        for j in range(n): hard[j]=post[j]<0
        valid=True
        for r in range(m):
            value=0
            for k in range(ptr[r],ptr[r+1]): value^=hard[cols[k]]
            if value!=s[r]: valid=False;break
        steps=it+1
        if steps==fast_iterations or (valid and steps<=fast_iterations):
            post1=post.copy();hard1=hard.copy();valid1=valid
        if valid: break
        for k in range(ne):
            value=max(-40.,min(40.,post[cols[k]]-msg[k]))
            v[k]=.8*value+.2*v[k]
    return post1,hard1,valid1,post,hard,valid,steps

@njit(cache=True)
def verify(e,s,cols,ptr):
    for r in range(len(s)):
        value=0
        for k in range(ptr[r],ptr[r+1]): value^=e[cols[k]]
        if value!=s[r]: return False
    return True

@njit(cache=True)
def logical_label(L,e):
    label=0
    for o in range(L.shape[0]):
        bit=0
        for j in range(len(e)): bit^=L[o,j]&e[j]
        label|=int(bit)<<o
    return label

@njit(cache=True)
def summarise(masses,L,e):
    y=np.zeros(4,np.uint8);cmin=1.;entropy=0.;first=0.;second=0.
    total=masses.sum()
    if total<=0:
        label=logical_label(L,e)
        for j in range(4): y[j]=(label>>j)&1
        return y,0.,1.,0.
    for label in range(16):
        q=masses[label]/total
        if q>0: entropy-=q*np.log(q)/np.log(16.)
        if q>first: second=first;first=q
        elif q>second: second=q
    for j in range(4):
        one=0.
        for label in range(16):
            if (label>>j)&1: one+=masses[label]/total
        y[j]=one>.5;cmin=min(cmin,abs(2*one-1))
    return y,cmin,entropy,first-second

@njit(cache=True)
def raw_features(s,adj,times):
    weight=0;edges=0;lo=1e9;hi=-1e9
    for i in range(len(s)):
        if s[i]:
            weight+=1;lo=min(lo,times[i]);hi=max(hi,times[i])
            for j in range(i):
                if s[j] and adj[i,j]: edges+=1
    seen=np.zeros(len(s),np.uint8);queue=np.empty(len(s),np.int64);largest=0
    for i in range(len(s)):
        if not s[i] or seen[i]: continue
        queue[0]=i;seen[i]=1;head=0;tail=1
        while head<tail:
            v=queue[head];head+=1
            for j in range(len(s)):
                if s[j] and not seen[j] and adj[v,j]:
                    seen[j]=1;queue[tail]=j;tail+=1
        largest=max(largest,tail)
    density=2.*edges/max(1,weight*(weight-1))
    span=hi-lo if weight else 0.
    return weight,largest,density,span

@njit(cache=True)
def specialist(H,L,s,prior,post,hard,e1,e2,cols,ptr,nfree,npair):
    # The expanded pool contains D2's pool: same raw syndrome, order and BP centre.
    best,masses,valid=search(H,L,s,post,prior,hard,nfree,npair)
    if not valid or not verify(best,s,cols,ptr): return e2.copy(),masses,False
    best_cost=0.
    for j in range(len(prior)): best_cost+=prior[j]*best[j]
    # A strictly better reference cannot already be in the enumerated pool.
    # Add it once; never impose reference bits or replace the original priors.
    for ref_id in range(2):
        ref=e1 if ref_id==0 else e2
        if not verify(ref,s,cols,ptr): continue
        cost=0.
        for j in range(len(prior)): cost+=prior[j]*ref[j]
        if cost<best_cost-1e-9:
            masses*=np.exp(cost-best_cost)
            masses[logical_label(L,ref)]+=1.
            best=ref.copy();best_cost=cost
    return best,masses,True

@njit(parallel=True,cache=True)
def batch(events,H,L,prior,cols,ptr,adj,times,fast_iterations,strong_iterations,nfree1,npair1,nfree2,npair2,nfree3,npair3,tau_conf,tau_weight,tau_component,force_disagreements):
    nshots=len(events);out=np.zeros((nshots,3,4),np.uint8)
    # weight, component, density, time span, c2, entropy2, BP2valid,
    # BP iterations, logical disagreement count, class margin, D1valid,
    # D2valid, routed, D3verified, c1
    diag=np.zeros((nshots,15))
    for shot in prange(nshots):
        s=events[shot] # Read-only throughout: no residual syndrome in this design.
        p1,h1,bp1,p2,h2,bp2,steps=paired_bp(s,prior,cols,ptr,fast_iterations,strong_iterations)
        e1,m1,v1=search(H,L,s,p1,prior,h1,nfree1,npair1)
        e2,m2,v2=search(H,L,s,p2,prior,h2,nfree2,npair2)
        v1=v1 and verify(e1,s,cols,ptr);v2=v2 and verify(e2,s,cols,ptr)
        y1,c1,_,_=summarise(m1,L,e1);y2,c2,entropy,margin=summarise(m2,L,e2)
        out[shot,0]=y1;out[shot,1]=y2;out[shot,2]=y2
        disagreement=0
        for j in range(4): disagreement+=y1[j]!=y2[j]
        weight,component,density,span=raw_features(s,adj,times)
        diag[shot,0]=weight;diag[shot,1]=component;diag[shot,2]=density;diag[shot,3]=span
        diag[shot,4]=c2;diag[shot,5]=entropy;diag[shot,6]=bp2;diag[shot,7]=steps
        diag[shot,8]=disagreement;diag[shot,9]=margin;diag[shot,10]=v1;diag[shot,11]=v2;diag[shot,14]=c1
        diagnostic=(not bp2) or c2<tau_conf
        complexity=weight>=tau_weight or component>=tau_component
        routed=(disagreement>0) and (force_disagreements or (diagnostic and complexity))
        if routed:
            diag[shot,12]=1
            e3,m3,v3=specialist(H,L,s,prior,p2,h2,e1,e2,cols,ptr,nfree3,npair3)
            diag[shot,13]=v3
            if v3: out[shot,2]=summarise(m3,L,e3)[0]
        # Every submission must contain bits; an invalid D2 is a hard failure.
        # Python wrapper raises, rather than silently submitting invalid candidates.
    return out,diag

BASE_CONFIG={'fast_iterations':8,'strong_iterations':60,'nfree1':32,'npair1':8,'nfree2':256,'npair2':64,'nfree3':1024,'npair3':128}
DEFAULT_GATE={'tau_conf':.5,'tau_weight':4.,'tau_component':3.}
CONFIG_PATH=Path(__file__).with_name('gate.json')
SAVED_CONFIG=json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else None
if SAVED_CONFIG is not None:
    BASE_CONFIG=SAVED_CONFIG['base_config']
GATE=SAVED_CONFIG['thresholds'] if SAVED_CONFIG is not None else DEFAULT_GATE
_CACHE={}
def model(circuit):
    key=str(circuit)
    if key not in _CACHE:
        H,L,p,c,ptr=problem(circuit)
        adj=(H.astype(np.int32)@H.T)>0;np.fill_diagonal(adj,False)
        coords=circuit.get_detector_coordinates()
        times=np.array([coords.get(i,[0,0,0])[-1] for i in range(H.shape[0])],float)
        _CACHE[key]=(H,L,p,c,ptr,adj,times)
    return _CACHE[key]

def predict(circuit,events,*,gate=None,force_disagreements=False):
    if gate is None and not force_disagreements and SAVED_CONFIG is None:
        raise RuntimeError('Missing gate.json; calibrate or supply the frozen configuration')
    data=np.asarray(events,dtype=np.uint8)
    if data.ndim!=2 or data.shape[1]!=70 or not np.isin(data,(0,1)).all(): raise ValueError('Expected binary shots x 70')
    packed,inverse=np.unique(np.packbits(data,axis=1),axis=0,return_inverse=True)
    unique=np.unpackbits(packed,axis=1)[:,:70].copy()
    out,diag=batch(unique,*model(circuit),**BASE_CONFIG,**(GATE if gate is None else gate),force_disagreements=force_disagreements)
    if not np.all(diag[:,11]): raise RuntimeError('D2 failed syndrome verification')
    return out[inverse],diag[inverse]

def decode(circuit,detector_events):
    return predict(circuit,detector_events)[0][:,2]

# Compile numerical kernels at import, outside timed decode calls.
batch(np.zeros((1,1),np.uint8),np.ones((1,1),np.uint8),np.zeros((4,1),np.uint8),np.ones(1),np.zeros(1,np.int64),np.array([0,1],np.int64),np.zeros((1,1),np.bool_),np.zeros(1),8,60,32,8,256,64,1024,128,.5,4.,3.,False)
