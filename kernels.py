"""Original partial-correction cascade and enlarged ordered-statistics search.
No third-party QEC decoder is used. Numba compiles our numerical loops.
"""
import numpy as np
from numba import njit, prange
from circuit_model import problem

@njit(cache=True)
def bp(s,prior,cols,ptr,iterations,damping,stop_on_valid):
    n=len(prior); ne=len(cols); m=len(ptr)-1
    v=prior[cols].copy(); msg=np.zeros(ne); post=prior.copy()
    hard=np.zeros(n,np.uint8); valid=False
    for it in range(iterations):
        for r in range(m):
            start=ptr[r]; end=ptr[r+1]
            prefix=np.ones(end-start+1); suffix=1.
            for k in range(start,end): prefix[k-start+1]=prefix[k-start]*np.tanh(v[k]*.5)
            for k in range(end-1,start-1,-1):
                prod=prefix[k-start]*suffix
                if s[r]: prod=-prod
                prod=max(-.999999999999,min(.999999999999,prod))
                msg[k]=2*np.arctanh(prod)
                suffix*=np.tanh(v[k]*.5)
        post=prior.copy()
        for k in range(ne): post[cols[k]]+=msg[k]
        for j in range(n): hard[j]=post[j]<0
        valid=True
        for r in range(m):
            val=0
            for k in range(ptr[r],ptr[r+1]): val^=hard[cols[k]]
            if val!=s[r]: valid=False; break
        if valid and stop_on_valid: break
        for k in range(ne):
            new=max(-40.,min(40.,post[cols[k]]-msg[k]))
            v[k]=(1.-damping)*new+damping*v[k]
    return post,hard,valid

@njit(cache=True)
def search(H,L,s,post,prior,hard,nfree,npair):
    """Unique free-variable candidates, with class probability sums and MAP."""
    m,n=H.shape; order=np.argsort(np.abs(post))
    A=H[:,order].copy(); b=s.copy()
    piv=np.empty(m,np.int64); rank=0
    for c in range(n):
        r=rank
        while r<m and A[r,c]==0: r+=1
        if r==m: continue
        if r!=rank:
            tmp=A[r].copy(); A[r]=A[rank]; A[rank]=tmp
            q=b[r]; b[r]=b[rank]; b[rank]=q
        for i in range(m):
            if i!=rank and A[i,c]: A[i]^=A[rank]; b[i]^=b[rank]
        piv[rank]=c; rank+=1
        if rank==m: break
    for i in range(rank,m):
        if b[i]: return hard.copy(),np.zeros(16),False
    x=hard[order].copy(); isp=np.zeros(n,np.uint8)
    for i in range(rank): x[piv[i]]=0; isp[piv[i]]=1
    for i in range(rank):
        val=b[i]
        for j in range(n): val^=A[i,j]&x[j]
        x[piv[i]]=val
    weights=prior[order]; labels=np.zeros(n,np.int64)
    for j in range(n):
        for o in range(L.shape[0]): labels[j]|=int(L[o,order[j]])<<o
    cost=0.; base_label=0
    for j in range(n):
        cost+=weights[j]*x[j]
        if x[j]: base_label^=labels[j]
    # Free variable offsets; each choice is a distinct full error vector.
    free=np.empty(nfree,np.int64); nf=0
    for j in range(n):
        if not isp[j] and np.any(A[:rank,j]):
            free[nf]=j; nf+=1
            if nf==nfree: break
    toggles=np.zeros(nf,np.int64)
    for a in range(nf):
        j=free[a]; toggles[a]=labels[j]
        for i in range(rank):
            if A[i,j]: toggles[a]^=labels[piv[i]]
    best_cost=cost; best_a=-1; best_b=-1
    masses=np.zeros(16); masses[base_label]=1.; reference=cost
    for a in range(nf):
        ja=free[a]
        for bb in range(-1,min(a,npair)):
            if bb>=0 and a>=npair: continue
            value=cost+weights[ja]*(1-2*int(x[ja])); label=base_label^toggles[a]
            if bb>=0:
                jb=free[bb]; value+=weights[jb]*(1-2*int(x[jb])); label^=toggles[bb]
            for i in range(rank):
                flip=A[i,ja]
                if bb>=0: flip^=A[i,free[bb]]
                if flip:
                    k=piv[i]; value+=weights[k]*(1-2*int(x[k]))
            if value<best_cost: best_cost=value; best_a=a; best_b=bb
            if value<reference:
                masses*=np.exp(value-reference); reference=value
            masses[label]+=np.exp(reference-value)
    best=x.copy()
    if best_a>=0:
        j=free[best_a]; best[j]^=1
        for i in range(rank): best[piv[i]]^=A[i,j]
        if best_b>=0:
            j=free[best_b]; best[j]^=1
            for i in range(rank): best[piv[i]]^=A[i,j]
    out=np.empty(n,np.uint8)
    for j in range(n): out[order[j]]=best[j]
    return out,masses,True

@njit(cache=True)
def consistent(H,e,s):
    for i in range(H.shape[0]):
        val=0
        for j in range(H.shape[1]): val^=H[i,j]&e[j]
        if val!=s[i]: return False
    return True

