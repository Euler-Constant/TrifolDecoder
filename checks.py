import itertools,json,time
from pathlib import Path
import numpy as np,numba,stim
from kernels import search,bp
from decoder import paired_bp,verify,predict,model,BASE_CONFIG,GATE
root=Path(__file__).resolve().parent
H=np.array([[1,0,0,1,1],[0,1,0,1,0],[0,0,1,0,1]],np.uint8)
L=np.array([[1,0,1,0,0],[0,1,0,0,1],[0,0,0,1,0],[1,0,0,0,1]],np.uint8)
p=np.array([.11,.07,.16,.09,.04]);w=np.log((1-p)/p)
es=np.array(list(itertools.product((0,1),repeat=5)),np.uint8)
for bits in itertools.product((0,1),repeat=3):
 s=np.array(bits,np.uint8);feasible=es[np.all((es@H.T)%2==s,axis=1)];cost=feasible@w
 best,mass,valid=search(H,L,s,w,w,np.zeros(5,np.uint8),5,5)
 assert valid and np.array_equal((H@best)%2,s) and abs(best@w-cost.min())<1e-10
 exact=np.zeros(16)
 for e,c in zip(feasible,cost):
  label=sum(int(b)<<i for i,b in enumerate((L@e)%2));exact[label]+=np.exp(-c)
 assert np.allclose(mass/mass.sum(),exact/exact.sum())
print('PASS: exact tiny-problem costs and class probability sums',flush=True)
for basis in 'XZ':
 c=stim.Circuit.from_file(str(root/f'{basis}_circuit.stim'))
 d,_=c.compile_detector_sampler(seed=44).sample(40,separate_observables=True)
 H,L,prior,cols,ptr,_,_=model(c)
 # Check that the shared computation preserves two independent BP runs.
 for s in d[:8]:
  a,b,va,p2,h2,v2,_=paired_bp(s,prior,cols,ptr,8,60)
  ap,bp_,av=bp(s,prior,cols,ptr,8,.2,True)
  pp,hh,vv=bp(s,prior,cols,ptr,60,.2,True)
  assert np.array_equal(a,ap) and np.array_equal(b,bp_) and va==av
  assert np.array_equal(p2,pp) and np.array_equal(h2,hh) and v2==vv
 before=d.copy();a,diag=predict(c,d);b,bd=predict(c,d[::-1])
 assert np.array_equal(d,before) and np.array_equal(a,b[::-1]) and np.array_equal(diag,bd[::-1])
 expected=(diag[:,8]>0)&((diag[:,6]==0)|(diag[:,4]<GATE['tau_conf']))&((diag[:,0]>=GATE['tau_weight'])|(diag[:,1]>=GATE['tau_component']))
 assert np.array_equal(diag[:,12].astype(bool),expected)
 assert np.array_equal(a[~expected,2],a[~expected,1])
 assert np.all(diag[expected,13]==1)
 print(basis,'PASS: raw passthrough, shared-BP equivalence, ordering, routing, fallback selection',flush=True)
print('PASS: all hierarchy checks',flush=True)
