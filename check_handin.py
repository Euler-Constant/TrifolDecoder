import time
from pathlib import Path
import numpy as np,stim
from decoder import decode
elapsed=0.;root=Path(__file__).resolve().parent
for b in 'XZ':
 c=stim.Circuit.from_file(str(root/f'{b}_circuit.stim'))
 d,t=c.compile_detector_sampler(seed=1).sample(100000,separate_observables=True)
 start=time.perf_counter();y=decode(c,d);elapsed+=time.perf_counter()-start
 assert y.shape==t.shape and np.isin(y,[0,1]).all()
 print(b,'LER',float(np.mean(y!=t)),flush=True)
print('Full decode seconds:',elapsed)
