import json,time
from pathlib import Path
import numpy as np,stim,numba
from decoder import predict
root=Path(__file__).resolve().parent
fixtures=[]
for basis in 'XZ':
 a=np.load(root/f'heldout_{basis}.npz')
 fixtures.append((stim.Circuit.from_file(str(root/f'{basis}_circuit.stim')),a['events'][:256]))
results={};reference=None
maximum=min(8,numba.config.NUMBA_NUM_THREADS)
for threads in [1,maximum]:
 numba.set_num_threads(threads)
 # Prime both circuit models and kernels outside measurements.
 for c,d in fixtures: predict(c,d[:1])
 times=[]
 for repeat in range(3):
  start=time.perf_counter();outs=[predict(c,d) for c,d in fixtures];times.append(time.perf_counter()-start)
  if reference is None: reference=outs
  for (y,di),(yr,dr) in zip(outs,reference):
   assert np.array_equal(y,yr) and np.array_equal(di,dr)
 results[str(threads)]={'median_seconds':float(np.median(times)),'runs':times,'shots':512}
numba.set_num_threads(maximum)
report={'results':results,'speedup':results['1']['median_seconds']/results[str(maximum)]['median_seconds'],'thread_invariant_predictions':True,'note':'local warmed benchmark; excludes imports; same 512 shots; no concurrent decoding job'}
(root/'thread_benchmark.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
