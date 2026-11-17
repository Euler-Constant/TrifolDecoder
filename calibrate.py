import json,time
from pathlib import Path
import numpy as np,stim
from decoder import predict,BASE_CONFIG
root=Path(__file__).resolve().parent;all_y=[];all_d=[];all_t=[];timings={}
for basis in 'XZ':
 a=np.load(root/f'calibration_input_{basis}.npz')
 d=a['events'];truth=a['truth']
 c=stim.Circuit.from_file(str(root/f'{basis}_circuit.stim'))
 before=d.copy();start=time.perf_counter();y,diag=predict(c,d,force_disagreements=True)
 assert np.array_equal(d,before)
 assert np.array_equal(y[:,1],a['predictions'][:,1]),'D2 differs from validated control'
 timings[basis]=time.perf_counter()-start
 np.savez_compressed(root/f'calibration_{basis}.npz',events=d,truth=truth,predictions=y,diagnostics=diag)
 all_y.append(y);all_d.append(diag);all_t.append(truth)
 print(basis,'D1',np.mean(y[:,0]!=truth),'D2',np.mean(y[:,1]!=truth),'D3-on-disagreement',np.mean(y[:,2]!=truth),'route',diag[:,12].mean(),'seconds',timings[basis],flush=True)
y=np.concatenate(all_y);f=np.concatenate(all_d);truth=np.concatenate(all_t)
base=np.mean(y[:,1]!=truth);records=[]
for conf in [.05,.15,.3,.5,.75,1.01]:
 for weight in [1.,4.,8.,12.]:
  for component in [2.,4.,8.]:
   route=(f[:,8]>0)&((f[:,6]==0)|(f[:,4]<conf))&((f[:,0]>=weight)|(f[:,1]>=component))
   selected=np.where(route[:,None],y[:,2],y[:,1]);ler=float(np.mean(selected!=truth))
   records.append({'thresholds':{'tau_conf':conf,'tau_weight':weight,'tau_component':component},'ler':ler,'route_rate':float(route.mean())})
# Accuracy first; ties prefer fewer D3 calls, then deterministic threshold order.
best=min(records,key=lambda r:(r['ler'],r['route_rate']))
report={'thresholds':best['thresholds'],'calibration_ler':best['ler'],'d2_calibration_ler':float(base),'calibration_route_rate':best['route_rate'],'calibration_shots':len(y),'base_config':BASE_CONFIG,'timing_seconds':timings,'selection':'minimum calibration bit error, tie-break lower routing rate','grid_results':records}
(root/'gate.json').write_text(json.dumps(report,indent=2))
print('FROZEN GATE',json.dumps({k:v for k,v in report.items() if k!='grid_results'}),flush=True)
