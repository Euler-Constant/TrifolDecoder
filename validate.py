import json,time
from pathlib import Path
import numpy as np,stim
from decoder import predict,GATE,BASE_CONFIG
root=Path(__file__).resolve().parent
assert (root/'gate.json').exists(), 'Calibrate and freeze gate first'
rows=[];delta_all=[]
for basis in 'XZ':
 c=stim.Circuit.from_file(str(root/f'{basis}_circuit.stim'))
 # New seeds: not the tuning or previous held-out data.
 d,t=c.compile_detector_sampler(seed=710123+ord(basis)).sample(10000,separate_observables=True)
 original=d.copy();start=time.perf_counter();y,diag=predict(c,d)
 assert np.array_equal(d,original)
 repeat,_=predict(c,d[:64][::-1]);assert np.array_equal(repeat,y[:64][::-1])
 assert np.all(diag[:,11]) and np.all(diag[diag[:,12]>0,13]==1)
 assert np.all(diag[diag[:,8]==0,12]==0)
 assert np.array_equal(y[diag[:,12]==0,2],y[diag[:,12]==0,1])
 errors=(y!=t[:,None,:]).sum(axis=2)
 delta=(errors[:,1]-errors[:,2])/4;delta_all.extend(delta)
 route=diag[:,12]>0;agree=diag[:,8]==0
 row={'basis':basis,'shots':len(d),'d1_ler':float(errors[:,0].mean()/4),'d2_ler':float(errors[:,1].mean()/4),'hierarchical_ler':float(errors[:,2].mean()/4),'route_rate':float(route.mean()),'disagreement_rate':float((~agree).mean()),'recovered_bits':int(((y[:,1]!=t)&(y[:,2]==t)).sum()),'introduced_bits':int(((y[:,1]==t)&(y[:,2]!=t)).sum()),'d2_wrong_bits_on_agreement':int(errors[agree,1].sum()),'d2_wrong_bits_not_routed':int(errors[~route,1].sum()),'paired_improvement':float(delta.mean()),'paired_se':float(delta.std(ddof=1)/np.sqrt(len(d))),'seconds':time.perf_counter()-start,'invalid_d3':int(((diag[:,12]>0)&(diag[:,13]==0)).sum())}
 rows.append(row);np.savez_compressed(root/f'heldout_{basis}.npz',events=d,truth=t,predictions=y,diagnostics=diag)
 print(json.dumps(row),flush=True)
 (root/'validation_results.json').write_text(json.dumps({'basis_results':rows,'gate':GATE,'base_config':BASE_CONFIG},indent=2))
delta=np.array(delta_all);se=delta.std(ddof=1)/np.sqrt(len(delta))
summary={'basis_results':rows,'gate':GATE,'base_config':BASE_CONFIG,'d2_ler':sum(r['d2_ler'] for r in rows)/2,'hierarchical_ler':sum(r['hierarchical_ler'] for r in rows)/2,'paired_improvement':float(delta.mean()),'paired_95_ci':[float(delta.mean()-1.96*se),float(delta.mean()+1.96*se)]}
(root/'validation_results.json').write_text(json.dumps(summary,indent=2));print('SUMMARY',json.dumps(summary),flush=True)
