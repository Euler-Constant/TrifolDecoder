import argparse,hashlib,json,time
from pathlib import Path
import numpy as np,pandas as pd,stim,numba
from decoder import decode,predict,GATE,BASE_CONFIG
p=argparse.ArgumentParser();p.add_argument('--test',required=True,type=Path);p.add_argument('--output',default='submission_hierarchical.csv',type=Path)
a=p.parse_args();root=Path(__file__).resolve().parent
assert (root/'gate.json').exists()
frame=pd.read_csv(a.test);obs=[f'L{i}' for i in range(4)];cols=[f'D{i}' for i in range(70)]
assert frame.id.is_unique and frame.basis.isin(['X','Z']).all() and frame[cols].isin([0,1]).all().all()
answer=pd.DataFrame({'id':frame.id})
for j in obs: answer[j]=np.uint8(0)
meta={'architecture':'raw_syndrome_hierarchical_trust','threads':numba.get_num_threads(),'gate':GATE,'base_config':BASE_CONFIG,'rows':len(frame),'basis':{},'source_hashes':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['decoder.py','kernels.py','circuit_model.py','gate.json']}}
for basis in 'XZ':
 mask=frame.basis.eq(basis);events=frame.loc[mask,cols].to_numpy(dtype=np.uint8)
 c=stim.Circuit.from_file(str(root/f'{basis}_circuit.stim'));start=time.perf_counter()
 packed,inverse=np.unique(np.packbits(events,axis=1),axis=0,return_inverse=True)
 unique=np.unpackbits(packed,axis=1)[:,:70].copy()
 predicted=np.empty((len(unique),4),np.uint8);diags=np.empty((len(unique),15))
 print(basis,'unique patterns',len(unique),'threads',numba.get_num_threads(),flush=True)
 for lo in range(0,len(unique),4096):
  hi=min(lo+4096,len(unique));original=unique[lo:hi].copy()
  y,di=predict(c,unique[lo:hi]);assert np.array_equal(original,unique[lo:hi])
  predicted[lo:hi]=y[:,2];diags[lo:hi]=di
  print(basis,hi,'/',len(unique),'elapsed',round(time.perf_counter()-start,1),'s',flush=True)
 predictions=predicted[inverse];diag=diags[inverse]
 assert np.array_equal(decode(c,events[:128][::-1]),predictions[:128][::-1])
 assert predictions.shape==(len(events),4) and np.isin(predictions,[0,1]).all()
 answer.loc[mask,obs]=predictions
 meta['basis'][basis]={'seconds':time.perf_counter()-start,'shots':len(events),'route_rate':float(diag[:,12].mean()),'disagreement_rate':float((diag[:,8]>0).mean()),'invalid_d3_fallbacks':int(((diag[:,12]>0)&(diag[:,13]==0)).sum())}
 print(basis,'COMPLETE',json.dumps(meta['basis'][basis]),flush=True)
assert answer.id.equals(frame.id) and answer[obs].isin([0,1]).all().all()
a.output.parent.mkdir(parents=True,exist_ok=True);answer.to_csv(a.output,index=False)
meta['csv_sha256']=hashlib.sha256(a.output.read_bytes()).hexdigest()
a.output.with_suffix('.metadata.json').write_text(json.dumps(meta,indent=2));print('VERIFIED',a.output,answer.shape,flush=True)
