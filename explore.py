import json,runpy,shutil,time
from pathlib import Path
import decoder
root=Path(__file__).resolve().parent
trials=[]
decoder.BASE_CONFIG['nfree3']=1024;decoder.BASE_CONFIG['npair3']=128
runpy.run_path(str(root/'calibrate.py'),run_name='__main__')
first=json.loads((root/'gate.json').read_text());trials.append(first)
for free,pairs in [(2048,256),(2390,384)]:
 decoder.BASE_CONFIG['nfree3']=free;decoder.BASE_CONFIG['npair3']=pairs
 runpy.run_path(str(root/'calibrate.py'),run_name='__main__')
 trial=json.loads((root/'gate.json').read_text());trials.append(trial)
 (root/f'gate_trial_{free}_{pairs}.json').write_text(json.dumps(trial,indent=2))
 for basis in 'XZ': shutil.copyfile(root/f'calibration_{basis}.npz',root/f'calibration_{basis}_{free}_{pairs}.npz')
# Pick accuracy first; ties prefer a smaller specialist and fewer routed shots.
best=min(trials,key=lambda r:(r['calibration_ler'],r['calibration_route_rate'],r['base_config']['npair3']))
(root/'gate.json').write_text(json.dumps(best,indent=2))
(root/'search_trials.json').write_text(json.dumps([{k:v for k,v in t.items() if k!='grid_results'} for t in trials],indent=2))
print('SELECTED',json.dumps({k:v for k,v in best.items() if k!='grid_results'}),flush=True)
