import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,stim
from decoder import decode
p=argparse.ArgumentParser();p.add_argument('--test',required=True,type=Path);p.add_argument('--submission',required=True,type=Path)
a=p.parse_args();root=Path(__file__).resolve().parent
frame=pd.read_csv(a.test);out=pd.read_csv(a.submission);obs=[f'L{i}' for i in range(4)]
assert list(out.columns)==['id',*obs] and out.id.equals(frame.id) and out.id.is_unique and out[obs].isin([0,1]).all().all()
indexed=out.set_index('id');count=0
for basis in 'XZ':
 rows=frame[frame.basis.eq(basis)].iloc[:256];c=stim.Circuit.from_file(str(root/f'{basis}_circuit.stim'))
 y=decode(c,rows[[f'D{i}' for i in range(70)]].to_numpy(dtype=np.uint8))
 assert np.array_equal(y,indexed.loc[rows.id,obs].to_numpy(dtype=np.uint8))
 count+=len(rows);print(basis,'reproduced',len(rows),'rows',flush=True)
meta=json.loads(a.submission.with_suffix('.metadata.json').read_text())
assert meta['csv_sha256']==hashlib.sha256(a.submission.read_bytes()).hexdigest()
for name,sha in meta['source_hashes'].items(): assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
(root/'csv_checks.json').write_text(json.dumps({'structure_passed':True,'rows':len(out),'fresh_subset_matches':count,'hashes_passed':True},indent=2))
print('PASS: CSV format, original IDs, binary outputs, source hashes, and subset reproduction')
