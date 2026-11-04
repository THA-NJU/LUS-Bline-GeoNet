import json,sys,hashlib,cv2,numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from data_v4 import targets
def one(p):
 p=Path(p);m=cv2.imread(str(p),0);assert m.shape==(512,512) and set(np.unique(m))<={0,1};t,r=targets(m);assert len(t)<=16
 return p.name,dict(targets=t,reliable=r,mask_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
if __name__=='__main__':
 root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.parent.mkdir(parents=True,exist_ok=True);meta={'queries':16,'definition':'six params, full normal width, linear LS over PCA cross-sections; train/val only'};audit={}
 for split in ['train','val']:
  paths=sorted((root/split/'masks').glob('*.png'));assert paths;assert {p.name for p in paths}=={p.name for p in (root/split/'images').glob('*.png')}
  with ProcessPoolExecutor(max_workers=8) as ex:meta[split]=dict(ex.map(one,map(str,paths),chunksize=32))
  v=list(meta[split].values());ious=[t['fit_iou'] for r in v for t in r['targets']];audit[split]=dict(images=len(v),accepted_instances=len(ious),images_with_targets=sum(bool(r['targets']) for r in v),fully_reliable=sum(r['reliable'] for r in v),fit_iou_mean=float(np.mean(ious)),fit_iou_quantiles=np.quantile(ious,[0,.25,.5,.75,1]).tolist());print(split,audit[split],flush=True)
 out.write_text(json.dumps(meta));out.with_suffix('.audit.json').write_text(json.dumps(audit,indent=2))
