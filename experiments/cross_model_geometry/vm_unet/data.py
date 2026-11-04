import cv2,numpy as np,torch
from pathlib import Path
from torch.utils.data import Dataset
cv2.setNumThreads(1)
def targets(mask):
 n,lab,stats,_=cv2.connectedComponentsWithStats(mask.astype('uint8'),8);out=[];reliable=True
 for k in range(1,n):
  y,x=np.where(lab==k)
  if len(x)<10:reliable=False;continue
  pts=np.stack([x,y],1).astype(float);mean=pts.mean(0);_,vec=np.linalg.eigh(np.cov(pts.T));axis=vec[:,-1]
  if axis[1]<0:axis=-axis
  normal=np.array([axis[1],-axis[0]]);t=(pts-mean)@axis;v=(pts-mean)@normal;lo,hi=t.min(),t.max();length=hi-lo
  if length<20 or axis[1]<.5:reliable=False;continue
  knots=np.linspace(lo,hi,8);ws=[];ts=[]
  for z in knots:
   vv=v[np.abs(t-z)<=max(2,length/14)]
   if len(vv):ws.append(2*max(.5,np.abs(vv).max()+.5));ts.append((z-lo)/length)
  if len(ws)!=8:reliable=False;continue
  # Least squares linear FULL width along normal; endpoints remain PCA extrema.
  fitcoef=np.linalg.lstsq(np.stack([1-np.array(ts),ts],1),np.array(ws),rcond=None)[0];w0,w1=fitcoef
  if min(w0,w1)<1.022 or max(w0,w1)>255.5 or length/max(np.mean(ws),1)<2:reliable=False;continue
  p0=mean+lo*axis;p1=mean+hi*axis
  if (np.r_[p0,p1]<0).any() or (np.r_[p0,p1]>511).any():reliable=False;continue
  poly=np.stack([p0-w0/2*normal,p0+w0/2*normal,p1+w1/2*normal,p1-w1/2*normal]);fit=np.zeros_like(mask);cv2.fillPoly(fit,[np.rint(poly).astype('int32')],1);truth=lab==k
  iou=((fit>0)&truth).sum()/max(((fit>0)|truth).sum(),1)
  if iou<.65:reliable=False;continue
  out.append(dict(component=k,params=(np.r_[p0,p1,w0,w1]/511).tolist(),fit_iou=float(iou)))
 return out,reliable
class BlineDataset(Dataset):
 def __init__(self,root,split,meta,augment=False):self.root=Path(root)/split;self.files=sorted((self.root/'images').glob('*.png'));self.meta=meta[split];self.augment=augment
 def __len__(self):return len(self.files)
 def __getitem__(self,i):
  p=self.files[i];im=cv2.imread(str(p),0);mask=cv2.imread(str(self.root/'masks'/p.name),0);assert im.shape==mask.shape==(512,512) and set(np.unique(mask))<={0,1}
  m=self.meta[p.name];_,lab,_,_=cv2.connectedComponentsWithStats(mask,8);params=np.array([x['params'] for x in m['targets']],dtype='float32').reshape(-1,6);ims=np.array([cv2.resize((lab==x['component']).astype('uint8'),(64,64),interpolation=cv2.INTER_NEAREST) for x in m['targets']],dtype='float32').reshape(-1,64,64)
  if self.augment and np.random.rand()<.5:im=im[:,::-1].copy();mask=mask[:,::-1].copy();params[:,[0,2]]=1-params[:,[0,2]];ims=ims[:,:,::-1].copy()
  return torch.from_numpy(im.copy()[None]).float()/255,torch.from_numpy(mask.copy()[None]).float(),torch.from_numpy(params),torch.from_numpy(ims),m['reliable'],p.name
def collate(batch):return tuple(torch.stack([b[i] for b in batch]) if i<2 else [b[i] for b in batch] for i in range(6))
