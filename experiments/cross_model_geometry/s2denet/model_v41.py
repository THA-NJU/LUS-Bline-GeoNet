"""Cartesian B-line segment endpoints + sampled left/right widths; no prompts or fan geometry."""
import torch
from torch import nn
from torch.nn import functional as F
class Block(nn.Module):
 def __init__(self,a,b):
  super().__init__();self.f=nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.GroupNorm(8,b),nn.GELU(),nn.Conv2d(b,b,3,padding=1,bias=False),nn.GroupNorm(8,b));self.skip=nn.Conv2d(a,b,1) if a!=b else nn.Identity()
 def forward(self,x):return F.gelu(self.f(x)+self.skip(x))
def ribbon(params,size=128):
 with torch.autocast(device_type=params.device.type,enabled=False):
  z=params.float();p0=z[...,:2];d=z[...,2:4]-p0;length=d.square().sum(-1).clamp_min(1e-8).sqrt();u=d/length[...,None];n=torch.stack([u[...,1],-u[...,0]],-1)
  yy,xx=torch.meshgrid(torch.linspace(0,1,size,device=z.device),torch.linspace(0,1,size,device=z.device),indexing='ij');v=torch.stack([xx,yy],-1)[None,None]-p0[...,None,None,:]
  along=(v*u[...,None,None,:]).sum(-1);across=(v*n[...,None,None,:]).sum(-1);t=(along/length[...,None,None]).clamp(0,1)
  width=((1-t)*z[...,4,None,None]+t*z[...,5,None,None])/2;tau=1./size
  return torch.sigmoid(along/tau)*torch.sigmoid((length[...,None,None]-along)/tau)*torch.sigmoid((across+width)/tau)*torch.sigmoid((width-across)/tau)
class RibbonNet(nn.Module):
 def __init__(self,queries=16):
  super().__init__();self.queries=queries;self.alpha=1.0
  self.enc=nn.ModuleList([Block(1,32),Block(32,64),Block(64,128),Block(128,256)]);self.dec=nn.ModuleList([Block(384,128),Block(192,64),Block(96,32)])
  self.pixel=nn.Conv2d(32,1,1);self.query=nn.Parameter(torch.randn(queries,256)*.02);self.position=nn.Conv2d(2,256,1)
  self.cross=nn.MultiheadAttention(256,4,batch_first=True);self.self_attn=nn.MultiheadAttention(256,4,batch_first=True);self.norm=nn.LayerNorm(256);self.ff=nn.Sequential(nn.Linear(256,512),nn.GELU(),nn.Linear(512,256));self.shape=nn.Linear(256,6);self.object=nn.Linear(256,1)
  nn.init.constant_(self.object.bias,-2.);nn.init.constant_(self.shape.bias,0.);self.shape.bias.data[4:]=-2.
  self.refine=nn.Sequential(Block(34,32),nn.Conv2d(32,1,1));nn.init.zeros_(self.refine[-1].weight);nn.init.zeros_(self.refine[-1].bias)
 def forward(self,image):
  f=[];x=image
  for i,layer in enumerate(self.enc):
   if i:x=F.avg_pool2d(x,2)
   x=layer(x);f.append(x)
  b=x.shape[0];yy,xx=torch.meshgrid(torch.linspace(-1,1,16,device=x.device),torch.linspace(-1,1,16,device=x.device),indexing='ij');pos=torch.stack([xx,yy])[None]
  memory=(F.adaptive_avg_pool2d(x,(16,16))+self.position(pos)).flatten(2).transpose(1,2);q=self.query[None].expand(b,-1,-1);q=q+self.cross(q,memory,memory,need_weights=False)[0];q=q+self.self_attn(q,q,q,need_weights=False)[0];q=q+self.ff(self.norm(q))
  z=self.shape(q).float();y0=.95*z[...,1].sigmoid();y1=y0+(1-y0)*(.01+.99*z[...,3].sigmoid());params=torch.cat([torch.stack([z[...,0].sigmoid(),y0,z[...,2].sigmoid(),y1],-1),.002+.498*z[...,4:].sigmoid()],-1);obj=self.object(q).squeeze(-1).float()
  masks=ribbon(params);shape=1-(1-masks*obj.sigmoid()[...,None,None]).prod(1,keepdim=True)
  for layer,skip in zip(self.dec,f[-2::-1]):x=layer(torch.cat([F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip],1))
  pixel=self.pixel(x);shape=F.interpolate(shape,size=image.shape[-2:],mode='bilinear',align_corners=False);final=pixel+self.alpha*self.refine(torch.cat([x,pixel.sigmoid(),shape.to(x.dtype)],1))
  return dict(logits=final,pixel=pixel,params=params,objects=obj,geometry=shape)
def segmentation(logits,target,A=1.):
 p=logits.float().sigmoid();return A*F.binary_cross_entropy_with_logits(logits.float(),target)+(1-(2*(p*target).sum((1,2,3))+1)/((p+target).sum((1,2,3))+1)).mean()
def losses(result,target,truth,instance_masks,reliable,A=1.,B=1.):
 from scipy.optimize import linear_sum_assignment
 seg=segmentation(result['logits'],target,A);pix=segmentation(result['pixel'],target,A);shape=seg*0;par=seg*0;obj=seg*0;smooth=seg*0
 pred=result['params'].float();scores=result['objects'];rm=ribbon(pred,64)
 for b,t in enumerate(truth):
  t=t.to(pred.device);tm=instance_masks[b].to(pred.device);selected=torch.zeros_like(scores[b],dtype=torch.bool);labels=torch.zeros_like(scores[b])
  if len(t):
   if len(t)>pred.shape[1]:raise RuntimeError('Query capacity exceeded; no silent truncation')
   pp=rm[b].flatten(1);tt=tm.flatten(1);dc=1-(2*(pp@tt.T)+1)/(pp.sum(1)[:,None]+tt.sum(1)[None]+1)
   cost=2*(pred[b,:,None,:4]-t[None,:,:4]).abs().mean(-1)+(pred[b,:,None,4:]-t[None,:,4:]).abs().mean(-1)+dc-.25*scores[b].sigmoid()[:,None]
   ii,jj=linear_sum_assignment(cost.detach().cpu().numpy());ii=torch.as_tensor(ii,device=pred.device);jj=torch.as_tensor(jj,device=pred.device);labels[ii]=1;selected[ii]=True
   p=pred[b,ii];gt=t[jj];pm=rm[b,ii];gm=tm[jj];shape=shape+(1-(2*(pm*gm).sum((1,2))+1)/((pm+gm).sum((1,2))+1)).mean()
   direction=1-F.cosine_similarity(p[:,2:4]-p[:,:2],gt[:,2:4]-gt[:,:2],dim=-1).mean()
   par=par+F.smooth_l1_loss(p[:,:4],gt[:,:4])+F.smooth_l1_loss(p[:,4:],gt[:,4:])+.25*direction
  if reliable[b]:selected[:]=True
  if selected.any():obj=obj+F.binary_cross_entropy_with_logits(scores[b,selected],labels[selected])
 n=len(truth);terms=dict(seg=seg,pixel=pix,shape=shape/n,params=par/n,object=obj/n,smooth=smooth/n)
 total=seg+B*(.5*pix+.3*terms['shape']+.2*terms['params']+.1*terms['object']);return total,terms


def consistency_loss(result,reliable):
 # Only fully reliable images; the geometric target side is detached.
 p=result['logits'].float().sigmoid()
 g=result['geometry'].float().detach()
 use=torch.as_tensor(reliable,device=p.device,dtype=torch.bool)
 if use.any():return (p[use]-g[use]).square().mean()
 return p.sum()*0.

def consistency_weight(epoch,target):
 return target*max(0.,min(1.,(epoch-10)/10.))
