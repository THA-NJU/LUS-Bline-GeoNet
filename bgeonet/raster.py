"""FP32 differentiable rasterization on a fixed Cartesian grid."""
import torch
from torch import nn
class RibbonRaster(nn.Module):
 def __init__(self,size=128):
  super().__init__();self.size=size
  yy,xx=torch.meshgrid(torch.linspace(0,1,size),torch.linspace(0,1,size),indexing='ij')
  self.register_buffer('grid',torch.stack([xx,yy],-1)[None,None],persistent=False)
 def _apply(self,fn,recurse=True):
  super()._apply(fn,recurse)
  yy,xx=torch.meshgrid(torch.linspace(0,1,self.size,device=self.grid.device),torch.linspace(0,1,self.size,device=self.grid.device),indexing='ij')
  self.grid=torch.stack([xx,yy],-1)[None,None]
  return self
 def forward(self,params):
  size=self.size
  with torch.autocast(device_type=params.device.type,enabled=False):
   z=params.float();p0=z[...,:2];d=z[...,2:4]-p0;length=d.square().sum(-1).clamp_min(1e-8).sqrt();u=d/length[...,None];n=torch.stack([u[...,1],-u[...,0]],-1)
   v=self.grid-p0[...,None,None,:]
   along=(v*u[...,None,None,:]).sum(-1);across=(v*n[...,None,None,:]).sum(-1);t=(along/length[...,None,None]).clamp(0,1)
   width=((1-t)*z[...,4,None,None]+t*z[...,5,None,None])/2;tau=1./size
   return torch.sigmoid(along/tau)*torch.sigmoid((length[...,None,None]-along)/tau)*torch.sigmoid((across+width)/tau)*torch.sigmoid((width-across)/tau)
