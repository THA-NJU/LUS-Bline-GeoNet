import torch
from torch import nn
from torch.nn import functional as F
from .geometry import RibbonNet as GeometryBase, losses
from .rolling import RollingUNet
from .raster import RibbonRaster
class BGeoNet(GeometryBase):
 def __init__(self,queries=16):
  super().__init__(queries)
  del self.enc,self.dec,self.pixel
  self.backbone=RollingUNet(in_channels=1,num_classes=1,img_size=512,deep_supervision=False)
  self.feature_adapter=nn.Conv2d(8,32,1)
  yy,xx=torch.meshgrid(torch.linspace(-1,1,16),torch.linspace(-1,1,16),indexing="ij")
  self.register_buffer("position_grid",torch.stack([xx,yy])[None],persistent=False)
  self.raster=RibbonRaster(128)
 def _apply(self,fn,recurse=True):
  super()._apply(fn,recurse)
  device=self.position_grid.device
  yy,xx=torch.meshgrid(torch.linspace(-1,1,16,device=device),torch.linspace(-1,1,16,device=device),indexing='ij')
  self.position_grid=torch.stack([xx,yy])[None]
  return self
 def forward(self,image):
  x,features,pixel=self.backbone(image)
  b=x.shape[0];pos=self.position_grid
  memory=(F.adaptive_avg_pool2d(x,(16,16))+self.position(pos)).flatten(2).transpose(1,2);q=self.query[None].expand(b,-1,-1);q=q+self.cross(q,memory,memory,need_weights=False)[0];q=q+self.self_attn(q,q,q,need_weights=False)[0];q=q+self.ff(self.norm(q))
  z=self.shape(q).float();y0=.95*z[...,1].sigmoid();y1=y0+(1-y0)*(.01+.99*z[...,3].sigmoid());params=torch.cat([torch.stack([z[...,0].sigmoid(),y0,z[...,2].sigmoid(),y1],-1),.002+.498*z[...,4:].sigmoid()],-1);obj=self.object(q).squeeze(-1).float()
  masks=self.raster(params);shape=1-(1-masks*obj.sigmoid()[...,None,None]).prod(1,keepdim=True)
  x=self.feature_adapter(features);shape=F.interpolate(shape,size=image.shape[-2:],mode='bilinear',align_corners=False);final=pixel+self.alpha*self.refine(torch.cat([x,pixel.sigmoid(),shape.to(x.dtype)],1))
  return dict(logits=final,pixel=pixel,params=params,objects=obj,geometry=shape)
