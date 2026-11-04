"""S2DENet native forward plus the unchanged V6 ab01 query/ribbon/refinement graph."""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).parent/'vendor'))
from s2denet import S2DENet
from model_v41 import RibbonNet as GeometryBase, ribbon
MODEL_NAMES=('s2denet',)
class GeometryAdapter(GeometryBase):
    def __init__(self,name='s2denet',queries=16):
        assert name=='s2denet' and queries==16
        super().__init__(queries)
        del self.enc,self.dec,self.pixel
        self.model_name=name
        self.initialization='random_only_no_pretrained_weights'
        self.backbone=S2DENet(in_channels=8,num_class=1,return_edge=True)
        self.deep_adapter=nn.Conv2d(128,256,1)
        self.feature_adapter=nn.Conv2d(8,32,1)
        self.alpha=1.
        self.last_feature_shapes={}
        self.register_buffer('input_mean',torch.tensor(.100638).reshape(1,1,1,1),persistent=False)
        self.register_buffer('input_std',torch.tensor(.145868).reshape(1,1,1,1),persistent=False)
        assert torch.count_nonzero(self.refine[-1].weight)==0
        assert torch.count_nonzero(self.refine[-1].bias)==0
    def preprocess(self,image):
        assert image.ndim==4 and image.shape[1:]==(1,512,512)
        return (image.repeat(1,3,1,1)-self.input_mean)/self.input_std
    def _native_features(self,image):
        taps={}
        def enc_hook(module,inputs,output):
            taps['deep']=torch.cat((output[3],output[7]),1)
        def head_pre(module,inputs):taps['feature']=inputs[0]
        def head_post(module,inputs,output):taps['mask_head_logits']=output
        handles=[self.backbone.encoder.register_forward_hook(enc_hook),self.backbone.decoder.head.register_forward_pre_hook(head_pre),self.backbone.decoder.head.register_forward_hook(head_post)]
        try:probability,edge=self.backbone(self.preprocess(image))
        finally:
            for h in handles:h.remove()
        deep,feature=taps['deep'],taps['feature']
        assert deep.shape[1:]==(128,64,64) and feature.shape[1:]==(8,512,512)
        # This is exactly the native pre-sigmoid sum, before the FP32 loss cast.
        pixel=(taps['mask_head_logits']+edge).float()
        self.last_feature_shapes={'deep':tuple(deep.shape),'decoder':tuple(feature.shape)}
        return deep,feature,pixel,edge,probability
    def forward(self, image):
        deep, feature, pixel, edge, native_probability = self._native_features(image)
        x = self.deep_adapter(deep)
        b = x.shape[0]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, 16, device=x.device),
            torch.linspace(-1, 1, 16, device=x.device), indexing="ij",
        )
        pos = torch.stack([xx, yy])[None]
        memory = (F.adaptive_avg_pool2d(x, (16, 16)) + self.position(pos)).flatten(2).transpose(1, 2)
        q = self.query[None].expand(b, -1, -1)
        q = q + self.cross(q, memory, memory, need_weights=False)[0]
        q = q + self.self_attn(q, q, q, need_weights=False)[0]
        q = q + self.ff(self.norm(q))
        z = self.shape(q).float()
        y0 = 0.95 * z[..., 1].sigmoid()
        y1 = y0 + (1 - y0) * (0.01 + 0.99 * z[..., 3].sigmoid())
        params = torch.cat([
            torch.stack([z[..., 0].sigmoid(), y0, z[..., 2].sigmoid(), y1], -1),
            0.002 + 0.498 * z[..., 4:].sigmoid(),
        ], -1)
        obj = self.object(q).squeeze(-1).float()
        masks = ribbon(params)
        shape = 1 - (1 - masks * obj.sigmoid()[..., None, None]).prod(1, keepdim=True)
        shape = F.interpolate(shape, size=image.shape[-2:], mode="bilinear", align_corners=False)
        features = F.interpolate(
            self.feature_adapter(feature), size=image.shape[-2:], mode="bilinear", align_corners=False,
        )
        residual = self.refine(torch.cat([
            features, pixel.sigmoid().to(features.dtype), shape.to(features.dtype),
        ], 1))
        final = pixel + self.alpha * residual
        return dict(logits=final, pixel=pixel, params=params, objects=obj, geometry=shape, native_edge=edge.float(), native_probability=native_probability.float())


def build_model(name):
    return GeometryAdapter(name)
