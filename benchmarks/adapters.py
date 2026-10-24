"""Frozen native architecture/input adapters. No training or test-time tuning."""
import sys, importlib.util, importlib, inspect
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from .paths import ROOT, EXTERNAL
R=E=ER=ROOT
APRIL=EXTERNAL/'april_medseg'
sys.path[:0]=[str(ROOT),str(APRIL)]
_load=torch.load
def safe_load(*a,**kw):
    kw['weights_only']=True
    return _load(*a,**kw)
torch.load=safe_load
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m
    spec.loader.exec_module(m);return m
def allow_yolo(path,names):
    # Only known installed neural-network classes, not arbitrary checkpoint callables.
    import ultralytics, collections
    allowed=[set,collections.OrderedDict]
    for name in names:
        if name=='builtins.set':continue
        if not name.startswith(('torch.nn.modules.','ultralytics.nn.')):raise RuntimeError('Unreviewed checkpoint global '+name)
        mod,attr=name.rsplit('.',1);obj=getattr(importlib.import_module(mod),attr)
        if not inspect.isclass(obj) or not issubclass(obj,torch.nn.Module):raise RuntimeError(name)
        allowed.append(obj)
    torch.serialization.add_safe_globals(allowed)
    return names
class Adapter:
    def __init__(self,job):
        self.job=job; n=job['model']; self.name=n;self.kind='logits';self.amp=job['precision']=='bf16'
        self.norm=job['preprocessing'];self.size=512;self.channels=3;self.yolo=None
        if n.startswith('yolo'):
            self.safe_globals=allow_yolo(job['checkpoint'],job['safe_globals'])
            from ultralytics import YOLO
            self.yolo=YOLO(job['checkpoint']);self.model=self.yolo.model;return
        c=safe_load(job['checkpoint'],map_location='cpu')
        if n.startswith('monai_'):
            build_monai_model=module("build_monai",E/"evaluation/families/evaluate_monai_family.py").build_monai_model
            m,_=build_monai_model(n,512)
        elif n in ['mamba_unet','vm_unet_v2','nnmamba_2d','swin_umamba']:
            build_mamba_model=module("build_mamba",E/"evaluation/families/evaluate_mamba_family.py").build_mamba_model
            m,_=build_mamba_model(n,512,APRIL)
        elif n in ['mednext','pvtb2_emcad','rolling_unet']:
            build_model=module("build_extra",E/"evaluation/families/evaluate_mednext_pvtb2_rolling.py").build_model
            m,_=build_model(n,APRIL)
        elif n in ['u_rwkv','rwkv_unet']:
            native=module("build_rwkv",E/"evaluation/families/evaluate_rwkv_family.py")
            native.configure_isolated_extension_cache(n,512,Path(__file__).parent)
            build_rwkv_model=native.build_rwkv_model
            m,_=build_rwkv_model(n,512,APRIL)
        elif n=='segformer_b2':
            from transformers import SegformerConfig,SegformerForSemanticSegmentation
            cfg=SegformerConfig(depths=[3,4,6,3],hidden_sizes=[64,128,320,512],decoder_hidden_size=768,num_labels=2)
            class Wrap(torch.nn.Module):
                def __init__(self):super().__init__();self.net=SegformerForSemanticSegmentation(cfg)
                def forward(self,x):return self.net(pixel_values=x).logits
            m=Wrap()
        elif n in ['fpn_resnet34','deeplabv3plus_resnet34']:
            native=module('legacy_resnet',E/'evaluation/legacy_resnet.py')
            m,_=native.build_model('FPN' if n=='fpn_resnet34' else 'DeepLabV3Plus',3,2)
        elif n=='s2denet':
            native=module('s2_native',ER/'experiments/s2denet/train.py')
            m=native.import_official_model(EXTERNAL/'S2DENet')(in_channels=8,num_class=1,return_edge=True)
            self.kind='probability'
        elif n=='usfm':
            sys.path.insert(0,str(EXTERNAL/'USFM'))
            native=importlib.import_module('benchmarks.usfm_native')
            m=native.USFMModel(num_classes=3);self.size=224
        elif n in ['sam2','medsam','samus','sam3']:
            native=importlib.import_module('benchmarks.'+n+'_native');self.kind='sigmoid'
            if n=='samus':
                raw=native.import_samus_registry(EXTERNAL/'SAMUS')[0]['vit_b'](args=SimpleNamespace(encoder_input_size=256),checkpoint=None)
                m=native.SAMUSFixedFullBox(raw,256);self.size=256;self.channels=1
            elif n=='medsam':
                raw=native.import_medsam_registry(EXTERNAL/'MedSAM')[0]['vit_b'](checkpoint=str(EXTERNAL/'weights/medsam_vit_b.pth'))
                m=native.MedSAMFixedFullBox(raw,1024);self.size=1024
            elif n=='sam2':
                from sam2.build_sam import build_sam2
                raw=build_sam2(c['args']['model_cfg'],ckpt_path=None,device='cpu')
                m=native.SAM2FixedFullBox(raw);self.size=1024
            else:
                m,_=native.build_sam3_subset(EXTERNAL/'sam3',EXTERNAL/'weights/sam3.pt',torch.device('cpu'),1008)
                native.load_model_delta(m,c['model_delta'],c['train_scope']);self.size=1008
            if n!='sam3':
                state=c['model']
                # Saved benchmark wrappers carry sam./samus. prefixes.
                if set(state)==set(m.state_dict()):m.load_state_dict(state,strict=True)
                else:raw.load_state_dict(state,strict=True)
            self.model=m.eval();return
        else:
            native=module('april_native',R/'experiments/april/train_general.py')
            native.set_task_context(512,ROOT/'data/Size_512',ROOT/'outputs/build_cache')
            m,_=native.build_april_model(n)
        state=next((c[k] for k in ['model_state_dict','model_state','state_dict','model'] if k in c),None)
        if state is None:state=c
        m.load_state_dict(state,strict=True);self.model=m.eval()
    def prepare(self,path):
        from torchvision.transforms import functional as TF,InterpolationMode
        with Image.open(path) as im:
            if self.yolo:return np.asarray(im.convert('RGB'))[:,:,::-1].copy()
            x=TF.pil_to_tensor(im.convert('L')).float()/255.
        if self.size!=512:
            if self.name=='usfm':
                import cv2
                x=torch.from_numpy(cv2.resize(x[0].numpy(),(224,224),interpolation=cv2.INTER_LINEAR))[None]
            else:x=TF.resize(x,[self.size,self.size],interpolation=InterpolationMode.BICUBIC if self.name=='medsam' else InterpolationMode.BILINEAR,antialias=True)
        if self.name=='medsam':x=(x-x.min())/(x.max()-x.min()).clamp_min(1e-8)
        x=x.repeat(self.channels,1,1)
        if self.norm['normalization_enabled']:x=TF.normalize(x,self.norm['mean'],self.norm['std'])
        return x[None].cuda()
    def start(self,x):
        if self.yolo:
            self.predict(x)
            if self.name.startswith('yolo26'):
                from evaluation.lesion.yolo_semantic_probability import install_probability_tap
                install_probability_tap(self.yolo,foreground_class=1)
        else:self.model.cuda()
    @torch.inference_mode()
    def predict(self,x):
        if self.yolo:
            kw=dict(imgsz=512,batch=1,device=0,verbose=False,stream=False)
            if '11' in self.name:kw.update(conf=self.job['instance_conf'],iou=self.job['instance_iou'],retina_masks=True)
            return self.yolo.predict(source=x,**kw)[0]
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.amp):y=self.model(x)
        while isinstance(y,(list,tuple)):y=y[0]
        if isinstance(y,dict):y=y['out']
        y=F.interpolate(y.float(),size=(512,512),mode='bilinear',align_corners=False)
        if self.kind=='sigmoid':p=y.sigmoid()[0,0]
        elif self.kind=='probability':p=y[0,0]
        else:p=y.softmax(1)[0,1]
        return p
    def unpack(self,y):
        if self.yolo:
            if self.name.startswith('yolo26'):
                from evaluation.lesion.yolo_semantic_probability import result_foreground_probability
                p=result_foreground_probability(y,(512,512));return p>.5,p
            native=module('yolo_native',ER/'experiments/yolo/train_family.py')
            return native.instance_prediction_and_score_map(y,512,self.job['instance_conf'])
        p=y.cpu().numpy();return p>.5,p
