"""B-GeoNet CUDA inference at batch 1 and 512 x 512."""
import torch
from .model import BGeoNet

class Runtime:
    @torch.inference_mode()
    def __init__(self, checkpoint, graph=True, compile_modules=True):
        checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.model = BGeoNet()
        self.model.load_state_dict(checkpoint.get('model', checkpoint), strict=True)
        self.model.alpha = float(checkpoint.get('config', {}).get('C', 1.0))
        self.model.eval().cuda().to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
        self.graph = None
        if compile_modules:
            self.model.raster = torch.compile(self.model.raster, fullgraph=True)
            self.model.refine = torch.compile(self.model.refine, fullgraph=True)
        self.static = torch.zeros(1, 1, 512, 512, device='cuda')
        for _ in range(15):
            self.forward(self.static)
        if graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(5):
                    self.forward(self.static)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = self.forward(self.static)

    @torch.inference_mode()
    def forward(self, image):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = self.model(image)
        return result['logits'].float().sigmoid()[0, 0]

    @torch.inference_mode()
    def __call__(self, image):
        if image.shape != (1, 1, 512, 512) or not image.is_cuda or image.dtype != torch.float32:
            raise ValueError('Expected CUDA float32 [1, 1, 512, 512] input in [0, 1]')
        if self.graph is None:
            return self.forward(image)
        self.static.copy_(image)
        self.graph.replay()
        return self.output  # Reusable buffer: clone before retaining across calls.
