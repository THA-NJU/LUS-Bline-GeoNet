# Geometry branches across segmentation backbones

One training implementation is provided for each of VM-UNet, S2DENet and
ResNet34-FPN. These RTX 4090 experiments train the backbone and the original
16-query, six-parameter geometry branch jointly from random initialization.
The geometry renderer and residual refinement are retained.

## Results

Validation-selected checkpoints; V6_Test contains 2,716 images. Values below
are percentages. SD is between-image population SD, not variation across seeds.
Lesion F1 uses pooled one-to-one matching at strict IoU > 0.2. No postprocessing.

| Model | Parameters | Best epoch | Validation Dice | Test Dice ± SD | Test IoU ± SD | Lesion F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VM-UNet + geometry | 9,581,406 | 15 | 64.44 | 66.85 ± 17.43 | 52.53 ± 18.02 | 67.38 |
| S2DENet + geometry | 996,734 | 73 | 61.41 | 65.52 ± 17.65 | 51.08 ± 18.23 | 55.87 |
| ResNet34-FPN + geometry | 24,107,690 | 23 | 64.35 | 67.02 ± 17.56 | 52.76 ± 18.18 | 71.25 |

Full precision summary: [results.json](results.json). Per-model training curves
and settings are in `history.json` and `training_config.json`. Each model also
includes all 59 recorded aggregate V6_Test metrics in `test_metrics.json`, with
units and inference precision. Timing values are specific to the recorded environment. These are single-seed
results from selected experimental configurations, not a multi-seed estimate.

## Environment

The recorded environment used Python with PyTorch 2.6.0 / CUDA 11.8, NumPy 1.26.4
and SciPy 1.15.3. Install OpenCV for input and geometry targets. VM-UNet additionally
requires mamba-ssm 2.2.5 and its CUDA selective-scan implementation; S2DENet needs
einops 0.8.1; ResNet34-FPN needs segmentation-models-pytorch 0.5.0 and timm 1.0.15.
Choose the matching CUDA PyTorch build before installing these dependencies.

## Data and training

Run from the repository root. Use 512 × 512 grayscale PNGs and binary masks
encoded as 0/1, with matching filenames under `train/images`, `train/masks`,
`val/images`, and `val/masks`. The recorded protocol requires exactly 27,814
training and 3,691 validation images; the training code checks these counts.
Preparation and optimization read only train/val. No image data, patient-level
metadata, weights, or code that mixes test images into training is included.

```bash
python experiments/cross_model_geometry/vm_unet/prepare.py data/Size_512 outputs/geometry.json
python experiments/cross_model_geometry/vm_unet/train.py --model vm_unet_v2 \
  --root data/Size_512 --metadata outputs/geometry.json --output outputs/vm_unet_geometry --batch-size 16

python experiments/cross_model_geometry/s2denet/train.py --model s2denet \
  --root data/Size_512 --metadata outputs/geometry.json --output outputs/s2denet_geometry --batch-size 16

python experiments/cross_model_geometry/resnet34_fpn/train.py --model fpn_resnet34 \
  --root data/Size_512 --metadata outputs/geometry.json --output outputs/resnet34_fpn_geometry --batch-size 32
```

Each folder includes an equivalent train/val-only `prepare.py`. Training uses
AdamW, learning rate and weight decay 1e-4, seed 42, 10 warmup epochs, and up to
600 epochs. Validation runs every epoch with batch 4. VM-UNet and S2DENet use
patience 10; ResNet34-FPN uses patience 15. Model-specific geometry loss schedules
and residual coefficients are preserved in the respective code/configuration.
VM-UNet's native backbone is VM-UNet V2; that architecture name is retained in
imports and checkpoint keys.

Use `--mode smoke` with a separate empty output directory to check full-state
restoration before training. Resume an interrupted formal run using the same
arguments and `--resume`; complete outputs are protected against overwriting.
The checkpoint contains model, optimizer, scheduler, history and RNG states.

## Loading for inference

Run Python with the selected model folder on `sys.path` (or from that folder):

```python
import torch
from models import build_model

checkpoint = torch.load("best.pth", map_location="cpu", weights_only=True)
config = checkpoint["config"]
model = build_model(config["model"])
model.load_state_dict(checkpoint["model"], strict=True)
model.alpha = config["C"]
model.eval()
with torch.inference_mode():
    # image: B x 1 x 512 x 512, grayscale float scaled by 1/255.
    # Model-specific normalization is performed inside the adapter.
    prediction = model(image)["logits"].float().sigmoid() > 0.5
```

Move both model and input to CUDA for VM-UNet's selective scan. Use a consistent
evaluation precision and report it. Test data is used only after training to
evaluate the validation-selected checkpoint.

## Attribution

VM-UNet implementation is adapted from APRIL-MedSeg; its license is retained at
`vm_unet/vendor/LICENSE.april_medseg`. S2DENet retains its source headers and MIT
license at `s2denet/vendor/S2DENet_LICENSE`. ResNet34-FPN is instantiated through
segmentation-models-pytorch with `encoder_weights=None`.

## Release checks

All supplied architecture, data and loss modules were checked against the source
hashes stored with the completed runs. Training functions are unchanged; the CLI
is restricted to the corresponding model in each folder. All three entry points
pass Python syntax and CLI checks. VM-UNet and S2DENet also passed fresh CPU strict
loading of their completed checkpoints and parameter-count checks. ResNet34-FPN
retains the recorded successful evaluation; its runtime was not re-executed in
the packaging environment. GPU training and evaluation were not rerun for this release.
