# Benchmark guide

The registry in `benchmarks/models.json` defines the 36 evaluation models, native preprocessing, precision and instance thresholds. Family training implementations are under `experiments/`; they retain model-specific optimizer, augmentation, batch and checkpoint handling.

## Dependencies

Install common packages with `python -m pip install -r requirements-benchmark.txt`. Mamba, RWKV and foundation models use their upstream CUDA extensions and environment requirements. Run each family in its corresponding environment rather than assuming a single package set covers every model.

Set `BGEONET_THIRD_PARTY` to an external dependency directory (default: `third_party/` under this repository). Use these subdirectory names:

| Path | Contents |
| --- | --- |
| `april_medseg/` | APRIL-MedSeg architecture repository, including its model registrations |
| `SAMUS/` | Official SAMUS repository |
| `MedSAM/` | Official MedSAM repository |
| `sam3/` | Official SAM3 repository |
| `USFM/` | USFM repository, with `usdsgen` and compatible MMSeg/MMCV dependencies |
| `S2DENet/` | Official S2DENet repository |
| `weights/medsam_vit_b.pth` | MedSAM initialization checkpoint |
| `weights/sam3.pt` | SAM3 base checkpoint used with its trained model delta |

SAM2 is imported from the installed `sam2` package. SegFormer-B2 is constructed with depths [3,4,6,3], channels [64,128,320,512] and a 768-channel decode head. YOLO26 semantic models require an Ultralytics installation supporting that model family and the supplied probability-tap integration.

## Models

| Evaluation key | Family | Training source |
| --- | --- | --- |
| `monai_unet` | CNN | [experiments/monai/train_512.py](../experiments/monai/train_512.py) |
| `monai_vnet` | CNN | [experiments/monai/train_512.py](../experiments/monai/train_512.py) |
| `monai_attention_unet` | CNN | [experiments/monai/train_512.py](../experiments/monai/train_512.py) |
| `monai_unetplusplus` | CNN | [experiments/monai/train_512.py](../experiments/monai/train_512.py) |
| `mednext` | CNN | [experiments/april/train_additional_512.py](../experiments/april/train_additional_512.py) |
| `nnunet_2d` | CNN | [experiments/april/train_general.py](../experiments/april/train_general.py) |
| `aau_net` | CNN | [experiments/april/train_general.py](../experiments/april/train_general.py) |
| `fpn_resnet34` | CNN | [experiments/april/train_general.py](../experiments/april/train_general.py) |
| `deeplabv3plus_resnet34` | CNN | [experiments/april/train_general.py](../experiments/april/train_general.py) |
| `swinunet` | Transformer | [experiments/april/train_general.py](../experiments/april/train_general.py) |
| `nnformer_2d` | Transformer | [experiments/april/train_general.py](../experiments/april/train_general.py) |
| `pvtb2_emcad` | Transformer | [experiments/april/train_additional_512.py](../experiments/april/train_additional_512.py) |
| `sepnet` | Transformer | [experiments/april/train_general.py](../experiments/april/train_general.py) |
| `nulite` | Transformer | [experiments/april/train_general.py](../experiments/april/train_general.py) |
| `segformer_b2` | Transformer | [experiments/segformer/train_512.py](../experiments/segformer/train_512.py) |
| `mamba_unet` | Mamba | [experiments/mamba/train_512.py](../experiments/mamba/train_512.py) |
| `vm_unet_v2` | Mamba | [experiments/mamba/train_512.py](../experiments/mamba/train_512.py) |
| `nnmamba_2d` | Mamba | [experiments/mamba/train_512.py](../experiments/mamba/train_512.py) |
| `swin_umamba` | Mamba | [experiments/mamba/train_512.py](../experiments/mamba/train_512.py) |
| `u_rwkv` | RWKV | [experiments/rwkv/train_512.py](../experiments/rwkv/train_512.py) |
| `rwkv_unet` | RWKV | [experiments/rwkv/train_512.py](../experiments/rwkv/train_512.py) |
| `sam2` | Foundation | [experiments/foundation/train_sam2_512.py](../experiments/foundation/train_sam2_512.py) |
| `medsam` | Foundation | [experiments/foundation/train_medsam_512.py](../experiments/foundation/train_medsam_512.py) |
| `samus` | Foundation | [experiments/foundation/train_samus_512.py](../experiments/foundation/train_samus_512.py) |
| `sam3` | Foundation | [experiments/foundation/sam3/train_interactive.py](../experiments/foundation/sam3/train_interactive.py) |
| `usfm` | Foundation | [experiments/april/train_general.py](../experiments/april/train_general.py) |
| `yolo26s-sem` | Yolo | [experiments/yolo26s/train_512.py](../experiments/yolo26s/train_512.py) |
| `yolo26m-sem` | Yolo | [experiments/yolo/train_family.py](../experiments/yolo/train_family.py) |
| `yolo26l-sem` | Yolo | [experiments/yolo/train_family.py](../experiments/yolo/train_family.py) |
| `yolo11s-seg` | Yolo | [experiments/yolo/train_family.py](../experiments/yolo/train_family.py) |
| `yolo11m-seg` | Yolo | [experiments/yolo/train_family.py](../experiments/yolo/train_family.py) |
| `yolo11l-seg` | Yolo | [experiments/yolo/train_family.py](../experiments/yolo/train_family.py) |
| `rolling_unet` | Other | [experiments/april/train_additional_512.py](../experiments/april/train_additional_512.py) |
| `ukan` | Other | [experiments/april/train_general.py](../experiments/april/train_general.py) |
| `xlstm_unet_bot` | Other | [experiments/april/train_general.py](../experiments/april/train_general.py) |
| `s2denet` | Other | [experiments/s2denet/train.py](../experiments/s2denet/train.py) |

## Training example

```bash
python -m experiments.monai.train_512 --april-root third_party/april_medseg \
  --data-root data/Size_512 --output-dir outputs/monai \
  --models monai_unet --run-mode formal
```

Run a family module with `--help` for its arguments. The APRIL `train_general.py` and `train_additional_512.py` runners expose their experiment settings at the top of the source; configure the dataset, selected models and output directory before running them. Use explicit data paths and validation-selected checkpoints for each evaluation.

## Common evaluation

```bash
python -m benchmarks.evaluate --model monai_unet \
  --checkpoint outputs/monai/monai_unet/best_model.pth \
  --images data/Size_512/test/images --masks data/Size_512/test/masks \
  --output outputs/unet_test --postprocess centerline100
```

For the Manual_test image track, run the same command with its image directory and the corresponding masks, using a separate output directory. Model-specific normalization and native input size come from the registry.

SAM adapters use full-image fixed-box prompts. YOLO11 uses native instance confidence 0.25 and NMS IoU 0.70; YOLO26 uses semantic foreground probabilities. USFM retains its three-class output and evaluates foreground channel 1. The benchmark's historical RWKV-UNet and USFM checkpoints retain their original training protocols.

The evaluator uses `weights_only=True` checkpoint loading. YOLO checkpoint classes are explicitly restricted to registered Torch/Ultralytics neural-network types. External architectures remain under their upstream licenses.
