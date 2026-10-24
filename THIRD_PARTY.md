# Third-party acknowledgments

B-GeoNet uses Rolling-UNet-S features. The Rolling architecture originates from [Rolling-Unet](https://github.com/Jiaoyang45/Rolling-Unet), AAAI 2024. The feature implementation used here was adapted from the APRIL-MedSeg implementation, with feature outputs and channel-shift vectorization for B-GeoNet. The retained APRIL-MedSeg license is in [LICENSES/APRIL-MedSeg.txt](LICENSES/APRIL-MedSeg.txt).

Benchmark model architectures are supplied by their upstream packages or external repositories. These include MONAI, APRIL-MedSeg, SegFormer/Transformers, segmentation-models-pytorch, Ultralytics, SAM2, MedSAM, SAMUS, SAM3, USFM and S2DENet. Each upstream project retains its own license and citation requirements. External repositories and model weights are installed separately as described in the benchmark guide.
