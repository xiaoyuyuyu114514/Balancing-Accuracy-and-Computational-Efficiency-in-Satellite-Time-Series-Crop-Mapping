# Balancing Accuracy and Computational Efficiency in Satellite Time-Series Crop Mapping

This repository contains the U-ESTAE and Pro architectures, the original and enhanced U-TAE recipes, and 75 inference checkpoints for 15 completed PASTIS five-fold configurations.

## Five-fold checkpoints

The table lists the 12 cosine configurations. Decay runs over epochs 61–100 unless stated. U-ESTAE's final training recipe combines boundary-weighted cross-entropy (E1), 20% GPU temporal dropout (TD20), and spatial flips of the projected feature map. The U-TAE enhanced recipe uses CPU TD20 and input-frame flips. U-ESTAE sizes denote windows × shallow feature channels, followed by the Gaussian window standard deviation.

| Model | Training recipe | Checkpoints | Test mIoU, mean ± SD (%) | Typical train + val (s/epoch) | Inference (ms/patch) | GFLOPs/patch |
|---|---|---|---:|---:|---:|---:|
| U-TAE | CE, epochs 41–100 | [utae_cosine60](weights/five_fold/utae_cosine60/) | 63.66 ± 0.97 | 82.29 | 16.16 | 278.72 |
| U-TAE enhanced | E1 + CPU TD20 + input flips | [utae_enhanced_flip_cosine40](weights/five_fold/utae_enhanced_flip_cosine40/) | 65.94 ± 1.36 | 85.74 | 16.69 | 278.72 |
| U-ESTAE 6×16, 60 d | CE | [u_estae_e0_cosine40](weights/five_fold/u_estae_e0_cosine40/) | 59.32 ± 0.93 | 22.77 | 3.05 | 9.69 |
| U-ESTAE 6×16, 60 d | E1 | [u_estae_6x16_e1_cosine40](weights/five_fold/u_estae_6x16_e1_cosine40/) | 60.74 ± 0.83 | 22.33 | 3.85 | 9.69 |
| U-ESTAE 6×16, 60 d | E1 + GPU TD20 | [u_estae_6x16_e1_td20_cosine40](weights/five_fold/u_estae_6x16_e1_td20_cosine40/) | 61.31 ± 1.03 | 22.85 | 3.07 | 9.69 |
| U-ESTAE 6×16, 60 d | Final | [u_estae_6x16_flip_cosine](weights/five_fold/u_estae_6x16_flip_cosine/) | 61.96 ± 1.17 | 20.28 | 3.02 | 9.69 |
| U-ESTAE 12×24, 60 d | CE | [u_estae_12x24_e0_cosine40](weights/five_fold/u_estae_12x24_e0_cosine40/) | 61.11 ± 1.16 | 29.12 | 4.48 | 12.13 |
| U-ESTAE 12×24, 60 d | Final | [u_estae_flip_cosine](weights/five_fold/u_estae_flip_cosine/) | 63.64 ± 1.32 | 31.03 | 4.53 | 12.13 |
| U-ESTAE Pro 24×24, 30 d | CE | [u_estae_pro_24x24_sigma30_e0_cosine40](weights/five_fold/u_estae_pro_24x24_sigma30_e0_cosine40/) | 61.66 ± 1.13 | 38.36 | 6.49 | 15.12 |
| U-ESTAE Pro 24×24, 30 d | Final | [u_estae_pro_24x24_sigma30](weights/five_fold/u_estae_pro_24x24_sigma30/) | 64.97 ± 0.86 | 37.61 | 6.46 | 15.12 |
| U-ESTAE Pro 32×32, 60 d | CE | [u_estae_pro_32x32_sigma60_e0_cosine40](weights/five_fold/u_estae_pro_32x32_sigma60_e0_cosine40/) | 62.29 ± 0.55 | 47.11 | 8.54 | 20.19 |
| U-ESTAE Pro 32×32, 60 d | Final | [u_estae_pro_32x32_sigma60](weights/five_fold/u_estae_pro_32x32_sigma60/) | 65.12 ± 0.98 | 48.35 | 8.59 | 20.19 |

Each fold file contains a CPU state dictionary and the epoch selected by validation mIoU for that test fold. The checkpoint manifest also lists three fixed-learning-rate configurations, per-fold metrics, SHA-256 checksums, training times, and accuracy. The older fixed-LR enhanced U-TAE uses its fold-5 rerun at epoch 79.

- [Checkpoint manifest and per-fold metrics](results/five_fold_checkpoints.json)
- [Five-fold confusion matrices](results/pastis_confusion_matrices.json)
- [Single-fold U-ESTAE ablations](results/uestae_ablations.json)

Typical train + validation time is the sum of the central 50% means of the training durations and epoch-51–100 validation durations in each fold, averaged across folds. This estimates one validation per epoch even for runs that skipped early validation. The manifest also retains actual train + validation times. Latency is the median of per-fold latency medians for a BF16, 46-date, 128 × 128 patch, excluding data loading. GFLOPs are estimated for the same input using 2 FLOPs per multiply–accumulate, PyTorch-supported operator counts, and transposed-convolution MACs. Normalization and some elementwise operations are excluded.

## U-ESTAE components

1. A pointwise convolution and GELU project the 10 input bands to per-date spectral features.
2. A depthwise spatial convolution and pointwise projection score the dates in each window. A Gaussian date prior and a masked temporal softmax produce the aggregation weights.
3. Each window contributes a weighted feature mean (State) and a regularized weighted linear slope (Trend). The model projects the concatenated window summaries to 32 channels and applies one 2D U-Net. The 6×16, 12×24, 24×24, and 32×32 configurations contribute 192, 576, 1,152, and 2,048 summary channels. [Model code](code/reference_model.py)
4. E1 assigns a weight of 3 to valid pixels within one pixel of a semantic class boundary. GPU temporal dropout masks 20% of valid dates during training. Projected-feature flips apply a horizontal or vertical flip to the 32-channel feature map and invert it on the logits. [Boundary loss](code/boundaries.py) · [Training steps](code/experiment.py)

## Setup

Install PyTorch 2.8 with the build appropriate for your CPU or CUDA environment, plus NumPy. Obtain [PASTIS](https://github.com/VSainteuf/pastis-benchmark) separately. Its root must contain `metadata.geojson`, `NORM_S2_patch.json`, `DATA_S2/`, and `ANNOTATIONS/`. From this repository's root, set the dataset location. In PowerShell:

```powershell
$env:PASTIS_ROOT = "D:/datasets/PASTIS"
```

On Linux, run `export PASTIS_ROOT=/path/to/PASTIS`.

## Inference

Predict a held-out PASTIS patch. The script selects the appropriate checkpoint from the patch's fold and writes a 128 × 128 `uint8` class-index mask. `--device cpu` is available.

```bash
python code/predict_cv.py --model u_estae_pro_24x24_sigma30 --patch 40173 --output runs/pro_40173.npy
python code/predict_cv.py --model u_estae_flip_cosine --patch 40173 --output runs/u_estae_flip_40173.npy
python code/predict_cv.py --model utae_cosine60 --patch 40173 --output runs/utae_cosine_40173.npy
python code/predict_cv.py --model utae_enhanced_flip_cosine40 --patch 40173 --output runs/utae_enhanced_40173.npy
```

For an external sequence, pass an `.npz` containing `s2` with shape `(T, 10, 128, 128)` and `dates` with `T` ISO `YYYY-MM-DD` strings in chronological order. Use the PASTIS reflectance scale and the band order B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12. For a patch with bottom or right padding, optionally include `valid_shape` as `[height, width]`. The script normalizes each member using its training folds' statistics, averages the five probability maps, and returns the class with the highest mean probability. Prepare and align external imagery before passing it to this entry point.

```bash
python code/predict_cv.py --model u_estae_pro_24x24_sigma30 --sequence inputs/patch.npz --output runs/ensemble.npy
```

## Training

`train_cv.py` runs the main recipes for 100 epochs with Adam, initial LR 0.001, batch size 2, seed 1, and validation mIoU checkpoint selection. Cosine schedules reach 0.0001 at epoch 100 and save the full training state at epoch 60. The portable entry point uses bounded input prefetch and requires CUDA with BF16 support. Published timings come from the original runs, where U-ESTAE preloaded imagery into RAM.

```bash
python code/train_cv.py --model u_estae_pro_24x24_sigma30 --test-fold 1
python code/train_cv.py --model utae_enhanced_flip_cosine40 --test-fold 5
python code/train_cv.py --model u_estae_e0_cosine40 --test-fold 1
```

Outputs go to `runs/five_fold/<model>/fold_<N>/` by default. An existing `last.pt` resumes the same configuration. Use separate output directories for concurrent runs. Class 0 is background, classes 1–18 are crop types, and class 19 is void. The main PASTIS mIoU convention removes the void row and column and averages the classes with nonzero union. The evaluation code also reports metrics with only the void target row excluded.

Cosine configurations validate from epoch 51. The published 12×24 final run validated every epoch for test fold 5, and the training entry point preserves that default for its historical configuration. Use `--validation-start` to change the validation schedule explicitly.

## Missing-date evaluation

The enhanced U-TAE and the four final U-ESTAE configurations share the masks in the [evaluation protocol](results/temporal_dropout/protocol.json). The complete sequence is run once. Each 10%–90% loss level is repeated ten times. PASTIS uses the held-out checkpoint for each fold. Beauce and Angers use five-checkpoint probability ensembles and audited labels.

- [PASTIS summary](results/temporal_dropout/pastis_summary.json) and [per-run scores](results/temporal_dropout/pastis_runs.json)
- [External summary](results/temporal_dropout/external_summary.json) and [per-run scores](results/temporal_dropout/external_runs.json)

The evaluation entry point requires CUDA with BF16 support. It resumes from its output JSON. For the external command, supply the two extracted regional archive directories.

```bash
python code/evaluate_outages.py --model u_estae_pro_24x24_sigma30 --output runs/outages_pastis.json
python code/evaluate_outages.py --model u_estae_pro_24x24_sigma30 --external beauce angers --output runs/outages_external.json
```

## Earlier single-fold files

The original `code/predict.py`, `code/train_boundary.py`, and their two checkpoints remain available. Those files refer to the earlier single-fold comparison and are described in [results/checkpoints.json](results/checkpoints.json).

## Data and attribution

Download the PASTIS benchmark from [Zenodo](https://doi.org/10.5281/zenodo.5012942). Download the 162 Beauce and Angers external test patches, audited labels, and checksums from the [external test dataset release](https://github.com/xiaoyuyuyu114514/Balancing-Accuracy-and-Computational-Efficiency-in-Satellite-Time-Series-Crop-Mapping/releases/tag/external-test-v1.0). See the [external dataset guide](external_test/README.md) for the file format, sources, and preprocessing for model inference.

The U-TAE implementation is derived from [utae-paps](https://github.com/VSainteuf/utae-paps), with its MIT license in [`code/src/LICENSE`](code/src/LICENSE). The optional earlier Lovász variants use [LovaszSoftmax](https://github.com/bermanmaxim/LovaszSoftmax), with its license in [`code/LOVASZ_LICENSE`](code/LOVASZ_LICENSE).
