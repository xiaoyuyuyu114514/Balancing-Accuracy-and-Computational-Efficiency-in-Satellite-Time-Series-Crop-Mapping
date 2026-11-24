# Balancing Accuracy and Computational Efficiency in Satellite Time-Series Crop Mapping

This repository contains U-ESTAE, U-TAE, and 25 inference checkpoints from the completed PASTIS five-fold experiments. The U-ESTAE experiments cover the original 6-window, 16-channel architecture and the 12-window, 24-channel configuration. The original and enhanced U-TAE share the same network architecture.

## Five-fold checkpoints

| Model | Training recipe | Checkpoints | Test mIoU, mean ± SD (%) | Pooled mIoU (%) | Train + val (s/epoch) | Inference (ms/patch) |
|---|---|---|---:|---:|---:|---:|
| U-TAE | Cross-entropy, full sequence, fixed LR | [utae](weights/five_fold/utae/) | 63.17 ± 1.95 | 63.27 | 86.06 | 17.68 |
| U-TAE | Cross-entropy, full sequence, cosine from epoch 41 | [utae_cosine60](weights/five_fold/utae_cosine60/) | 63.66 ± 0.97 | 63.80 | 77.69 | 16.16 |
| U-TAE enhanced | E1 and CPU temporal dropout (20%), fixed LR | [utae_enhanced](weights/five_fold/utae_enhanced/) | 65.07 ± 1.25 | 65.24 | 82.96 | 16.73 |
| U-ESTAE (12 × 24) | E1 and GPU temporal dropout (20%), fixed LR | [u_estae](weights/five_fold/u_estae/) | 62.98 ± 1.30 | 63.12 | 30.50 | 4.53 |
| U-ESTAE (12 × 24) | E1, GPU temporal dropout (20%), spatial flips, cosine from epoch 61 | [u_estae_flip_cosine](weights/five_fold/u_estae_flip_cosine/) | 63.64 ± 1.32 | 63.74 | 29.54 | 4.53 |

Each `fold_N.pt` contains a CPU `state_dict` and the epoch selected by validation mIoU for **test fold N**. Each run uses three training folds and the preceding validation fold, with fold 5 preceding fold 1. U-TAE enhanced uses the fold-5 rerun selected at epoch 79.

- [Checkpoint manifest and per-fold metrics](results/five_fold_checkpoints.json) includes accuracy, class IoU, selected epochs, timing, and SHA-256 hashes.
- [Five-fold confusion matrices](results/pastis_confusion_matrices.json) store pixel counts with reference classes on rows and predictions on columns. Pooled mIoU is calculated after adding the five matrices and excluding class 19.
- [U-ESTAE single-fold ablations](results/uestae_ablations.json) contains the completed structural and training-pipeline experiments and their settings.

Training time is the sum of training and actual validation time divided by the number of training epochs. The cosine U-TAE validates from epoch 51. The cosine U-ESTAE validates from epoch 51 on folds 1–4 and every epoch on fold 5. Inference time is the median of the per-fold latency medians for a BF16, 46-date, 128 × 128 patch, excluding data loading.

## U-ESTAE components

1. A pointwise convolution and GELU project the 10 input bands to per-date spectral features.
2. A depthwise spatial convolution and pointwise projection score the dates in each window. A Gaussian date prior and a masked temporal softmax produce the aggregation weights.
3. Each window contributes a weighted feature mean (State) and a regularized weighted linear slope (Trend). The model projects the resulting 192 channels (6 × 16 × 2) or 576 channels (12 × 24 × 2) to 32 and applies one 2D U-Net. [Model code](code/reference_model.py)
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
python code/predict_cv.py --model u_estae --patch 40173 --output runs/u_estae_40173.npy
python code/predict_cv.py --model u_estae_flip_cosine --patch 40173 --output runs/u_estae_flip_40173.npy
python code/predict_cv.py --model utae_cosine60 --patch 40173 --output runs/utae_cosine_40173.npy
python code/predict_cv.py --model utae_enhanced --patch 40173 --output runs/utae_enhanced_40173.npy
```

For an external sequence, pass an `.npz` containing `s2` with shape `(T, 10, 128, 128)` and `dates` with `T` ISO `YYYY-MM-DD` strings in chronological order. Use the PASTIS reflectance scale and the band order B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12. For a patch with bottom or right padding, optionally include `valid_shape` as `[height, width]`. The script normalizes each member using its training folds' statistics, averages the five probability maps, and returns the class with the highest mean probability. Prepare and align external imagery before passing it to this entry point.

```bash
python code/predict_cv.py --model u_estae --sequence inputs/patch.npz --output runs/ensemble.npy
```

## Training

`train_cv.py` runs the main recipes for 100 epochs with Adam, initial LR 0.001, batch size 2, seed 1, and validation mIoU checkpoint selection. Cosine schedules reach 0.0001 at epoch 100 and save the full training state at epoch 60. The portable entry point uses bounded input prefetch and requires CUDA with BF16 support. Published timings come from the original runs, where U-ESTAE preloaded imagery into RAM.

```bash
python code/train_cv.py --model u_estae_flip_cosine --test-fold 1
python code/train_cv.py --model utae_cosine60 --test-fold 5
python code/train_cv.py --model u_estae_e0_cosine60 --test-fold 1
```

Outputs go to `runs/five_fold/<model>/fold_<N>/` by default. An existing `last.pt` resumes the same configuration. Use separate output directories for concurrent runs. Class 0 is background, classes 1–18 are crop types, and class 19 is void. The main PASTIS mIoU convention removes the void row and column and averages the classes with nonzero union. The evaluation code also reports metrics with only the void target row excluded.

The original U-ESTAE E0 recipe (`u_estae_e0_cosine60`, 6 windows × 16 channels, cross-entropy, full time series) runs cosine decay from epoch 41 and validates from epoch 51. Its five-fold evaluation was still running at the time of this update.

## Earlier single-fold files

The original `code/predict.py`, `code/train_boundary.py`, and their two checkpoints remain available. Those files refer to the earlier single-fold comparison and are described in [results/checkpoints.json](results/checkpoints.json).

## Data and attribution

Download the PASTIS benchmark from [Zenodo](https://doi.org/10.5281/zenodo.5012942). Download the 162 Beauce and Angers external test patches, audited labels, and checksums from the [external test dataset release](https://github.com/xiaoyuyuyu114514/Balancing-Accuracy-and-Computational-Efficiency-in-Satellite-Time-Series-Crop-Mapping/releases/tag/external-test-v1.0). See the [external dataset guide](external_test/README.md) for the file format, sources, and preprocessing for model inference.

The U-TAE implementation is derived from [utae-paps](https://github.com/VSainteuf/utae-paps), with its MIT license in [`code/src/LICENSE`](code/src/LICENSE). The optional earlier Lovász variants use [LovaszSoftmax](https://github.com/bermanmaxim/LovaszSoftmax), with its license in [`code/LOVASZ_LICENSE`](code/LOVASZ_LICENSE).
