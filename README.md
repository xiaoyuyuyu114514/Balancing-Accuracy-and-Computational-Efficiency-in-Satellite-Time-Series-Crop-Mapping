# Balancing Accuracy and Computational Efficiency in Satellite Time-Series Crop Mapping

This repository contains the code and 15 checkpoints for the final five-fold comparison on PASTIS. U-ESTAE aggregates per-pixel features into state and trend summaries in 12 soft date windows before applying a single spatial U-Net. Each date has 24 feature channels, and the aggregated 576-channel summary is projected to 32 channels. The original and enhanced U-TAE share the same network architecture.

## Five-fold checkpoints

| Model | Training recipe | Weights | PASTIS test mIoU, mean ± SD (%) |
|---|---|---|---:|
| U-TAE | Cross-entropy, full sequence | [utae](weights/five_fold/utae/) | 63.17 ± 1.95 |
| U-TAE enhanced | Boundary-weighted cross-entropy (E1) and 20% temporal dropout (CPU) | [utae_enhanced](weights/five_fold/utae_enhanced/) | 65.03 ± 1.27 |
| U-ESTAE | E1 and 20% temporal dropout (GPU mask) | [u_estae](weights/five_fold/u_estae/) | 62.98 ± 1.30 |

Each `fold_N.pt` contains a CPU `state_dict` and the selected epoch for **test fold N**. Each run uses three training folds and the preceding validation fold, with fold 5 preceding fold 1. The per-fold validation and test scores, split assignments, selected epochs, and SHA-256 hashes are in the [checkpoint manifest](results/five_fold_checkpoints.json).

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
python code/predict_cv.py --model utae --patch 40173 --output runs/utae_40173.npy
python code/predict_cv.py --model utae_enhanced --patch 40173 --output runs/utae_enhanced_40173.npy
```

For an external sequence, pass an `.npz` containing `s2` with shape `(T, 10, 128, 128)` and `dates` with `T` ISO `YYYY-MM-DD` strings in chronological order. Use the PASTIS reflectance scale and the band order B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12. For a patch with bottom or right padding, optionally include `valid_shape` as `[height, width]`. The script normalizes each member using its training folds' statistics, averages the five probability maps, and returns the class with the highest mean probability. Prepare and align external imagery before passing it to this entry point.

```bash
python code/predict_cv.py --model u_estae --sequence inputs/patch.npz --output runs/ensemble.npy
```

## Training

`train_cv.py` implements the three training recipes with 100 epochs, Adam at 0.001, batch size 2, seed 1, and validation mIoU checkpoint selection. It requires CUDA with BF16 support and uses bounded input prefetch. The included training entry point focuses on reproducing the recipes and splits. The efficiency measurements in the study used the corresponding experiment data-pipeline configurations.

```bash
python code/train_cv.py --model u_estae --test-fold 5
python code/train_cv.py --model utae --test-fold 5 --smoke
```

Outputs go to `runs/five_fold/<model>/fold_<N>/` by default. An existing `last.pt` resumes the same configuration. Use separate output directories for concurrent runs. Class 0 is background, classes 1–18 are crop types, and class 19 is void. The main PASTIS mIoU convention removes the void row and column and averages the classes with nonzero union. The evaluation code also reports metrics with only the void target row excluded.

## Earlier single-fold files

The original `code/predict.py`, `code/train_boundary.py`, and their two checkpoints remain available. Those files refer to the earlier single-fold comparison and are described in [results/checkpoints.json](results/checkpoints.json).

## Data and attribution

The U-TAE implementation is derived from [utae-paps](https://github.com/VSainteuf/utae-paps), with its MIT license in [`code/src/LICENSE`](code/src/LICENSE). The optional earlier Lovász variants use [LovaszSoftmax](https://github.com/bermanmaxim/LovaszSoftmax), with its license in [`code/LOVASZ_LICENSE`](code/LOVASZ_LICENSE). The PASTIS and external test imagery and labels are not included in this repository.
