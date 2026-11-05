# Balancing Accuracy and Computational Efficiency in Satellite Time-Series Crop Mapping

U-ESTAE (TP-EAN in the experiment logs) summarizes a Sentinel-2 sequence with six date windows. Each window provides a weighted state and trend before a single spatial U-Net. The second model uses the U-TAE backbone. Both released checkpoints use boundary-weighted cross-entropy and training-time temporal dropout (20%).

## Checkpoints

We selected the variant and epoch within each model family by validation mIoU among completed, aligned 100-epoch experiments. Test scores refer to those selected checkpoints.

| Model | Weights | Epoch | Validation mIoU (%) | Test mIoU (%) |
|---|---|---:|---:|---:|
| U-ESTAE | weights/u_estae.pt | 98 | 62.47 | 61.00 |
| Boundary-aware U-TAE | weights/utae_boundary.pt | 96 | 66.56 | 64.53 |

Each file contains a PyTorch inference state dictionary. SHA-256 hashes and source run IDs are in [results/checkpoints.json](results/checkpoints.json).

## Data and usage

Use Python with PyTorch 2.8 and NumPy. Training uses a CUDA GPU with BF16 support. The dataset root must contain metadata.geojson, NORM_S2_patch.json, DATA_S2/ and ANNOTATIONS/. From the repository root, set the data location in PowerShell:

    $env:PASTIS_ROOT = "D:/datasets/PASTIS"

On Linux use export PASTIS_ROOT=/path/to/PASTIS. Predict a 128-by-128 class-index mask on a fold-5 patch:

    python code/predict.py --model u_estae --patch 40173 --output runs/u_estae_40173.npy
    python code/predict.py --model utae_boundary --patch 40173 --output runs/utae_40173.npy

Reproduce the training protocol using the included epoch sampling order:

    python code/train_boundary.py --family ours --variant e1_td20
    python code/train_boundary.py --family utae --variant e1_td20

Outputs are written to runs/ (override with RUNS_DIR). An existing last.pt resumes that run. Add --smoke for a two-batch training and one-batch validation check. Use separate output folders for concurrent runs of the same configuration.

Training uses folds 1-3, validation fold 4, test fold 5, seed 1, Adam at 0.001, and batch size 2. The reference loader applies fold-specific normalization, counts dates from 2018-09-01, and masks padded frames. The reported mIoU follows the original benchmark convention in code/experiment.py (remove row and column 19; average classes with nonzero union). Background is valid.

## Attribution

The U-TAE implementation is derived from [utae-paps](https://github.com/VSainteuf/utae-paps); its license is in code/src/LICENSE. The optional Lovasz training variants use [LovaszSoftmax](https://github.com/bermanmaxim/LovaszSoftmax); see code/LOVASZ_LICENSE. PASTIS data must be obtained separately from the [dataset authors](https://github.com/VSainteuf/pastis-benchmark).
