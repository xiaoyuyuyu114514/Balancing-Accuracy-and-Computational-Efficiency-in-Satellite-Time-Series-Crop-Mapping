"""Predict a held-out PASTIS patch or run a five-weight ensemble on a prepared sequence."""

import argparse
from datetime import date
import json
from pathlib import Path

import numpy as np
import torch

from cv_models import FOLDS, build_model, fold_split
from data import DATA
from fast_data import RawPastis


WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "five_fold"
REFERENCE_DATE = date(2018, 9, 1)


@torch.inference_mode()
def predict(name, sequence, dates, stats, folds, device, valid_shape=(128, 128)):
    if sequence.shape != (len(dates), 10, 128, 128) or not len(dates):
        raise ValueError("Expected nonempty s2 data shaped (T, 10, 128, 128) with T dates")
    if dates != sorted(dates) or not np.isfinite(sequence).all():
        raise ValueError("Observations must be finite and in chronological order")
    height, width = valid_shape
    if not (1 <= height <= 128 and 1 <= width <= 128):
        raise ValueError("valid_shape must contain height and width between 1 and 128")

    x = torch.as_tensor(sequence, dtype=torch.float32, device=device)
    positions = torch.tensor([dates], dtype=torch.long, device=device)
    valid = torch.ones_like(positions, dtype=torch.bool)
    autocast = device.type == "cuda" and torch.cuda.is_bf16_supported()
    total = torch.zeros((1, 20, 128, 128), dtype=torch.float32, device=device)
    ensemble = len(folds) > 1
    for fold in folds:
        # The held-out PASTIS protocol uses that split's supplied statistics.
        # For external sequences, each member uses its training folds' statistics.
        stat_folds = fold_split(fold)[0] if ensemble else (fold,)
        mean, std = [torch.as_tensor(np.mean([stats[f"Fold_{k}"][key] for k in stat_folds], axis=0),
                                    dtype=torch.float32, device=device)[None, :, None, None]
                     for key in ("mean", "std")]
        z = (x - mean) / std
        z[:, :, height:, :] = 0
        z[:, :, :, width:] = 0
        checkpoint = torch.load(WEIGHTS / name / f"fold_{fold}.pt", map_location="cpu", weights_only=True)
        model = build_model(name)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model = model.to(device).eval()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
            logits = model(z.unsqueeze(0), batch_positions=positions, valid_mask=valid)
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite logits")
        total += logits.float().softmax(dim=1)
    return total[0].argmax(dim=0).byte().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    manifest = json.loads((WEIGHTS.parents[1] / "results" / "five_fold_checkpoints.json").read_text(encoding="utf-8"))
    available = [key for key, entry in manifest.items() if len(entry.get("folds", [])) == len(FOLDS)]
    parser.add_argument("--model", choices=available, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--patch", type=int, help="PASTIS patch ID, using its held-out checkpoint")
    inputs.add_argument("--sequence", type=Path, help="Prepared NPZ sequence, using all five checkpoints")
    parser.add_argument("--output", type=Path, required=True, help="Output class mask (.npy)")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    stats = json.loads((DATA / "NORM_S2_patch.json").read_text(encoding="utf-8"))

    valid_shape = (128, 128)
    if args.patch is not None:
        dataset = RawPastis(FOLDS)
        index = next((i for i, row in enumerate(dataset.rows) if int(row["ID_PATCH"]) == args.patch), None)
        if index is None:
            parser.error("Unknown PASTIS patch ID")
        fold = dataset.rows[index]["Fold"]
        raw, dates, _ = dataset[index]
        x, positions, folds = raw.numpy(), dates.tolist(), (fold,)
    else:
        with np.load(args.sequence, allow_pickle=False) as inputs:
            x = inputs["s2"]
            positions = [(date.fromisoformat(value) - REFERENCE_DATE).days for value in inputs["dates"].tolist()]
            if "valid_shape" in inputs:
                valid_shape = tuple(int(v) for v in inputs["valid_shape"])
        folds = tuple(FOLDS)
    mask = predict(args.model, x, positions, stats, folds, torch.device(device), valid_shape)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, mask)


if __name__ == "__main__":
    main()
