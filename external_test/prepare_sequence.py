"""Calibrate an archived external test patch for predict_cv.py."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patch", type=Path, help="A patch in an extracted regional archive")
    parser.add_argument("output", type=Path, help="Calibrated input NPZ for predict_cv.py")
    args = parser.parse_args()

    region_dir = args.patch.parent.parent
    manifest = json.loads((region_dir / "manifest.json").read_text(encoding="utf-8"))
    calibration = json.loads((region_dir / "calibration.json").read_text(encoding="utf-8"))
    patch_path = args.patch.relative_to(region_dir).as_posix()
    record = next((entry for entry in manifest["patches"] if entry["file"] == patch_path), None)
    if record is None:
        parser.error("Patch is absent from the regional manifest")

    with np.load(args.patch, allow_pickle=False) as raw:
        dates = raw["dates"]
        bands = raw["bands"].tolist()
        sequence = raw["s2"].astype(np.float32)
    if bands != calibration["bands"] or dates.tolist() != record["dates"]:
        parser.error("Patch dates or band order do not match the release metadata")

    coeff = np.asarray([calibration["observations"][day]["coefficients"] for day in dates],
                       dtype=np.float32)
    sequence = sequence * coeff[:, :, 0, None, None] + coeff[:, :, 1, None, None]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, s2=sequence, dates=dates, valid_shape=record["shape"])


if __name__ == "__main__":
    main()
