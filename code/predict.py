"""Predict the semantic mask of one normalized PASTIS test patch."""
import argparse
from pathlib import Path

import numpy as np
import torch

from boundary_models import base_model
from data import Pastis, collate

MODELS = {"u_estae": "ours", "utae_boundary": "utae"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--patch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = Pastis([5])
    index = next((i for i, row in enumerate(dataset.rows)
                  if int(row["ID_PATCH"]) == args.patch), None)
    if index is None:
        parser.error("Patch ID must belong to fold 5")
    model = base_model(MODELS[args.model]).to(device).eval()
    weights = Path(__file__).resolve().parents[1] / "weights" / (args.model + ".pt")
    model.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"), strict=True)
    x, dates, valid, _ = collate([dataset[index]])
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        logits = model(x.to(device), batch_positions=dates.to(device), valid_mask=valid.to(device))
    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite logits")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, logits[0].argmax(0).byte().cpu().numpy())


if __name__ == "__main__":
    main()
