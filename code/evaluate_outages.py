"""Reproduce paired date-outage evaluations with published five-fold checkpoints."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from itertools import product
import json
from pathlib import Path

import numpy as np
import torch

from boundaries import BoundaryMetrics
from cv_models import FOLDS, build_model, fold_split
from data import DATA
from experiment import segmentation_metrics
from fast_data import GPUNormalizer, RawPastis, ThreadBatchLoader
from predict_cv import REFERENCE_DATE, WEIGHTS

PROTOCOL = WEIGHTS.parents[1] / "results" / "temporal_dropout" / "protocol.json"
BATCH_SIZE = 2


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def network(name, fold):
    model = build_model(name)
    weights = torch.load(WEIGHTS / name / f"fold_{fold}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(weights["state_dict"], strict=True)
    return model.cuda().eval()


def retained_orders(plan, repeat, percent):
    # All folds and all model configurations share one deletion order per tile.
    result = {}
    for tile, order in plan["deletion_orders"][repeat - 1].items():
        count = min(len(order) - 1, (len(order) * percent + 50) // 100)
        result[tile] = torch.tensor(sorted(order[count:]), dtype=torch.long)
    return result


def confusion_add(matrix, pred, label):
    matrix += torch.bincount((label * 20 + pred).flatten(), minlength=400).reshape(20, 20)


@torch.inference_mode()
def score_pastis(model, dataset, samples, choices):
    tiles = [row["TILE"].upper() for row in dataset.rows]
    class Selected:
        def __getitem__(self, i):
            x, dates, y = samples[i]
            if choices is not None:
                idx = choices[tiles[i]]
                return x.index_select(0, idx), dates.index_select(0, idx), y
            return x, dates, y

    norm = GPUNormalizer(dataset)
    matrix = torch.zeros((20, 20), dtype=torch.long, device="cuda")
    loader = ThreadBatchLoader(Selected(), range(len(samples)), batch_size=BATCH_SIZE)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in loader:
            x, dates, valid, y = norm(batch)
            logits = model(x, batch_positions=dates, valid_mask=valid)
            confusion_add(matrix, logits.argmax(dim=1), y)
    return segmentation_metrics(matrix)["target_masked_mIoU"]


def evaluate_pastis(name, protocol, cases, seen, save, limit=None):
    plan = protocol["pastis_plan"]
    for fold in FOLDS:
        todo = [(repeat, percent) for repeat, percent in cases if (fold, repeat, percent) not in seen]
        if not todo:
            continue
        dataset = RawPastis([fold])
        for row, dates in zip(dataset.rows, dataset.dates):
            if dates.tolist() != plan["calendar_date_offsets"][row["TILE"].upper()]:
                raise ValueError("The input dates do not match the published PASTIS plan")
        with ThreadPoolExecutor(4) as pool:
            samples = list(pool.map(dataset.__getitem__, range(min(limit or len(dataset), len(dataset)))))
        model = network(name, fold)
        for repeat, percent in todo:
            indices = retained_orders(plan, repeat, percent) if percent else None
            score = score_pastis(model, dataset, samples, indices)
            save(dict(fold=fold, repeat=repeat, missing_pct=percent, mIoU=score))
        del samples, model
        torch.cuda.empty_cache()


def external_samples(root, calendar, limit):
    manifest = read(root / "manifest.json")
    calibration = read(root / "calibration.json")
    bands = calibration["bands"]
    coefficients = torch.tensor([calibration["observations"][day]["coefficients"] for day in calendar],
                                device="cuda", dtype=torch.float32)
    patches = []
    for record in manifest["patches"][:limit]:
        with np.load(root / record["file"], allow_pickle=False) as item:
            if item["dates"].tolist() != record["dates"] or item["bands"].tolist() != bands:
                raise ValueError(f"Patch metadata mismatch: {record['file']}")
            raw = item["s2"]
        positions = np.array([calendar.index(day) for day in record["dates"]])
        label = np.load(root / record["audited_label_file"], allow_pickle=False)
        patches.append((raw, positions, label, record["shape"]))
    return patches, coefficients


def ensemble(name):
    stats = read(DATA / "NORM_S2_patch.json")
    members = []
    for fold in FOLDS:
        training, _, _ = fold_split(fold)
        norm = [torch.tensor(np.mean([stats[f"Fold_{i}"][key] for i in training], axis=0),
                             device="cuda", dtype=torch.float32)[None, :, None, None]
                for key in ("mean", "std")]
        members.append((network(name, fold), *norm))
    return members


@torch.inference_mode()
def score_external(members, data, calendar, deleted):
    samples, coefficients = data
    matrix = torch.zeros((20, 20), dtype=torch.long, device="cuda")
    boundary = BoundaryMetrics("cuda")
    for raw, positions, y, (h, w) in samples:
        mask = ~np.isin(positions, deleted)
        kept = positions[mask]
        if not len(kept):
            raise ValueError("A patch has no retained observations")
        x = torch.as_tensor(raw[mask], device="cuda", dtype=torch.float32)
        scale = coefficients[kept]
        x.mul_(scale[:, :, 0, None, None]).add_(scale[:, :, 1, None, None])
        dates = torch.tensor([[(date.fromisoformat(calendar[i]) - REFERENCE_DATE).days for i in kept]],
                             device="cuda")
        valid = torch.ones_like(dates, dtype=torch.bool)
        total = torch.zeros((1, 20, 128, 128), device="cuda")
        for model, mu, sd in members:
            z = (x - mu) / sd
            z[:, :, h:, :] = 0
            z[:, :, :, w:] = 0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(z.unsqueeze(0), batch_positions=dates, valid_mask=valid)
            total += logits.float().softmax(1)
        pred = total.argmax(1)
        target = torch.as_tensor(y, device="cuda", dtype=torch.long)[None]
        confusion_add(matrix, pred, target)
        boundary.add(pred, target)
    return matrix, boundary


def external_metrics(cm, boundary):
    c = cm.cpu().numpy()[:19, :]
    intersection = np.diag(c[:, :19])
    union = c.sum(1) + c[:, :19].sum(0) - intersection
    active = c.sum(1) > 0
    miou = 100 * (intersection[active] / union[active]).mean()
    return dict(mIoU=float(miou), OA=segmentation_metrics(cm)["target_masked_accuracy"],
                BF1=boundary.value()["boundary_f1"])


def evaluate_external(name, protocol, cases, seen, save, roots, limit=None):
    plans = {(p["region"], p["repeat"]): p for p in protocol["external_masks"]}
    calendars = {r: plans[r, 1]["calendar"] for r in roots}
    data = {r: external_samples(d, calendars[r], limit) for r, d in roots.items()}
    members = ensemble(name)
    for repeat, percent in cases:
        if (None, repeat, percent) in seen:
            continue
        results = {}
        combined = torch.zeros((20, 20), dtype=torch.long, device="cuda")
        both = BoundaryMetrics("cuda")
        for r, calendar in calendars.items():
            deleted = plans[r, repeat or 1]["order"][:percent // 2]
            cm, boundary = score_external(members, data[r], calendar, deleted)
            results[r] = external_metrics(cm, boundary)
            combined += cm
            both.counts += boundary.counts
        results["combined"] = external_metrics(combined, both)
        save(dict(repeat=repeat, missing_pct=percent, results=results))


def main():
    protocol = read(PROTOCOL)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=protocol["models"])
    parser.add_argument("--external", type=Path, nargs=2, metavar=("BEAUCE", "ANGERS"),
                        help="Extracted regional archive folders. Without this option, evaluate PASTIS.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true", help="Evaluate two patches per fold or region at 0%% and 90%%.")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("Published evaluation uses CUDA and BF16")
    torch.set_num_threads(4)
    identity = {"model": args.model, "dataset": "external" if args.external else "pastis", "smoke": args.smoke}
    data = read(args.output) if args.output.is_file() else dict(**identity, runs=[])
    if any(data[k] != v for k, v in identity.items()):
        parser.error("The output file belongs to a different run")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = [(0, 0), (1, 90)] if args.smoke else [(0, 0), *product(range(1, 11), range(10, 91, 10))]
    seen = {(r.get("fold"), r["repeat"], r["missing_pct"]) for r in data["runs"]}

    def save(row):
        data["runs"].append(row)
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, allow_nan=False) + "\n", encoding="utf-8")
        tmp.replace(args.output)
        print(f"{args.model}: completed case {len(data['runs'])}", flush=True)

    params = (args.model, protocol, cases, seen, save)
    limit = 2 if args.smoke else None
    if args.external:
        evaluate_external(*params, dict(zip(("Beauce", "Angers"), args.external)), limit)
    else:
        evaluate_pastis(*params, limit)

if __name__ == "__main__":
    main()
