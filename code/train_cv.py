"""Train and evaluate one fold of a final model configuration on PASTIS."""

import argparse
from pathlib import Path
import random

import numpy as np
import torch

from cv_models import FOLDS, TRAINING_VARIANTS, build_model, fold_split
from experiment import save, stage
from fast_data import GPUNormalizer, RawPastis
from reference_model import StateTrendUNet


SEED = 1
EPOCHS = 100
LR = .001


def initial_model(name):
    if name != "u_estae":
        return build_model(name)
    reference = StateTrendUNet().state_dict()
    torch.manual_seed(SEED)
    model = build_model(name)
    with torch.no_grad():
        for key, parameter in model.named_parameters():
            if key in reference and reference[key].shape == parameter.shape:
                parameter.copy_(reference[key])
    return model


def store_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=TRAINING_VARIANTS, required=True)
    parser.add_argument("--test-fold", type=int, choices=FOLDS, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/five_fold"))
    parser.add_argument("--smoke", action="store_true", help="Run two training batches and one validation batch")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("Training requires a CUDA GPU with BF16 support")
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.set_per_process_memory_fraction(.85)

    train_folds, val_fold, test_fold = fold_split(args.test_fold)
    tag = f"fold_{test_fold}" + ("_smoke" if args.smoke else "")
    folder = args.output_dir / args.model / tag
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(model=args.model, folds=dict(train=train_folds, val=val_fold, test=test_fold),
                  epochs=EPOCHS, seed=SEED, batch_size=2, lr=LR, optimizer="Adam",
                  training_variant=TRAINING_VARIANTS[args.model])
    train, validation, test = [RawPastis(folds) for folds in (train_folds, [val_fold], [test_fold])]
    norms = [GPUNormalizer(dataset) for dataset in (train, validation, test)]
    val_indices, test_indices = [list(range(len(dataset))) for dataset in (validation, test)]
    if test_fold == 5:
        orders = np.load(Path(__file__).with_name("epoch_orders.npy"))
    else:
        rng = np.random.default_rng(SEED)
        orders = [rng.permutation(len(train)) for _ in range(EPOCHS)]
    model = initial_model(args.model).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    variant = TRAINING_VARIANTS[args.model]
    if args.smoke:
        stage(model, train, orders[0].tolist(), norms[0], folder, "smoke_train", 1, variant, optimizer, limit=2)
        stage(model, validation, val_indices, norms[1], folder, "smoke_val", 1, limit=1)
        return

    last = folder / "last.pt"
    history, first, best = [], 1, None
    if last.exists():
        checkpoint = torch.load(last, map_location="cpu", weights_only=True)
        if checkpoint["config"] != config:
            raise ValueError("Cannot resume because the saved training configuration differs")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        history, best, first = checkpoint["history"], checkpoint["best_checkpoint"], checkpoint["epoch"] + 1
    save(folder / "config.json", config)
    for epoch in range(first, EPOCHS + 1):
        tr, _ = stage(model, train, orders[epoch-1].tolist(), norms[0], folder, "train", epoch, variant, optimizer)
        va, _ = stage(model, validation, val_indices, norms[1], folder, "val", epoch)
        history.append(dict(epoch=epoch, train=tr, val=va))
        if best is None or va["mIoU"] > best["val"]["mIoU"]:
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best = dict(epoch=epoch, state_dict=state, val=va)
        store_checkpoint(last, dict(config=config, epoch=epoch, best_checkpoint=best, state_dict=model.state_dict(),
                                    optimizer=optimizer.state_dict(), history=history, rng=torch.get_rng_state(),
                                    cuda_rng=torch.cuda.get_rng_state_all()))
        save(folder / "history.json", history)

    store_checkpoint(folder / "best.pt", best)
    model.load_state_dict(best["state_dict"], strict=True)
    test_metrics, confusion = stage(model, test, test_indices, norms[2], folder, "test", best["epoch"])
    save(folder / "test_metrics.json", test_metrics)
    np.save(folder / "confusion_test.npy", confusion)
    print(f"Test fold {test_fold}, epoch {best['epoch']}: mIoU {test_metrics['mIoU']:.2f}%", flush=True)


if __name__ == "__main__":
    main()
