"""Architectures and training recipes for the PASTIS experiments."""

import math

from base_models import get_model
from reference_model import StateTrendUNet


FOLDS = range(1, 6)
TRAINING_VARIANTS = {
    "utae": "e0",
    "utae_enhanced": "e1_td20",
    "u_estae": "e1_td20_gpu",
    "utae_cosine60": "e0",
    "u_estae_flip_cosine": "e1_td20_gpu",
    "u_estae_e0_cosine60": "e0",
    "u_estae_e0_cosine40": "e0",
}
COSINE_START = {
    "utae_cosine60": 41,
    "u_estae_flip_cosine": 61,
    "u_estae_e0_cosine60": 41,
    "u_estae_e0_cosine40": 61,
}
FLIP_TRAINING = "u_estae_flip_cosine"


def learning_rate(name, epoch):
    start = COSINE_START.get(name)
    if start is None or epoch < start:
        return .001
    fraction = (epoch - start + 1) / (101 - start)
    return .0001 + .00045 * (1 + math.cos(math.pi * fraction))


def fold_split(test_fold):
    if test_fold not in FOLDS:
        raise ValueError("Test fold must be between 1 and 5")
    validation = (test_fold - 2) % 5 + 1
    training = [fold for fold in FOLDS if fold not in (test_fold, validation)]
    return training, validation, test_fold


def build_model(name):
    if name in ("u_estae", FLIP_TRAINING):
        return StateTrendUNet(windows=12, features=24, sigma_days=60.0)
    if name in ("u_estae_e0_cosine60", "u_estae_e0_cosine40"):
        return StateTrendUNet()
    if name in ("utae", "utae_enhanced", "utae_cosine60"):
        return get_model("utae")
    raise ValueError(f"Unknown model: {name}")
