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
    "utae_enhanced_flip_cosine40": "e1_td20",
    "u_estae_flip_cosine": "e1_td20_gpu",
    "u_estae_e0_cosine60": "e0",
    "u_estae_e0_cosine40": "e0",
    "u_estae_6x16_e1_cosine40": "e1",
    "u_estae_6x16_e1_td20_cosine40": "e1_td20_gpu",
    "u_estae_6x16_flip_cosine": "e1_td20_gpu",
    "u_estae_12x24_e0_cosine40": "e0",
    "u_estae_pro_24x24_sigma30_e0_cosine40": "e0",
    "u_estae_pro_24x24_sigma30": "e1_td20_gpu",
    "u_estae_pro_32x32_sigma60_e0_cosine40": "e0",
    "u_estae_pro_32x32_sigma60": "e1_td20_gpu",
}
COSINE_START = {
    "utae_cosine60": 41,
    "u_estae_flip_cosine": 61,
    "u_estae_e0_cosine60": 41,
    "u_estae_e0_cosine40": 61,
    "utae_enhanced_flip_cosine40": 61,
    **dict.fromkeys(("u_estae_6x16_e1_cosine40", "u_estae_6x16_e1_td20_cosine40",
                     "u_estae_6x16_flip_cosine", "u_estae_12x24_e0_cosine40",
                     "u_estae_pro_24x24_sigma30_e0_cosine40", "u_estae_pro_24x24_sigma30",
                     "u_estae_pro_32x32_sigma60_e0_cosine40", "u_estae_pro_32x32_sigma60"), 61),
}
FLIP_TRAINING = "u_estae_flip_cosine"
PROJECTED_FLIPS = {
    FLIP_TRAINING, "u_estae_6x16_flip_cosine", "u_estae_pro_24x24_sigma30", "u_estae_pro_32x32_sigma60",
}
INPUT_FLIPS = {"utae_enhanced_flip_cosine40"}
UESTAE_ARCHITECTURES = {
    **{key: (6, 16, 60.) for key in (
        "u_estae_e0_cosine40", "u_estae_e0_cosine60", "u_estae_6x16_e1_cosine40",
        "u_estae_6x16_e1_td20_cosine40", "u_estae_6x16_flip_cosine")},
    **{key: (12, 24, 60.) for key in ("u_estae", FLIP_TRAINING, "u_estae_12x24_e0_cosine40")},
    **{key: (24, 24, 30.) for key in ("u_estae_pro_24x24_sigma30_e0_cosine40", "u_estae_pro_24x24_sigma30")},
    **{key: (32, 32, 60.) for key in ("u_estae_pro_32x32_sigma60_e0_cosine40", "u_estae_pro_32x32_sigma60")},
}


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
    if name in UESTAE_ARCHITECTURES:
        windows, features, sigma = UESTAE_ARCHITECTURES[name]
        return StateTrendUNet(windows=windows, features=features, sigma_days=sigma)
    if name in ("utae", "utae_enhanced", "utae_cosine60", "utae_enhanced_flip_cosine40"):
        return get_model("utae")
    raise ValueError(f"Unknown model: {name}")
