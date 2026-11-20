"""Model and fold definitions for the final three-way comparison."""

from base_models import get_model
from reference_model import StateTrendUNet


FOLDS = range(1, 6)
TRAINING_VARIANTS = {
    "utae": "e0",
    "utae_enhanced": "e1_td20",
    "u_estae": "e1_td20_gpu",
}


def fold_split(test_fold):
    if test_fold not in FOLDS:
        raise ValueError("Test fold must be between 1 and 5")
    validation = (test_fold - 2) % 5 + 1
    training = [fold for fold in FOLDS if fold not in (test_fold, validation)]
    return training, validation, test_fold


def build_model(name):
    if name == "u_estae":
        return StateTrendUNet(windows=12, features=24, sigma_days=60.0)
    if name in ("utae", "utae_enhanced"):
        return get_model("utae")
    raise ValueError(f"Unknown model: {name}")
