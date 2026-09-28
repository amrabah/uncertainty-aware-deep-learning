import json
import random
from pathlib import Path
import numpy as np
import torch
from .data import load_data
from .metrics import accuracy, brier_score, expected_calibration_error, selective_accuracy
from .model import DropoutMLP
from .train import predict, train_model
from .uncertainty import mc_predict

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    set_seed()
    data = load_data(seed=SEED)
    model = DropoutMLP(data.input_dim)
    model = train_model(model, data.train, data.val)

    det_probs, y = predict(model, data.test)
    mc_probs, mc_y, entropy, mutual_information = mc_predict(model, data.test, passes=50)
    assert np.array_equal(y, mc_y)

    metrics = {
        "deterministic": {
            "accuracy": accuracy(det_probs, y),
            "ece": expected_calibration_error(det_probs, y),
            "brier": brier_score(det_probs, y),
        },
        "mc_dropout": {
            "accuracy": accuracy(mc_probs, y),
            "ece": expected_calibration_error(mc_probs, y),
            "brier": brier_score(mc_probs, y),
            "mean_predictive_entropy": float(entropy.mean()),
            "mean_mutual_information": float(mutual_information.mean()),
            "selective_accuracy_by_coverage": selective_accuracy(
                mc_probs, y, mutual_information
            ),
        },
        "seed": SEED,
        "mc_passes": 50,
    }

    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
