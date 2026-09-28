import numpy as np
import torch
from torch import nn

def enable_dropout(model: nn.Module) -> None:
    """Enable dropout layers while leaving the rest of the model in eval mode."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

@torch.no_grad()
def mc_predict(model, loader, passes: int = 50):
    if passes < 2:
        raise ValueError("MC Dropout requires at least two stochastic passes.")
    all_passes = []
    labels = None
    for _ in range(passes):
        enable_dropout(model)
        probs, ys = [], []
        for x, y in loader:
            probs.append(torch.softmax(model(x), dim=-1).cpu().numpy())
            ys.append(y.cpu().numpy())
        all_passes.append(np.concatenate(probs))
        if labels is None:
            labels = np.concatenate(ys)
    samples = np.stack(all_passes, axis=0)
    mean_probs = samples.mean(axis=0)
    entropy = predictive_entropy(mean_probs)
    expected_entropy = predictive_entropy(samples).mean(axis=0)
    mutual_information = np.maximum(entropy - expected_entropy, 0.0)
    return mean_probs, labels, entropy, mutual_information

def predictive_entropy(probs, eps: float = 1e-12):
    probs = np.clip(probs, eps, 1.0)
    return -(probs * np.log(probs)).sum(axis=-1)
