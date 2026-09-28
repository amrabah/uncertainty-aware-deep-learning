import copy
import numpy as np
import torch
from torch import nn

def train_model(model, train_loader, val_loader, epochs=150, lr=1e-3, patience=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_loss, best_state, stale = float("inf"), None, 0

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

        val_loss = evaluate_loss(model, val_loader, loss_fn)
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

@torch.no_grad()
def evaluate_loss(model, loader, loss_fn=None):
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    model.eval()
    losses = [loss_fn(model(x), y).item() * len(y) for x, y in loader]
    n = sum(len(y) for _, y in loader)
    return sum(losses) / n

@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x), dim=-1).cpu().numpy())
        labels.append(y.cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels)
