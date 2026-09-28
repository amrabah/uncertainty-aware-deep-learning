from dataclasses import dataclass
import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    input_dim: int

def _loader(x, y, batch_size, shuffle=False):
    ds = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def load_data(batch_size: int = 64, seed: int = 42) -> DataBundle:
    data = load_breast_cancer()
    x_train, x_tmp, y_train, y_tmp = train_test_split(
        data.data, data.target, test_size=0.30, stratify=data.target, random_state=seed
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed
    )
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    return DataBundle(
        _loader(x_train, y_train, batch_size, True),
        _loader(x_val, y_val, batch_size),
        _loader(x_test, y_test, batch_size),
        x_train.shape[1],
    )
