import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from .model_trimodal import TriModalHABNet

def scale_modalities(X_sat, X_in, X_meta):
    s1, s2, s3 = StandardScaler(), StandardScaler(), StandardScaler()
    Xs = s1.fit_transform(X_sat)
    Xi = s2.fit_transform(X_in)
    Xm = s3.fit_transform(X_meta)
    return Xs.astype(np.float32), Xi.astype(np.float32), Xm.astype(np.float32), (s1, s2, s3)

def apply_scalers(X_sat, X_in, X_meta, scalers):
    s1, s2, s3 = scalers
    return (
        s1.transform(X_sat).astype(np.float32),
        s2.transform(X_in).astype(np.float32),
        s3.transform(X_meta).astype(np.float32),
    )

def train_model(X_sat, X_in, X_meta, y, epochs=10, lr=2e-3, batch_size=128, seed=42):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TriModalHABNet(
        sat_dim=X_sat.shape[1],
        insitu_dim=X_in.shape[1],
        meta_dim=X_meta.shape[1],
        hid=64,
    ).to(device)

    ds = TensorDataset(
        torch.from_numpy(X_sat), torch.from_numpy(X_in), torch.from_numpy(X_meta),
        torch.from_numpy(y.astype(np.int64)),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        for b_sat, b_in, b_meta, b_y in dl:
            b_sat, b_in, b_meta = b_sat.to(device), b_in.to(device), b_meta.to(device)
            b_y = b_y.float().to(device)
            logit, _ = model(b_sat, b_in, b_meta)
            loss = loss_fn(logit, b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    return model
