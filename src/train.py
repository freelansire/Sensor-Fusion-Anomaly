import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .model import HABFusionNet

def train_fusion_net(X_ts, X_static, y, lr=2e-3, epochs=6, batch_size=128, seed=42):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HABFusionNet(ts_dim=X_ts.shape[-1], static_dim=X_static.shape[-1], hid=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    ds = TensorDataset(torch.from_numpy(X_ts), torch.from_numpy(X_static), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for _ in range(epochs):
        for b_ts, b_st, b_y in dl:
            b_ts, b_st = b_ts.to(device), b_st.to(device)
            b_y = b_y.float().to(device)
            logit, _ = model(b_ts, b_st)
            loss = loss_fn(logit, b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    return model
