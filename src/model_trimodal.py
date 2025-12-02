import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim: int, hid: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hid, hid),
            nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)

class AttentionFusion(nn.Module):
    def __init__(self, hid: int = 64):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hid, hid),
            nn.Tanh(),
            nn.Linear(hid, 1)
        )
    def forward(self, embs):  # (B, M, hid)
        logits = self.score(embs).squeeze(-1)      # (B, M)
        w = torch.softmax(logits, dim=-1)          # (B, M)
        fused = (embs * w.unsqueeze(-1)).sum(dim=1)
        return fused, w

class TriModalHABNet(nn.Module):
    def __init__(self, sat_dim: int, insitu_dim: int, meta_dim: int, hid: int = 64):
        super().__init__()
        self.sat = MLP(sat_dim, hid)
        self.insitu = MLP(insitu_dim, hid)
        self.meta = MLP(meta_dim, hid)
        self.fuse = AttentionFusion(hid)
        self.head = nn.Sequential(
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hid, 1)
        )

    def forward(self, x_sat, x_in, x_meta):
        e1 = self.sat(x_sat)
        e2 = self.insitu(x_in)
        e3 = self.meta(x_meta)
        embs = torch.stack([e1, e2, e3], dim=1)  # (B, 3, hid)
        fused, w = self.fuse(embs)
        logit = self.head(fused).squeeze(-1)
        return logit, w
