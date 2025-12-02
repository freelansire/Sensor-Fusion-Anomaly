import torch
import torch.nn as nn

class TemporalEncoder(nn.Module):
    def __init__(self, in_dim=3, hid=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hid, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hid, hid, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(hid, hid)

    def forward(self, x):  # x: (B, T, C)
        x = x.transpose(1, 2)          # (B, C, T)
        h = self.net(x).squeeze(-1)    # (B, hid)
        return self.proj(h)            # (B, hid)

class StaticEncoder(nn.Module):
    def __init__(self, in_dim=5, hid=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, hid),
            nn.GELU(),
        )

    def forward(self, x):  # (B, in_dim)
        return self.net(x)

class GatedAttentionFusion(nn.Module):
    """
    Learns modality weights (softmax) over [temporal, static].
    """
    def __init__(self, hid=64):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hid, hid),
            nn.Tanh(),
            nn.Linear(hid, 1)
        )

    def forward(self, embs):  # embs: (B, M, hid)
        logits = self.score(embs).squeeze(-1)     # (B, M)
        w = torch.softmax(logits, dim=-1)         # (B, M)
        fused = (embs * w.unsqueeze(-1)).sum(dim=1)  # (B, hid)
        return fused, w

class HABFusionNet(nn.Module):
    def __init__(self, ts_dim=3, static_dim=5, hid=64):
        super().__init__()
        self.temporal = TemporalEncoder(ts_dim, hid)
        self.static = StaticEncoder(static_dim, hid)
        self.fusion = GatedAttentionFusion(hid)
        self.head = nn.Sequential(
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hid, 1)
        )

    def forward(self, x_ts, x_static):
        e_ts = self.temporal(x_ts)
        e_st = self.static(x_static)
        embs = torch.stack([e_ts, e_st], dim=1)  # (B, 2, hid)
        fused, weights = self.fusion(embs)
        logit = self.head(fused).squeeze(-1)
        return logit, weights
