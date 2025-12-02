import numpy as np

# def drift_score(train_ref: np.ndarray, recent: np.ndarray):
#     """
#     Quick drift proxy: compare mean/std distance across features.
#     train_ref: (N, D), recent: (M, D)
#     """
#     mu0, sd0 = train_ref.mean(axis=0), train_ref.std(axis=0) + 1e-9
#     mu1, sd1 = recent.mean(axis=0), recent.std(axis=0) + 1e-9

#     z_mu = np.abs(mu1 - mu0) / sd0
#     z_sd = np.abs(sd1 - sd0) / sd0
#     return float((z_mu.mean() + z_sd.mean()) / 2.0)

def drift_score(train_ref: np.ndarray, recent: np.ndarray):
    mu0, sd0 = train_ref.mean(axis=0), train_ref.std(axis=0) + 1e-9
    mu1, sd1 = recent.mean(axis=0), recent.std(axis=0) + 1e-9
    z_mu = np.abs(mu1 - mu0) / sd0
    z_sd = np.abs(sd1 - sd0) / sd0
    return float((z_mu.mean() + z_sd.mean()) / 2.0)