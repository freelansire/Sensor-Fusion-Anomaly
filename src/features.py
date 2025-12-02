import numpy as np
from sklearn.preprocessing import StandardScaler

def make_windows(df, feature_cols, label_col, window: int):
    X_ts, X_static, y = [], [], []
    arr = df[feature_cols].to_numpy()
    labels = df[label_col].to_numpy()

    # We’ll treat last 3 as metadata-ish (wind,rainfall,season), and sat proxies as static too.
    # Time-series inputs: IoT sensors only (temperature,turbidity,DO) -> first 3 cols
    # Static inputs: satellite + metadata -> remaining cols
    for i in range(window, len(df)):
        ts = arr[i-window:i, 0:3]      # (window, 3)
        static = arr[i, 3:]            # (5,)
        X_ts.append(ts)
        X_static.append(static)
        y.append(labels[i])

    return np.array(X_ts, dtype=np.float32), np.array(X_static, dtype=np.float32), np.array(y, dtype=np.int64)

def fit_transform_scalers(X_ts, X_static):
    ts_scaler = StandardScaler()
    st_scaler = StandardScaler()

    # flatten time for scaling
    n, w, c = X_ts.shape
    X_ts_flat = X_ts.reshape(n*w, c)
    X_ts_scaled = ts_scaler.fit_transform(X_ts_flat).reshape(n, w, c)

    X_static_scaled = st_scaler.fit_transform(X_static)
    return X_ts_scaled.astype(np.float32), X_static_scaled.astype(np.float32), ts_scaler, st_scaler

def transform_with_scalers(X_ts, X_static, ts_scaler, st_scaler):
    n, w, c = X_ts.shape
    X_ts_scaled = ts_scaler.transform(X_ts.reshape(n*w, c)).reshape(n, w, c)
    X_static_scaled = st_scaler.transform(X_static)
    return X_ts_scaled.astype(np.float32), X_static_scaled.astype(np.float32)
