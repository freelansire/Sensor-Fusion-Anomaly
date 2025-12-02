import time
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.simulate import simulate_hab_dataset
from src.features import make_windows, fit_transform_scalers, transform_with_scalers
from src.train import train_fusion_net
from src.baselines import EnsembleBaselines
from src.drift import drift_score

st.set_page_config(page_title="Sensor Fusion HAB Early-Warning", layout="wide")

st.title("🚨Sensor-Fusion-Anomaly: HAB Early-Warning")
st.caption("Hybrid fusion of IoT sensors + satellite proxies + environmental metadata • Attention fusion + ensemble baselines • Drift monitoring")

with st.sidebar:
    st.header("Simulation")
    n_steps = st.slider("Stream length (steps)", 800, 2500, 1500, 100)
    window = st.slider("Time window (steps)", 24, 96, 48, 6)
    lead_time = st.slider("Early-warning lead time", 6, 48, 12, 3)
    drift_strength = st.slider("Drift strength", 0.0, 2.0, 0.6, 0.1)

    st.header("Model")
    epochs = st.slider("Train epochs (fast)", 2, 15, 6, 1)

    st.header("Live")
    step_speed = st.slider("Stream speed", 0.01, 0.2, 0.05, 0.01)
    drift_threshold = st.slider("Drift alert threshold", 0.2, 2.0, 0.75, 0.05)
    risk_threshold = st.slider("HAB risk threshold", 0.1, 0.9, 0.6, 0.05)

colA, colB = st.columns([1.2, 1.0], gap="large")

if "state" not in st.session_state:
    st.session_state.state = {}

def reset_state():
    st.session_state.state = {}

if st.button("🔄 Reset session", type="secondary"):
    reset_state()

# ---- Generate data ----
if st.button("⚙️ Generate synthetic multimodal stream", type="primary"):
    df, feat_cols, label_col, _ = simulate_hab_dataset(
        n_steps=n_steps, window=window, lead_time=lead_time, drift_strength=drift_strength
    )
    st.session_state.state["df"] = df
    st.session_state.state["feat_cols"] = feat_cols
    st.session_state.state["label_col"] = label_col
    st.session_state.state["idx"] = window + 1
    st.success("Generated synthetic IoT + satellite + metadata stream.")

state = st.session_state.state
if "df" not in state:
    st.info("Click **Generate synthetic multimodal stream** to begin.")
    st.stop()

df: pd.DataFrame = state["df"]
feat_cols = state["feat_cols"]
label_col = state["label_col"]

# ---- Build windows ----
X_ts, X_static, y = make_windows(df, feat_cols, label_col, window=window)
X_ts, X_static, ts_scaler, st_scaler = fit_transform_scalers(X_ts, X_static)

# Train/test split
n = len(y)
split = int(n * 0.75)
Xts_tr, Xst_tr, y_tr = X_ts[:split], X_static[:split], y[:split]
Xts_te, Xst_te, y_te = X_ts[split:], X_static[split:], y[split:]

with colA:
    st.subheader("1) Data + Events")
    st.line_chart(df.set_index("step")[["temperature","turbidity","dissolved_oxygen"]].tail(400))
    st.line_chart(df.set_index("step")[["chl_a","wind","rainfall"]].tail(400))
    st.caption("HAB events are rare; label is **future event** (early warning).")

# ---- Train models ----
if st.button("🧠 Train attention fusion + baselines"):
    with st.spinner("Training fusion model..."):
        model = train_fusion_net(Xts_tr, Xst_tr, y_tr, epochs=epochs)
    base = EnsembleBaselines().fit(Xts_tr, Xst_tr, y_tr)

    state["model"] = model
    state["base"] = base
    state["ts_scaler"] = ts_scaler
    state["st_scaler"] = st_scaler
    state["train_ref"] = np.hstack([Xts_tr.reshape(len(Xts_tr), -1), Xst_tr])
    st.success("Models trained.")

if "model" not in state:
    st.warning("Train models to enable live early-warning + attention weights.")
    st.stop()

model = state["model"]
base = state["base"]
train_ref = state["train_ref"]
model.eval()

# ---- Quick evaluation ----
with colB:
    st.subheader("2) Quick Evaluation (Holdout)")
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        xt = torch.from_numpy(Xts_te).to(device)
        xs = torch.from_numpy(Xst_te).to(device)
        logit, w = model(xt, xs)
        p_fusion = torch.sigmoid(logit).detach().cpu().numpy()

    p_base = base.predict_proba(Xts_te, Xst_te)
    # simple AUC-like proxy without importing extra metrics: rank correlation-ish
    def approx_auc(y_true, p):
        # Mann–Whitney U / AUC approx
        y_true = np.asarray(y_true)
        pos = p[y_true==1]
        neg = p[y_true==0]
        if len(pos)==0 or len(neg)==0:
            return np.nan
        return float((pos.reshape(-1,1) > neg.reshape(1,-1)).mean())

    st.metric("Fusion approx-AUC", f"{approx_auc(y_te, p_fusion):.3f}")
    st.metric("Ensemble approx-AUC", f"{approx_auc(y_te, p_base):.3f}")
    st.caption("Approx-AUC is a quick ranking proxy")

# ---- Live streaming ----
st.divider()
st.subheader("3) Live Streaming Early-Warning + Drift Alert")

if "idx" not in state:
    state["idx"] = window + 1

run = st.toggle("▶️ Start live stream", value=False)
step_btn = st.button("Step once")

# live placeholders
ph_metrics = st.empty()
ph_chart = st.empty()
ph_attn = st.empty()

def predict_at(index_in_df: int):
    # index_in_df maps to windowed index index_in_df - window
    widx = index_in_df - window
    x_ts = X_ts[widx:widx+1]
    x_st = X_static[widx:widx+1]

    with torch.no_grad():
        device = next(model.parameters()).device
        logit, weights = model(
            torch.from_numpy(x_ts).to(device),
            torch.from_numpy(x_st).to(device),
        )
        p_f = float(torch.sigmoid(logit).cpu().numpy()[0])
        att = weights.cpu().numpy()[0]  # [temporal, static]
    p_b = float(base.predict_proba(x_ts, x_st)[0])
    return p_f, p_b, att

def update_ui(i):
    p_f, p_b, att = predict_at(i)

    # drift monitor uses recent flattened features
    recent_widx = max(0, (i-window) - 180)
    recent = np.hstack([
        X_ts[recent_widx:(i-window)].reshape(-1, X_ts.shape[1]*X_ts.shape[2]),
        X_static[recent_widx:(i-window)]
    ]) if (i-window) > recent_widx else train_ref[:50]

    dscore = drift_score(train_ref, recent)

    true_future = int(df.loc[i, label_col]) if i < len(df) else 0
    alert = (p_f >= risk_threshold) or (p_b >= risk_threshold)
    drift_alert = dscore >= drift_threshold

    ph_metrics.markdown(
        f"""
        **Step:** `{i}`  \n
        **Fusion HAB risk:** `{p_f:.3f}`  • **Ensemble risk:** `{p_b:.3f}`  \n
        **Early-warning (future label):** `{true_future}`  \n
        **Drift score:** `{dscore:.3f}`  {'🚨 **DRIFT**' if drift_alert else ''}  \n
        **Alert:** {'🚨 **HAB RISK**' if alert else '—'}
        """
    )

    tail = df.set_index("step").iloc[max(0, i-250):i+1][
        ["temperature","turbidity","dissolved_oxygen","chl_a","wind","hab_event"]
    ]
    ph_chart.line_chart(tail)

    ph_attn.markdown(
        f"**Attention weights**:  \n"
        f"- Temporal (IoT window): `{att[0]:.3f}`  \n"
        f"- Static (satellite + metadata): `{att[1]:.3f}`"
    )

if step_btn:
    state["idx"] = min(state["idx"] + 1, len(df)-1)
    update_ui(state["idx"])

if run:
    # bounded loop so it stays responsive
    for _ in range(120):
        state["idx"] = min(state["idx"] + 1, len(df)-1)
        update_ui(state["idx"])
        time.sleep(step_speed)
        if state["idx"] >= len(df)-2:
            break
