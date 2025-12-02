import time
import numpy as np
import pandas as pd
import streamlit as st
import torch

from sklearn.model_selection import StratifiedShuffleSplit

from src.epa_hab_dataset import load_epa_model_data, suggest_label_columns, infer_modalities, make_xy
from src.train_tabular import scale_modalities, apply_scalers, train_model
from src.drift import drift_score
from src.baselines import RFGBEnsemble
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


st.set_page_config(page_title="HAB Detection via Sensor Fusion", layout="wide")
st.title("🚨 HAB Detection (Real Data): Satellite + In-situ + Metadata Fusion")
st.caption("Real EPA HAB-related dataset • Tri-modal attention fusion • RF/GB baseline • Live streaming demo • Drift monitoring")


st.sidebar.markdown(
    """
    👤
    <a href="https://github.com/freelansire/Sensor-Fusion-Anomaly" target="_blank" style="text-decoration:none;">GitHub</a>
    <a href="https://freelansire.com" target="_blank" style="text-decoration:none;">Website</a>
    """,
    unsafe_allow_html=True
)

with st.sidebar:

    st.header("Dataset")
    data_dir = st.text_input("Data folder", "data")
    epochs = st.slider("Attention model epochs", 3, 25, 10, 1)
    speed = st.slider("Stream speed", 0.01, 0.3, 0.05, 0.01)
    risk_threshold = st.slider("Alert threshold", 0.1, 0.9, 0.6, 0.05)
    drift_threshold = st.slider("Drift alert threshold", 0.2, 2.0, 0.75, 0.05)


@st.cache_data(show_spinner=True)
def _load_df(data_dir: str):
    return load_epa_model_data(data_dir)

df0 = _load_df(data_dir)
st.success(f"Loaded EPA dataset with {len(df0):,} rows and {df0.shape[1]:,} columns.")

label_suggestions = suggest_label_columns(df0)
if not label_suggestions:
    st.error("No HAB-like label columns were auto-detected. Try a different dataset file.")
    st.stop()

label_col = st.selectbox(
    "Choose HAB label column (auto-suggested)",
    options=label_suggestions,
    index=0,
)

mods = infer_modalities(df0, label_col=label_col)
df, X_sat, X_in, X_meta, y = make_xy(df0, mods)

pos_rate = float(y.mean()) if len(y) else 0.0
st.write("**Modalities (auto-grouped):**")
st.write({
    "rows_used_after_cleaning": len(df),
    "satellite_features": len(mods.sat_cols),
    "in_situ_features": len(mods.insitu_cols),
    "metadata_features": len(mods.meta_cols),
    "label": mods.label_col,
    "positive_rate": round(pos_rate, 4),
})

if pos_rate <= 0.001 or pos_rate >= 0.999:
    st.warning(
        f"Label `{label_col}` yields a nearly single-class target (pos_rate={pos_rate:.3f}). "
        "Pick another label from the dropdown for a meaningful HAB detection task."
    )

if len(np.unique(y)) < 2:
    st.error("Selected label produces only one class after cleaning. Choose another label.")
    st.stop()

# ---- Stratified split ----
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
tr_idx, te_idx = next(sss.split(np.zeros(len(y)), y))

# ---- Scale modalities for attention model ----
Xs_tr, Xi_tr, Xm_tr, scalers = scale_modalities(X_sat[tr_idx], X_in[tr_idx], X_meta[tr_idx])
Xs_te, Xi_te, Xm_te = apply_scalers(X_sat[te_idx], X_in[te_idx], X_meta[te_idx], scalers)
y_tr, y_te = y[tr_idx], y[te_idx]

# scaled full arrays for streaming
Xs_all, Xi_all, Xm_all = apply_scalers(X_sat, X_in, X_meta, scalers)

# ---- Baseline features (flatten tri-modal scaled features) ----
X_base_tr = np.hstack([Xs_tr, Xi_tr, Xm_tr])
X_base_te = np.hstack([Xs_te, Xi_te, Xm_te])
X_base_all = np.hstack([Xs_all, Xi_all, Xm_all])

def approx_auc(y_true, p):
    y_true = np.asarray(y_true)
    pos = p[y_true == 1]
    neg = p[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float((pos.reshape(-1, 1) > neg.reshape(1, -1)).mean())

if st.button("🧠 Train attention model + RF/GB baseline", type="primary"):
    # attention model
    model = train_model(Xs_tr, Xi_tr, Xm_tr, y_tr, epochs=epochs)
    st.session_state["model"] = model
    st.session_state["train_ref"] = np.hstack([Xs_tr, Xi_tr, Xm_tr])

    # baseline
    base = RFGBEnsemble(seed=42).fit(X_base_tr, y_tr)
    st.session_state["baseline"] = base

    st.session_state["stream_i"] = 0
    st.success("Both models trained.")

if "model" not in st.session_state or "baseline" not in st.session_state:
    st.info("Train the models to enable evaluation + live detection.")
    st.stop()


# ---- helper function for Precision / Recall / F1 + Confusion Matrix ----

model = st.session_state["model"]
baseline = st.session_state["baseline"]
train_ref = st.session_state["train_ref"]

def cls_metrics(y_true, p, threshold: float):
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p)
    y_pred = (p >= threshold).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    }


# ---- Holdout evaluation side-by-side ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

with torch.no_grad():
    logit, w = model(
        torch.from_numpy(Xs_te).to(device),
        torch.from_numpy(Xi_te).to(device),
        torch.from_numpy(Xm_te).to(device),
    )
    p_attn = torch.sigmoid(logit).cpu().numpy()

p_base = baseline.predict_proba(X_base_te)

att_m = cls_metrics(y_te, p_attn, threshold=risk_threshold)
base_m = cls_metrics(y_te, p_base, threshold=risk_threshold)

st.subheader("2) Holdout Metrics (at alert threshold)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Attention approx-AUC", f"{approx_auc(y_te, p_attn):.3f}")
c2.metric("Baseline approx-AUC", f"{approx_auc(y_te, p_base):.3f}")
c3.metric("Positive rate", f"{pos_rate:.3f}")
c4.metric("Threshold", f"{risk_threshold:.2f}")

a1, a2, a3 = st.columns(3)
a1.metric("Attention Precision", f"{att_m['precision']:.3f}")
a2.metric("Attention Recall", f"{att_m['recall']:.3f}")
a3.metric("Attention F1", f"{att_m['f1']:.3f}")

b1, b2, b3 = st.columns(3)
b1.metric("Baseline Precision", f"{base_m['precision']:.3f}")
b2.metric("Baseline Recall", f"{base_m['recall']:.3f}")
b3.metric("Baseline F1", f"{base_m['f1']:.3f}")

# Confusion matrices (compact + reviewer-friendly)
cm1, cm2 = st.columns(2)
cm1.markdown("**Attention Confusion (TN / FP / FN / TP)**")
cm1.code(f"TN={att_m['tn']}  FP={att_m['fp']}  FN={att_m['fn']}  TP={att_m['tp']}")
cm2.markdown("**Baseline Confusion (TN / FP / FN / TP)**")
cm2.code(f"TN={base_m['tn']}  FP={base_m['fp']}  FN={base_m['fn']}  TP={base_m['tp']}")


st.divider()
st.subheader("▶️ Live detection stream (row-by-row)")

run = st.toggle("Start streaming", value=False)
step = st.button("Step once")

ph = st.empty()
ph2 = st.empty()

def update(i):
    # Attention model prediction
    xS = Xs_all[i:i+1]
    xI = Xi_all[i:i+1]
    xM = Xm_all[i:i+1]

    with torch.no_grad():
        logit, att = model(
            torch.from_numpy(xS).to(device),
            torch.from_numpy(xI).to(device),
            torch.from_numpy(xM).to(device),
        )
        prob_attn = float(torch.sigmoid(logit).cpu().numpy()[0])
        att = att.cpu().numpy()[0]  # [sat, insitu, meta]

    # Baseline prediction
    prob_base = float(baseline.predict_proba(X_base_all[i:i+1])[0])

    # Drift monitor (last 150 rows)
    start = max(0, i - 150)
    recent = np.hstack([Xs_all[start:i+1], Xi_all[start:i+1], Xm_all[start:i+1]])
    d = drift_score(train_ref, recent)

    alert_attn = prob_attn >= risk_threshold
    alert_base = prob_base >= risk_threshold
    drift_alert = d >= drift_threshold

    ph.markdown(
        f"""
**Row:** `{i}`  

**Attention HAB risk:** `{prob_attn:.3f}`  {'🚨 **ALERT**' if alert_attn else ''}  
**Baseline (RF/GB) risk:** `{prob_base:.3f}`  {'🚨 **ALERT**' if alert_base else ''}  

**Drift score:** `{d:.3f}`  {'🚨 **DRIFT**' if drift_alert else ''}  

**Attention weights:**
- Satellite: `{att[0]:.3f}`
- In-situ: `{att[1]:.3f}`
- Metadata: `{att[2]:.3f}`
        """
    )

    cols_preview = [mods.label_col] + mods.sat_cols[:6] + mods.insitu_cols[:6] + mods.meta_cols[:6]
    cols_preview = [c for c in cols_preview if c in df.columns]
    ph2.dataframe(df.iloc[i:i+1][cols_preview], use_container_width=True)

if "stream_i" not in st.session_state:
    st.session_state["stream_i"] = 0

if step:
    st.session_state["stream_i"] = (st.session_state["stream_i"] + 1) % len(df)
    update(st.session_state["stream_i"])

if run:
    for _ in range(60):
        st.session_state["stream_i"] = (st.session_state["stream_i"] + 1) % len(df)
        update(st.session_state["stream_i"])
        time.sleep(speed)
