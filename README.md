## Sensor-Fusion-Anomaly 🚨
#### HAB Detection using Multi-Modal Fusion (Satellite + In-situ + Environmental Metadata) 

This repository provides a **research-style prototype** for detecting **harmful algal bloom (HAB) risk** using **multi-modal feature fusion**:
- **Satellite-derived proxies** (e.g., remote sensing indices / cyanobacteria-aligned signals where available)
- **In-situ water quality variables** (e.g., chlorophyll-a, cyanobacteria density proxies, toxin-aligned fields where present)
- **Environmental / lake metadata** (e.g., geography/geomorphology features and other context signals)

It includes:
✅ **Tri-modal attention fusion network (PyTorch)**  
✅ **Strong classical baseline ensemble (RF + Gradient Boosting)**  
✅ **Live “streaming” inference demo**  
✅ **Operational thresholding (alerts)** + quick metrics (**AUC proxy, Precision/Recall/F1**, confusion counts)  
✅ **Drift monitoring** (distribution shift alert)  

---

### research prototype:

- **Explicit modality split**: Satellite vs In-situ vs Metadata
- **Attention-based fusion**: the model learns which modality matters *per sample*
- **Baseline comparisons**: RF/GB ensemble beside a neural attention model
- **Operational framing**: thresholded alerts + confusion counts for decision support
- **Robustness hooks**: drift score to flag distribution shift over recent samples

---

### Dataset (Real HAB-related data)
The app uses a public EPA dataset intended for modeling lake HAB risk from satellite and survey features.
The app **downloads the CSVs automatically** to `data/` on first run using URLs embedded in `src/epa_hab_dataset.py`.

**Important:** The dataset contains multiple HAB-relevant fields (e.g., chlorophyll-related, cyanobacteria-related, and toxin-aligned fields depending on the column).  
You choose the **label column** from the UI. For the most defensible “harmful” framing, prefer toxin/cyanobacteria-related labels (e.g., microcystin / cyanobacteria proxies) when available.

---

#### Project Layout
```bash
.
├── app.py                       # Streamlit UI (live demo + metrics)
├── requirements.txt
├── src/
│   ├── epa_hab_dataset.py        # download + label suggestions + modality inference + cleaning
│   ├── model_trimodal.py         # tri-modal attention fusion network
│   ├── train_tabular.py          # scaling + training loop
│   ├── baselines.py              # RF/GB baseline ensemble
│   └── drift.py                  # lightweight drift score
└── data/                         # auto-created (downloaded CSVs)
```

---

### How it works
#### 1) Load + Clean
```bash
- Downloads EPA model data CSV
- Filters out bad label candidates (e.g., sampling index fields)
- Drops rows where label is missing
- Converts continuous outcomes into a binary “event” label via a high-quantile threshold (default: 85th percentile)
- Median-imputes missing feature values to prevent NaNs in training
```

#### 2) Feature Modalities
```bash
We group numeric features into three buckets:
- **Satellite**: remote sensing aligned fields / indices
- **In-situ**: water quality / toxin / cyanobacteria / chlorophyll aligned fields
- **Metadata**: spatial + context + remaining numeric features
```
### 3) Models
```bash
#### A) Attention Fusion Model (PyTorch)
- Each modality passes through its own encoder (MLP)
- An attention module assigns weights across the three modalities
- A prediction head outputs HAB risk probability
```
#### B) Baseline Ensemble (Scikit-learn)
```bash
- Random Forest + Gradient Boosting
- Ensemble probability = average(RF, GB)
```

### 4) Evaluation
```bash
- Stratified train/test split
- Metrics reported at your alert threshold:
  - Approx-AUC (fast rank proxy)
  - Precision / Recall / F1
  - Confusion counts (TN/FP/FN/TP)
```
### 5) Live Streaming Demo
```bash
The app iterates through rows:
- Shows attention risk + baseline risk
- Raises alerts based on your threshold
- Shows modality attention weights
- Shows drift score over a rolling window
```
---

## Quickstart (Local)
```bash
pip install -r requirements.txt
streamlit run app.py
```
