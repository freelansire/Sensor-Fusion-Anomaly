from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import pandas as pd
import numpy as np
import requests

EPA_FILES = {
    "model_data": "https://pasteur.epa.gov/uploads/10.23719/1524299/HABsRisk_CyAN-NLA_ModelData.csv",
    "model_data_meta": "https://pasteur.epa.gov/uploads/10.23719/1524299/HABsRisk_CyAN-NLA_ModelData_meta.csv",
    "model_output": "https://pasteur.epa.gov/uploads/10.23719/1524299/HABsRisk_CyAN-NLA_ModelOutput.csv",
    "model_output_meta": "https://pasteur.epa.gov/uploads/10.23719/1524299/HABsRisk_CyAN-NLA_ModelOutput_meta.csv",
}

@dataclass
class Modalities:
    sat_cols: list[str]
    insitu_cols: list[str]
    meta_cols: list[str]
    label_col: str

def _download(url: str, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        return
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out.write_bytes(r.content)

def ensure_epa_data(data_dir: str = "data") -> dict[str, Path]:
    data_dir = Path(data_dir)
    paths = {}
    for k, url in EPA_FILES.items():
        p = data_dir / f"{k}.csv"
        _download(url, p)
        paths[k] = p
    return paths

def load_epa_model_data(data_dir: str = "data") -> pd.DataFrame:
    paths = ensure_epa_data(data_dir)
    # low_memory=False to avoid mixed dtype weirdness
    df = pd.read_csv(paths["model_data"], low_memory=False)
    return df

def suggest_label_columns(df: pd.DataFrame) -> list[str]:
    """
    Prefer real outcome columns and avoid sample-index fields like INDXSAMP_*.
    Rank by (non-null rate) * (value variability).
    """
    key = re.compile(r"(microcystin|toxin|cyan|cyano|chlor|chl|bloom|hab)", re.IGNORECASE)
    candidates = [c for c in df.columns if key.search(c)]

    # drop common "not-label" patterns
    bad = re.compile(r"^(INDXSAMP_|INDEXSAMP_|SAMP_|SAMPLE_|FLAG_)", re.IGNORECASE)
    candidates = [c for c in candidates if not bad.search(c)]

    scored = []
    for c in candidates:
        s = df[c]
        # coerce to numeric where possible
        if not pd.api.types.is_numeric_dtype(s):
            s = pd.to_numeric(s, errors="coerce")
        non_null = float(s.notna().mean())
        uniq = s.dropna().nunique()
        # prefer columns with signal and enough coverage
        score = non_null * np.log1p(uniq)
        if non_null >= 0.2 and uniq >= 2:
            scored.append((score, c))

    scored.sort(reverse=True)
    return [c for _, c in scored] if scored else candidates

def infer_modalities(df: pd.DataFrame, label_col: str) -> Modalities:
    cols = [c for c in df.columns if c != label_col]

    # numeric features only
    num_cols = []
    for c in cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            num_cols.append(c)
        else:
            # allow coercion to numeric
            coerced = pd.to_numeric(s, errors="coerce")
            if coerced.notna().mean() > 0.8:
                df[c] = coerced
                num_cols.append(c)

    sat_pat = re.compile(r"(cyan|cyAN|sat|remote|reflect|index|ndvi|ndwi|pixel|band|chl)", re.IGNORECASE)
    insitu_pat = re.compile(r"(microcystin|toxin|phyc|secchi|turb|do\b|oxygen|temp|conduct|tp\b|tn\b|nitrate|ammonia|phosph|nitrogen|cell)", re.IGNORECASE)
    meta_pat = re.compile(r"(lat|lon|long|elev|depth|area|region|ecoreg|huc|state|basin|watershed|climate|land|use)", re.IGNORECASE)

    sat_cols = [c for c in num_cols if sat_pat.search(c)]
    insitu_cols = [c for c in num_cols if insitu_pat.search(c) and c not in sat_cols]
    meta_cols = [c for c in num_cols if meta_pat.search(c) and c not in sat_cols and c not in insitu_cols]

    remaining = [c for c in num_cols if c not in sat_cols and c not in insitu_cols and c not in meta_cols]
    meta_cols += remaining

    # ensure non-empty groups
    if len(sat_cols) == 0:
        sat_cols = meta_cols[: max(1, len(meta_cols)//6)]
        meta_cols = [c for c in meta_cols if c not in sat_cols]
    if len(insitu_cols) == 0:
        insitu_cols = meta_cols[: max(1, len(meta_cols)//6)]
        meta_cols = [c for c in meta_cols if c not in insitu_cols]

    return Modalities(sat_cols=sat_cols, insitu_cols=insitu_cols, meta_cols=meta_cols, label_col=label_col)

def _median_impute(X: np.ndarray) -> np.ndarray:
    X = X.copy().astype(np.float32)
    med = np.nanmedian(X, axis=0)
    # if a whole column is NaN, nanmedian returns NaN; replace those with 0
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(med, inds[1])
    return X

def make_xy(df: pd.DataFrame, mods: Modalities):
    """
    Builds X_sat, X_in, X_meta, y and returns also a filtered df with valid rows.
    - Drops rows where label is missing (for non-binary labels)
    - Converts continuous labels into an event label via high-quantile threshold
    - Median-imputes missing features to prevent NaNs in training/prediction
    """
    # label
    y_raw = df[mods.label_col]
    if not pd.api.types.is_numeric_dtype(y_raw):
        y_raw = pd.to_numeric(y_raw, errors="coerce")

    # Decide if binary
    vals = set(pd.Series(y_raw.dropna().unique()).head(20).tolist())
    is_binary = vals.issubset({0, 1})

    if is_binary:
        mask = y_raw.notna()
        y = y_raw[mask].astype(np.int64).to_numpy()
    else:
        mask = y_raw.notna()
        y_cont = y_raw[mask].to_numpy(dtype=np.float32)
        # choose a threshold that guarantees positives (unless constant)
        if np.nanstd(y_cont) < 1e-8:
            # constant label -> can't learn; create tiny positives to avoid crashes (still warn in app)
            thr = np.nanmax(y_cont)
        else:
            thr = float(np.nanquantile(y_cont, 0.85))
        y = (y_cont >= thr).astype(np.int64)

    df_f = df.loc[mask].reset_index(drop=True)

    X_sat = df_f[mods.sat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    X_in  = df_f[mods.insitu_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    X_meta= df_f[mods.meta_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    X_sat = _median_impute(X_sat)
    X_in  = _median_impute(X_in)
    X_meta= _median_impute(X_meta)

    return df_f, X_sat, X_in, X_meta, y
