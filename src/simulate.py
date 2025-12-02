import numpy as np
import pandas as pd

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def simulate_hab_dataset(
    n_steps: int = 1500,
    window: int = 48,
    lead_time: int = 12,
    drift_strength: float = 0.6,
    seed: int = 42,
):
    """
    Returns:
      df: dataframe with timestamped features + labels
      feature_cols: list of feature columns
      label_col: label column (hab_event_future)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)

    # ---- Environmental metadata (slow-moving) ----
    wind = np.clip(rng.normal(6, 2, n_steps) + 1.5*np.sin(t/60), 0, None)      # m/s
    rainfall = np.clip(rng.normal(2, 1.2, n_steps) + 1.0*np.sin(t/90+0.7), 0, None)  # mm
    season = (np.sin(2*np.pi*t/365) + 1) / 2  # 0..1

    # ---- IoT sensors (higher frequency, correlated) ----
    temp_base = 18 + 6*season + 0.8*np.sin(t/40)
    # add slow drift to temperature (e.g., seasonal warming / sensor drift)
    temp_drift = drift_strength * (t/n_steps) * 2.0
    temperature = temp_base + temp_drift + rng.normal(0, 0.6, n_steps)

    turbidity = np.clip(3 + 1.2*rainfall + 0.25*np.sin(t/30) + rng.normal(0, 0.5, n_steps), 0, None)
    dissolved_oxygen = np.clip(9.5 - 0.12*(temperature-18) - 0.08*turbidity + rng.normal(0, 0.35, n_steps), 2, None)

    # ---- Satellite proxies (lower frequency: sample/hold every 12 steps) ----
    sat_update = 12
    chl_a = np.zeros(n_steps)
    ndwi = np.zeros(n_steps)  # water index proxy
    for i in range(0, n_steps, sat_update):
        # chl_a increases with warm water + nutrients proxy (rainfall) + calm wind
        chl_level = 1.5 + 0.25*(temperature[i]-18) + 0.15*rainfall[i] - 0.12*wind[i] + rng.normal(0, 0.25)
        chl_level = np.clip(chl_level, 0.2, None)
        chl_a[i:i+sat_update] = chl_level

        ndwi_level = 0.4 + 0.06*rainfall[i] - 0.03*wind[i] + rng.normal(0, 0.05)
        ndwi[i:i+sat_update] = np.clip(ndwi_level, 0.0, 1.0)

    # ---- HAB risk model (latent) ----
    # Risk rises with: high temperature, high chl-a, low DO, low wind, higher turbidity.
    z = (
        0.55*(temperature - np.mean(temperature))/np.std(temperature)
        + 0.75*(chl_a - np.mean(chl_a))/np.std(chl_a)
        - 0.65*(dissolved_oxygen - np.mean(dissolved_oxygen))/np.std(dissolved_oxygen)
        - 0.35*(wind - np.mean(wind))/np.std(wind)
        + 0.25*(turbidity - np.mean(turbidity))/np.std(turbidity)
        + rng.normal(0, 0.25, n_steps)
    )
    hab_prob = sigmoid(z)

    # Rare-ish events; threshold can be tuned
    hab_event = (hab_prob > np.quantile(hab_prob, 0.93)).astype(int)

    # Early warning label: event in the future (lead_time)
    hab_future = np.zeros(n_steps, dtype=int)
    hab_future[:-lead_time] = hab_event[lead_time:]

    df = pd.DataFrame({
        "step": t,
        "temperature": temperature,
        "turbidity": turbidity,
        "dissolved_oxygen": dissolved_oxygen,
        "chl_a": chl_a,
        "ndwi": ndwi,
        "wind": wind,
        "rainfall": rainfall,
        "season": season,
        "hab_event": hab_event,
        "hab_event_future": hab_future,
        "hab_prob_latent": hab_prob,  # for debugging/visuals only
    })

    feature_cols = [
        "temperature","turbidity","dissolved_oxygen",
        "chl_a","ndwi",
        "wind","rainfall","season",
    ]
    label_col = "hab_event_future"
    return df, feature_cols, label_col, window
