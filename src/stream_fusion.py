import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from time import sleep
from datetime import datetime
from utils import plot_live

# === Step 1: Generate synthetic live data ===
def generate_sensor_data(n=1000):
    np.random.seed(42)
    timestamps = pd.date_range(start=datetime.now(), periods=n, freq='S')
    temperature = np.random.normal(25, 1, n)
    humidity = np.random.normal(50, 5, n)
    pressure = np.random.normal(1013, 3, n)

    # Inject anomalies
    for i in range(0, n, 150):
        temperature[i] += np.random.normal(10, 2)
        humidity[i] += np.random.normal(15, 3)
        pressure[i] -= np.random.normal(15, 4)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure
    })
    return df

# === Step 2: Real-time streaming & anomaly detection ===
def stream_and_detect(df):
    clf = IsolationForest(contamination=0.02, random_state=42)
    window = []

    for i in range(len(df)):
        # simulate streaming
        current = df.iloc[i]
        window.append(current[['temperature', 'humidity', 'pressure']].values)
        if len(window) > 50:
            window.pop(0)
        data = np.array(window)
        clf.fit(data)
        pred = clf.predict(data)[-1]

        # visualize current reading
        is_anomaly = pred == -1
        plot_live(current['timestamp'], current['temperature'], is_anomaly)

        sleep(0.05)

if __name__ == "__main__":
    df = generate_sensor_data(500)
    stream_and_detect(df)
