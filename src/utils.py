import matplotlib.pyplot as plt

plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))
x_data, y_data = [], []
anomalies_x, anomalies_y = [], []

def plot_live(timestamp, value, is_anomaly):
    x_data.append(timestamp)
    y_data.append(value)

    ax.clear()
    ax.plot(x_data, y_data, label="Temperature", color="blue")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Real-Time Sensor Fusion Anomaly Detection")

    if is_anomaly:
        anomalies_x.append(timestamp)
        anomalies_y.append(value)

    ax.scatter(anomalies_x, anomalies_y, color="red", label="Anomalies", marker="x")
    ax.legend()
    plt.tight_layout()
    plt.pause(0.001)
