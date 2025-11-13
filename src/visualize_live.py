import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/sensor_anomalies.csv', parse_dates=['timestamp'])

plt.figure(figsize=(12,6))
plt.plot(df['timestamp'], df['temperature'], label='Temperature', alpha=0.7)
plt.scatter(df['timestamp'][df['anomaly']==1], 
            df['temperature'][df['anomaly']==1], 
            color='red', label='Anomaly', marker='x')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Sensor Fusion Anomaly Detection (Temperature)')
plt.legend()
plt.tight_layout()
plt.show()
