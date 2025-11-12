# Sensor-Fusion-Anomaly
Multi-sensor anomaly detection using synthetic data. The project simulates temperature, humidity, and pressure sensors and applies Isolation Forest to detect anomalies.  

## Features
- Multi-sensor fusion
- Real-time anomaly detection (demonstration)
- Visualization of detected anomalies

## Structure
Sensor-Fusion-Anomaly/
├── data/
│   └── synthetic_sensors.csv
├── src/
│   ├── generate_data.py
│   ├── anomaly_detection.py
│   └── visualize.py
├── README.md
├── requirements.txt
└── .gitignore

## How to run
```bash
pip install -r requirements.txt
python src/generate_data.py
python src/anomaly_detection.py
python src/visualize.py



