# Sensor-Fusion-Anomaly
Multi-sensor anomaly detection using synthetic data. The project simulates temperature, humidity, and pressure sensors and applies Isolation Forest to detect anomalies.  

## Features
- Multi-sensor fusion
- Real-time anomaly detection (demonstration)
- Visualization of detected anomalies

## Structure
Sensor-Fusion-Anomaly/<br/>
├── data/ <br/>
│   └── synthetic_sensors.csv <br/>
├── src/ <br/>
│   ├── generate_data.py <br/>
│   ├── anomaly_detection.py <br/>
│   └── visualize.py <br/>
├── README.md <br/>
├── requirements.txt <br/>
└── .gitignore

## How to run
```bash
pip install -r requirements.txt
python src/generate_data.py
python src/anomaly_detection.py
python src/visualize.py



