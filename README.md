# Sensor-Fusion-Anomaly
This project demonstrates **real-time multi-sensor anomaly detection** using lightweight data fusion and streaming simulation.

## Features
- Simulated streaming from temperature, humidity, and pressure sensors
- Adaptive anomaly detection using Isolation Forest
- Live updating Matplotlib visualization
- Lightweight, reproducible, and educational for IoT / AI research demos

## Structure
Sensor-Fusion-Anomaly/<br/>
├── data/ <br/>
│   └── synthetic_sensors.csv <br/>
├── src/ <br/>
│   ├── stream_fusion.py <br/>
│   ├── utils.py <br/>
│   └── visualize_live.py <br/>
├── README.md <br/>
├── requirements.txt <br/>
└── .gitignore

## 🧠 How It Works
1. Synthetic data is generated to simulate 3 correlated sensors.
2. A small sliding window of recent readings is used to detect anomalies adaptively.
3. Detected anomalies are plotted live in red.

## How to run
```bash
pip install -r requirements.txt
cd src
python stream_fusion.py



