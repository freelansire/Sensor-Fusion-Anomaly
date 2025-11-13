# 🛰️ Sensor-Fusion-Anomaly
*A real-time lightweight anomaly detection framework for multi-sensor data streams.*

![Demo](./demo.gif)

## 🔍 Overview
This repository demonstrates a **real-time anomaly detection prototype** that fuses readings from multiple simulated sensors (temperature, humidity, pressure) and detects irregularities using a lightweight **Isolation Forest** model.  
It showcases a practical application of **sensor fusion**, **stream processing**, and **adaptive machine learning** — principles central to modern IoT and cyber-physical systems.

---

## ⚙️ Features
- 📡 Synthetic multi-sensor streaming (temperature, humidity, pressure)  
- 🧠 Adaptive anomaly detection using Isolation Forest  
- 🔁 Real-time visualisation with Matplotlib (auto-refreshing live plot)  
- 🧩 Modular and lightweight – runs in <100 lines of Python  
- 💡 Demonstrates principles of edge AI, IoT monitoring, and streaming analytics  

---

## 🧠 Methodology
1. **Sensor Simulation:** Sensors generate time-correlated signals with random Gaussian noise and injected outliers.  
2. **Feature Fusion:** Incoming data are combined into a small rolling window to represent the system state.  
3. **Adaptive Learning:** Isolation Forest updates on the window to detect deviations from learned normal patterns.  
4. **Real-Time Feedback:** Anomalies appear live as red markers on a continuous temperature plot.  

---

## 🚀 Run Locally
```bash
git clone https://github.com/yourusername/Sensor-Fusion-Anomaly.git
cd Sensor-Fusion-Anomaly
pip install -r requirements.txt
cd src
python stream_fusion.py


## Structure
Sensor-Fusion-Anomaly/
├── data/ 
│   └── synthetic_sensors.csv 
├── src/ 
│   ├── stream_fusion.py 
│   ├── utils.py 
│   └── visualize_live.py 
├── README.md 
├── requirements.txt 
└── .gitignore
