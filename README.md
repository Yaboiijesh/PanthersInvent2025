# Panthers Invent 2025 - Structural Health Monitoring System

A real-time anomaly detection system for structural health monitoring using Acoustic Emission (AE) data. Designed for deployment on Raspberry Pi with configurable threshold-based alerts.

## 🏗️ Project Overview

This system analyzes Acoustic Emission data to detect structural anomalies in real-time:
- **Amplitude (dB)**: Signal strength analysis
- **Duration (μs)**: Signal duration analysis
- **Threshold-based alerts**: Configurable limits for immediate notifications
- **Machine Learning**: Statistical anomaly detection using your baseline data

## 🚀 Quick Start

### Installation
```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# For Raspberry Pi
pip install RPi.GPIO psutil requests
```

### Basic Usage
```python
from structural_health_monitor import StructuralHealthMonitor

# Initialize monitor
monitor = StructuralHealthMonitor()

# Establish baseline from your data
monitor.establish_baseline('AE Data_16FT - Sheet1.csv')

# Process a reading
result = monitor.process_reading(amplitude=65.5, duration=3500.0)

if result['alert_triggered']:
    print(f"ALERT: {result['alert_reason']}")
```

## 📁 Core Files

- **`structural_health_monitor.py`** - Main monitoring system
- **`raspberry_pi_monitor.py`** - Pi-optimized version with GPIO
- **`anomaly_detection.py`** - ML anomaly detection algorithms
- **`evaluation_metrics.py`** - Performance evaluation tools
- **`test_shm_system.py`** - Test suite
- **`run_anomaly_detection.py`** - Complete pipeline runner

## 🍓 Raspberry Pi Deployment

### Configuration
Edit `raspberry_pi_config.json`:
```json
{
  "amplitude_threshold": {
    "upper": 75.0,
    "lower": 25.0
  },
  "duration_threshold": {
    "upper": 8000.0,
    "lower": 200.0
  }
}
```

### Running on Pi
```bash
python3 raspberry_pi_monitor.py
```

## ⚙️ Alert Types

1. **Threshold Exceeded**: Readings outside normal amplitude/duration ranges
2. **Statistical Anomaly**: ML-detected unusual patterns
3. **System Warning**: High CPU, memory, or temperature

## 🧪 Testing

```bash
python test_shm_system.py
```

## 📊 Data Format

CSV files with columns:
- `Duration (microsecond)`: Signal duration in microseconds
- `Amplitude (dB)`: Signal amplitude in decibels

## 📚 Documentation

- [Raspberry Pi Deployment Guide](RASPBERRY_PI_DEPLOYMENT.md)
- Configuration examples in `raspberry_pi_config.json`

## 🎯 Features

- ✅ Real-time monitoring
- ✅ Configurable thresholds
- ✅ Machine learning detection
- ✅ Raspberry Pi optimization
- ✅ Physical alerts (LEDs, buzzer)
- ✅ Data logging
- ✅ Performance monitoring

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

**Built for Panthers Invent 2025** 🏗️