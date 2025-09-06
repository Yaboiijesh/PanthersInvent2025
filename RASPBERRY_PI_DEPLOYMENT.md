# Raspberry Pi Structural Health Monitoring Deployment Guide

## Overview
This guide will help you deploy the Structural Health Monitoring (SHM) system on a Raspberry Pi for real-time monitoring of acoustic emission data.

## System Requirements

### Hardware
- Raspberry Pi 4 (recommended) or Pi 3B+
- MicroSD card (32GB+ recommended)
- Power supply (5V, 3A)
- AE sensors and data acquisition hardware
- Optional: LEDs, buzzer for alerts

### Software
- Raspberry Pi OS (latest version)
- Python 3.7+
- Required Python packages (see requirements.txt)

## Installation Steps

### 1. Prepare Raspberry Pi
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python dependencies
sudo apt install python3-pip python3-venv -y

# Create project directory
mkdir -p /home/pi/shm_system
cd /home/pi/shm_system
```

### 2. Install Python Packages
```bash
# Create virtual environment (recommended)
python3 -m venv shm_env
source shm_env/bin/activate

# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn
pip install RPi.GPIO psutil requests

# Or install from requirements file
pip install -r requirements.txt
```

### 3. Setup GPIO Pins (Optional)
If using physical alert indicators:

| Component | GPIO Pin | Purpose |
|-----------|----------|---------|
| Alert LED | GPIO 18 | Flashes when anomaly detected |
| Status LED | GPIO 24 | Shows system status |
| Buzzer | GPIO 25 | Sounds alert |

### 4. Configure System
```bash
# Copy configuration files
cp raspberry_pi_config.json /home/pi/shm_system/

# Create log directories
mkdir -p /home/pi/shm_logs
mkdir -p /home/pi/shm_data
mkdir -p /home/pi/shm_config

# Set permissions
chmod +x raspberry_pi_monitor.py
```

### 5. Test Installation
```bash
# Test the system
python3 test_shm_system.py

# Test Raspberry Pi specific features
python3 raspberry_pi_monitor.py
```

## Configuration

### Threshold Settings
Edit `raspberry_pi_config.json` to customize:

```json
{
  "alert_threshold": 0.05,
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

### Performance Settings
```json
{
  "raspberry_pi": {
    "performance": {
      "max_cpu_usage": 80,
      "memory_limit_mb": 512,
      "temperature_limit_c": 70
    }
  }
}
```

## Data Integration

### Connecting AE Sensors
1. **Serial Connection**: Connect AE sensors via USB/Serial
2. **Network Connection**: Connect via Ethernet/WiFi
3. **File Monitoring**: Monitor CSV files from data acquisition system

### Example Data Integration
```python
import serial
import time

# Serial connection example
ser = serial.Serial('/dev/ttyUSB0', 9600)

while True:
    if ser.in_waiting:
        data = ser.readline().decode('utf-8').strip()
        # Parse amplitude and duration from data
        amplitude, duration = parse_ae_data(data)
        
        # Process with SHM system
        result = monitor.process_reading(amplitude, duration)
        
        if result['alert_triggered']:
            print(f"ALERT: {result['alert_reason']}")
```

## Running the System

### Manual Start
```bash
cd /home/pi/shm_system
source shm_env/bin/activate
python3 raspberry_pi_monitor.py
```

### Auto-start with systemd
Create service file `/etc/systemd/system/shm-monitor.service`:

```ini
[Unit]
Description=Structural Health Monitor
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/shm_system
Environment=PATH=/home/pi/shm_system/shm_env/bin
ExecStart=/home/pi/shm_system/shm_env/bin/python raspberry_pi_monitor.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start service:
```bash
sudo systemctl enable shm-monitor.service
sudo systemctl start shm-monitor.service
sudo systemctl status shm-monitor.service
```

## Monitoring and Alerts

### Log Files
- `/home/pi/shm_logs/shm_log.txt` - System logs
- `/home/pi/shm_logs/shm_data.csv` - All sensor readings
- `/home/pi/shm_logs/shm_alerts.json` - Alert history

### Alert Types
1. **Threshold Exceeded**: Amplitude or duration outside normal range
2. **Statistical Anomaly**: Unusual pattern detected by ML algorithm
3. **System Warning**: High CPU, memory, or temperature

### Network Alerts
Configure webhook URL in config for remote notifications:
```json
{
  "raspberry_pi": {
    "network": {
      "webhook_url": "https://your-server.com/webhook",
      "email_alerts": true,
      "sms_alerts": false
    }
  }
}
```

## Performance Optimization

### For Raspberry Pi 3B+
```json
{
  "buffer_size": 200,
  "baseline_samples": 1000,
  "raspberry_pi": {
    "performance": {
      "max_cpu_usage": 70,
      "memory_limit_mb": 256
    }
  }
}
```

### For Raspberry Pi 4
```json
{
  "buffer_size": 500,
  "baseline_samples": 2000,
  "raspberry_pi": {
    "performance": {
      "max_cpu_usage": 80,
      "memory_limit_mb": 512
    }
  }
}
```

## Troubleshooting

### Common Issues

1. **GPIO Permission Error**
   ```bash
   sudo usermod -a -G gpio pi
   # Logout and login again
   ```

2. **High CPU Usage**
   - Reduce `baseline_samples` in config
   - Increase `alert_cooldown` period
   - Use fewer anomaly detection algorithms

3. **Memory Issues**
   - Reduce `buffer_size` in config
   - Clear old log files regularly
   - Use swap file if needed

4. **Temperature Warnings**
   - Ensure proper cooling
   - Reduce CPU-intensive operations
   - Check for background processes

### Monitoring Commands
```bash
# Check system status
sudo systemctl status shm-monitor.service

# View logs
tail -f /home/pi/shm_logs/shm_log.txt

# Check system performance
htop
vcgencmd measure_temp

# Check disk space
df -h
```

## Security Considerations

1. **Network Security**
   - Use HTTPS for webhook URLs
   - Implement authentication for remote access
   - Use VPN for remote monitoring

2. **Data Protection**
   - Encrypt sensitive configuration files
   - Regular backups of log files
   - Secure physical access to Pi

3. **System Security**
   - Keep system updated
   - Use strong passwords
   - Disable unnecessary services

## Maintenance

### Regular Tasks
1. **Weekly**: Check log files and system performance
2. **Monthly**: Update system and packages
3. **Quarterly**: Review and adjust thresholds
4. **Annually**: Replace SD card and backup data

### Backup Strategy
```bash
# Backup configuration and logs
tar -czf shm_backup_$(date +%Y%m%d).tar.gz \
  /home/pi/shm_system/ \
  /home/pi/shm_logs/ \
  /home/pi/shm_data/
```

## Support and Updates

For issues or updates:
1. Check log files for error messages
2. Verify configuration settings
3. Test with sample data
4. Contact system administrator

## Example Deployment Script

```bash
#!/bin/bash
# deploy_shm.sh

echo "Deploying Structural Health Monitor..."

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install python3-pip python3-venv -y

# Create directories
mkdir -p /home/pi/shm_system
mkdir -p /home/pi/shm_logs
mkdir -p /home/pi/shm_data

# Setup virtual environment
cd /home/pi/shm_system
python3 -m venv shm_env
source shm_env/bin/activate

# Install packages
pip install pandas numpy matplotlib seaborn scikit-learn RPi.GPIO psutil requests

# Copy files
cp *.py /home/pi/shm_system/
cp *.json /home/pi/shm_system/

# Set permissions
chmod +x raspberry_pi_monitor.py

# Create systemd service
sudo cp shm-monitor.service /etc/systemd/system/
sudo systemctl enable shm-monitor.service

echo "Deployment complete!"
echo "Start monitoring with: sudo systemctl start shm-monitor.service"
```

This deployment guide provides everything you need to set up a robust structural health monitoring system on your Raspberry Pi!
