#!/usr/bin/env python3
"""
Raspberry Pi Structural Health Monitor
=====================================

Optimized version for Raspberry Pi deployment with GPIO integration,
performance monitoring, and system resource management.

Author: AI Assistant
Date: 2025
"""

import time
import json
import os
import psutil
import subprocess
from datetime import datetime
import logging
from structural_health_monitor import StructuralHealthMonitor

# GPIO imports (install with: pip install RPi.GPIO)
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("⚠️  RPi.GPIO not available. Install with: pip install RPi.GPIO")

class RaspberryPiMonitor(StructuralHealthMonitor):
    """
    Raspberry Pi optimized structural health monitor with GPIO integration.
    """
    
    def __init__(self, config_file: str = "raspberry_pi_config.json"):
        """
        Initialize Raspberry Pi monitor.
        """
        super().__init__(config_file)
        
        # Load Pi-specific config
        self.pi_config = self.config.get('raspberry_pi', {})
        
        # Initialize GPIO if available
        if GPIO_AVAILABLE:
            self.setup_gpio()
        else:
            self.gpio_pins = {}
        
        # Performance monitoring
        self.performance_log = []
        self.start_time = datetime.now()
        
        print("🍓 Raspberry Pi Structural Health Monitor initialized")
    
    def setup_gpio(self):
        """
        Setup GPIO pins for alerts and status indicators.
        """
        GPIO.setmode(GPIO.BCM)
        
        # Configure GPIO pins
        self.gpio_pins = self.pi_config.get('gpio_pins', {})
        
        # Setup output pins
        for pin_name, pin_number in self.gpio_pins.items():
            GPIO.setup(pin_number, GPIO.OUT)
            GPIO.output(pin_number, GPIO.LOW)
        
        print("✅ GPIO pins configured")
    
    def check_system_performance(self) -> Dict:
        """
        Check Raspberry Pi system performance.
        """
        performance = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_mb': psutil.virtual_memory().used / 1024 / 1024,
            'disk_percent': psutil.disk_usage('/').percent,
            'temperature': self.get_cpu_temperature(),
            'uptime': (datetime.now() - self.start_time).total_seconds()
        }
        
        # Check performance limits
        limits = self.pi_config.get('performance', {})
        performance['warnings'] = []
        
        if performance['cpu_percent'] > limits.get('max_cpu_usage', 80):
            performance['warnings'].append(f"High CPU usage: {performance['cpu_percent']:.1f}%")
        
        if performance['memory_used_mb'] > limits.get('memory_limit_mb', 512):
            performance['warnings'].append(f"High memory usage: {performance['memory_used_mb']:.1f} MB")
        
        if performance['temperature'] > limits.get('temperature_limit_c', 70):
            performance['warnings'].append(f"High temperature: {performance['temperature']:.1f}°C")
        
        return performance
    
    def get_cpu_temperature(self) -> float:
        """
        Get Raspberry Pi CPU temperature.
        """
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = int(f.read()) / 1000.0
            return temp
        except:
            return 0.0
    
    def trigger_alert_indicators(self, alert_type: str = "anomaly"):
        """
        Trigger physical alert indicators.
        """
        if not GPIO_AVAILABLE:
            return
        
        try:
            # Flash alert LED
            if 'alert_led' in self.gpio_pins:
                for _ in range(5):
                    GPIO.output(self.gpio_pins['alert_led'], GPIO.HIGH)
                    time.sleep(0.2)
                    GPIO.output(self.gpio_pins['alert_led'], GPIO.LOW)
                    time.sleep(0.2)
            
            # Sound buzzer
            if 'buzzer' in self.gpio_pins:
                GPIO.output(self.gpio_pins['buzzer'], GPIO.HIGH)
                time.sleep(0.5)
                GPIO.output(self.gpio_pins['buzzer'], GPIO.LOW)
        
        except Exception as e:
            self.logger.error(f"GPIO error: {e}")
    
    def set_status_led(self, status: str):
        """
        Set status LED based on system status.
        
        Args:
            status: 'normal', 'warning', 'error', 'monitoring'
        """
        if not GPIO_AVAILABLE or 'status_led' not in self.gpio_pins:
            return
        
        try:
            if status == 'normal':
                GPIO.output(self.gpio_pins['status_led'], GPIO.LOW)
            elif status == 'monitoring':
                # Blink slowly
                GPIO.output(self.gpio_pins['status_led'], GPIO.HIGH)
                time.sleep(0.5)
                GPIO.output(self.gpio_pins['status_led'], GPIO.LOW)
            elif status == 'warning':
                # Blink rapidly
                for _ in range(3):
                    GPIO.output(self.gpio_pins['status_led'], GPIO.HIGH)
                    time.sleep(0.1)
                    GPIO.output(self.gpio_pins['status_led'], GPIO.LOW)
                    time.sleep(0.1)
            elif status == 'error':
                # Solid on
                GPIO.output(self.gpio_pins['status_led'], GPIO.HIGH)
        
        except Exception as e:
            self.logger.error(f"Status LED error: {e}")
    
    def send_network_alert(self, alert_data: Dict):
        """
        Send alert via network (webhook, email, SMS).
        """
        network_config = self.pi_config.get('network', {})
        
        # Webhook notification
        webhook_url = network_config.get('webhook_url')
        if webhook_url:
            try:
                import requests
                payload = {
                    'timestamp': alert_data['timestamp'],
                    'type': 'structural_health_alert',
                    'amplitude': alert_data['amplitude'],
                    'duration': alert_data['duration'],
                    'reason': alert_data['reason'],
                    'system_status': self.get_status()
                }
                response = requests.post(webhook_url, json=payload, timeout=10)
                if response.status_code == 200:
                    self.logger.info("Webhook alert sent successfully")
                else:
                    self.logger.error(f"Webhook failed: {response.status_code}")
            except Exception as e:
                self.logger.error(f"Webhook error: {e}")
    
    def process_reading(self, amplitude: float, duration: float, source: str = "sensor") -> Dict:
        """
        Process reading with Raspberry Pi specific features.
        """
        # Check system performance
        performance = self.check_system_performance()
        if performance['warnings']:
            self.logger.warning(f"Performance warnings: {performance['warnings']}")
            self.set_status_led('warning')
        
        # Process the reading
        result = super().process_reading(amplitude, duration, source)
        
        # Handle alerts
        if result['alert_triggered']:
            self.trigger_alert_indicators()
            self.set_status_led('error')
            
            # Send network alert
            alert_data = {
                'timestamp': result['timestamp'],
                'amplitude': result['amplitude'],
                'duration': result['duration'],
                'reason': result['alert_reason']
            }
            self.send_network_alert(alert_data)
        else:
            self.set_status_led('monitoring')
        
        # Add performance data to result
        result['system_performance'] = performance
        
        return result
    
    def cleanup(self):
        """
        Cleanup GPIO and resources.
        """
        if GPIO_AVAILABLE:
            GPIO.cleanup()
        self.logger.info("Raspberry Pi monitor cleanup completed")
    
    def get_detailed_status(self) -> Dict:
        """
        Get detailed system status including Pi-specific information.
        """
        status = self.get_status()
        performance = self.check_system_performance()
        
        return {
            **status,
            'system_performance': performance,
            'gpio_available': GPIO_AVAILABLE,
            'config_file': self.config_file,
            'uptime_hours': performance['uptime'] / 3600
        }

def create_directories():
    """
    Create necessary directories for logging.
    """
    directories = [
        '/home/pi/shm_logs',
        '/home/pi/shm_data',
        '/home/pi/shm_config'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Directory created: {directory}")

def install_dependencies():
    """
    Install required dependencies for Raspberry Pi.
    """
    packages = [
        'RPi.GPIO',
        'psutil',
        'requests'
    ]
    
    for package in packages:
        try:
            subprocess.check_call(['pip3', 'install', package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

def main():
    """
    Main function for Raspberry Pi deployment.
    """
    print("🍓 Raspberry Pi Structural Health Monitor")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Initialize monitor
    monitor = RaspberryPiMonitor()
    
    try:
        # Establish baseline
        monitor.establish_baseline('AE Data_16FT - Sheet1.csv')
        
        # Start monitoring
        monitor.start_monitoring()
        monitor.set_status_led('monitoring')
        
        print("🔄 Monitoring started. Press Ctrl+C to stop.")
        
        # Simulate continuous monitoring
        reading_count = 0
        while True:
            # Simulate sensor reading
            amplitude = np.random.normal(50, 15)  # Simulate real sensor data
            duration = np.random.normal(3000, 1500)
            
            result = monitor.process_reading(amplitude, duration, f"pi_sensor_{reading_count}")
            
            if reading_count % 10 == 0:  # Print status every 10 readings
                status = monitor.get_detailed_status()
                print(f"Reading {reading_count}: Anomaly rate: {status['anomaly_rate']:.1%}, "
                      f"CPU: {status['system_performance']['cpu_percent']:.1f}%, "
                      f"Temp: {status['system_performance']['temperature']:.1f}°C")
            
            reading_count += 1
            time.sleep(1)  # 1 second between readings
    
    except KeyboardInterrupt:
        print("\n⏹️  Stopping monitor...")
        monitor.stop_monitoring()
        monitor.set_status_led('normal')
        monitor.cleanup()
        print("✅ Monitor stopped successfully")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        monitor.set_status_led('error')
        monitor.cleanup()

if __name__ == "__main__":
    main()
