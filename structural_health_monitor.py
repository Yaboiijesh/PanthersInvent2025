#!/usr/bin/env python3
"""
Structural Health Monitoring System
==================================

Real-time anomaly detection system for structural health monitoring
using Acoustic Emission (AE) data. Designed for Raspberry Pi deployment.

Features:
- Real-time threshold-based monitoring
- Configurable alert system
- Data logging and historical analysis
- Raspberry Pi optimized

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import sklearn with error handling
try:
    import sklearn
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    print("✅ sklearn imported successfully")
except ImportError as e:
    print(f"❌ sklearn import failed: {e}")
    print("Please install scikit-learn: pip install scikit-learn")
    exit(1)

class StructuralHealthMonitor:
    """
    Real-time structural health monitoring system for AE data.
    """
    
    def __init__(self, config_file: str = "shm_config.json"):
        """
        Initialize the structural health monitor.
        
        Args:
            config_file (str): Path to configuration file
        """
        self.config_file = config_file
        self.config = self.load_config()
        self.setup_logging()
        
        # Initialize monitoring components
        self.baseline_data = None
        self.scaler = StandardScaler()
        self.anomaly_detector = None
        self.alert_history = []
        self.data_buffer = []
        self.is_monitoring = False
        
        # Statistics
        self.total_readings = 0
        self.anomaly_count = 0
        self.last_alert_time = None
        
        print("🏗️  Structural Health Monitor initialized")
        print(f"   Alert threshold: {self.config['alert_threshold']}")
        print(f"   Buffer size: {self.config['buffer_size']}")
    
    def load_config(self) -> Dict:
        """
        Load configuration from JSON file or create default.
        """
        default_config = {
            "alert_threshold": 0.1,  # 10% of readings as anomalies triggers alert
            "amplitude_threshold": {
                "upper": 80.0,  # dB - upper limit for normal amplitude
                "lower": 20.0   # dB - lower limit for normal amplitude
            },
            "duration_threshold": {
                "upper": 10000.0,  # μs - upper limit for normal duration
                "lower": 100.0     # μs - lower limit for normal duration
            },
            "buffer_size": 1000,  # Number of readings to keep in buffer
            "baseline_samples": 5000,  # Number of samples for baseline
            "alert_cooldown": 300,  # Seconds between alerts (5 minutes)
            "log_file": "shm_log.txt",
            "data_log_file": "shm_data.csv",
            "alert_log_file": "shm_alerts.json"
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                print(f"✅ Configuration loaded from {self.config_file}")
                return {**default_config, **config}  # Merge with defaults
            except Exception as e:
                print(f"⚠️  Error loading config: {e}. Using defaults.")
                return default_config
        else:
            # Create default config file
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"✅ Default configuration created: {self.config_file}")
            return default_config
    
    def setup_logging(self):
        """
        Setup logging system.
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config['log_file']),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def establish_baseline(self, data_file: str = None, data: pd.DataFrame = None):
        """
        Establish baseline from historical data or provided data.
        
        Args:
            data_file (str): Path to CSV file with historical data
            data (pd.DataFrame): DataFrame with historical data
        """
        print("📊 Establishing baseline...")
        
        if data is not None:
            baseline_data = data
        elif data_file and os.path.exists(data_file):
            baseline_data = pd.read_csv(data_file)
        else:
            raise ValueError("No data provided for baseline establishment")
        
        # Extract relevant columns
        if 'Duration (microsecond)' in baseline_data.columns and 'Amplitude (dB)' in baseline_data.columns:
            self.baseline_data = baseline_data[['Duration (microsecond)', 'Amplitude (dB)']].copy()
        else:
            raise ValueError("Required columns 'Duration (microsecond)' and 'Amplitude (dB)' not found")
        
        # Remove missing values
        self.baseline_data = self.baseline_data.dropna()
        
        # Limit baseline size for performance
        if len(self.baseline_data) > self.config['baseline_samples']:
            self.baseline_data = self.baseline_data.sample(n=self.config['baseline_samples'], random_state=42)
        
        # Train anomaly detector on baseline
        self.scaler.fit(self.baseline_data)
        baseline_scaled = self.scaler.transform(self.baseline_data)
        
        self.anomaly_detector = IsolationForest(
            contamination=self.config['alert_threshold'],
            random_state=42,
            n_estimators=50  # Reduced for Raspberry Pi performance
        )
        self.anomaly_detector.fit(baseline_scaled)
        
        # Calculate baseline statistics
        self.baseline_stats = {
            'amplitude_mean': self.baseline_data['Amplitude (dB)'].mean(),
            'amplitude_std': self.baseline_data['Amplitude (dB)'].std(),
            'duration_mean': self.baseline_data['Duration (microsecond)'].mean(),
            'duration_std': self.baseline_data['Duration (microsecond)'].std(),
            'sample_count': len(self.baseline_data)
        }
        
        print(f"✅ Baseline established with {len(self.baseline_data)} samples")
        print(f"   Amplitude: {self.baseline_stats['amplitude_mean']:.2f} ± {self.baseline_stats['amplitude_std']:.2f} dB")
        print(f"   Duration: {self.baseline_stats['duration_mean']:.2f} ± {self.baseline_stats['duration_std']:.2f} μs")
        
        self.logger.info(f"Baseline established with {len(self.baseline_data)} samples")
    
    def check_thresholds(self, amplitude: float, duration: float) -> Dict[str, bool]:
        """
        Check if readings exceed configured thresholds.
        
        Args:
            amplitude (float): Amplitude reading in dB
            duration (float): Duration reading in μs
            
        Returns:
            Dict with threshold check results
        """
        results = {
            'amplitude_high': amplitude > self.config['amplitude_threshold']['upper'],
            'amplitude_low': amplitude < self.config['amplitude_threshold']['lower'],
            'duration_high': duration > self.config['duration_threshold']['upper'],
            'duration_low': duration < self.config['duration_threshold']['lower'],
            'any_threshold_exceeded': False
        }
        
        results['any_threshold_exceeded'] = any([
            results['amplitude_high'], results['amplitude_low'],
            results['duration_high'], results['duration_low']
        ])
        
        return results
    
    def detect_anomaly(self, amplitude: float, duration: float) -> Dict:
        """
        Detect if a single reading is anomalous.
        
        Args:
            amplitude (float): Amplitude reading in dB
            duration (float): Duration reading in μs
            
        Returns:
            Dict with anomaly detection results
        """
        if self.anomaly_detector is None:
            raise ValueError("Baseline not established. Call establish_baseline() first.")
        
        # Prepare data
        reading = np.array([[duration, amplitude]])
        reading_scaled = self.scaler.transform(reading)
        
        # Detect anomaly
        prediction = self.anomaly_detector.predict(reading_scaled)[0]
        score = self.anomaly_detector.decision_function(reading_scaled)[0]
        
        # Check thresholds
        threshold_results = self.check_thresholds(amplitude, duration)
        
        is_anomaly = prediction == -1 or threshold_results['any_threshold_exceeded']
        
        return {
            'is_anomaly': is_anomaly,
            'prediction': prediction,
            'score': score,
            'threshold_results': threshold_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def process_reading(self, amplitude: float, duration: float, source: str = "sensor") -> Dict:
        """
        Process a single reading and determine if alert should be triggered.
        
        Args:
            amplitude (float): Amplitude reading in dB
            duration (float): Duration reading in μs
            source (str): Source identifier for the reading
            
        Returns:
            Dict with processing results and alert status
        """
        self.total_readings += 1
        
        # Detect anomaly
        anomaly_result = self.detect_anomaly(amplitude, duration)
        
        # Add to buffer
        reading_data = {
            'timestamp': datetime.now(),
            'amplitude': amplitude,
            'duration': duration,
            'source': source,
            'is_anomaly': anomaly_result['is_anomaly'],
            'score': anomaly_result['score']
        }
        
        self.data_buffer.append(reading_data)
        
        # Maintain buffer size
        if len(self.data_buffer) > self.config['buffer_size']:
            self.data_buffer.pop(0)
        
        # Count anomalies
        if anomaly_result['is_anomaly']:
            self.anomaly_count += 1
        
        # Check if alert should be triggered
        alert_triggered = False
        alert_reason = None
        
        if anomaly_result['is_anomaly']:
            # Check cooldown period
            if (self.last_alert_time is None or 
                (datetime.now() - self.last_alert_time).seconds > self.config['alert_cooldown']):
                
                alert_triggered = True
                self.last_alert_time = datetime.now()
                
                if anomaly_result['threshold_results']['any_threshold_exceeded']:
                    alert_reason = "Threshold exceeded"
                else:
                    alert_reason = "Statistical anomaly detected"
        
        # Log reading
        self.log_reading(reading_data, alert_triggered, alert_reason)
        
        return {
            'reading_id': self.total_readings,
            'timestamp': reading_data['timestamp'].isoformat(),
            'amplitude': amplitude,
            'duration': duration,
            'is_anomaly': anomaly_result['is_anomaly'],
            'anomaly_score': anomaly_result['score'],
            'threshold_results': anomaly_result['threshold_results'],
            'alert_triggered': alert_triggered,
            'alert_reason': alert_reason,
            'anomaly_rate': self.anomaly_count / self.total_readings if self.total_readings > 0 else 0
        }
    
    def log_reading(self, reading_data: Dict, alert_triggered: bool, alert_reason: str = None):
        """
        Log reading data and alerts.
        """
        # Log to CSV file
        log_entry = {
            'timestamp': reading_data['timestamp'].isoformat(),
            'amplitude': reading_data['amplitude'],
            'duration': reading_data['duration'],
            'source': reading_data['source'],
            'is_anomaly': reading_data['is_anomaly'],
            'score': reading_data['score'],
            'alert_triggered': alert_triggered,
            'alert_reason': alert_reason or ""
        }
        
        # Append to CSV log
        log_df = pd.DataFrame([log_entry])
        if not os.path.exists(self.config['data_log_file']):
            log_df.to_csv(self.config['data_log_file'], index=False)
        else:
            log_df.to_csv(self.config['data_log_file'], mode='a', header=False, index=False)
        
        # Log alert if triggered
        if alert_triggered:
            alert_data = {
                'timestamp': reading_data['timestamp'].isoformat(),
                'amplitude': reading_data['amplitude'],
                'duration': reading_data['duration'],
                'reason': alert_reason,
                'anomaly_score': reading_data['score']
            }
            
            self.alert_history.append(alert_data)
            
            # Save alert to JSON file
            with open(self.config['alert_log_file'], 'w') as f:
                json.dump(self.alert_history, f, indent=2)
            
            # Log to console and file
            alert_msg = f"🚨 ALERT: {alert_reason} - Amplitude: {reading_data['amplitude']:.2f} dB, Duration: {reading_data['duration']:.2f} μs"
            self.logger.warning(alert_msg)
            print(alert_msg)
    
    def get_status(self) -> Dict:
        """
        Get current system status.
        """
        return {
            'is_monitoring': self.is_monitoring,
            'total_readings': self.total_readings,
            'anomaly_count': self.anomaly_count,
            'anomaly_rate': self.anomaly_count / self.total_readings if self.total_readings > 0 else 0,
            'buffer_size': len(self.data_buffer),
            'last_alert_time': self.last_alert_time.isoformat() if self.last_alert_time else None,
            'baseline_established': self.baseline_data is not None,
            'baseline_stats': self.baseline_stats if hasattr(self, 'baseline_stats') else None
        }
    
    def get_recent_anomalies(self, count: int = 10) -> List[Dict]:
        """
        Get recent anomaly readings.
        """
        anomalies = [r for r in self.data_buffer if r['is_anomaly']]
        return anomalies[-count:] if anomalies else []
    
    def start_monitoring(self):
        """
        Start continuous monitoring (placeholder for real-time implementation).
        """
        self.is_monitoring = True
        self.logger.info("Monitoring started")
        print("🔄 Monitoring started")
    
    def stop_monitoring(self):
        """
        Stop continuous monitoring.
        """
        self.is_monitoring = False
        self.logger.info("Monitoring stopped")
        print("⏹️  Monitoring stopped")

def main():
    """
    Example usage of the Structural Health Monitor.
    """
    print("🏗️  Structural Health Monitoring System")
    print("=" * 50)
    
    # Initialize monitor
    monitor = StructuralHealthMonitor()
    
    # Establish baseline from your data
    try:
        monitor.establish_baseline('AE Data_16FT - Sheet1.csv')
    except Exception as e:
        print(f"❌ Error establishing baseline: {e}")
        return
    
    # Simulate some readings
    print("\n📊 Simulating readings...")
    
    # Normal readings
    for i in range(5):
        amplitude = np.random.normal(50, 10)  # Normal amplitude around 50 dB
        duration = np.random.normal(3000, 1000)  # Normal duration around 3000 μs
        result = monitor.process_reading(amplitude, duration, f"sim_{i}")
        print(f"Reading {i+1}: Amplitude={amplitude:.1f} dB, Duration={duration:.1f} μs, Anomaly={result['is_anomaly']}")
    
    # Anomalous readings
    for i in range(3):
        amplitude = np.random.normal(90, 5)  # High amplitude
        duration = np.random.normal(15000, 2000)  # Long duration
        result = monitor.process_reading(amplitude, duration, f"anomaly_{i}")
        print(f"Anomaly {i+1}: Amplitude={amplitude:.1f} dB, Duration={duration:.1f} μs, Alert={result['alert_triggered']}")
    
    # Show status
    status = monitor.get_status()
    print(f"\n📈 System Status:")
    print(f"   Total readings: {status['total_readings']}")
    print(f"   Anomalies: {status['anomaly_count']} ({status['anomaly_rate']:.1%})")
    print(f"   Recent alerts: {len(monitor.alert_history)}")
    
    print("\n✅ Structural Health Monitor test completed!")

if __name__ == "__main__":
    main()
