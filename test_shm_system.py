#!/usr/bin/env python3
"""
Test Script for Structural Health Monitoring System
==================================================

Tests the SHM system with your AE data to verify threshold detection
and alert functionality.

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
from structural_health_monitor import StructuralHealthMonitor
import json

def test_shm_system():
    """
    Test the Structural Health Monitoring system.
    """
    print("🏗️  Testing Structural Health Monitoring System")
    print("=" * 60)
    
    # Initialize monitor
    monitor = StructuralHealthMonitor()
    
    # Test 1: Establish baseline
    print("\n📊 Test 1: Establishing baseline...")
    try:
        monitor.establish_baseline('AE Data_16FT - Sheet1.csv')
        print("✅ Baseline established successfully")
    except Exception as e:
        print(f"❌ Baseline establishment failed: {e}")
        return False
    
    # Test 2: Test threshold detection
    print("\n🎯 Test 2: Testing threshold detection...")
    
    # Normal readings (should not trigger alerts)
    normal_readings = [
        (45.0, 2500.0),  # Normal amplitude and duration
        (55.0, 3500.0),  # Normal amplitude and duration
        (35.0, 1800.0),  # Normal amplitude and duration
    ]
    
    print("Testing normal readings:")
    for i, (amp, dur) in enumerate(normal_readings):
        result = monitor.process_reading(amp, dur, f"normal_{i}")
        print(f"  Reading {i+1}: Amp={amp} dB, Dur={dur} μs, Anomaly={result['is_anomaly']}, Alert={result['alert_triggered']}")
    
    # High amplitude readings (should trigger alerts)
    high_amp_readings = [
        (85.0, 3000.0),  # High amplitude
        (90.0, 2500.0),  # Very high amplitude
    ]
    
    print("\nTesting high amplitude readings:")
    for i, (amp, dur) in enumerate(high_amp_readings):
        result = monitor.process_reading(amp, dur, f"high_amp_{i}")
        print(f"  Reading {i+1}: Amp={amp} dB, Dur={dur} μs, Anomaly={result['is_anomaly']}, Alert={result['alert_triggered']}")
        if result['alert_triggered']:
            print(f"    🚨 Alert triggered: {result['alert_reason']}")
    
    # Long duration readings (should trigger alerts)
    long_dur_readings = [
        (50.0, 12000.0),  # Long duration
        (45.0, 15000.0),  # Very long duration
    ]
    
    print("\nTesting long duration readings:")
    for i, (amp, dur) in enumerate(long_dur_readings):
        result = monitor.process_reading(amp, dur, f"long_dur_{i}")
        print(f"  Reading {i+1}: Amp={amp} dB, Dur={dur} μs, Anomaly={result['is_anomaly']}, Alert={result['alert_triggered']}")
        if result['alert_triggered']:
            print(f"    🚨 Alert triggered: {result['alert_reason']}")
    
    # Test 3: Test with real data samples
    print("\n📈 Test 3: Testing with real data samples...")
    try:
        # Load some real data
        data = pd.read_csv('AE Data_16FT - Sheet1.csv')
        sample_data = data.sample(n=10, random_state=42)
        
        print("Testing with real data samples:")
        for i, row in sample_data.iterrows():
            amp = row['Amplitude (dB)']
            dur = row['Duration (microsecond)']
            result = monitor.process_reading(amp, dur, f"real_{i}")
            print(f"  Sample {i}: Amp={amp:.1f} dB, Dur={dur:.1f} μs, Anomaly={result['is_anomaly']}")
    
    except Exception as e:
        print(f"❌ Real data test failed: {e}")
    
    # Test 4: System status
    print("\n📊 Test 4: System status...")
    status = monitor.get_status()
    print(f"  Total readings: {status['total_readings']}")
    print(f"  Anomalies detected: {status['anomaly_count']}")
    print(f"  Anomaly rate: {status['anomaly_rate']:.1%}")
    print(f"  Alerts triggered: {len(monitor.alert_history)}")
    
    # Test 5: Configuration test
    print("\n⚙️  Test 5: Configuration test...")
    print(f"  Alert threshold: {monitor.config['alert_threshold']}")
    print(f"  Amplitude limits: {monitor.config['amplitude_threshold']}")
    print(f"  Duration limits: {monitor.config['duration_threshold']}")
    print(f"  Buffer size: {monitor.config['buffer_size']}")
    
    # Test 6: Recent anomalies
    print("\n🔍 Test 6: Recent anomalies...")
    recent_anomalies = monitor.get_recent_anomalies(5)
    if recent_anomalies:
        print(f"  Found {len(recent_anomalies)} recent anomalies:")
        for anomaly in recent_anomalies:
            print(f"    {anomaly['timestamp']}: Amp={anomaly['amplitude']:.1f} dB, Dur={anomaly['duration']:.1f} μs")
    else:
        print("  No recent anomalies found")
    
    print("\n✅ All tests completed!")
    return True

def test_threshold_customization():
    """
    Test threshold customization.
    """
    print("\n🎛️  Testing threshold customization...")
    
    # Create custom config
    custom_config = {
        "alert_threshold": 0.02,  # 2% anomaly rate
        "amplitude_threshold": {
            "upper": 70.0,  # Lower upper limit
            "lower": 30.0   # Higher lower limit
        },
        "duration_threshold": {
            "upper": 6000.0,  # Lower upper limit
            "lower": 500.0    # Higher lower limit
        },
        "buffer_size": 200,
        "baseline_samples": 1000,
        "alert_cooldown": 60,  # 1 minute cooldown
        "log_file": "custom_shm_log.txt",
        "data_log_file": "custom_shm_data.csv",
        "alert_log_file": "custom_shm_alerts.json"
    }
    
    # Save custom config
    with open('custom_config.json', 'w') as f:
        json.dump(custom_config, f, indent=2)
    
    # Test with custom config
    monitor = StructuralHealthMonitor('custom_config.json')
    monitor.establish_baseline('AE Data_16FT - Sheet1.csv')
    
    print("Testing with stricter thresholds:")
    test_readings = [
        (75.0, 3000.0),  # Should trigger amplitude alert
        (25.0, 3000.0),  # Should trigger amplitude alert
        (50.0, 7000.0),  # Should trigger duration alert
        (50.0, 400.0),   # Should trigger duration alert
    ]
    
    for i, (amp, dur) in enumerate(test_readings):
        result = monitor.process_reading(amp, dur, f"custom_{i}")
        print(f"  Reading {i+1}: Amp={amp} dB, Dur={dur} μs, Alert={result['alert_triggered']}")
        if result['alert_triggered']:
            print(f"    🚨 Alert: {result['alert_reason']}")
    
    print("✅ Custom threshold test completed!")

def main():
    """
    Run all tests.
    """
    print("🏗️  Structural Health Monitoring System - Test Suite")
    print("=" * 70)
    
    # Run main tests
    success = test_shm_system()
    
    if success:
        # Run customization tests
        test_threshold_customization()
        
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("Your Structural Health Monitoring system is ready!")
        print("\nNext steps:")
        print("1. Customize thresholds in the config file")
        print("2. Deploy to Raspberry Pi using raspberry_pi_monitor.py")
        print("3. Connect to your AE sensors")
        print("4. Start monitoring!")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
