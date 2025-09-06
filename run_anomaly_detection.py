#!/usr/bin/env python3
"""
Complete Anomaly Detection Pipeline
==================================

This script runs the complete anomaly detection pipeline including
evaluation metrics and comprehensive reporting.

Author: AI Assistant
Date: 2025
"""

from anomaly_detection import AnomalyDetector
from evaluation_metrics import AnomalyEvaluator
import sys
import os

def main():
    """
    Run the complete anomaly detection and evaluation pipeline.
    """
    print("="*60)
    print("COMPLETE AE DATA ANOMALY DETECTION PIPELINE")
    print("="*60)
    
    # Check if data file exists
    data_file = 'AE Data_16FT - Sheet1.csv'
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found!")
        print("Please ensure your CSV file is in the current directory.")
        return
    
    try:
        # Step 1: Initialize and run anomaly detection
        print("\nSTEP 1: ANOMALY DETECTION")
        print("-" * 40)
        
        detector = AnomalyDetector(data_file)
        detector.load_data()
        detector.preprocess_data()
        detector.train_models()
        detector.detect_anomalies()
        detector.visualize_results()
        detector.generate_report()
        detector.save_anomaly_data()
        
        # Step 2: Run comprehensive evaluation
        print("\nSTEP 2: PERFORMANCE EVALUATION")
        print("-" * 40)
        
        evaluator = AnomalyEvaluator(detector.features, detector.anomaly_scores)
        evaluator.calculate_basic_metrics()
        evaluator.calculate_clustering_metrics()
        evaluator.calculate_statistical_metrics()
        evaluator.create_evaluation_plots()
        evaluator.generate_evaluation_report()
        evaluator.save_evaluation_results()
        
        # Step 3: Summary
        print("\nSTEP 3: ANALYSIS COMPLETE")
        print("-" * 40)
        print("✅ Anomaly detection completed successfully!")
        print("✅ Performance evaluation completed successfully!")
        print("\nGenerated files:")
        print("📊 anomaly_detection_results.png - Main visualization")
        print("📊 evaluation_metrics.png - Performance evaluation plots")
        print("📄 anomaly_results.csv - Detailed results with predictions")
        print("📄 evaluation_results.csv - Performance metrics summary")
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE - Check the generated files for results!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        print("Please check your data file and try again.")
        return

if __name__ == "__main__":
    main()
