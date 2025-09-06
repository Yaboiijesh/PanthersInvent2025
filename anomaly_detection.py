#!/usr/bin/env python3
"""
Anomaly Detection System for AE Data
====================================

This script implements multiple anomaly detection algorithms to identify
anomalous patterns in Acoustic Emission (AE) data based on amplitude and duration.

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    """
    A comprehensive anomaly detection system for AE data.
    """
    
    def __init__(self, data_file):
        """
        Initialize the anomaly detector with data file.
        
        Args:
            data_file (str): Path to the CSV file containing AE data
        """
        self.data_file = data_file
        self.data = None
        self.features = None
        self.scaler = StandardScaler()
        self.models = {}
        self.anomaly_scores = {}
        
    def load_data(self):
        """
        Load and preprocess the AE data from CSV file.
        """
        print("Loading data...")
        self.data = pd.read_csv(self.data_file)
        
        # Extract relevant columns
        self.features = self.data[['Duration (microsecond)', 'Amplitude (dB)']].copy()
        
        # Remove any rows with missing values
        self.features = self.features.dropna()
        
        print(f"Data loaded successfully!")
        print(f"Dataset shape: {self.features.shape}")
        print(f"Features: {list(self.features.columns)}")
        print(f"Duration range: {self.features['Duration (microsecond)'].min():.2f} - {self.features['Duration (microsecond)'].max():.2f} μs")
        print(f"Amplitude range: {self.features['Amplitude (dB)'].min():.2f} - {self.features['Amplitude (dB)'].max():.2f} dB")
        
        return self.features
    
    def preprocess_data(self):
        """
        Preprocess the data for anomaly detection.
        """
        print("\nPreprocessing data...")
        
        # Standardize the features
        self.features_scaled = self.scaler.fit_transform(self.features)
        
        print("Data preprocessing completed!")
        
    def train_models(self):
        """
        Train multiple anomaly detection models.
        """
        print("\nTraining anomaly detection models...")
        
        # 1. Isolation Forest
        print("Training Isolation Forest...")
        self.models['isolation_forest'] = IsolationForest(
            contamination=0.1,  # Assume 10% of data is anomalous
            random_state=42,
            n_estimators=100
        )
        self.models['isolation_forest'].fit(self.features_scaled)
        
        # 2. One-Class SVM
        print("Training One-Class SVM...")
        self.models['one_class_svm'] = OneClassSVM(
            nu=0.1,  # Proportion of outliers
            kernel='rbf',
            gamma='scale'
        )
        self.models['one_class_svm'].fit(self.features_scaled)
        
        # 3. Local Outlier Factor
        print("Training Local Outlier Factor...")
        self.models['lof'] = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.1
        )
        self.models['lof'].fit(self.features_scaled)
        
        print("All models trained successfully!")
        
    def detect_anomalies(self):
        """
        Detect anomalies using all trained models.
        """
        print("\nDetecting anomalies...")
        
        # Isolation Forest
        if_pred = self.models['isolation_forest'].predict(self.features_scaled)
        if_scores = self.models['isolation_forest'].decision_function(self.features_scaled)
        self.anomaly_scores['isolation_forest'] = {
            'predictions': if_pred,
            'scores': if_scores,
            'anomalies': if_pred == -1
        }
        
        # One-Class SVM
        svm_pred = self.models['one_class_svm'].predict(self.features_scaled)
        svm_scores = self.models['one_class_svm'].decision_function(self.features_scaled)
        self.anomaly_scores['one_class_svm'] = {
            'predictions': svm_pred,
            'scores': svm_scores,
            'anomalies': svm_pred == -1
        }
        
        # Local Outlier Factor
        lof_pred = self.models['lof'].predict(self.features_scaled)
        lof_scores = self.models['lof'].negative_outlier_factor_
        self.anomaly_scores['lof'] = {
            'predictions': lof_pred,
            'scores': lof_scores,
            'anomalies': lof_pred == -1
        }
        
        # Ensemble prediction (majority vote)
        ensemble_pred = np.zeros(len(self.features_scaled))
        for model_name in self.anomaly_scores:
            ensemble_pred += (self.anomaly_scores[model_name]['anomalies'] * 2 - 1)
        
        ensemble_anomalies = ensemble_pred > 0  # Majority vote for anomaly
        self.anomaly_scores['ensemble'] = {
            'predictions': np.where(ensemble_anomalies, -1, 1),
            'scores': ensemble_pred,
            'anomalies': ensemble_anomalies
        }
        
        print("Anomaly detection completed!")
        
        # Print summary statistics
        for model_name, results in self.anomaly_scores.items():
            n_anomalies = np.sum(results['anomalies'])
            percentage = (n_anomalies / len(self.features)) * 100
            print(f"{model_name}: {n_anomalies} anomalies ({percentage:.2f}%)")
    
    def visualize_results(self, save_plots=True):
        """
        Create comprehensive visualizations of the anomaly detection results.
        """
        print("\nCreating visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Original data distribution
        plt.subplot(3, 3, 1)
        plt.scatter(self.features['Duration (microsecond)'], 
                   self.features['Amplitude (dB)'], 
                   alpha=0.6, s=1)
        plt.xlabel('Duration (μs)')
        plt.ylabel('Amplitude (dB)')
        plt.title('Original Data Distribution')
        plt.grid(True, alpha=0.3)
        
        # 2. Data distribution with histograms
        plt.subplot(3, 3, 2)
        plt.hist2d(self.features['Duration (microsecond)'], 
                  self.features['Amplitude (dB)'], 
                  bins=50, cmap='Blues')
        plt.xlabel('Duration (μs)')
        plt.ylabel('Amplitude (dB)')
        plt.title('Data Density Heatmap')
        plt.colorbar()
        
        # 3. Feature distributions
        plt.subplot(3, 3, 3)
        plt.hist(self.features['Duration (microsecond)'], bins=50, alpha=0.7, label='Duration')
        plt.hist(self.features['Amplitude (dB)'], bins=50, alpha=0.7, label='Amplitude')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title('Feature Distributions')
        plt.legend()
        plt.yscale('log')
        
        # 4-7. Anomaly detection results for each model
        model_names = ['isolation_forest', 'one_class_svm', 'lof', 'ensemble']
        colors = ['red', 'blue', 'green', 'purple']
        
        for i, (model_name, color) in enumerate(zip(model_names, colors)):
            plt.subplot(3, 3, 4 + i)
            
            # Normal points
            normal_mask = ~self.anomaly_scores[model_name]['anomalies']
            plt.scatter(self.features.loc[normal_mask, 'Duration (microsecond)'], 
                       self.features.loc[normal_mask, 'Amplitude (dB)'], 
                       c='lightblue', alpha=0.6, s=1, label='Normal')
            
            # Anomalous points
            anomaly_mask = self.anomaly_scores[model_name]['anomalies']
            plt.scatter(self.features.loc[anomaly_mask, 'Duration (microsecond)'], 
                       self.features.loc[anomaly_mask, 'Amplitude (dB)'], 
                       c=color, alpha=0.8, s=3, label='Anomaly')
            
            plt.xlabel('Duration (μs)')
            plt.ylabel('Amplitude (dB)')
            plt.title(f'{model_name.replace("_", " ").title()} Results')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 8. Anomaly score distributions
        plt.subplot(3, 3, 8)
        for model_name, color in zip(model_names, colors):
            scores = self.anomaly_scores[model_name]['scores']
            plt.hist(scores, bins=50, alpha=0.6, label=model_name.replace('_', ' ').title())
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title('Anomaly Score Distributions')
        plt.legend()
        plt.yscale('log')
        
        # 9. Model comparison
        plt.subplot(3, 3, 9)
        model_counts = []
        for model_name in model_names:
            n_anomalies = np.sum(self.anomaly_scores[model_name]['anomalies'])
            model_counts.append(n_anomalies)
        
        bars = plt.bar([name.replace('_', ' ').title() for name in model_names], 
                      model_counts, color=colors, alpha=0.7)
        plt.ylabel('Number of Anomalies')
        plt.title('Anomaly Count by Model')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, model_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('anomaly_detection_results.png', dpi=300, bbox_inches='tight')
            print("Visualization saved as 'anomaly_detection_results.png'")
        
        plt.show()
    
    def generate_report(self):
        """
        Generate a detailed anomaly detection report.
        """
        print("\n" + "="*60)
        print("ANOMALY DETECTION REPORT")
        print("="*60)
        
        print(f"\nDataset Information:")
        print(f"- Total samples: {len(self.features):,}")
        print(f"- Features: Duration (μs), Amplitude (dB)")
        print(f"- Duration range: {self.features['Duration (microsecond)'].min():.2f} - {self.features['Duration (microsecond)'].max():.2f} μs")
        print(f"- Amplitude range: {self.features['Amplitude (dB)'].min():.2f} - {self.features['Amplitude (dB)'].max():.2f} dB")
        
        print(f"\nAnomaly Detection Results:")
        print("-" * 40)
        
        for model_name, results in self.anomaly_scores.items():
            n_anomalies = np.sum(results['anomalies'])
            percentage = (n_anomalies / len(self.features)) * 100
            print(f"{model_name.replace('_', ' ').title():20}: {n_anomalies:6,} anomalies ({percentage:5.2f}%)")
        
        # Statistical summary of anomalies
        print(f"\nAnomaly Statistics:")
        print("-" * 40)
        
        for model_name, results in self.anomaly_scores.items():
            anomaly_data = self.features[results['anomalies']]
            if len(anomaly_data) > 0:
                print(f"\n{model_name.replace('_', ' ').title()} Anomalies:")
                print(f"  Duration - Mean: {anomaly_data['Duration (microsecond)'].mean():.2f} μs, "
                      f"Std: {anomaly_data['Duration (microsecond)'].std():.2f} μs")
                print(f"  Amplitude - Mean: {anomaly_data['Amplitude (dB)'].mean():.2f} dB, "
                      f"Std: {anomaly_data['Amplitude (dB)'].std():.2f} dB")
        
        print("\n" + "="*60)
    
    def save_anomaly_data(self, filename='anomaly_results.csv'):
        """
        Save the anomaly detection results to a CSV file.
        """
        print(f"\nSaving anomaly results to '{filename}'...")
        
        # Create a comprehensive results dataframe
        results_df = self.features.copy()
        
        # Add anomaly predictions and scores for each model
        for model_name, results in self.anomaly_scores.items():
            results_df[f'{model_name}_prediction'] = results['predictions']
            results_df[f'{model_name}_score'] = results['scores']
            results_df[f'{model_name}_is_anomaly'] = results['anomalies']
        
        # Save to CSV
        results_df.to_csv(filename, index=False)
        print(f"Results saved successfully!")
        
        return results_df

def main():
    """
    Main function to run the anomaly detection system.
    """
    print("AE Data Anomaly Detection System")
    print("=" * 40)
    
    # Initialize the detector
    detector = AnomalyDetector('AE Data_16FT - Sheet1.csv')
    
    # Run the complete pipeline
    detector.load_data()
    detector.preprocess_data()
    detector.train_models()
    detector.detect_anomalies()
    detector.visualize_results()
    detector.generate_report()
    detector.save_anomaly_data()
    
    print("\nAnomaly detection analysis completed successfully!")
    print("Check the generated files:")
    print("- anomaly_detection_results.png (visualizations)")
    print("- anomaly_results.csv (detailed results)")

if __name__ == "__main__":
    main()
