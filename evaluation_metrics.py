#!/usr/bin/env python3
"""
Evaluation Metrics for Anomaly Detection
========================================

This module provides comprehensive evaluation metrics and performance
comparison tools for anomaly detection models.

Author: AI Assistant
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    silhouette_score, calinski_harabasz_score
)
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class AnomalyEvaluator:
    """
    Comprehensive evaluation system for anomaly detection models.
    """
    
    def __init__(self, features, anomaly_scores):
        """
        Initialize the evaluator.
        
        Args:
            features (pd.DataFrame): Original feature data
            anomaly_scores (dict): Dictionary containing anomaly detection results
        """
        self.features = features
        self.anomaly_scores = anomaly_scores
        self.evaluation_results = {}
        
    def calculate_basic_metrics(self):
        """
        Calculate basic evaluation metrics for each model.
        """
        print("Calculating basic evaluation metrics...")
        
        for model_name, results in self.anomaly_scores.items():
            if model_name == 'ensemble':
                continue  # Skip ensemble for now
                
            predictions = results['predictions']
            scores = results['scores']
            anomalies = results['anomalies']
            
            # Convert predictions to binary (0 = normal, 1 = anomaly)
            y_pred = (predictions == -1).astype(int)
            
            # For unsupervised methods, we'll use the scores as confidence
            # and calculate metrics based on different thresholds
            thresholds = np.percentile(scores, [90, 95, 99])  # Top 10%, 5%, 1%
            
            model_metrics = {}
            for i, threshold in enumerate(thresholds):
                # Create binary predictions based on threshold
                if model_name == 'isolation_forest':
                    # Lower scores indicate more anomalous
                    y_pred_thresh = (scores < threshold).astype(int)
                else:
                    # Higher scores indicate more anomalous
                    y_pred_thresh = (scores > threshold).astype(int)
                
                # Calculate metrics
                precision = precision_score(anomalies, y_pred_thresh, zero_division=0)
                recall = recall_score(anomalies, y_pred_thresh, zero_division=0)
                f1 = f1_score(anomalies, y_pred_thresh, zero_division=0)
                
                model_metrics[f'threshold_{i+1}'] = {
                    'threshold': threshold,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'n_predicted_anomalies': np.sum(y_pred_thresh)
                }
            
            self.evaluation_results[model_name] = model_metrics
        
        return self.evaluation_results
    
    def calculate_clustering_metrics(self):
        """
        Calculate clustering-based evaluation metrics.
        """
        print("Calculating clustering evaluation metrics...")
        
        # Prepare data for clustering
        X = self.features.values
        
        # Calculate silhouette score for different numbers of clusters
        silhouette_scores = []
        calinski_scores = []
        k_range = range(2, min(11, len(X)//10))  # Reasonable range for k
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            if len(np.unique(cluster_labels)) > 1:  # Ensure we have multiple clusters
                sil_score = silhouette_score(X, cluster_labels)
                cal_score = calinski_harabasz_score(X, cluster_labels)
                
                silhouette_scores.append(sil_score)
                calinski_scores.append(cal_score)
            else:
                silhouette_scores.append(0)
                calinski_scores.append(0)
        
        # Find optimal number of clusters
        optimal_k_sil = k_range[np.argmax(silhouette_scores)]
        optimal_k_cal = k_range[np.argmax(calinski_scores)]
        
        clustering_metrics = {
            'silhouette_scores': dict(zip(k_range, silhouette_scores)),
            'calinski_scores': dict(zip(k_range, calinski_scores)),
            'optimal_k_silhouette': optimal_k_sil,
            'optimal_k_calinski': optimal_k_cal,
            'max_silhouette_score': max(silhouette_scores),
            'max_calinski_score': max(calinski_scores)
        }
        
        self.evaluation_results['clustering'] = clustering_metrics
        return clustering_metrics
    
    def calculate_statistical_metrics(self):
        """
        Calculate statistical properties of the detected anomalies.
        """
        print("Calculating statistical evaluation metrics...")
        
        statistical_metrics = {}
        
        for model_name, results in self.anomaly_scores.items():
            anomalies = results['anomalies']
            scores = results['scores']
            
            if np.sum(anomalies) > 0:
                anomaly_data = self.features[anomalies]
                normal_data = self.features[~anomalies]
                
                # Statistical properties
                model_stats = {
                    'n_anomalies': np.sum(anomalies),
                    'anomaly_rate': np.mean(anomalies),
                    
                    # Duration statistics
                    'duration_anomaly_mean': anomaly_data['Duration (microsecond)'].mean(),
                    'duration_anomaly_std': anomaly_data['Duration (microsecond)'].std(),
                    'duration_normal_mean': normal_data['Duration (microsecond)'].mean(),
                    'duration_normal_std': normal_data['Duration (microsecond)'].std(),
                    
                    # Amplitude statistics
                    'amplitude_anomaly_mean': anomaly_data['Amplitude (dB)'].mean(),
                    'amplitude_anomaly_std': anomaly_data['Amplitude (dB)'].std(),
                    'amplitude_normal_mean': normal_data['Amplitude (dB)'].mean(),
                    'amplitude_normal_std': normal_data['Amplitude (dB)'].std(),
                    
                    # Score statistics
                    'score_mean': np.mean(scores),
                    'score_std': np.std(scores),
                    'score_min': np.min(scores),
                    'score_max': np.max(scores)
                }
                
                # Effect sizes (Cohen's d)
                duration_effect_size = abs(
                    (model_stats['duration_anomaly_mean'] - model_stats['duration_normal_mean']) /
                    np.sqrt((model_stats['duration_anomaly_std']**2 + model_stats['duration_normal_std']**2) / 2)
                )
                
                amplitude_effect_size = abs(
                    (model_stats['amplitude_anomaly_mean'] - model_stats['amplitude_normal_mean']) /
                    np.sqrt((model_stats['amplitude_anomaly_std']**2 + model_stats['amplitude_normal_std']**2) / 2)
                )
                
                model_stats['duration_effect_size'] = duration_effect_size
                model_stats['amplitude_effect_size'] = amplitude_effect_size
                
                statistical_metrics[model_name] = model_stats
        
        self.evaluation_results['statistical'] = statistical_metrics
        return statistical_metrics
    
    def create_evaluation_plots(self, save_plots=True):
        """
        Create comprehensive evaluation visualizations.
        """
        print("Creating evaluation visualizations...")
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Model comparison - F1 scores
        plt.subplot(3, 4, 1)
        model_names = []
        f1_scores = []
        
        for model_name, metrics in self.evaluation_results.items():
            if model_name != 'clustering' and model_name != 'statistical':
                for threshold_name, threshold_metrics in metrics.items():
                    model_names.append(f"{model_name}\n({threshold_name})")
                    f1_scores.append(threshold_metrics['f1_score'])
        
        if f1_scores:
            bars = plt.bar(range(len(model_names)), f1_scores, alpha=0.7)
            plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
            plt.ylabel('F1 Score')
            plt.title('Model Performance Comparison (F1 Score)')
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, score in zip(bars, f1_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
        
        # 2. Precision-Recall curves
        plt.subplot(3, 4, 2)
        for model_name, results in self.anomaly_scores.items():
            if model_name != 'ensemble':
                scores = results['scores']
                anomalies = results['anomalies']
                
                # Normalize scores to [0, 1] for ROC/PR curves
                scores_norm = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
                
                precision, recall, _ = precision_recall_curve(anomalies, scores_norm)
                plt.plot(recall, precision, label=model_name.replace('_', ' ').title(), linewidth=2)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. ROC curves
        plt.subplot(3, 4, 3)
        for model_name, results in self.anomaly_scores.items():
            if model_name != 'ensemble':
                scores = results['scores']
                anomalies = results['anomalies']
                
                # Normalize scores
                scores_norm = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
                
                fpr, tpr, _ = roc_curve(anomalies, scores_norm)
                auc = roc_auc_score(anomalies, scores_norm)
                plt.plot(fpr, tpr, label=f"{model_name.replace('_', ' ').title()} (AUC={auc:.3f})", linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Clustering evaluation
        if 'clustering' in self.evaluation_results:
            plt.subplot(3, 4, 4)
            clustering_metrics = self.evaluation_results['clustering']
            k_range = list(clustering_metrics['silhouette_scores'].keys())
            sil_scores = list(clustering_metrics['silhouette_scores'].values())
            cal_scores = list(clustering_metrics['calinski_scores'].values())
            
            ax1 = plt.gca()
            color = 'tab:blue'
            ax1.set_xlabel('Number of Clusters (k)')
            ax1.set_ylabel('Silhouette Score', color=color)
            line1 = ax1.plot(k_range, sil_scores, 'o-', color=color, label='Silhouette')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, alpha=0.3)
            
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Calinski-Harabasz Score', color=color)
            line2 = ax2.plot(k_range, cal_scores, 's-', color=color, label='Calinski-Harabasz')
            ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title('Clustering Quality Metrics')
        
        # 5-8. Statistical comparison plots
        if 'statistical' in self.evaluation_results:
            stat_metrics = self.evaluation_results['statistical']
            
            # Effect sizes
            plt.subplot(3, 4, 5)
            models = list(stat_metrics.keys())
            duration_effects = [stat_metrics[model]['duration_effect_size'] for model in models]
            amplitude_effects = [stat_metrics[model]['amplitude_effect_size'] for model in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            plt.bar(x - width/2, duration_effects, width, label='Duration', alpha=0.7)
            plt.bar(x + width/2, amplitude_effects, width, label='Amplitude', alpha=0.7)
            
            plt.xlabel('Models')
            plt.ylabel('Effect Size (Cohen\'s d)')
            plt.title('Effect Sizes: Anomaly vs Normal')
            plt.xticks(x, [m.replace('_', ' ').title() for m in models], rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Anomaly rates
            plt.subplot(3, 4, 6)
            anomaly_rates = [stat_metrics[model]['anomaly_rate'] for model in models]
            bars = plt.bar(models, anomaly_rates, alpha=0.7)
            plt.ylabel('Anomaly Rate')
            plt.title('Anomaly Detection Rates')
            plt.xticks(rotation=45)
            
            for bar, rate in zip(bars, anomaly_rates):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{rate:.3f}', ha='center', va='bottom')
        
        # 9-12. Score distributions
        for i, (model_name, results) in enumerate(self.anomaly_scores.items()):
            if i >= 4:  # Limit to 4 models
                break
                
            plt.subplot(3, 4, 9 + i)
            scores = results['scores']
            anomalies = results['anomalies']
            
            plt.hist(scores[~anomalies], bins=50, alpha=0.7, label='Normal', density=True)
            plt.hist(scores[anomalies], bins=50, alpha=0.7, label='Anomaly', density=True)
            plt.xlabel('Anomaly Score')
            plt.ylabel('Density')
            plt.title(f'{model_name.replace("_", " ").title()} Score Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('evaluation_metrics.png', dpi=300, bbox_inches='tight')
            print("Evaluation plots saved as 'evaluation_metrics.png'")
        
        plt.show()
    
    def generate_evaluation_report(self):
        """
        Generate a comprehensive evaluation report.
        """
        print("\n" + "="*70)
        print("ANOMALY DETECTION EVALUATION REPORT")
        print("="*70)
        
        # Basic metrics
        if any(key in self.evaluation_results for key in ['isolation_forest', 'one_class_svm', 'lof']):
            print("\nMODEL PERFORMANCE METRICS:")
            print("-" * 50)
            
            for model_name, metrics in self.evaluation_results.items():
                if model_name not in ['clustering', 'statistical']:
                    print(f"\n{model_name.replace('_', ' ').title()}:")
                    for threshold_name, threshold_metrics in metrics.items():
                        print(f"  {threshold_name}:")
                        print(f"    Precision: {threshold_metrics['precision']:.4f}")
                        print(f"    Recall:    {threshold_metrics['recall']:.4f}")
                        print(f"    F1 Score:  {threshold_metrics['f1_score']:.4f}")
                        print(f"    Threshold: {threshold_metrics['threshold']:.4f}")
        
        # Clustering metrics
        if 'clustering' in self.evaluation_results:
            print(f"\nCLUSTERING EVALUATION:")
            print("-" * 50)
            clustering = self.evaluation_results['clustering']
            print(f"Optimal number of clusters (Silhouette): {clustering['optimal_k_silhouette']}")
            print(f"Optimal number of clusters (Calinski): {clustering['optimal_k_calinski']}")
            print(f"Maximum Silhouette Score: {clustering['max_silhouette_score']:.4f}")
            print(f"Maximum Calinski-Harabasz Score: {clustering['max_calinski_score']:.4f}")
        
        # Statistical metrics
        if 'statistical' in self.evaluation_results:
            print(f"\nSTATISTICAL ANALYSIS:")
            print("-" * 50)
            
            for model_name, stats in self.evaluation_results['statistical'].items():
                print(f"\n{model_name.replace('_', ' ').title()}:")
                print(f"  Anomaly Rate: {stats['anomaly_rate']:.4f} ({stats['n_anomalies']:,} anomalies)")
                print(f"  Duration Effect Size: {stats['duration_effect_size']:.4f}")
                print(f"  Amplitude Effect Size: {stats['amplitude_effect_size']:.4f}")
                print(f"  Score Range: [{stats['score_min']:.4f}, {stats['score_max']:.4f}]")
        
        print("\n" + "="*70)
    
    def save_evaluation_results(self, filename='evaluation_results.csv'):
        """
        Save evaluation results to CSV files.
        """
        print(f"\nSaving evaluation results to '{filename}'...")
        
        # Create summary dataframe
        summary_data = []
        
        for model_name, metrics in self.evaluation_results.items():
            if model_name not in ['clustering', 'statistical']:
                for threshold_name, threshold_metrics in metrics.items():
                    summary_data.append({
                        'model': model_name,
                        'threshold': threshold_name,
                        'precision': threshold_metrics['precision'],
                        'recall': threshold_metrics['recall'],
                        'f1_score': threshold_metrics['f1_score'],
                        'threshold_value': threshold_metrics['threshold'],
                        'n_predicted_anomalies': threshold_metrics['n_predicted_anomalies']
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(filename, index=False)
            print(f"Evaluation summary saved to '{filename}'")
        
        return summary_df if summary_data else None

def main():
    """
    Example usage of the evaluation system.
    """
    print("Anomaly Detection Evaluation System")
    print("=" * 40)
    print("This module is designed to be used with the main anomaly detection script.")
    print("Import and use the AnomalyEvaluator class to evaluate your models.")

if __name__ == "__main__":
    main()
