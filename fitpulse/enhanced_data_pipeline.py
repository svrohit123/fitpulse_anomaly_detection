"""
Enhanced Data Pipeline for FitPulse Anomaly Detection

This module integrates TSFresh feature extraction, Prophet modeling,
and clustering analysis into a comprehensive data processing pipeline.
"""

import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st
from tsfresh_features import TSFreshFeatureExtractor, extract_fitness_features
from prophet_modeling import ProphetFitnessModeler, model_fitness_patterns
from clustering_analysis import FitnessClusteringAnalyzer, analyze_fitness_clusters
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class EnhancedFitnessPipeline:
    """
    Enhanced pipeline that combines traditional anomaly detection with
    advanced feature extraction, seasonal modeling, and clustering analysis.
    """
    
    def __init__(self):
        """Initialize the enhanced pipeline."""
        self.tsfresh_extractor = TSFreshFeatureExtractor()
        self.prophet_modeler = ProphetFitnessModeler()
        self.clustering_analyzer = FitnessClusteringAnalyzer()
        self.features = None
        self.prophet_results = None
        self.clustering_results = None
        self.anomaly_results = None
    
    def load_data(self, file_path):
        """
        Load data from CSV or JSON file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            DataFrame with loaded data
        """
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format. Use .csv or .json")
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def data_quality_report(self, df):
        """
        Generate comprehensive data quality report.
        
        Args:
            df: DataFrame with fitness data
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            total_records = len(df)
            missing_counts = df.isnull().sum().to_dict()
            total_missing = sum(missing_counts.values())
            quality_score = round(100 * (1 - total_missing / (df.shape[0] * df.shape[1])), 2) if df.shape[0] > 0 else 0
            
            # Additional quality metrics
            duplicate_rows = df.duplicated().sum()
            data_types = df.dtypes.to_dict()
            
            # Check for outliers using IQR method
            outlier_info = {}
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col != 'timestamp':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    outlier_info[col] = {
                        'count': int(outliers),
                        'percentage': float((outliers / len(df)) * 100)
                    }
            
            report = {
                "total_records": total_records,
                "missing": missing_counts,
                "duplicate_rows": int(duplicate_rows),
                "data_types": {str(k): str(v) for k, v in data_types.items()},
                "outlier_info": outlier_info,
                "quality_score": f"{quality_score}%",
                "describe": df.describe().to_dict()
            }
            return report
            
        except Exception as e:
            st.error(f"Error in data quality report: {str(e)}")
            return {
                "total_records": 0,
                "missing": {},
                "duplicate_rows": 0,
                "data_types": {},
                "outlier_info": {},
                "quality_score": "0%",
                "describe": {}
            }
        
        report = {
            "total_records": total_records,
            "shape": df.shape,
            "missing": df.isnull().sum().to_dict(),
            "duplicate_rows": duplicate_rows,
            "data_types": data_types,
            "outlier_info": outlier_info,
            "quality_score": f"{quality_score}%",
            "describe": df.describe(include='all').to_dict()
        }
        
        return report
    
    def detect_anomalies(self, df, column, threshold=3, custom_bounds=None):
        """
        Enhanced anomaly detection using multiple methods.
        
        Args:
            df: DataFrame with fitness data
            column: Column name to analyze
            threshold: Z-score threshold
            custom_bounds: Custom bounds for anomaly detection
            
        Returns:
            Dictionary with anomaly detection results
        """
        anomalies = np.zeros(len(df), dtype=bool)
        anomaly_scores = np.zeros(len(df))
        
        # Z-score based anomalies
        z_scores = np.abs(stats.zscore(df[column]))
        z_anomalies = z_scores > threshold
        anomalies = anomalies | z_anomalies
        anomaly_scores += z_scores
        
        # Custom bounds based anomalies
        if custom_bounds:
            lower, upper = custom_bounds
            bound_anomalies = (df[column] < lower) | (df[column] > upper)
            anomalies = anomalies | bound_anomalies
            # Add to anomaly scores
            bound_scores = np.where(bound_anomalies, 2.0, 0.0)
            anomaly_scores += bound_scores
        
        # IQR based anomalies
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_anomalies = (df[column] < lower_bound) | (df[column] > upper_bound)
        anomalies = anomalies | iqr_anomalies
        
        # Calculate anomaly severity
        anomaly_severity = np.where(anomalies, anomaly_scores, 0)
        
        return {
            'anomalies': anomalies,
            'anomaly_scores': anomaly_scores,
            'anomaly_severity': anomaly_severity,
            'z_scores': z_scores,
            'iqr_anomalies': iqr_anomalies,
            'bound_anomalies': bound_anomalies if custom_bounds else None
        }
    
    def extract_features(self, df, method='simple', value_columns=None):
        """
        Extract features using TSFresh or simple methods.
        
        Args:
            df: DataFrame with fitness data
            method: 'simple' or 'tsfresh'
            value_columns: List of columns to extract features from
            
        Returns:
            DataFrame with extracted features
        """
        if value_columns is None:
            value_columns = ['heart_rate', 'sleep_duration', 'step_count']
        
        self.features = extract_fitness_features(df, method=method, value_columns=value_columns)
        return self.features
    
    def model_seasonal_patterns(self, df, value_columns=None, forecast_periods=30):
        """
        Model seasonal patterns using Prophet.
        
        Args:
            df: DataFrame with fitness data
            value_columns: List of columns to model
            forecast_periods: Number of periods to forecast
            
        Returns:
            Dictionary with Prophet results
        """
        if value_columns is None:
            value_columns = ['heart_rate', 'sleep_duration', 'step_count']
        
        self.prophet_results = model_fitness_patterns(df, value_columns, forecast_periods)
        return self.prophet_results
    
    def perform_clustering(self, df, feature_columns=None, methods=['kmeans', 'dbscan']):
        """
        Perform clustering analysis on fitness data.
        
        Args:
            df: DataFrame with fitness data
            feature_columns: List of columns to use as features
            methods: List of clustering methods to use
            
        Returns:
            Dictionary with clustering results
        """
        self.clustering_results = analyze_fitness_clusters(df, feature_columns, methods)
        return self.clustering_results
    
    def run_comprehensive_analysis(self, df, value_columns=None, 
                                 anomaly_configs=None, forecast_periods=30):
        """
        Run comprehensive analysis including all features.
        
        Args:
            df: DataFrame with fitness data
            value_columns: List of columns to analyze
            anomaly_configs: Configuration for anomaly detection
            forecast_periods: Number of periods to forecast
            
        Returns:
            Dictionary with all analysis results
        """
        if value_columns is None:
            value_columns = ['heart_rate', 'sleep_duration', 'step_count']
        
        if anomaly_configs is None:
            anomaly_configs = {
                'heart_rate': {'bounds': (50, 100), 'z_threshold': 2.5},
                'sleep_duration': {'bounds': (4, 10), 'z_threshold': 2.5},
                'step_count': {'bounds': (1000, 20000), 'z_threshold': 2.5}
            }
        
        results = {
            'data_quality': self.data_quality_report(df),
            'anomaly_detection': {},
            'feature_extraction': {},
            'seasonal_modeling': {},
            'clustering_analysis': {}
        }
        
        # Anomaly detection
        for col in value_columns:
            if col in df.columns:
                config = anomaly_configs.get(col, {})
                results['anomaly_detection'][col] = self.detect_anomalies(
                    df, col, 
                    threshold=config.get('z_threshold', 2.5),
                    custom_bounds=config.get('bounds')
                )
        
        # Feature extraction
        try:
            results['feature_extraction']['simple'] = self.extract_features(df, method='simple', value_columns=value_columns)
        except Exception as e:
            st.warning(f"Simple feature extraction failed: {str(e)}")
            results['feature_extraction']['simple'] = None
        
        # Seasonal modeling
        try:
            results['seasonal_modeling'] = self.model_seasonal_patterns(df, value_columns, forecast_periods)
        except Exception as e:
            st.warning(f"Prophet modeling failed: {str(e)}")
            results['seasonal_modeling'] = {}
        
        # Clustering analysis
        try:
            results['clustering_analysis'] = self.perform_clustering(df, value_columns)
        except Exception as e:
            st.warning(f"Clustering analysis failed: {str(e)}")
            results['clustering_analysis'] = {}
        
        return results
    
    def create_enhanced_visualizations(self, df, results, value_columns=None):
        """
        Create enhanced visualizations for the analysis results.
        
        Args:
            df: DataFrame with fitness data
            results: Dictionary with analysis results
            value_columns: List of columns to visualize
            
        Returns:
            Dictionary with Plotly figures
        """
        if value_columns is None:
            value_columns = ['heart_rate', 'sleep_duration', 'step_count']
        
        figures = {}
        
        # Anomaly detection plots
        for col in value_columns:
            if col in df.columns and col in results['anomaly_detection']:
                anomaly_data = results['anomaly_detection'][col]
                
                fig = go.Figure()
                
                # Add normal data points
                normal_mask = ~anomaly_data['anomalies']
                normal_data = df[normal_mask]
                fig.add_trace(go.Scatter(
                    x=normal_data['timestamp'],
                    y=normal_data[col],
                    mode='markers',
                    name='Normal',
                    marker=dict(color='blue', size=6)
                ))
                
                # Add anomaly points
                anomaly_mask = anomaly_data['anomalies']
                anomaly_data_points = df[anomaly_mask]
                fig.add_trace(go.Scatter(
                    x=anomaly_data_points['timestamp'],
                    y=anomaly_data_points[col],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(color='red', size=8)
                ))
                
                fig.update_layout(
                    title=f"Enhanced Anomaly Detection - {col.replace('_', ' ').title()}",
                    xaxis_title="Timestamp",
                    yaxis_title=col.replace('_', ' ').title(),
                    hovermode='x unified',
                    paper_bgcolor='#1a1a1a',
                    plot_bgcolor='#2d2d2d',
                    font=dict(color='#ffffff'),
                    xaxis=dict(gridcolor='#444444', zerolinecolor='#444444'),
                    yaxis=dict(gridcolor='#444444', zerolinecolor='#444444')
                )
                
                figures[f'anomaly_{col}'] = fig
        
        # Prophet forecast plots
        if 'seasonal_modeling' in results and results['seasonal_modeling']:
            for col, prophet_result in results['seasonal_modeling'].items():
                if 'forecast' in prophet_result:
                    forecast = prophet_result['forecast']
                    
                    fig = go.Figure()
                    
                    # Add historical data
                    historical_data = forecast[forecast['y'].notna()]
                    fig.add_trace(go.Scatter(
                        x=historical_data['ds'],
                        y=historical_data['y'],
                        mode='markers',
                        name='Historical Data',
                        marker=dict(color='blue', size=6)
                    ))
                    
                    # Add forecast
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat'],
                        mode='lines',
                        name='Forecast',
                        line=dict(color='red', width=2)
                    ))
                    
                    # Add confidence intervals
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat_upper'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat_lower'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(255,0,0,0.2)',
                        name='Confidence Interval'
                    ))
                    
                    fig.update_layout(
                        title=f"Prophet Forecast - {col.replace('_', ' ').title()}",
                        xaxis_title="Date",
                        yaxis_title=col.replace('_', ' ').title(),
                        hovermode='x unified',
                        paper_bgcolor='#1a1a1a',
                        plot_bgcolor='#2d2d2d',
                        font=dict(color='#ffffff'),
                        xaxis=dict(gridcolor='#444444', zerolinecolor='#444444'),
                        yaxis=dict(gridcolor='#444444', zerolinecolor='#444444')
                    )
                    
                    figures[f'prophet_{col}'] = fig
        
        # Clustering plots
        if 'clustering_analysis' in results and results['clustering_analysis']:
            for method, cluster_result in results['clustering_analysis'].items():
                if 'results_df' in cluster_result:
                    results_df = cluster_result['results_df']
                    features_used = cluster_result['features_used']
                    
                    # Create 2D cluster plot
                    fig = self.clustering_analyzer.plot_clusters_2d(
                        results_df, features_used, method=method
                    )
                    figures[f'clusters_{method}'] = fig
        
        return figures
    
    def generate_analysis_report(self, results):
        """
        Generate a comprehensive analysis report.
        
        Args:
            results: Dictionary with analysis results
            
        Returns:
            Dictionary with report summary
        """
        report = {
            'summary': {
                'total_records': results['data_quality']['total_records'],
                'quality_score': results['data_quality']['quality_score'],
                'features_extracted': len(results['feature_extraction'].get('simple', {}).columns) if results['feature_extraction'].get('simple') is not None else 0,
                'prophet_models': len(results['seasonal_modeling']),
                'clustering_methods': len(results['clustering_analysis'])
            },
            'anomaly_summary': {},
            'clustering_summary': {}
        }
        
        # Anomaly summary
        for col, anomaly_data in results['anomaly_detection'].items():
            report['anomaly_summary'][col] = {
                'total_anomalies': anomaly_data['anomalies'].sum(),
                'anomaly_rate': (anomaly_data['anomalies'].sum() / len(anomaly_data['anomalies'])) * 100,
                'max_anomaly_score': anomaly_data['anomaly_scores'].max()
            }
        
        # Clustering summary
        for method, cluster_result in results['clustering_analysis'].items():
            report['clustering_summary'][method] = {
                'n_clusters': cluster_result['n_clusters'],
                'silhouette_score': cluster_result['silhouette_score'],
                'calinski_harabasz_score': cluster_result['calinski_harabasz_score']
            }
            if method == 'dbscan':
                report['clustering_summary'][method]['n_noise'] = cluster_result['n_noise']
        
        return report

# Convenience functions for backward compatibility
def load_data(file_path):
    """Load data from file."""
    pipeline = EnhancedFitnessPipeline()
    return pipeline.load_data(file_path)

def data_quality_report(df):
    """Generate data quality report."""
    pipeline = EnhancedFitnessPipeline()
    return pipeline.data_quality_report(df)

def detect_anomalies(df, column, threshold=3, custom_bounds=None):
    """Detect anomalies in data."""
    pipeline = EnhancedFitnessPipeline()
    return pipeline.detect_anomalies(df, column, threshold, custom_bounds)

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Load sample data
    df = pd.read_csv('fitness_data.csv')
    
    # Create enhanced pipeline
    pipeline = EnhancedFitnessPipeline()
    
    # Run comprehensive analysis
    print("Running comprehensive analysis...")
    results = pipeline.run_comprehensive_analysis(df)
    
    # Generate report
    report = pipeline.generate_analysis_report(results)
    print("\nAnalysis Report:")
    print(f"Total Records: {report['summary']['total_records']}")
    print(f"Quality Score: {report['summary']['quality_score']}")
    print(f"Features Extracted: {report['summary']['features_extracted']}")
    print(f"Prophet Models: {report['summary']['prophet_models']}")
    print(f"Clustering Methods: {report['summary']['clustering_methods']}")
    
    # Print anomaly summary
    print("\nAnomaly Summary:")
    for col, summary in report['anomaly_summary'].items():
        print(f"{col}: {summary['total_anomalies']} anomalies ({summary['anomaly_rate']:.1f}%)")
    
    # Print clustering summary
    print("\nClustering Summary:")
    for method, summary in report['clustering_summary'].items():
        print(f"{method}: {summary['n_clusters']} clusters (silhouette: {summary['silhouette_score']:.3f})")
