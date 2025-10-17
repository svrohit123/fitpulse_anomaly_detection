"""
Demo script for FitPulse Advanced Features

This script demonstrates the new advanced features including TSFresh feature extraction,
Prophet modeling, and clustering analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_sample_data(n_days=30):
    """
    Create sample fitness data for demonstration.
    
    Args:
        n_days: Number of days of data to generate
        
    Returns:
        DataFrame with sample fitness data
    """
    # Generate timestamps
    start_date = datetime.now() - timedelta(days=n_days)
    timestamps = [start_date + timedelta(hours=i) for i in range(n_days * 24)]
    
    # Generate sample data with some patterns and anomalies
    np.random.seed(42)
    n_points = len(timestamps)
    
    # Heart rate with daily pattern and some anomalies
    base_hr = 70
    daily_pattern = 10 * np.sin(2 * np.pi * np.arange(n_points) / 24)  # Daily cycle
    weekly_pattern = 5 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 7))  # Weekly cycle
    noise = np.random.normal(0, 5, n_points)
    heart_rate = base_hr + daily_pattern + weekly_pattern + noise
    
    # Add some anomalies
    anomaly_indices = np.random.choice(n_points, size=int(0.05 * n_points), replace=False)
    heart_rate[anomaly_indices] += np.random.normal(0, 30, len(anomaly_indices))
    
    # Sleep duration with weekly pattern
    base_sleep = 7.5
    weekly_sleep_pattern = 1 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 7))
    sleep_noise = np.random.normal(0, 0.5, n_points)
    sleep_duration = base_sleep + weekly_sleep_pattern + sleep_noise
    
    # Add some sleep anomalies
    sleep_anomaly_indices = np.random.choice(n_points, size=int(0.03 * n_points), replace=False)
    sleep_duration[sleep_anomaly_indices] += np.random.normal(0, 2, len(sleep_anomaly_indices))
    
    # Step count with daily and weekly patterns
    base_steps = 8000
    daily_step_pattern = 3000 * np.sin(2 * np.pi * np.arange(n_points) / 24)
    weekly_step_pattern = 2000 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 7))
    step_noise = np.random.normal(0, 500, n_points)
    step_count = base_steps + daily_step_pattern + weekly_step_pattern + step_noise
    
    # Ensure positive values
    step_count = np.maximum(step_count, 0)
    
    # Add some step anomalies
    step_anomaly_indices = np.random.choice(n_points, size=int(0.04 * n_points), replace=False)
    step_count[step_anomaly_indices] += np.random.normal(0, 5000, len(step_anomaly_indices))
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'heart_rate': heart_rate,
        'sleep_duration': sleep_duration,
        'step_count': step_count
    })
    
    return df

def demo_tsfresh_features():
    """Demonstrate TSFresh feature extraction."""
    print("=" * 60)
    print("TSFresh Feature Extraction Demo")
    print("=" * 60)
    
    # Create sample data
    df = create_sample_data(n_days=7)  # Use smaller dataset for demo
    print(f"Created sample data with {len(df)} data points")
    
    # Extract simple features
    from tsfresh_features import extract_fitness_features
    
    print("\n1. Extracting simple features...")
    simple_features = extract_fitness_features(df, method='simple')
    print(f"Extracted {len(simple_features.columns)} simple features:")
    print(simple_features.columns.tolist())
    
    # Show feature values
    print("\nSimple feature values:")
    for col in simple_features.columns:
        print(f"  {col}: {simple_features[col].iloc[0]:.3f}")
    
    return simple_features

def demo_prophet_modeling():
    """Demonstrate Prophet modeling."""
    print("\n" + "=" * 60)
    print("Prophet Seasonal Modeling Demo")
    print("=" * 60)
    
    # Create sample data
    df = create_sample_data(n_days=14)  # Use 2 weeks for better seasonal patterns
    print(f"Created sample data with {len(df)} data points")
    
    from prophet_modeling import ProphetFitnessModeler
    
    # Initialize modeler
    modeler = ProphetFitnessModeler()
    
    # Model heart rate
    print("\n1. Modeling heart rate patterns...")
    try:
        model = modeler.fit_model(df, 'heart_rate')
        forecast = modeler.make_forecast('heart_rate', periods=7)
        anomalies = modeler.detect_anomalies_with_prophet(df, 'heart_rate')
        
        print(f"  - Model fitted successfully")
        print(f"  - Forecast created for 7 periods")
        print(f"  - Detected {anomalies['anomaly'].sum()} anomalies")
        print(f"  - Forecast shape: {forecast.shape}")
        
    except Exception as e:
        print(f"  - Prophet modeling failed: {str(e)}")
    
    return modeler

def demo_clustering_analysis():
    """Demonstrate clustering analysis."""
    print("\n" + "=" * 60)
    print("Clustering Analysis Demo")
    print("=" * 60)
    
    # Create sample data
    df = create_sample_data(n_days=10)
    print(f"Created sample data with {len(df)} data points")
    
    from clustering_analysis import FitnessClusteringAnalyzer
    
    # Initialize analyzer
    analyzer = FitnessClusteringAnalyzer()
    
    # Perform KMeans clustering
    print("\n1. Performing KMeans clustering...")
    try:
        kmeans_result = analyzer.perform_kmeans_clustering(df, n_clusters=3)
        print(f"  - KMeans completed successfully")
        print(f"  - Number of clusters: {kmeans_result['n_clusters']}")
        print(f"  - Silhouette score: {kmeans_result['silhouette_score']:.3f}")
        print(f"  - Features used: {len(kmeans_result['features_used'])}")
        
        # Show cluster statistics
        print("\n  Cluster statistics:")
        for cluster_id, stats in kmeans_result['cluster_stats'].items():
            print(f"    Cluster {cluster_id}: {stats['size']} points ({stats['percentage']:.1f}%)")
        
    except Exception as e:
        print(f"  - KMeans clustering failed: {str(e)}")
    
    # Perform DBSCAN clustering
    print("\n2. Performing DBSCAN clustering...")
    try:
        dbscan_result = analyzer.perform_dbscan_clustering(df)
        print(f"  - DBSCAN completed successfully")
        print(f"  - Number of clusters: {dbscan_result['n_clusters']}")
        print(f"  - Noise points: {dbscan_result['n_noise']}")
        print(f"  - Silhouette score: {dbscan_result['silhouette_score']:.3f}")
        
    except Exception as e:
        print(f"  - DBSCAN clustering failed: {str(e)}")
    
    return analyzer

def demo_enhanced_pipeline():
    """Demonstrate the enhanced pipeline."""
    print("\n" + "=" * 60)
    print("Enhanced Pipeline Demo")
    print("=" * 60)
    
    # Create sample data
    df = create_sample_data(n_days=5)
    print(f"Created sample data with {len(df)} data points")
    
    from enhanced_data_pipeline import EnhancedFitnessPipeline
    
    # Initialize pipeline
    pipeline = EnhancedFitnessPipeline()
    
    # Run comprehensive analysis
    print("\n1. Running comprehensive analysis...")
    try:
        results = pipeline.run_comprehensive_analysis(df)
        
        # Generate report
        report = pipeline.generate_analysis_report(results)
        
        print(f"  - Analysis completed successfully")
        print(f"  - Total records: {report['summary']['total_records']}")
        print(f"  - Quality score: {report['summary']['quality_score']}")
        print(f"  - Features extracted: {report['summary']['features_extracted']}")
        print(f"  - Prophet models: {report['summary']['prophet_models']}")
        print(f"  - Clustering methods: {report['summary']['clustering_methods']}")
        
        # Show anomaly summary
        print("\n  Anomaly summary:")
        for col, summary in report['anomaly_summary'].items():
            print(f"    {col}: {summary['total_anomalies']} anomalies ({summary['anomaly_rate']:.1f}%)")
        
        # Show clustering summary
        if report['clustering_summary']:
            print("\n  Clustering summary:")
            for method, summary in report['clustering_summary'].items():
                print(f"    {method}: {summary['n_clusters']} clusters (silhouette: {summary['silhouette_score']:.3f})")
        
    except Exception as e:
        print(f"  - Enhanced pipeline failed: {str(e)}")
    
    return pipeline

def main():
    """Run all demos."""
    print("FitPulse Advanced Features Demo")
    print("This demo showcases the new advanced features:")
    print("- TSFresh feature extraction")
    print("- Prophet seasonal modeling")
    print("- Clustering analysis")
    print("- Enhanced data pipeline")
    
    try:
        # Demo TSFresh features
        demo_tsfresh_features()
        
        # Demo Prophet modeling
        demo_prophet_modeling()
        
        # Demo clustering analysis
        demo_clustering_analysis()
        
        # Demo enhanced pipeline
        demo_enhanced_pipeline()
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        print("\nTo run the full application:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run Streamlit app: streamlit run app.py")
        print("3. Upload your fitness data and explore the new features!")
        
    except Exception as e:
        print(f"\nDemo failed with error: {str(e)}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
