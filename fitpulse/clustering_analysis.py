"""
Clustering Analysis Module for FitPulse Anomaly Detection

This module provides clustering capabilities using KMeans and DBSCAN
to group similar fitness behaviors and identify patterns.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class FitnessClusteringAnalyzer:
    """
    A class to perform clustering analysis on fitness data using KMeans and DBSCAN.
    """
    
    def __init__(self, scaler_type='standard'):
        """
        Initialize the clustering analyzer.
        
        Args:
            scaler_type: Type of scaler to use ('standard' or 'minmax')
        """
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        self.pca = None
        self.clusters = {}
        self.cluster_centers = {}
        self.cluster_labels = {}
        self.feature_importance = {}
    
    def prepare_features(self, df, feature_columns=None, time_window=24):
        """
        Prepare features for clustering analysis.
        
        Args:
            df: DataFrame with fitness data
            feature_columns: List of columns to use as features
            time_window: Time window for rolling features (in hours)
            
        Returns:
            DataFrame with prepared features
        """
        if feature_columns is None:
            feature_columns = ['heart_rate', 'sleep_duration', 'step_count']
        
        # Create a copy of the dataframe
        df_features = df.copy()
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df_features.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_features['timestamp']):
                df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
        
        # Create time-based features
        if 'timestamp' in df_features.columns:
            df_features['hour'] = df_features['timestamp'].dt.hour
            df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
            df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
        
        # Create rolling window features
        for col in feature_columns:
            if col in df_features.columns:
                # Rolling statistics
                df_features[f'{col}_rolling_mean'] = df_features[col].rolling(window=time_window, min_periods=1).mean()
                df_features[f'{col}_rolling_std'] = df_features[col].rolling(window=time_window, min_periods=1).std()
                df_features[f'{col}_rolling_min'] = df_features[col].rolling(window=time_window, min_periods=1).min()
                df_features[f'{col}_rolling_max'] = df_features[col].rolling(window=time_window, min_periods=1).max()
                
                # Lag features
                df_features[f'{col}_lag_1'] = df_features[col].shift(1)
                df_features[f'{col}_lag_2'] = df_features[col].shift(2)
                
                # Difference features
                df_features[f'{col}_diff'] = df_features[col].diff()
                df_features[f'{col}_diff_2'] = df_features[col].diff(2)
        
        # Create interaction features
        if 'heart_rate' in df_features.columns and 'step_count' in df_features.columns:
            df_features['hr_steps_ratio'] = df_features['heart_rate'] / (df_features['step_count'] + 1)
        
        if 'sleep_duration' in df_features.columns and 'step_count' in df_features.columns:
            df_features['sleep_steps_ratio'] = df_features['sleep_duration'] / (df_features['step_count'] + 1)
        
        # Create activity level categories
        if 'step_count' in df_features.columns:
            df_features['activity_level'] = pd.cut(
                df_features['step_count'],
                bins=[0, 5000, 10000, 15000, float('inf')],
                labels=['Low', 'Moderate', 'High', 'Very High']
            )
            df_features['activity_level_encoded'] = df_features['activity_level'].cat.codes
        
        # Select features for clustering
        clustering_features = []
        for col in df_features.columns:
            if any(keyword in col for keyword in ['rolling', 'lag', 'diff', 'ratio', 'hour', 'day_of_week', 'is_weekend', 'activity_level_encoded']):
                clustering_features.append(col)
        
        # Add original features
        clustering_features.extend(feature_columns)
        
        # Remove duplicates and ensure columns exist
        clustering_features = list(set(clustering_features))
        clustering_features = [col for col in clustering_features if col in df_features.columns]
        
        # Fill NaN values
        df_features[clustering_features] = df_features[clustering_features].fillna(df_features[clustering_features].mean())
        
        return df_features, clustering_features
    
    def perform_kmeans_clustering(self, df, feature_columns=None, n_clusters=None, 
                                 max_clusters=10, random_state=42):
        """
        Perform KMeans clustering on fitness data.
        
        Args:
            df: DataFrame with fitness data
            feature_columns: List of columns to use as features
            n_clusters: Number of clusters (if None, will be determined automatically)
            max_clusters: Maximum number of clusters to test
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with clustering results
        """
        # Prepare features
        df_features, clustering_features = self.prepare_features(df, feature_columns)
        
        # Scale features
        X = self.scaler.fit_transform(df_features[clustering_features])
        
        # Determine optimal number of clusters if not specified
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(X, max_clusters, random_state)
        
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Store results
        self.clusters['kmeans'] = kmeans
        self.cluster_centers['kmeans'] = kmeans.cluster_centers_
        self.cluster_labels['kmeans'] = cluster_labels
        
        # Calculate metrics
        silhouette_avg = silhouette_score(X, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(X, cluster_labels)
        
        # Create results dataframe
        results_df = df_features.copy()
        results_df['cluster'] = cluster_labels
        
        # Calculate cluster statistics
        cluster_stats = self._calculate_cluster_statistics(results_df, clustering_features)
        
        return {
            'model': kmeans,
            'labels': cluster_labels,
            'centers': kmeans.cluster_centers_,
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_harabasz,
            'n_clusters': n_clusters,
            'results_df': results_df,
            'cluster_stats': cluster_stats,
            'features_used': clustering_features
        }
    
    def perform_dbscan_clustering(self, df, feature_columns=None, eps=0.5, min_samples=5):
        """
        Perform DBSCAN clustering on fitness data.
        
        Args:
            df: DataFrame with fitness data
            feature_columns: List of columns to use as features
            eps: Maximum distance between two samples for one to be considered in the neighborhood
            min_samples: Minimum number of samples in a neighborhood for a core point
            
        Returns:
            Dictionary with clustering results
        """
        # Prepare features
        df_features, clustering_features = self.prepare_features(df, feature_columns)
        
        # Scale features
        X = self.scaler.fit_transform(df_features[clustering_features])
        
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(X)
        
        # Store results
        self.clusters['dbscan'] = dbscan
        self.cluster_labels['dbscan'] = cluster_labels
        
        # Calculate metrics (only for non-noise points)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        if n_clusters > 1:
            # Filter out noise points for metric calculation
            non_noise_mask = cluster_labels != -1
            if non_noise_mask.sum() > 1:
                silhouette_avg = silhouette_score(X[non_noise_mask], cluster_labels[non_noise_mask])
                calinski_harabasz = calinski_harabasz_score(X[non_noise_mask], cluster_labels[non_noise_mask])
            else:
                silhouette_avg = -1
                calinski_harabasz = 0
        else:
            silhouette_avg = -1
            calinski_harabasz = 0
        
        # Create results dataframe
        results_df = df_features.copy()
        results_df['cluster'] = cluster_labels
        
        # Calculate cluster statistics
        cluster_stats = self._calculate_cluster_statistics(results_df, clustering_features)
        
        return {
            'model': dbscan,
            'labels': cluster_labels,
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_harabasz,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'results_df': results_df,
            'cluster_stats': cluster_stats,
            'features_used': clustering_features
        }
    
    def _find_optimal_clusters(self, X, max_clusters, random_state):
        """
        Find optimal number of clusters using elbow method and silhouette analysis.
        
        Args:
            X: Scaled feature matrix
            max_clusters: Maximum number of clusters to test
            random_state: Random state for reproducibility
            
        Returns:
            Optimal number of clusters
        """
        if len(X) < 2:
            return 1
        
        max_clusters = min(max_clusters, len(X) - 1)
        
        # Calculate inertia and silhouette scores
        inertias = []
        silhouette_scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        
        # Find optimal k using silhouette score
        optimal_k = np.argmax(silhouette_scores) + 2
        
        return optimal_k
    
    def _calculate_cluster_statistics(self, results_df, feature_columns):
        """
        Calculate statistics for each cluster.
        
        Args:
            results_df: DataFrame with cluster labels
            feature_columns: List of feature columns
            
        Returns:
            Dictionary with cluster statistics
        """
        cluster_stats = {}
        
        for cluster_id in results_df['cluster'].unique():
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue
            
            cluster_data = results_df[results_df['cluster'] == cluster_id]
            
            stats = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(results_df) * 100
            }
            
            # Calculate statistics for each feature
            for col in feature_columns:
                if col in cluster_data.columns:
                    stats[f'{col}_mean'] = cluster_data[col].mean()
                    stats[f'{col}_std'] = cluster_data[col].std()
                    stats[f'{col}_min'] = cluster_data[col].min()
                    stats[f'{col}_max'] = cluster_data[col].max()
            
            cluster_stats[cluster_id] = stats
        
        return cluster_stats
    
    def plot_clusters_2d(self, results_df, feature_columns, method='kmeans', 
                        title=None, height=600):
        """
        Plot clusters in 2D using PCA.
        
        Args:
            results_df: DataFrame with cluster labels
            feature_columns: List of feature columns
            method: Clustering method ('kmeans' or 'dbscan')
            title: Title for the plot
            height: Height of the plot
            
        Returns:
            Plotly figure
        """
        if title is None:
            title = f"{method.upper()} Clustering Results"
        
        # Prepare data
        X = self.scaler.transform(results_df[feature_columns])
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Create dataframe for plotting
        plot_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'cluster': results_df['cluster']
        })
        
        # Create plot
        fig = px.scatter(
            plot_df,
            x='PC1',
            y='PC2',
            color='cluster',
            title=title,
            labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                   'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'},
            height=height
        )
        
        # Update layout
        fig.update_layout(
            paper_bgcolor='#1a1a1a',
            plot_bgcolor='#2d2d2d',
            font=dict(color='#ffffff'),
            xaxis=dict(gridcolor='#444444', zerolinecolor='#444444'),
            yaxis=dict(gridcolor='#444444', zerolinecolor='#444444')
        )
        
        return fig
    
    def plot_cluster_characteristics(self, cluster_stats, method='kmeans', title=None):
        """
        Plot cluster characteristics.
        
        Args:
            cluster_stats: Dictionary with cluster statistics
            method: Clustering method ('kmeans' or 'dbscan')
            title: Title for the plot
            
        Returns:
            Plotly figure
        """
        if title is None:
            title = f"{method.upper()} Cluster Characteristics"
        
        # Prepare data for plotting
        cluster_ids = list(cluster_stats.keys())
        cluster_sizes = [cluster_stats[cid]['size'] for cid in cluster_ids]
        cluster_percentages = [cluster_stats[cid]['percentage'] for cid in cluster_ids]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Cluster Sizes', 'Cluster Percentages'],
            vertical_spacing=0.1
        )
        
        # Add cluster sizes
        fig.add_trace(go.Bar(
            x=cluster_ids,
            y=cluster_sizes,
            name='Size',
            marker_color='lightblue'
        ), row=1, col=1)
        
        # Add cluster percentages
        fig.add_trace(go.Bar(
            x=cluster_ids,
            y=cluster_percentages,
            name='Percentage',
            marker_color='lightgreen'
        ), row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=title,
            height=600,
            paper_bgcolor='#1a1a1a',
            plot_bgcolor='#2d2d2d',
            font=dict(color='#ffffff'),
            showlegend=False
        )
        
        # Update axes
        for i in range(1, 3):
            fig.update_xaxes(gridcolor='#444444', zerolinecolor='#444444', row=i, col=1)
            fig.update_yaxes(gridcolor='#444444', zerolinecolor='#444444', row=i, col=1)
        
        return fig
    
    def get_cluster_summary(self, cluster_stats, method='kmeans'):
        """
        Get a summary of cluster analysis results.
        
        Args:
            cluster_stats: Dictionary with cluster statistics
            method: Clustering method ('kmeans' or 'dbscan')
            
        Returns:
            Dictionary with summary information
        """
        summary = {
            'method': method,
            'n_clusters': len(cluster_stats),
            'total_points': sum(stats['size'] for stats in cluster_stats.values()),
            'cluster_details': {}
        }
        
        for cluster_id, stats in cluster_stats.items():
            summary['cluster_details'][cluster_id] = {
                'size': stats['size'],
                'percentage': stats['percentage'],
                'characteristics': {k: v for k, v in stats.items() 
                                  if not k.startswith(('size', 'percentage'))}
            }
        
        return summary

def analyze_fitness_clusters(df, feature_columns=None, methods=['kmeans', 'dbscan']):
    """
    Convenience function to perform clustering analysis on fitness data.
    
    Args:
        df: DataFrame with fitness data
        feature_columns: List of columns to use as features
        methods: List of clustering methods to use
        
    Returns:
        Dictionary with clustering results
    """
    analyzer = FitnessClusteringAnalyzer()
    results = {}
    
    for method in methods:
        print(f"Performing {method.upper()} clustering...")
        
        if method == 'kmeans':
            result = analyzer.perform_kmeans_clustering(df, feature_columns)
        elif method == 'dbscan':
            result = analyzer.perform_dbscan_clustering(df, feature_columns)
        else:
            print(f"Unknown method: {method}")
            continue
        
        results[method] = result
        
        print(f"- Number of clusters: {result['n_clusters']}")
        print(f"- Silhouette score: {result['silhouette_score']:.3f}")
        if method == 'dbscan':
            print(f"- Noise points: {result['n_noise']}")
    
    return results

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Load sample data
    df = pd.read_csv('fitness_data.csv')
    
    # Perform clustering analysis
    print("Performing clustering analysis...")
    results = analyze_fitness_clusters(df)
    
    for method, result in results.items():
        print(f"\n{method.upper()} Results:")
        print(f"- Clusters: {result['n_clusters']}")
        print(f"- Silhouette Score: {result['silhouette_score']:.3f}")
        if method == 'dbscan':
            print(f"- Noise Points: {result['n_noise']}")
