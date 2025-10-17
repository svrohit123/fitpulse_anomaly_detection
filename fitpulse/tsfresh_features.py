"""
TSFresh Feature Extraction Module for FitPulse Anomaly Detection

This module provides statistical feature extraction capabilities using TSFresh
for time series fitness data analysis.
"""

import pandas as pd
import numpy as np
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
import warnings
warnings.filterwarnings('ignore')

class TSFreshFeatureExtractor:
    """
    A class to extract statistical features from time series fitness data using TSFresh.
    """
    
    def __init__(self, default_fc_parameters=None):
        """
        Initialize the TSFresh feature extractor.
        
        Args:
            default_fc_parameters: Dictionary of feature calculators to use.
                                 If None, uses comprehensive parameters.
        """
        if default_fc_parameters is None:
            # Use comprehensive feature calculator parameters
            self.fc_parameters = ComprehensiveFCParameters()
        else:
            self.fc_parameters = default_fc_parameters
    
    def prepare_data_for_tsfresh(self, df, id_column='id', time_column='timestamp', value_columns=None):
        """
        Prepare the fitness data for TSFresh feature extraction.
        
        Args:
            df: DataFrame with fitness data
            id_column: Column name for series ID (default: 'id')
            time_column: Column name for timestamp (default: 'timestamp')
            value_columns: List of columns to extract features from
            
        Returns:
            DataFrame in TSFresh format
        """
        if value_columns is None:
            value_columns = ['heart_rate', 'sleep_duration', 'step_count']
        
        # Create a copy of the dataframe
        df_tsfresh = df.copy()
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df_tsfresh[time_column]):
            df_tsfresh[time_column] = pd.to_datetime(df_tsfresh[time_column])
        
        # Create a unique ID for each time series (assuming single user for now)
        df_tsfresh[id_column] = 0
        
        # Create time index (minutes since start)
        start_time = df_tsfresh[time_column].min()
        df_tsfresh['time'] = (df_tsfresh[time_column] - start_time).dt.total_seconds() / 60
        
        # Melt the dataframe to long format for TSFresh
        df_melted = pd.melt(
            df_tsfresh,
            id_vars=[id_column, 'time'],
            value_vars=value_columns,
            var_name='kind',
            value_name='value'
        )
        
        # Remove rows with NaN values
        df_melted = df_melted.dropna()
        
        return df_melted
    
    def extract_statistical_features(self, df, value_columns=None, feature_selection=True):
        """
        Extract statistical features from the fitness data.
        
        Args:
            df: DataFrame with fitness data
            value_columns: List of columns to extract features from
            feature_selection: Whether to perform feature selection
            
        Returns:
            DataFrame with extracted features
        """
        if value_columns is None:
            value_columns = ['heart_rate', 'sleep_duration', 'step_count']
        
        # Prepare data for TSFresh
        df_tsfresh = self.prepare_data_for_tsfresh(df, value_columns=value_columns)
        
        print(f"Extracting features from {len(df_tsfresh)} data points...")
        
        # Extract features
        extracted_features = extract_features(
            df_tsfresh,
            column_id='id',
            column_sort='time',
            column_value='value',
            column_kind='kind',
            default_fc_parameters=self.fc_parameters,
            n_jobs=1  # Use single job for stability
        )
        
        # Impute missing values
        extracted_features = impute(extracted_features)
        
        # Feature selection if requested
        if feature_selection:
            # TSFresh creates one row per series id. With a single id, there will be only
            # one row of extracted features; feature selection requires at least two samples.
            if len(extracted_features) < 2:
                # Not enough samples for selection; return extracted features as-is
                return extracted_features
            
            print("Performing feature selection...")
            # Create dummy target for feature selection (using heart_rate as example)
            if 'heart_rate__mean' in extracted_features.columns and 'heart_rate' in df.columns:
                y = df['heart_rate'].values
                # Align y with extracted features (ensure at least two samples)
                if len(y) != len(extracted_features):
                    y = y[:len(extracted_features)]
                if len(y) < 2:
                    return extracted_features
                
                selected_features = select_features(
                    extracted_features,
                    y,
                    fdr_level=0.05
                )
                extracted_features = selected_features
        
        return extracted_features
    
    def get_feature_summary(self, features_df):
        """
        Get a summary of the extracted features.
        
        Args:
            features_df: DataFrame with extracted features
            
        Returns:
            Dictionary with feature summary information
        """
        summary = {
            'total_features': len(features_df.columns),
            'feature_names': list(features_df.columns),
            'feature_types': {},
            'missing_values': features_df.isnull().sum().sum(),
            'feature_statistics': features_df.describe()
        }
        
        # Categorize features by type
        for col in features_df.columns:
            if '__mean' in col:
                summary['feature_types'].setdefault('mean', []).append(col)
            elif '__std' in col:
                summary['feature_types'].setdefault('std', []).append(col)
            elif '__min' in col:
                summary['feature_types'].setdefault('min', []).append(col)
            elif '__max' in col:
                summary['feature_types'].setdefault('max', []).append(col)
            elif '__median' in col:
                summary['feature_types'].setdefault('median', []).append(col)
            elif '__var' in col:
                summary['feature_types'].setdefault('variance', []).append(col)
            elif '__skewness' in col:
                summary['feature_types'].setdefault('skewness', []).append(col)
            elif '__kurtosis' in col:
                summary['feature_types'].setdefault('kurtosis', []).append(col)
            else:
                summary['feature_types'].setdefault('other', []).append(col)
        
        return summary
    
    def extract_simple_features(self, df, value_columns=None):
        """
        Extract a simplified set of features for faster processing.
        
        Args:
            df: DataFrame with fitness data
            value_columns: List of columns to extract features from
            
        Returns:
            DataFrame with simplified features
        """
        if value_columns is None:
            value_columns = ['heart_rate', 'sleep_duration', 'step_count']
        
        # Simple feature extraction without TSFresh for basic analysis
        features = {}
        
        for col in value_columns:
            if col in df.columns:
                # Basic statistical features
                features[f'{col}_mean'] = df[col].mean()
                features[f'{col}_std'] = df[col].std()
                features[f'{col}_min'] = df[col].min()
                features[f'{col}_max'] = df[col].max()
                features[f'{col}_median'] = df[col].median()
                features[f'{col}_var'] = df[col].var()
                features[f'{col}_skew'] = df[col].skew()
                features[f'{col}_kurtosis'] = df[col].kurtosis()
                
                # Trend features
                features[f'{col}_trend'] = np.polyfit(range(len(df)), df[col], 1)[0]
                
                # Range features
                features[f'{col}_range'] = df[col].max() - df[col].min()
                features[f'{col}_iqr'] = df[col].quantile(0.75) - df[col].quantile(0.25)
                
                # Count features
                features[f'{col}_count'] = len(df[col])
                features[f'{col}_non_zero_count'] = (df[col] != 0).sum()
        
        return pd.DataFrame([features])

def extract_fitness_features(df, method='simple', value_columns=None):
    """
    Convenience function to extract features from fitness data.
    
    Args:
        df: DataFrame with fitness data
        method: 'simple' for basic features, 'tsfresh' for comprehensive features
        value_columns: List of columns to extract features from
        
    Returns:
        DataFrame with extracted features
    """
    if method == 'simple':
        extractor = TSFreshFeatureExtractor()
        return extractor.extract_simple_features(df, value_columns)
    elif method == 'tsfresh':
        extractor = TSFreshFeatureExtractor()
        return extractor.extract_statistical_features(df, value_columns)
    else:
        raise ValueError("Method must be 'simple' or 'tsfresh'")

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Load sample data
    df = pd.read_csv('fitness_data.csv')
    
    # Extract simple features
    print("Extracting simple features...")
    simple_features = extract_fitness_features(df, method='simple')
    print("Simple features extracted:")
    print(simple_features)
    
    # Extract TSFresh features (commented out for demo as it requires more data)
    # print("\nExtracting TSFresh features...")
    # tsfresh_features = extract_fitness_features(df, method='tsfresh')
    # print("TSFresh features extracted:")
    # print(tsfresh_features.shape)
