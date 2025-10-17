"""
Facebook Prophet Modeling Module for FitPulse Anomaly Detection

This module provides seasonal pattern modeling and forecasting capabilities
using Facebook Prophet for fitness time series data.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class ProphetFitnessModeler:
    """
    A class to model seasonal patterns in fitness data using Facebook Prophet.
    """
    
    def __init__(self, seasonality_mode='multiplicative', yearly_seasonality=True, 
                 weekly_seasonality=True, daily_seasonality=False):
        """
        Initialize the Prophet modeler.
        
        Args:
            seasonality_mode: 'additive' or 'multiplicative'
            yearly_seasonality: Whether to include yearly seasonality
            weekly_seasonality: Whether to include weekly seasonality
            daily_seasonality: Whether to include daily seasonality
        """
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.models = {}
        self.forecasts = {}
        self.training_data = {}
    
    def prepare_data_for_prophet(self, df, value_column, time_column='timestamp'):
        """
        Prepare data for Prophet modeling.
        
        Args:
            df: DataFrame with fitness data
            value_column: Column name for the value to model
            time_column: Column name for timestamp
            
        Returns:
            DataFrame in Prophet format (ds, y)
        """
        df_prophet = df[[time_column, value_column]].copy()
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df_prophet[time_column]):
            df_prophet[time_column] = pd.to_datetime(df_prophet[time_column])
        
        # Rename columns for Prophet
        df_prophet = df_prophet.rename(columns={
            time_column: 'ds',
            value_column: 'y'
        })
        
        # Remove NaN values
        df_prophet = df_prophet.dropna()
        
        return df_prophet
    
    def fit_model(self, df, value_column, time_column='timestamp', **prophet_params):
        """
        Fit a Prophet model for the given fitness metric.
        
        Args:
            df: DataFrame with fitness data
            value_column: Column name for the value to model
            time_column: Column name for timestamp
            **prophet_params: Additional parameters for Prophet model
            
        Returns:
            Fitted Prophet model
        """
        # Prepare data
        df_prophet = self.prepare_data_for_prophet(df, value_column, time_column)
        # Ensure we have enough data points
        if len(df_prophet) < 2:
            raise ValueError(f"Not enough data points to fit Prophet model for {value_column} (need >= 2)")
        
        # Initialize Prophet model
        model = Prophet(
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            **prophet_params
        )
        
        # Fit the model
        model.fit(df_prophet)
        
        # Store the model
        self.models[value_column] = model
        # Store training data for plotting
        self.training_data[value_column] = df_prophet
        
        return model
    
    def make_forecast(self, value_column, periods=30, freq='D'):
        """
        Make a forecast using the fitted model.
        
        Args:
            value_column: Column name for the value to forecast
            periods: Number of periods to forecast
            freq: Frequency of the forecast ('D' for daily, 'H' for hourly)
            
        Returns:
            DataFrame with forecast
        """
        if value_column not in self.models:
            raise ValueError(f"No model found for {value_column}. Fit the model first.")
        
        model = self.models[value_column]
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq=freq)
        
        # Make forecast
        forecast = model.predict(future)
        
        # Store the forecast
        self.forecasts[value_column] = forecast
        
        return forecast
    
    def get_seasonal_components(self, value_column):
        """
        Extract seasonal components from the model.
        
        Args:
            value_column: Column name for the value
            
        Returns:
            Dictionary with seasonal components
        """
        if value_column not in self.models:
            raise ValueError(f"No model found for {value_column}. Fit the model first.")
        
        model = self.models[value_column]
        
        # Get seasonal components
        components = {}
        
        if hasattr(model, 'seasonalities'):
            for seasonality_name in model.seasonalities:
                if seasonality_name in ['yearly', 'weekly', 'daily']:
                    # Extract seasonal component
                    seasonal_data = model.predict_seasonal_components(
                        model.make_future_dataframe(periods=365 if seasonality_name == 'yearly' else 7)
                    )
                    components[seasonality_name] = seasonal_data
        
        return components
    
    def detect_anomalies_with_prophet(self, df, value_column, time_column='timestamp', 
                                    anomaly_threshold=2.0):
        """
        Detect anomalies using Prophet's prediction intervals.
        
        Args:
            df: DataFrame with fitness data
            value_column: Column name for the value to analyze
            time_column: Column name for timestamp
            anomaly_threshold: Threshold for anomaly detection (in standard deviations)
            
        Returns:
            DataFrame with anomaly flags
        """
        # Fit model if not already fitted
        if value_column not in self.models:
            self.fit_model(df, value_column, time_column)
        
        model = self.models[value_column]
        
        # Prepare data
        df_prophet = self.prepare_data_for_prophet(df, value_column, time_column)
        
        # Make prediction
        forecast = model.predict(df_prophet)
        
        # Calculate residuals
        residuals = df_prophet['y'] - forecast['yhat']
        
        # Calculate anomaly threshold
        residual_std = residuals.std()
        threshold = anomaly_threshold * residual_std
        
        # Detect anomalies
        anomalies = np.abs(residuals) > threshold
        
        # Create result dataframe
        result = df_prophet.copy()
        result['predicted'] = forecast['yhat']
        result['residual'] = residuals
        result['anomaly'] = anomalies
        result['anomaly_score'] = np.abs(residuals) / residual_std
        
        return result
    
    def plot_forecast(self, value_column, title=None):
        """
        Plot the forecast using Plotly.
        
        Args:
            value_column: Column name for the value
            title: Title for the plot
            
        Returns:
            Plotly figure
        """
        if value_column not in self.forecasts:
            raise ValueError(f"No forecast found for {value_column}. Make forecast first.")
        
        forecast = self.forecasts[value_column]
        
        if title is None:
            title = f"Prophet Forecast for {value_column.replace('_', ' ').title()}"
        
        # Create figure
        fig = go.Figure()
        
        # Add historical data from stored training data
        if value_column not in self.training_data:
            raise ValueError(f"No training data found for {value_column}. Fit the model first.")
        historical_data = self.training_data[value_column]
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
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title=value_column.replace('_', ' ').title(),
            hovermode='x unified',
            paper_bgcolor='#1a1a1a',
            plot_bgcolor='#2d2d2d',
            font=dict(color='#ffffff'),
            xaxis=dict(gridcolor='#444444', zerolinecolor='#444444'),
            yaxis=dict(gridcolor='#444444', zerolinecolor='#444444')
        )
        
        return fig
    
    def plot_components(self, value_column):
        """
        Plot the components of the Prophet model.
        
        Args:
            value_column: Column name for the value
            
        Returns:
            Plotly figure with subplots
        """
        if value_column not in self.forecasts:
            raise ValueError(f"No forecast found for {value_column}. Make forecast first.")
        
        forecast = self.forecasts[value_column]
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Trend', 'Weekly Seasonality', 'Yearly Seasonality'],
            vertical_spacing=0.1
        )
        
        # Add trend
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['trend'],
            mode='lines',
            name='Trend',
            line=dict(color='blue')
        ), row=1, col=1)
        
        # Add weekly seasonality if available
        if 'weekly' in forecast.columns:
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['weekly'],
                mode='lines',
                name='Weekly',
                line=dict(color='green')
            ), row=2, col=1)
        
        # Add yearly seasonality if available
        if 'yearly' in forecast.columns:
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yearly'],
                mode='lines',
                name='Yearly',
                line=dict(color='red')
            ), row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title=f"Prophet Components for {value_column.replace('_', ' ').title()}",
            height=800,
            paper_bgcolor='#1a1a1a',
            plot_bgcolor='#2d2d2d',
            font=dict(color='#ffffff'),
            showlegend=False
        )
        
        # Update axes
        for i in range(1, 4):
            fig.update_xaxes(gridcolor='#444444', zerolinecolor='#444444', row=i, col=1)
            fig.update_yaxes(gridcolor='#444444', zerolinecolor='#444444', row=i, col=1)
        
        return fig

def model_fitness_patterns(df, value_columns=None, forecast_periods=30):
    """
    Convenience function to model fitness patterns using Prophet.
    
    Args:
        df: DataFrame with fitness data
        value_columns: List of columns to model
        forecast_periods: Number of periods to forecast
        
    Returns:
        Dictionary with models and forecasts
    """
    if value_columns is None:
        value_columns = ['heart_rate', 'sleep_duration', 'step_count']
    
    modeler = ProphetFitnessModeler()
    results = {}
    
    for col in value_columns:
        if col in df.columns:
            print(f"Modeling {col}...")
            
            # Fit model
            model = modeler.fit_model(df, col)
            
            # Make forecast
            forecast = modeler.make_forecast(col, periods=forecast_periods)
            
            # Detect anomalies
            anomalies = modeler.detect_anomalies_with_prophet(df, col)
            
            results[col] = {
                'model': model,
                'forecast': forecast,
                'anomalies': anomalies
            }
    
    return results

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Load sample data
    df = pd.read_csv('fitness_data.csv')
    
    # Model fitness patterns
    print("Modeling fitness patterns with Prophet...")
    results = model_fitness_patterns(df, forecast_periods=7)
    
    for col, result in results.items():
        print(f"\n{col} Model Results:")
        print(f"- Anomalies detected: {result['anomalies']['anomaly'].sum()}")
        print(f"- Forecast shape: {result['forecast'].shape}")
