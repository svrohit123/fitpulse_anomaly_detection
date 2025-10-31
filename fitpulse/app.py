import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from enhanced_data_pipeline import EnhancedFitnessPipeline, load_data, detect_anomalies, data_quality_report
from reporting import build_anomaly_summary_dataframe, dataframe_to_csv_bytes
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="FitPulse Anomaly Detection", page_icon="💓", layout="wide")

# Sidebar
with st.sidebar:
    # Logo section
    st.markdown("""
    <div style='text-align: center; margin-bottom: 20px;'>
        <div style='display: inline-block; padding: 10px; background: linear-gradient(135deg, var(--accent), var(--accent2)); border-radius: 12px; margin-bottom: 10px;'>
            <div style='color: white; font-size: 24px; font-weight: bold;'>💓</div>
        </div>
        <h1 style='color: var(--accent); margin: 0;'>FitPulse</h1>
        <h3 style='color: var(--muted); margin: 0; font-size: 14px; font-weight: normal;'>Anomaly Detection</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='padding: 10px;'>
        <h4 style='color: var(--accent);'>🎯 Project Overview</h4>
        <p>FitPulse is an advanced fitness data analysis system that detects anomalies in:</p>
        <ul>
            <li>Heart Rate Patterns</li>
            <li>Sleep Duration</li>
            <li>Daily Step Counts</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='padding: 10px;'>
        <h4 style='color: var(--accent);'>🔍 Features</h4>
        <ul>
            <li>Real-time data analysis</li>
            <li>Interactive visualizations</li>
            <li>Customizable thresholds</li>
            <li>CSV & JSON support</li>
            <li>Detailed anomaly reports</li>
            <li>TSFresh feature extraction</li>
            <li>Prophet seasonal modeling</li>
            <li>KMeans & DBSCAN clustering</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='padding: 10px;'>
        <h4 style='color: #00ccff;'>📝 How to Use</h4>
        <ol>
            <li>Upload your fitness data file (CSV/JSON)</li>
            <li>Adjust detection thresholds as needed</li>
            <li>Explore the visualizations in each tab</li>
            <li>Review anomaly reports and statistics</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Add threshold explanations
    with st.expander("ℹ️ Normal Ranges"):
        st.markdown("""
        <div style='padding: 5px;'>
            <p><strong>Heart Rate:</strong> 50-100 bpm</p>
            <small>Normal resting heart rate for adults</small>
            
            <p><strong>Sleep Duration:</strong> 4-10 hours</p>
            <small>Typical daily sleep patterns</small>
            
            <p><strong>Step Count:</strong> 1,000-20,000 steps</p>
            <small>Common daily step range</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Add about section
    with st.expander("ℹ️ About"):
        st.markdown("""
        <div style='padding: 5px;'>
            <p>FitPulse Anomaly Detection is a data analysis tool designed to help identify unusual patterns in fitness tracking data.</p>
            <p>Version: 1.0.0</p>
            <p>Created by: FitPulse Team</p>
        </div>
        """, unsafe_allow_html=True)

# Theme toggle and dynamic CSS
with st.sidebar:
    theme_choice = st.selectbox("Theme", ["Dark", "Light"], index=0)

theme = {
    "Dark": {
        "bg": "#101014",            # nearly black
        "card": "#18181b",          # dark neutral for cards
        "text": "#ffffff",           # white
        "muted": "#b0b0b8",         # medium-light gray
        "border": "#2563eb",        # blue-600 border
        "accent": "#2563eb",        # blue-600 main accent
        "accent2": "#1e3a8a",       # blue-900 secondary
        "plot_bg": "#18181b",
        "paper_bg": "#101014",
        "grid": "#233763"           # blue-gray grid
    },
    "Light": {
        "bg": "#101014",            # unify for now
        "card": "#18181b",
        "text": "#ffffff",
        "muted": "#b0b0b8",
        "border": "#2563eb",
        "accent": "#2563eb",
        "accent2": "#1e3a8a",
        "plot_bg": "#18181b",
        "paper_bg": "#101014",
        "grid": "#233763"
    }
}[theme_choice]

# --- Streamlit and App-wide CSS Styling ---
st.markdown(f"""
<style>
    :root {{
        --bg: {theme['bg']};
        --card: {theme['card']};
        --text: {theme['text']};
        --muted: {theme['muted']};
        --border: {theme['border']};
        --accent: {theme['accent']};
        --accent2: {theme['accent2']};
    }}
    .main {{
        background-color: var(--bg);
        color: var(--text);
    }}
    .sidebar .sidebar-content {{
        background-color: var(--card);
        color: var(--text);
        border-right: 1px solid var(--border);
    }}
    h1 {{ color: var(--accent); font-weight: 700; }}
    h2, h3, h4 {{ color: var(--accent); }}
    .metric-card {{
        background-color: var(--card);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid var(--border);
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        transition: transform .2s ease, box-shadow .2s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    }}
    .stButton>button {{
        background: var(--accent);
        background-image: linear-gradient(90deg, var(--accent), var(--accent2));
        color: #fff;
        font-weight: 700;
        padding: 10px 18px;
        border-radius: 8px;
        border: none;
        box-shadow: 0 0 8px 2px var(--accent2), 0 2px 10px 0 var(--accent);
        transition: box-shadow 0.18s, background 0.18s, filter 0.16s;
        filter: brightness(1.08);
        outline: none;
        position: relative;
        z-index: 1;
    }}
    .stButton>button:hover, .stButton>button:focus {{
        filter: brightness(1.18) saturate(1.2);
        box-shadow: 0 0 16px 6px var(--accent2), 0 2px 18px 6px var(--accent), 0 0 0 2.5px #fff8, 0 0 8px 2px var(--accent2);
        cursor: pointer;
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: var(--card);
        border: 1px solid var(--border);
        color: var(--text);
        border-radius: 8px;
        padding: 8px 14px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: var(--accent);
        color: #fff;
        border-color: transparent;
    }}
</style>
""", unsafe_allow_html=True)

def apply_plot_theme(fig):
    fig.update_layout(
        paper_bgcolor=theme['paper_bg'],
        plot_bgcolor=theme['plot_bg'],
        font=dict(color=theme['text']),
        xaxis=dict(gridcolor=theme['grid'], zerolinecolor=theme['grid']),
        yaxis=dict(gridcolor=theme['grid'], zerolinecolor=theme['grid'])
    )
    return fig

# Title and description
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1> FitPulse Anomaly Detection</h1>
    <p style='font-size: 1.1em; color: var(--muted);'>
        Analyze anomalies in heart rate, sleep duration, and step count.<br>
        Upload your fitness data file (CSV or JSON) to get started!
    </p>
</div>
""", unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader("Choose a file", type=['csv', 'json'])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    try:
        # Load and analyze data
        df = load_data(uploaded_file.name)
        report = data_quality_report(df)
        
        # Display basic statistics
        st.markdown("""
        <div style='padding: 20px 0;'>
            <h2 style='color: #00ccff; text-align: center; margin-bottom: 20px;'>📊 Data Overview</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h4 style='color: #7F8C8D; margin: 0;'>Total Records</h4>
                <h2 style='color: #2980B9; margin: 10px 0;'>{report['total_records']}</h2>
                <p style='color: #95A5A6; margin: 0;'>Total data points analyzed</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            quality_score = report['quality_score'].replace('%', '')
            color = theme['accent2'] if float(quality_score) > 90 else '#E67E22' if float(quality_score) > 70 else '#E74C3C'
            st.markdown(f"""
            <div class='metric-card'>
                <h4 style='color: #7F8C8D; margin: 0;'>Quality Score</h4>
                <h2 style='color: {color}; margin: 10px 0;'>{report['quality_score']}</h2>
                <p style='color: #95A5A6; margin: 0;'>Data quality assessment</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            missing_values = sum(report['missing'].values()) if isinstance(report['missing'], dict) and report['missing'] is not None else 0
            st.markdown(f"""
            <div class='metric-card'>
                <h4 style='color: #7F8C8D; margin: 0;'>Missing Values</h4>
                <h2 style='color: #8E44AD; margin: 10px 0;'>{missing_values}</h2>
                <p style='color: #95A5A6; margin: 0;'>Total missing data points</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Anomaly Detection Configuration
        st.markdown("""
        <div style='padding: 20px 0;'>
            <h2 style='color: #00ccff; text-align: center; margin-bottom: 20px;'>
                🎯 Anomaly Detection Settings
            </h2>
            <p style='text-align: center; color: #7F8C8D;'>
                Adjust the thresholds below to customize anomaly detection parameters
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <style>
        .stSlider {
            padding: 16px;
            background-color: var(--card);
            border-radius: 10px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            border: 1px solid var(--border);
            margin-bottom: 16px;
        }
        .stSlider > div > div > div { background-color: var(--accent2); }
        </style>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div style='background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #E74C3C; text-align: center;'>❤️ Heart Rate Limits</h4>", unsafe_allow_html=True)
            hr_threshold = st.slider("Heart Rate Range (bpm)", 30, 200, (50, 100), help="Set the normal range for heart rate in beats per minute")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div style='background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #3498DB; text-align: center;'>😴 Sleep Duration Limits</h4>", unsafe_allow_html=True)
            sleep_threshold = st.slider("Sleep Duration Range (hours)", 0, 24, (4, 10), help="Set the normal range for sleep duration in hours")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div style='background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #27AE60; text-align: center;'>👣 Step Count Limits</h4>", unsafe_allow_html=True)
            steps_threshold = st.slider("Step Count Range", 0, 25000, (1000, 20000), help="Set the normal range for daily step count")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Detect anomalies
        anomaly_configs = {
            'heart_rate': {
                'label': 'Heart Rate',
                'bounds': hr_threshold,
                'z_threshold': 2.5
            },
            'sleep_duration': {
                'label': 'Sleep Duration',
                'bounds': sleep_threshold,
                'z_threshold': 2.5
            },
            'step_count': {
                'label': 'Step Count',
                'bounds': steps_threshold,
                'z_threshold': 2.5
            }
        }
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Heart Rate", "Sleep Duration", "Step Count", 
            "Advanced Features", "Seasonal Modeling", "Clustering Analysis", "Reports"
        ])
        
        for (col, config), tab in zip(anomaly_configs.items(), [tab1, tab2, tab3]):
            if col in df.columns:
                with tab:
                    anomaly_result = detect_anomalies(df, col, custom_bounds=config['bounds'])
                    anomaly_mask = anomaly_result['anomalies'] if isinstance(anomaly_result, dict) else anomaly_result
                    anomaly_count = anomaly_mask.sum()
                    
                    st.markdown(f"### {config['label']} Analysis")
                    st.metric("Anomalies Detected", anomaly_count)
                    
                    # Create time series plot
                    fig = go.Figure()
                    
                    # Add normal data points
                    normal_data = df[~anomaly_mask]
                    fig.add_trace(go.Scatter(
                        x=normal_data['timestamp'],
                        y=normal_data[col],
                        mode='markers',
                        name='Normal',
                        marker=dict(color='blue', size=8)
                    ))
                    
                    # Add anomaly points
                    anomaly_data = df[anomaly_mask]
                    fig.add_trace(go.Scatter(
                        x=anomaly_data['timestamp'],
                        y=anomaly_data[col],
                        mode='markers',
                        name='Anomaly',
                        marker=dict(color='red', size=10)
                    ))
                    
                    # Add threshold lines
                    fig.add_hline(y=config['bounds'][0], line_dash="dash", line_color="#ef4444",
                                annotation_text=f"Lower Threshold: {config['bounds'][0]}")
                    fig.add_hline(y=config['bounds'][1], line_dash="dash", line_color="#ef4444",
                                annotation_text=f"Upper Threshold: {config['bounds'][1]}")

                    fig.update_layout(
                        title=f"{config['label']} Over Time",
                        xaxis_title="Timestamp",
                        yaxis_title=config['label'],
                        hovermode='x unified'
                    )
                    apply_plot_theme(fig)
                    
                    st.plotly_chart(fig, use_container_width=True)

                    # Optional Matplotlib view
                    with st.expander("Show Matplotlib Chart"):
                        fig_m, ax = plt.subplots()
                        ax.scatter(normal_data['timestamp'], normal_data[col], s=10, c='blue', label='Normal')
                        ax.scatter(anomaly_data['timestamp'], anomaly_data[col], s=15, c='red', label='Anomaly')
                        ax.axhline(config['bounds'][0], linestyle='--', color='red')
                        ax.axhline(config['bounds'][1], linestyle='--', color='red')
                        ax.set_title(f"{config['label']} Over Time (Matplotlib)")
                        ax.set_xlabel("Timestamp")
                        ax.set_ylabel(config['label'])
                        ax.legend()
                        st.pyplot(fig_m, use_container_width=True)
                    
                    if anomaly_count > 0:
                        st.markdown("### Anomalous Values")
                        anomaly_df = pd.DataFrame({
                            'Timestamp': anomaly_data['timestamp'],
                            'Value': anomaly_data[col]
                        })
                        st.dataframe(anomaly_df)
        
        # Advanced Features Tab
        with tab4:
            st.markdown("### 🔬 Advanced Feature Analysis")
            st.markdown("This section provides advanced statistical features extracted using TSFresh and other methods.")
            
            # Initialize enhanced pipeline
            pipeline = EnhancedFitnessPipeline()
            
            # Feature extraction options
            feature_method = st.selectbox(
                "Select Feature Extraction Method:",
                ["Simple", "TSFresh"],
                help="Simple: Basic statistical features, TSFresh: Comprehensive time series features"
            )
            
            if st.button("Extract Features", key="extract_features"):
                with st.spinner("Extracting features..."):
                    try:
                        method = 'simple' if feature_method == 'Simple' else 'tsfresh'
                        features = pipeline.extract_features(df, method=method)
                        
                        st.success(f"Successfully extracted {len(features.columns)} features!")
                        
                        # Display feature summary
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Features", len(features.columns))
                        with col2:
                            st.metric("Feature Method", feature_method)
                        
                        # Display features
                        st.markdown("#### Extracted Features")
                        st.dataframe(features, use_container_width=True)
                        
                        # Feature importance plot
                        if len(features.columns) > 1:
                            st.markdown("#### Feature Importance")
                            
                            # Since we have a single row of features, we'll use the absolute values as importance
                            # and normalize them for better visualization
                            feature_values = features.iloc[0].abs()
                            
                            # Normalize the values to 0-1 range for better visualization
                            if feature_values.max() > 0:
                                feature_importance = (feature_values / feature_values.max()).sort_values(ascending=False)
                            else:
                                feature_importance = feature_values.sort_values(ascending=False)
                            
                            # Create the plot
                            fig = px.bar(
                                x=feature_importance.values,
                                y=feature_importance.index,
                                orientation='h',
                                title="Feature Importance (Normalized Values)",
                                labels={'x': 'Importance Score', 'y': 'Features'},
                                color=feature_importance.values,
                                color_continuous_scale='viridis'
                            )
                            
                            # Update layout for better readability
                            fig.update_layout(
                                height=max(400, len(feature_importance) * 25),
                                showlegend=False,
                                xaxis_title="Importance Score (0-1)",
                                yaxis_title="Features"
                            )
                            
                            apply_plot_theme(fig)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Also show a summary table with original values
                            st.markdown("#### Feature Importance Summary")
                            importance_df = pd.DataFrame({
                                'Feature': feature_importance.index,
                                'Original_Value': features.iloc[0][feature_importance.index],
                                'Importance_Score': feature_importance.values,
                                'Rank': range(1, len(feature_importance) + 1)
                            })
                            st.dataframe(importance_df, use_container_width=True)
                            
                            # Show top 5 most important features
                            st.markdown("#### Top 5 Most Important Features")
                            top_features = importance_df.head(5)
                            for idx, row in top_features.iterrows():
                                st.markdown(f"**{row['Rank']}.** {row['Feature']}: {row['Original_Value']:.4f}")
                        
                    except Exception as e:
                        st.error(f"Feature extraction failed: {str(e)}")
        
        # Seasonal Modeling Tab
        with tab5:
            st.markdown("### 📈 Seasonal Pattern Modeling")
            st.markdown("This section uses Facebook Prophet to model seasonal patterns and make forecasts.")
            
            # Prophet modeling options
            col1, col2 = st.columns(2)
            with col1:
                forecast_periods = st.slider("Forecast Periods", 1, 30, 7, help="Number of periods to forecast")
            with col2:
                seasonality_mode = st.selectbox("Seasonality Mode", ["additive", "multiplicative"])
            
            if st.button("Run Prophet Analysis", key="prophet_analysis"):
                with st.spinner("Running Prophet analysis..."):
                    try:
                        # Initialize Prophet modeler
                        from prophet_modeling import ProphetFitnessModeler
                        modeler = ProphetFitnessModeler(seasonality_mode=seasonality_mode)
                        
                        # Model each metric
                        prophet_results = {}
                        for col in ['heart_rate', 'sleep_duration', 'step_count']:
                            if col in df.columns:
                                st.markdown(f"#### {col.replace('_', ' ').title()} Analysis")
                                
                                # Fit model
                                model = modeler.fit_model(df, col)
                                
                                # Make forecast
                                forecast = modeler.make_forecast(col, periods=forecast_periods)
                                
                                # Detect anomalies
                                anomalies = modeler.detect_anomalies_with_prophet(df, col)
                                
                                prophet_results[col] = {
                                    'model': model,
                                    'forecast': forecast,
                                    'anomalies': anomalies
                                }
                                
                                # Display results
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Prophet Anomalies", anomalies['anomaly'].sum())
                                with col_b:
                                    st.metric("Forecast Periods", forecast_periods)
                                
                                # Plot forecast
                                fig = modeler.plot_forecast(col)
                                apply_plot_theme(fig)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Plot components
                                fig_components = modeler.plot_components(col)
                                apply_plot_theme(fig_components)
                                st.plotly_chart(fig_components, use_container_width=True)
                        
                        st.success("Prophet analysis completed successfully!")
                        
                    except Exception as e:
                        st.error(f"Prophet analysis failed: {str(e)}")
        
        # Clustering Analysis Tab
        with tab6:
            st.markdown("### 🎯 Clustering Analysis")
            st.markdown("This section groups similar fitness behaviors using KMeans and DBSCAN clustering.")
            
            # Clustering options
            col1, col2 = st.columns(2)
            with col1:
                clustering_methods = st.multiselect(
                    "Select Clustering Methods:",
                    ["KMeans", "DBSCAN"],
                    default=["KMeans"],
                    help="Choose which clustering algorithms to use"
                )
            with col2:
                n_clusters = st.slider("Number of Clusters (KMeans)", 2, 10, 3, help="Number of clusters for KMeans")
            
            if st.button("Run Clustering Analysis", key="clustering_analysis"):
                with st.spinner("Running clustering analysis..."):
                    try:
                        # Initialize clustering analyzer
                        from clustering_analysis import FitnessClusteringAnalyzer
                        analyzer = FitnessClusteringAnalyzer()
                        
                        # Perform clustering
                        clustering_results = {}
                        for method in clustering_methods:
                            method_lower = method.lower()
                            st.markdown(f"#### {method} Results")
                            
                            if method_lower == 'kmeans':
                                result = analyzer.perform_kmeans_clustering(
                                    df, n_clusters=n_clusters
                                )
                            elif method_lower == 'dbscan':
                                result = analyzer.perform_dbscan_clustering(df)
                            
                            clustering_results[method_lower] = result
                            
                            # Display results
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Number of Clusters", result['n_clusters'])
                            with col_b:
                                st.metric("Silhouette Score", f"{result['silhouette_score']:.3f}")
                            with col_c:
                                if method_lower == 'dbscan':
                                    st.metric("Noise Points", result['n_noise'])
                                else:
                                    st.metric("Calinski-Harabasz Score", f"{result['calinski_harabasz_score']:.1f}")
                            
                            # Plot clusters
                            fig = analyzer.plot_clusters_2d(
                                result['results_df'], 
                                result['features_used'], 
                                method=method_lower
                            )
                            apply_plot_theme(fig)
                            st.plotly_chart(fig, use_container_width=True)
                            

                            
                            # Display cluster summary
                            st.markdown("#### Cluster Summary")
                            
                            # Common metrics for both methods
                            total_points = len(result['labels'])
                            unique_clusters = len(set(result['labels'])) - (1 if method_lower == 'dbscan' and -1 in result['labels'] else 0)
                            
                            if method_lower == 'dbscan':
                                # DBSCAN specific metrics
                                noise_mask = result['labels'] == -1
                                n_noise = noise_mask.sum()
                                non_noise_points = total_points - n_noise
                                
                                st.markdown("**DBSCAN Clustering Summary:**")
                                
                                # Core metrics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Points", total_points)
                                    st.metric("Number of Clusters", unique_clusters)
                                with col2:
                                    st.metric("Clustered Points", non_noise_points)
                                    st.metric("Clustering Rate", f"{(non_noise_points/total_points)*100:.1f}%")
                                with col3:
                                    st.metric("Noise Points", n_noise)
                                    st.metric("Noise Rate", f"{(n_noise/total_points)*100:.1f}%")
                            else:
                                # KMeans specific metrics
                                st.markdown("**KMeans Clustering Summary:**")
                                
                                # Calculate average distance to centroids
                                distances = []
                                # Use only the numeric features that were used in clustering
                                numeric_features = []
                                for feat in result['features_used']:
                                    # Check if the feature contains only numeric values
                                    if pd.api.types.is_numeric_dtype(result['results_df'][feat]):
                                        numeric_features.append(feat)
                                
                                for i in range(unique_clusters):
                                    cluster_points = result['results_df'][result['results_df']['cluster'] == i]
                                    if not cluster_points.empty:
                                        # Use only numeric features
                                        cluster_data = cluster_points[numeric_features].values
                                        # Get corresponding centroid values for numeric features
                                        centroid = result['centers'][i][:len(numeric_features)]
                                        
                                        # Reshape centroid to match the number of features
                                        centroid = centroid.reshape(1, -1)
                                        # Calculate distances using broadcasting
                                        cluster_distances = np.sqrt(np.sum((cluster_data - centroid) ** 2, axis=1))
                                        distances.append(np.mean(cluster_distances))
                                
                                avg_distance = np.mean(distances) if distances else 0
                                
                                # Core metrics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Points", total_points)
                                    st.metric("Number of Clusters", unique_clusters)
                                with col2:
                                    st.metric("Avg Points per Cluster", f"{total_points // unique_clusters}")
                                    st.metric("Silhouette Score", f"{result['silhouette_score']:.3f}")
                                with col3:
                                    st.metric("Calinski-Harabasz Score", f"{result['calinski_harabasz_score']:.1f}")
                                    st.metric("Avg Distance to Centroid", f"{avg_distance:.2f}")
                            
                            # Create cluster size distribution plot
                            cluster_sizes = pd.Series(result['labels']).value_counts()
                            fig_dist = px.bar(
                                x=cluster_sizes.index.astype(str),
                                y=cluster_sizes.values,
                                title="Cluster Size Distribution",
                                labels={'x': 'Cluster ID' + (' (-1 = Noise)' if method_lower == 'dbscan' else ''),
                                       'y': 'Number of Points'},
                                color=cluster_sizes.values,
                                color_continuous_scale='Viridis'
                            )
                            apply_plot_theme(fig_dist)
                            st.plotly_chart(fig_dist, use_container_width=True)
                            
                            # --- UI Enhancement: Section Title & Guide ---
                            st.markdown("<hr>", unsafe_allow_html=True)
                            st.markdown("<h2 style='color: #2ec4b6; margin-bottom: 0;'>📈 Feature Distribution Across Clusters</h2>", unsafe_allow_html=True)
                            st.markdown(
                                "<p style='color: #bcbcbc; font-size: 1.1em; margin-bottom: 1.25em;'>Visualize how your selected feature distributes across clusters. Use the controls to choose a feature and customize the plot.</p>",
                                unsafe_allow_html=True
                            )

                            with st.container():
                                st.markdown("<div style='padding: 18px; background: rgba(46,196,182,0.07); border-radius: 10px;'><h4 style='margin-bottom:10px;'>🔧 Choose Feature and Plot Type</h4>", unsafe_allow_html=True)
                                colA, colB, colC = st.columns([1.1, 2, 1.2])
                                with colA:
                                    # Get all features except cluster and timestamp
                                    all_features = [col for col in result['results_df'].columns 
                                                if col not in ['cluster', 'timestamp']]
                                    
                                    # Separate numeric and categorical features
                                    numeric_features = [col for col in all_features 
                                                    if pd.api.types.is_numeric_dtype(result['results_df'][col])]
                                    categorical_features = [col for col in all_features 
                                                        if not pd.api.types.is_numeric_dtype(result['results_df'][col])]
                                    
                                    feature_type = st.radio(
                                        "Feature Type 🏷️",
                                        ["Numeric", "Categorical"] if categorical_features else ["Numeric"],
                                        key=f"feature_type_{method_lower}",
                                        help="Choose whether to explore numeric or categorical features."
                                    )
                                with colB:
                                    feature_list = numeric_features if feature_type == "Numeric" else categorical_features
                                    if feature_list:
                                        selected_feature = st.selectbox(
                                            "Select Feature",
                                            feature_list,
                                            key=f"feature_select_{method_lower}",
                                            help="Pick the specific feature you wish to visualize."
                                        )
                                    else:
                                        st.info(f"No {feature_type.lower()} features available in your data.")
                                        selected_feature = None
                                with colC:
                                    if feature_type == "Numeric":
                                        plot_type = st.radio(
                                            "Plot Type 📊",
                                            ["Box Plot", "Violin Plot"],
                                            key=f"plot_type_{method_lower}",
                                            help="Box Plot and Violin Plot are available for numeric features."
                                        )
                                    else:
                                        plot_type = st.radio(
                                            "Plot Type 📊",
                                            ["Bar Plot (Count)",],
                                            key=f"plot_type_{method_lower}",
                                            help="Bar/Count plot is available for categorical features."
                                        )
                                st.markdown("</div>", unsafe_allow_html=True)

                            st.markdown('<br>', unsafe_allow_html=True)
                            st.markdown('---')

                            # --- Dynamic Plot Subheader ---
                            plot_heading = None
                            if selected_feature:
                                if feature_type == "Numeric":
                                    if plot_type == "Box Plot":
                                        plot_heading = f"<h3 style='font-weight:bold;'>📦 Box Plot of <span style='color:#2ec4b6'>{selected_feature}</span> by Cluster</h3>"
                                    else:
                                        plot_heading = f"<h3 style='font-weight:bold;'>🎻 Violin Plot of <span style='color:#2ec4b6'>{selected_feature}</span> by Cluster</h3>"
                                else:
                                    plot_heading = f"<h3 style='font-weight:bold;'>📊 Bar Plot of <span style='color:#2ec4b6'>{selected_feature}</span> Category Counts by Cluster</h3>"
                            if plot_heading:
                                st.markdown(plot_heading, unsafe_allow_html=True)
                                st.caption(f"Showing {plot_type} for feature '{selected_feature}', grouped by cluster assignment.")

                            with st.container():
                                plot_data = result['results_df'].copy()
                                plot_data['cluster'] = plot_data['cluster'].astype(str)

                                fig_dist = None
                                if selected_feature:  # Only attempt if something is selected
                                    if feature_type == "Numeric":
                                        if plot_type == "Box Plot":
                                            fig_dist = px.box(
                                                plot_data,
                                                x='cluster',
                                                y=selected_feature,
                                                title=f"{selected_feature} Distribution by Cluster",
                                                labels={'cluster': 'Cluster ID' + (' (-1 = Noise)' if method_lower == 'dbscan' else ''),
                                                       'y': selected_feature},
                                                category_orders={'cluster': sorted(plot_data['cluster'].unique())}
                                            )
                                        else:  # Violin Plot
                                            fig_dist = go.Figure()
                                            for cluster in sorted(plot_data['cluster'].unique()):
                                                cluster_data = plot_data[plot_data['cluster'] == cluster][selected_feature]
                                                fig_dist.add_trace(go.Violin(
                                                    x=[cluster] * len(cluster_data),
                                                    y=cluster_data,
                                                    name=f"Cluster {cluster}",
                                                    box_visible=True,
                                                    meanline_visible=True,
                                                    points="all"
                                                ))
                                            fig_dist.update_layout(
                                                title=f"{selected_feature} Distribution by Cluster",
                                                xaxis_title='Cluster ID' + (' (-1 = Noise)' if method_lower == 'dbscan' else ''),
                                                yaxis_title=selected_feature,
                                                showlegend=False,
                                                title_font_size=22,
                                                title_font_color="#232323"
                                            )
                                    else:  # Categorical
                                        if plot_type == "Bar Plot (Count)":
                                            count_df = plot_data.groupby(['cluster', selected_feature]).size().reset_index(name='count')
                                            fig_dist = px.bar(
                                                count_df,
                                                x='cluster',
                                                y='count',
                                                color=selected_feature,
                                                barmode='group',
                                                title=f"{selected_feature} Count by Cluster",
                                                category_orders={'cluster': sorted(plot_data['cluster'].unique())},
                                                labels={'count': 'Count', 'cluster': 'Cluster ID' + (' (-1 = Noise)' if method_lower == 'dbscan' else '')},
                                                color_discrete_sequence=px.colors.qualitative.Set2
                                            )
                                            fig_dist.update_layout(
                                                title_font_size=22,
                                                title_font_color="#232323"
                                            )
                                        else:
                                            st.info("Only count/bar plots are supported for categorical features.")
                                            fig_dist = None
                                if fig_dist is not None:
                                    apply_plot_theme(fig_dist)
                                    st.plotly_chart(fig_dist, use_container_width=True)
                            
                            # Feature correlations within clusters
                            st.markdown("**Feature Correlations Within Clusters:**")
                            selected_cluster = st.selectbox(
                                "Select Cluster:",
                                sorted(result['results_df']['cluster'].unique()),
                                key=f"cluster_select_{method_lower}"
                            )
                            
                            # Get numeric features only
                            all_features = [col for col in result['results_df'].columns 
                                          if col not in ['cluster', 'timestamp']]
                            numeric_features = [col for col in all_features 
                                             if pd.api.types.is_numeric_dtype(result['results_df'][col])]
                            
                            if numeric_features:  # Only show correlation if we have numeric features
                                cluster_data = result['results_df'][result['results_df']['cluster'] == selected_cluster]
                                correlation_matrix = cluster_data[numeric_features].corr()
                                
                                fig_corr = px.imshow(
                                    correlation_matrix,
                                    title=f"Feature Correlations in Cluster {selected_cluster}",
                                    color_continuous_scale="RdBu",
                                    aspect="auto"
                                )
                            apply_plot_theme(fig_corr)
                            st.plotly_chart(fig_corr, use_container_width=True)
                            
                            # Show detailed cluster statistics
                            st.markdown("**Detailed Cluster Statistics:**")
                            st.dataframe(
                                result['cluster_stats'],
                                use_container_width=True,
                                height=400
                            )
                        
                        st.success("Clustering analysis completed successfully!")
                        
                    except Exception as e:
                        st.error(f"Clustering analysis failed: {str(e)}")
        
        # Distribution plots
        st.subheader("Data Distribution Analysis")
        col1, col2, col3 = st.columns(3)
        
        for (col, config), plot_col in zip(anomaly_configs.items(), [col1, col2, col3]):
            if col in df.columns:
                with plot_col:
                    fig = px.histogram(df, x=col, title=f"{config['label']} Distribution")
                    fig.add_vline(x=config['bounds'][0], line_dash="dash", line_color="red")
                    fig.add_vline(x=config['bounds'][1], line_dash="dash", line_color="red")
                    apply_plot_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)

        # Reports tab
        with tab7:
            st.markdown("### 📄 Reports & Exports")

            # Run comprehensive analysis to aggregate results
            pipeline = EnhancedFitnessPipeline()
            analysis_results = pipeline.run_comprehensive_analysis(df, anomaly_configs=anomaly_configs)

            # Build anomaly summary dataframe
            summary_df = build_anomaly_summary_dataframe(analysis_results)
            st.dataframe(summary_df, use_container_width=True)

            # CSV downloads
            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button(
                    label="Download Anomaly Summary (CSV)",
                    data=dataframe_to_csv_bytes(summary_df),
                    file_name="anomaly_summary.csv",
                    mime="text/csv"
                )
            with col_b:
                # Lazy import PDF builder and handle missing dependency
                try:
                    from reporting import build_pdf_report
                    overview = {
                        "total_records": analysis_results["data_quality"]["total_records"],
                        "quality_score": analysis_results["data_quality"]["quality_score"],
                        "features_extracted": len(analysis_results.get("feature_extraction", {}).get("simple", pd.DataFrame()).columns) if analysis_results.get("feature_extraction", {}).get("simple") is not None else 0,
                    }
                    pdf_bytes = build_pdf_report(
                        title="FitPulse Anomaly Report",
                        overview=overview,
                        anomaly_summary_df=summary_df
                    )
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name="fitpulse_anomaly_report.pdf",
                        mime="application/pdf"
                    )
                except ImportError:
                    st.info("PDF export requires the 'reportlab' package. Install with: pip install reportlab")
        
    except Exception as e:
        st.error(f"Error analyzing file: {str(e)}")
else:
    st.info("Please upload a file to begin analysis")