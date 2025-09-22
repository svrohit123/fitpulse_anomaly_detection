import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_load_pipeline import load_data, detect_anomalies, data_quality_report

# Page config
st.set_page_config(page_title="FitPulse Anomaly Detection", page_icon="🏃", layout="wide")

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #00ccff;'>📊 FitPulse</h1>
        <h3 style='color: #00ccff;'>Anomaly Detection System</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='padding: 10px;'>
        <h4 style='color: #00ccff;'>🎯 Project Overview</h4>
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
        <h4 style='color: #00ccff;'>🔍 Features</h4>
        <ul>
            <li>Real-time data analysis</li>
            <li>Interactive visualizations</li>
            <li>Customizable thresholds</li>
            <li>CSV & JSON support</li>
            <li>Detailed anomaly reports</li>
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

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .sidebar h1 {
        margin-bottom: 0;
        font-size: 1.5em;
        color: #00ff00;
    }
    .sidebar h3 {
        margin-top: 0;
        font-size: 1.1em;
        font-weight: normal;
        color: #00ccff;
    }
    .sidebar h4 {
        color: #ffcc00;
    }
    .sidebar hr {
        margin: 15px 0;
        border-color: #444444;
    }
    .sidebar ul, .sidebar ol {
        padding-left: 20px;
        color: #ffffff;
    }
    .sidebar li {
        margin-bottom: 5px;
    }
    .sidebar .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 5px;
        margin-top: 10px;
    }
    .sidebar .streamlit-expanderContent {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    .css-1d391kg, .css-12oz5g7 {
        background-color: #2d2d2d;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0, 255, 0, 0.1);
        border: 1px solid #444444;
    }
    .metric-card {
        background-color: #2d2d2d;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease;
        border: 1px solid #444444;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0, 255, 0, 0.2);
    }
    h1 {
        color: #00ff00;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(0, 255, 0, 0.3);
    }
    h2, h3 {
        color: #00ccff;
        text-shadow: 0 0 8px rgba(0, 204, 255, 0.3);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #2d2d2d;
        border-radius: 5px;
        padding: 10px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
        color: #ffffff;
    }
    .stTabs [aria-selected="true"] {
        background-color: #006600;
        color: #ffffff;
        text-shadow: 0 0 8px rgba(0, 255, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1 style='color: #00ccff;'>🏃 FitPulse Anomaly Detection</h1>
    <p style='font-size: 1.2em; color: #00ccff;'>
        This application analyzes fitness data for anomalies in heart rate, sleep duration, and step count.<br>
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
            <h2 style='color: #2C3E50; text-align: center; margin-bottom: 20px;'>📊 Data Overview</h2>
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
            color = '#27AE60' if float(quality_score) > 90 else '#E67E22' if float(quality_score) > 70 else '#E74C3C'
            st.markdown(f"""
            <div class='metric-card'>
                <h4 style='color: #7F8C8D; margin: 0;'>Quality Score</h4>
                <h2 style='color: {color}; margin: 10px 0;'>{report['quality_score']}</h2>
                <p style='color: #95A5A6; margin: 0;'>Data quality assessment</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            missing_values = sum(report['missing'].values())
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
            padding: 20px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        .stSlider > div > div > div {
            background-color: #4CAF50;
        }
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
        tab1, tab2, tab3 = st.tabs(["Heart Rate", "Sleep Duration", "Step Count"])
        
        for (col, config), tab in zip(anomaly_configs.items(), [tab1, tab2, tab3]):
            if col in df.columns:
                with tab:
                    anomalies = detect_anomalies(df, col, custom_bounds=config['bounds'])
                    anomaly_count = anomalies.sum()
                    
                    st.markdown(f"### {config['label']} Analysis")
                    st.metric("Anomalies Detected", anomaly_count)
                    
                    # Create time series plot
                    fig = go.Figure()
                    
                    # Add normal data points
                    normal_data = df[~anomalies]
                    fig.add_trace(go.Scatter(
                        x=normal_data['timestamp'],
                        y=normal_data[col],
                        mode='markers',
                        name='Normal',
                        marker=dict(color='blue', size=8)
                    ))
                    
                    # Add anomaly points
                    anomaly_data = df[anomalies]
                    fig.add_trace(go.Scatter(
                        x=anomaly_data['timestamp'],
                        y=anomaly_data[col],
                        mode='markers',
                        name='Anomaly',
                        marker=dict(color='red', size=10)
                    ))
                    
                    # Add threshold lines
                    fig.add_hline(y=config['bounds'][0], line_dash="dash", line_color="red",
                                annotation_text=f"Lower Threshold: {config['bounds'][0]}")
                    fig.add_hline(y=config['bounds'][1], line_dash="dash", line_color="red",
                                annotation_text=f"Upper Threshold: {config['bounds'][1]}")
                    
                    fig.update_layout(
                        title=f"{config['label']} Over Time",
                        xaxis_title="Timestamp",
                        yaxis_title=config['label'],
                        hovermode='x unified',
                        paper_bgcolor='#1a1a1a',
                        plot_bgcolor='#2d2d2d',
                        font=dict(color='#ffffff'),
                        xaxis=dict(gridcolor='#444444', zerolinecolor='#444444'),
                        yaxis=dict(gridcolor='#444444', zerolinecolor='#444444')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if anomaly_count > 0:
                        st.markdown("### Anomalous Values")
                        anomaly_df = pd.DataFrame({
                            'Timestamp': anomaly_data['timestamp'],
                            'Value': anomaly_data[col]
                        })
                        st.dataframe(anomaly_df)
        
        # Distribution plots
        st.subheader("Data Distribution Analysis")
        col1, col2, col3 = st.columns(3)
        
        for (col, config), plot_col in zip(anomaly_configs.items(), [col1, col2, col3]):
            if col in df.columns:
                with plot_col:
                    fig = px.histogram(df, x=col, title=f"{config['label']} Distribution")
                    fig.add_vline(x=config['bounds'][0], line_dash="dash", line_color="red")
                    fig.add_vline(x=config['bounds'][1], line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error analyzing file: {str(e)}")
else:
    st.info("Please upload a file to begin analysis")