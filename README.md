# FitPulse Anomaly Detection

## Overview
FitPulse Anomaly Detection is an interactive web application that helps users identify unusual patterns in their fitness data. Using advanced statistical analysis, the system can detect anomalies in various health metrics, providing valuable insights for both fitness enthusiasts and healthcare professionals.

The latest update focuses on a cleaner, high-contrast visual design with a black-and-blue theme, glowing highlight elements, and improved plotting controls for both numeric and categorical features.

## Features
- **Interactive Web Interface**: Built with Streamlit for an intuitive user experience
- **Flexible Data Input**: Support for both CSV and JSON file formats
- **Advanced Anomaly Detection**: Statistical analysis to identify outliers in fitness metrics
- **Real-time Visualization**: Interactive plots using Plotly for detailed data exploration
- **Black & Blue Theme (New)**: High-contrast black background with vibrant blue accents (#2563eb) across the entire app
- **Glowing Highlights (New)**:
  - Glowing blue summary metrics in the Data Overview (Total Records, Total Data Points, Quality Score, Missing Values)
  - Buttons with gradient/glow styling, including glowing blue Download buttons in Reports & Exports
- **Improved Feature Distribution (New)**:
  - Numeric features: Box Plot or Violin Plot by cluster
  - Categorical features: Bar (count) plot by cluster
- **User-friendly Dashboard**: Easy-to-use interface with informative sidebar
- **TSFresh Feature Extraction**: Advanced time series feature extraction for comprehensive analysis
- **Prophet Seasonal Modeling**: Facebook Prophet integration for seasonal pattern detection and forecasting
- **Clustering Analysis**: KMeans and DBSCAN clustering to group similar fitness behaviors
- **Enhanced Data Pipeline**: Comprehensive analysis combining multiple advanced techniques

## Installation

### Prerequisites
- Python 3.x
- pip (Python package manager)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/svrohit123/fitpulse_anomaly_detection.git
cd fitpulse_anomaly_detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

Or install individual packages:
```bash
pip install streamlit pandas plotly numpy scipy tsfresh prophet scikit-learn
```

## Usage

1. Navigate to the project directory:
```bash
cd fitpulse
```

2. Run the Streamlit application:
```bash
python -m streamlit run app.py
```

3. Open your web browser and go to:
```
http://localhost:8501
```

4. Upload your fitness data file (CSV or JSON format)
5. Explore the analysis results across multiple tabs:
   - **Heart Rate, Sleep Duration, Step Count**: Basic anomaly detection and visualization
   - **Advanced Features**: TSFresh feature extraction and analysis
   - **Seasonal Modeling**: Prophet-based seasonal pattern detection and forecasting
   - **Clustering Analysis**: KMeans and DBSCAN clustering for behavior grouping
   - **Reports & Exports (Updated)**: Glowing blue Download buttons to export summary and full data

### UI Notes (New)
- The app uses a black background with vibrant blue accents. Buttons and key metrics have a subtle glow for readability and emphasis.
- In Feature Distribution Across Clusters:
  - Choose feature type (Numeric or Categorical)
  - For Numeric: Box Plot or Violin Plot per cluster
  - For Categorical: Bar Plot (count) per cluster

## Data Format
The application accepts two types of files:

### CSV Format
Your CSV file should include columns for:
- Timestamp
- Heart rate
- Activity levels
- Other fitness metrics

### JSON Format
JSON files should follow the structure:
```json
{
    "timestamp": [...],
    "heart_rate": [...],
    "activity_level": [...],
    ...
}
```

## Technical Architecture
- **Frontend**: Streamlit
- **Data Processing**: Python (Pandas, NumPy)
- **Visualization**: Plotly
- **Statistical Analysis**: SciPy
- **Feature Extraction**: TSFresh
- **Seasonal Modeling**: Facebook Prophet
- **Clustering**: Scikit-learn (KMeans, DBSCAN)
- **Machine Learning**: Scikit-learn

### Theming and Styling (Updated)
- Global theme variables define colors for background, cards, text, borders, and accents.
- Accent color uses a vibrant blue (#2563eb) with a deeper blue (#1e3a8a) for secondary accents.
- Buttons and key metrics use a blue glow via CSS for enhanced emphasis.

## Project Structure
```
fitpulse_anomaly_detection/
├── fitpulse/
│   ├── app.py                    # Main Streamlit application
│   ├── data_load_pipeline.py     # Original data processing and analysis
│   ├── enhanced_data_pipeline.py # Enhanced pipeline with new features
│   ├── tsfresh_features.py       # TSFresh feature extraction module
│   ├── prophet_modeling.py       # Facebook Prophet modeling module
│   ├── clustering_analysis.py    # Clustering analysis module
│   ├── Create_csv.py             # CSV data creation utility
│   ├── Create_Json.py            # JSON data creation utility
│   ├── load_csv.py               # CSV data loader
│   └── load_json.py              # JSON data loader
├── requirements.txt              # Python dependencies
└── README.md
```

## Advanced Features

### TSFresh Feature Extraction
The application now includes advanced time series feature extraction using TSFresh:
- **Statistical Features**: Mean, standard deviation, skewness, kurtosis, and more
- **Trend Features**: Linear and polynomial trend analysis
- **Seasonal Features**: Seasonal decomposition and pattern detection
- **Custom Features**: User-defined feature extraction parameters

### Prophet Seasonal Modeling
Facebook Prophet integration for advanced time series analysis:
- **Seasonal Pattern Detection**: Automatic detection of yearly, weekly, and daily patterns
- **Forecasting**: Future value prediction with confidence intervals
- **Anomaly Detection**: Prophet-based anomaly detection using prediction intervals
- **Component Analysis**: Trend, seasonal, and holiday component visualization

### Clustering Analysis
Behavioral clustering using machine learning algorithms:
- **KMeans Clustering**: Groups similar fitness behaviors into clusters
- **DBSCAN Clustering**: Density-based clustering with noise detection
- **Feature Engineering**: Automatic feature creation for clustering
- **Visualization**: 2D cluster plots and characteristic analysis

### Enhanced Data Pipeline
Comprehensive analysis combining all advanced techniques:
- **Multi-method Anomaly Detection**: Z-score, IQR, and custom bounds
- **Feature Integration**: Seamless integration of all analysis methods
- **Performance Optimization**: Efficient processing for large datasets
- **Error Handling**: Robust error handling and user feedback

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Streamlit for the amazing web framework
- Plotly for interactive visualization capabilities
- The open-source community for various tools and libraries

## Contact
For any queries or suggestions, please open an issue in the GitHub repository.

---
Created and maintained by [svrohit123](https://github.com/svrohit123)
