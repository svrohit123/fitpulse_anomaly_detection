# FitPulse Anomaly Detection

## Overview
FitPulse Anomaly Detection is an interactive web application that helps users identify unusual patterns in their fitness data. Using advanced statistical analysis, the system can detect anomalies in various health metrics, providing valuable insights for both fitness enthusiasts and healthcare professionals.

## Features
- **Interactive Web Interface**: Built with Streamlit for an intuitive user experience
- **Flexible Data Input**: Support for both CSV and JSON file formats
- **Advanced Anomaly Detection**: Statistical analysis to identify outliers in fitness metrics
- **Real-time Visualization**: Interactive plots using Plotly for detailed data exploration
- **Dark Theme**: High-contrast visual design for better readability
- **User-friendly Dashboard**: Easy-to-use interface with informative sidebar

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
pip install streamlit pandas plotly numpy scipy
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
5. View the analysis results and interactive visualizations

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

## Project Structure
```
fitpulse_anomaly_detection/
├── fitpulse/
│   ├── app.py                 # Main Streamlit application
│   ├── data_load_pipeline.py  # Data processing and analysis
│   ├── Create_csv.py          # CSV data creation utility
│   ├── Create_Json.py         # JSON data creation utility
│   ├── load_csv.py            # CSV data loader
│   └── load_json.py           # JSON data loader
└── README.md
```

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
