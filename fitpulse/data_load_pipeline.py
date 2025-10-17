import pandas as pd
import numpy as np
from scipy import stats

def load_data(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .json")
    return df

def data_quality_report(df):
    total_records = len(df)
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    quality_score = round(100 * (1 - total_missing / (df.shape[0] * df.shape[1])), 2) if df.shape[0] > 0 else 0
    report = {
        "total_records": total_records,
        "shape": df.shape,
        "missing": missing_values.to_dict(),
        "missing_total": int(total_missing),
        "describe": df.describe(include='all').to_dict(),
        "quality_score": f"{quality_score}%"
    }
    return report

def detect_anomalies(df, column, threshold=3, custom_bounds=None):
    """
    Detect anomalies using Z-score method and custom bounds
    Returns a boolean mask where True indicates an anomaly
    """
    anomalies = np.zeros(len(df), dtype=bool)
    
    # Z-score based anomalies
    z_scores = np.abs(stats.zscore(df[column]))
    anomalies = z_scores > threshold
    
    # Custom bounds based anomalies
    if custom_bounds:
        lower, upper = custom_bounds
        bound_anomalies = (df[column] < lower) | (df[column] > upper)
        anomalies = anomalies | bound_anomalies
    
    return anomalies

def quality_check(df):
    errors = []
    # Handle missing columns gracefully
    for col in ['SleepHours', 'HeartRate', 'Steps']:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
            df[col] = pd.NA

    sleep_min, sleep_max = 4, 10
    hr_min, hr_max = 50, 100
    steps_min, steps_max = 1000, 20000

    sleep_err = ~df['SleepHours'].between(sleep_min, sleep_max)
    hr_err = ~df['HeartRate'].between(hr_min, hr_max)
    steps_err = ~df['Steps'].between(steps_min, steps_max)

    if sleep_err.any():
        errors.append("SleepHours out of range (4-10 hours).")
    if hr_err.any():
        errors.append("HeartRate out of range (50-100 bpm).")
    if steps_err.any():
        errors.append("Steps out of range (1000-20000).")

    error_rows = df[sleep_err | hr_err | steps_err]
    return errors, error_rows

def run_pipeline(file_path):
    df = load_data(file_path)
    report = data_quality_report(df)
    errors, error_rows = quality_check(df)

    print(f"\nData Quality Report for {file_path}:")
    print(f"Total Records: {report['total_records']}")
    print(f"Shape: {report['shape']}")
    print(f"Missing Values: {report['missing']}")
    print(f"Quality Score: {report['quality_score']}")

    # Define anomaly detection parameters for each metric
    anomaly_configs = {
        'heart_rate': {
            'label': 'Heart Rate',
            'bounds': (50, 100),  # Normal heart rate range
            'z_threshold': 2.5
        },
        'sleep_duration': {
            'label': 'Sleep Duration',
            'bounds': (4, 10),    # Normal sleep duration range (hours)
            'z_threshold': 2.5
        },
        'step_count': {
            'label': 'Step Count',
            'bounds': (1000, 20000),  # Normal daily steps range
            'z_threshold': 2.5
        }
    }

    print("\nAnomaly Detection Results:")
    for col, config in anomaly_configs.items():
        if col in df.columns:
            anomalies = detect_anomalies(
                df, 
                col, 
                threshold=config['z_threshold'],
                custom_bounds=config['bounds']
            )
            anomaly_count = anomalies.sum()
            
            print(f"\n{config['label']} Anomalies:")
            print(f"- Found {anomaly_count} anomalies")
            print(f"- Normal range: {config['bounds'][0]} to {config['bounds'][1]}")
            
            if anomaly_count > 0:
                print("- Anomalous values:")
                anomaly_data = df[anomalies]
                anomaly_report = pd.DataFrame({
                    'Timestamp': anomaly_data['timestamp'],
                    'Value': anomaly_data[col]
                })
                print(anomaly_report.to_string(index=False))

    if not errors:
        print(f"\n{file_path} - No quality errors found. Showing 5 data lines:\n")
        print(df.head())
    else:
        print(f"\n{file_path} - Data Quality Errors Detected:")
        for err in errors:
            print("-", err)
        print("\nRows with errors:\n", error_rows)

if __name__ == "__main__":
    run_pipeline('fitness_data.csv')
