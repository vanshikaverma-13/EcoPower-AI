"""
Data Preprocessing Module for EcoPower AI
Handles dataset ingestion, schema validation, missing values, and date parsing.
"""

import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    'datetime', 'temperature_c', 'humidity_pct', 'hour',
    'day_of_week', 'month', 'is_weekend', 'is_business_hours',
    'occupancy_estimate'
]

def load_data(filepath):
    """Load CSV dataset from file path."""
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        raise ValueError(f"Error reading file at {filepath}: {str(e)}")

def validate_schema(df, require_target=False):
    """
    Validates that the dataframe contains required columns.
    Returns (is_valid: bool, missing_columns: list)
    """
    required = list(REQUIRED_COLUMNS)
    if require_target:
        required.append('energy_consumption_kwh')
    
    missing = [col for col in required if col not in df.columns]
    return len(missing) == 0, missing

def clean_data(df):
    """
    Cleans dataset by converting datetimes, imputing missing values, and removing duplicates.
    """
    df = df.copy()
    
    # 1. Check empty
    if df.empty:
        raise ValueError("Dataframe is empty.")

    # 2. Datetime conversion
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        # Drop rows where datetime completely failed to parse
        df = df.dropna(subset=['datetime']).copy()
        df = df.sort_values('datetime').reset_index(drop=True)
    
    # 3. Deduplicate
    df = df.drop_duplicates().reset_index(drop=True)
    
    # 4. Numeric conversions and null handling
    numeric_cols = ['temperature_c', 'humidity_pct', 'occupancy_estimate']
    if 'energy_consumption_kwh' in df.columns:
        numeric_cols.append('energy_consumption_kwh')
        
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Forward fill then median fill fallback
            df[col] = df[col].ffill().bfill()
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
                
    # 5. Extract temporal features if missing
    if 'datetime' in df.columns:
        if 'hour' not in df.columns:
            df['hour'] = df['datetime'].dt.hour
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df['datetime'].dt.dayofweek
        if 'month' not in df.columns:
            df['month'] = df['datetime'].dt.month
        if 'is_weekend' not in df.columns:
            df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        if 'is_business_hours' not in df.columns:
            df['is_business_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 18) & (df['is_weekend'] == 0)).astype(int)
        if 'peak_hours_flag' not in df.columns:
            df['peak_hours_flag'] = ((df['hour'] >= 10) & (df['hour'] <= 16) & (df['is_weekend'] == 0)).astype(int)

    return df
