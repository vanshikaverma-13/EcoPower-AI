"""
Feature Engineering Module for EcoPower AI
Constructs physics-informed energy metrics and cyclical temporal encodings.
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    'temperature_c', 'humidity_pct', 'occupancy_estimate',
    'is_weekend', 'is_business_hours', 'peak_hours_flag',
    'cdd_cooling_load', 'hdd_heating_load',
    'sin_hour', 'cos_hour', 'sin_month', 'cos_month',
    'heat_index_proxy'
]

def engineer_features(df):
    """
    Transforms clean dataframe into feature matrix for ML modeling/inference.
    """
    df = df.copy()
    
    # 1. Cyclical encodings
    # Hour cycle (24h period)
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    
    # Month cycle (12m period)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    # 2. Cooling & Heating Degree Proxy
    # Base cooling setpoint: 24.0°C (BEE national recommendation)
    df['cdd_cooling_load'] = np.maximum(0.0, df['temperature_c'] - 24.0)
    # Base heating threshold: 15.0°C
    df['hdd_heating_load'] = np.maximum(0.0, 15.0 - df['temperature_c'])
    
    # 3. Heat Index / Thermal Comfort Proxy
    df['heat_index_proxy'] = df['temperature_c'] + 0.05 * df['humidity_pct']
    
    # Ensure all required features are present
    for col in ['is_weekend', 'is_business_hours', 'peak_hours_flag']:
        if col not in df.columns:
            if col == 'is_weekend':
                df[col] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
            elif col == 'is_business_hours':
                df[col] = ((df['hour'] >= 8) & (df['hour'] <= 18) & (df['is_weekend'] == 0)).astype(int)
            elif col == 'peak_hours_flag':
                df[col] = ((df['hour'] >= 10) & (df['hour'] <= 16) & (df['is_weekend'] == 0)).astype(int)

    return df

def get_feature_matrix(df):
    """Returns engineered feature matrix X and target y (if present)."""
    df_feat = engineer_features(df)
    X = df_feat[FEATURE_COLUMNS]
    y = df_feat['energy_consumption_kwh'] if 'energy_consumption_kwh' in df_feat.columns else None
    return X, y
