"""
Inference Engine for EcoPower AI
Provides real-time single-instance and batch prediction with uncertainty estimation.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from src.feature_engineering import engineer_features, FEATURE_COLUMNS

MODELS_DIR = r"C:\Users\ASUS\Desktop\EcoPower-AI\models"

class EnergyPredictor:
    def __init__(self, models_dir=MODELS_DIR):
        self.model_path = os.path.join(models_dir, "best_model.pkl")
        self.scaler_path = os.path.join(models_dir, "scaler.pkl")
        self.metadata_path = os.path.join(models_dir, "model_metadata.json")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Run model_training.py first.")
            
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
        self.uses_scaler = self.metadata.get("uses_scaler", False)
        self.rmse = self.metadata["best_metrics"]["RMSE"]

    def predict_single(self, temperature_c, humidity_pct, hour, day_of_week, month,
                       is_weekend, occupancy_estimate):
        """
        Generates single prediction from feature dictionary with 95% confidence interval.
        """
        # Cast inputs
        hour = int(hour)
        day_of_week = int(day_of_week)
        month = int(month)
        is_weekend = int(is_weekend)
        temperature_c = float(temperature_c)
        humidity_pct = float(humidity_pct)
        occupancy_estimate = float(occupancy_estimate)

        # Derived indicators
        is_biz = 1 if (8 <= hour <= 18 and is_weekend == 0) else 0
        peak_flag = 1 if (10 <= hour <= 16 and is_weekend == 0) else 0
        
        row_df = pd.DataFrame([{
            'temperature_c': float(temperature_c),
            'humidity_pct': float(humidity_pct),
            'hour': int(hour),
            'day_of_week': int(day_of_week),
            'month': int(month),
            'is_weekend': int(is_weekend),
            'is_business_hours': int(is_biz),
            'peak_hours_flag': int(peak_flag),
            'occupancy_estimate': float(occupancy_estimate)
        }])
        
        feat_df = engineer_features(row_df)
        X = feat_df[FEATURE_COLUMNS]
        
        if self.uses_scaler:
            X_in = self.scaler.transform(X)
        else:
            X_in = X
            
        pred = float(self.model.predict(X_in)[0])
        pred = max(20.0, round(pred, 2))  # Physically valid non-negative load
        
        # 95% confidence interval bounds (~1.96 * RMSE)
        margin = round(1.96 * self.rmse, 2)
        lower_bound = max(15.0, round(pred - margin, 2))
        upper_bound = round(pred + margin, 2)
        
        return {
            "predicted_kwh": pred,
            "lower_bound_95": lower_bound,
            "upper_bound_95": upper_bound,
            "model_used": self.metadata["best_model_name"],
            "model_r2": self.metadata["best_metrics"]["R2"],
            "is_peak_hour": bool(peak_flag)
        }

    def predict_batch(self, df):
        """
        Performs batch inference on dataframe with validation.
        """
        df = df.copy()
        feat_df = engineer_features(df)
        X = feat_df[FEATURE_COLUMNS]
        
        if self.uses_scaler:
            X_in = self.scaler.transform(X)
        else:
            X_in = X
            
        preds = self.model.predict(X_in)
        preds = np.clip(preds, 20.0, None)
        df['predicted_energy_kwh'] = np.round(preds, 2)
        df['pred_lower_95'] = np.round(np.clip(preds - 1.96 * self.rmse, 15.0, None), 2)
        df['pred_upper_95'] = np.round(preds + 1.96 * self.rmse, 2)
        return df
