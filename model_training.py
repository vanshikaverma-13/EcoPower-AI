"""
Model Training & Benchmarking Module for EcoPower AI
Trains and evaluates Linear Regression, Random Forest, and Gradient Boosting Regressors
using a strict time-series split to prevent data leakage.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import get_feature_matrix, FEATURE_COLUMNS

MODELS_DIR = r"C:\Users\ASUS\Desktop\EcoPower-AI\models"

def train_and_evaluate_models(data_path):
    """
    Orchestrates full ML pipeline:
    1. Ingestion & cleaning
    2. Feature engineering
    3. Chronological 80/20 train/test split
    4. Model benchmarking
    5. Best model serialization
    """
    print(f"Loading data from {data_path}...")
    raw_df = load_data(data_path)
    cleaned_df = clean_data(raw_df)
    
    X, y = get_feature_matrix(cleaned_df)
    
    # Chronological Split (80% train, 20% test) - zero time leakage
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train size: {len(X_train)} samples, Test size: {len(X_test)} samples")
    
    # Fit Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Candidates
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42)
    }
    
    results = {}
    fitted_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        # Linear regression uses scaled, tree-based models can use raw or scaled
        if name == "Linear Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        
        results[name] = {
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "R2": round(r2, 4)
        }
        fitted_models[name] = model
        print(f"{name} -> MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.4f}")
        
    # Best model selection by lowest RMSE
    best_model_name = min(results, key=lambda k: results[k]["RMSE"])
    best_model = fitted_models[best_model_name]
    print(f"\n--> Best Model Selected: {best_model_name} (Lowest RMSE: {results[best_model_name]['RMSE']})")
    
    # Feature Importances (if applicable)
    feature_importance = {}
    if hasattr(best_model, "feature_importances_"):
        for f, imp in zip(FEATURE_COLUMNS, best_model.feature_importances_):
            feature_importance[f] = round(float(imp), 4)
        # Sort descending
        feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))
        
    # Save artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    
    joblib.dump(best_model, best_model_path)
    joblib.dump(scaler, scaler_path)
    
    metadata = {
        "best_model_name": best_model_name,
        "metrics_comparison": results,
        "best_metrics": results[best_model_name],
        "feature_names": FEATURE_COLUMNS,
        "feature_importance": feature_importance,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "uses_scaler": best_model_name == "Linear Regression"
    }
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Artifacts successfully saved to {MODELS_DIR}")
    return metadata

if __name__ == "__main__":
    data_file = r"C:\Users\ASUS\Desktop\EcoPower-AI\data\energy_consumption.csv"
    train_and_evaluate_models(data_file)
