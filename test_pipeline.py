"""
Comprehensive Test Suite for EcoPower AI
Covers:
1. Valid CSV loading & cleaning
2. Missing columns detection
3. Missing values & null imputation
4. Incorrect data types coercion
5. Empty dataset error handling
6. Valid sustainability RAG query
7. Out-of-scope / unrelated query guardrail
8. Low-information / insufficient retrieval guardrail
9. Energy prediction range & physical bounds
10. Invalid single prediction inputs
"""

import unittest
import os
import pandas as pd
import numpy as np

from src.data_preprocessing import load_data, validate_schema, clean_data
from src.feature_engineering import engineer_features, get_feature_matrix
from src.prediction import EnergyPredictor
from src.rag_pipeline import SustainabilityRAG

BASE_DIR = r"C:\Users\ASUS\Desktop\EcoPower-AI"
DATA_PATH = os.path.join(BASE_DIR, "data", "energy_consumption.csv")

class TestEcoPowerAIPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = EnergyPredictor()
        cls.rag = SustainabilityRAG()

    # 1. Valid CSV Test
    def test_01_valid_csv_loading(self):
        df = load_data(DATA_PATH)
        self.assertGreater(len(df), 8000)
        is_valid, missing = validate_schema(df, require_target=True)
        self.assertTrue(is_valid)
        self.assertEqual(len(missing), 0)

    # 2. Missing Columns Handling
    def test_02_missing_columns_validation(self):
        incomplete_df = pd.DataFrame({
            'datetime': ['2024-01-01 00:00:00'],
            'temperature_c': [22.5]
            # missing humidity, hour, etc.
        })
        is_valid, missing = validate_schema(incomplete_df)
        self.assertFalse(is_valid)
        self.assertIn('humidity_pct', missing)
        self.assertIn('occupancy_estimate', missing)

    # 3. Missing Values & Imputation
    def test_03_missing_values_imputation(self):
        dirty_df = pd.DataFrame({
            'datetime': ['2024-01-01 00:00:00', '2024-01-01 01:00:00', '2024-01-01 02:00:00'],
            'temperature_c': [25.0, np.nan, 27.0],
            'humidity_pct': [np.nan, 60.0, 65.0],
            'occupancy_estimate': [50.0, 55.0, np.nan],
            'hour': [0, 1, 2],
            'day_of_week': [0, 0, 0],
            'month': [1, 1, 1],
            'is_weekend': [0, 0, 0],
            'is_business_hours': [0, 0, 0],
            'peak_hours_flag': [0, 0, 0]
        })
        cleaned = clean_data(dirty_df)
        self.assertFalse(cleaned['temperature_c'].isna().any())
        self.assertFalse(cleaned['humidity_pct'].isna().any())
        self.assertFalse(cleaned['occupancy_estimate'].isna().any())

    # 4. Incorrect Data Types Coercion
    def test_04_incorrect_data_types(self):
        stringy_df = pd.DataFrame({
            'datetime': ['2024-06-01 12:00:00'],
            'temperature_c': ['38.5'],  # String instead of float
            'humidity_pct': ['45.2'],
            'occupancy_estimate': ['85'],
            'hour': ['12'],
            'day_of_week': ['5'],
            'month': ['6'],
            'is_weekend': ['1'],
            'is_business_hours': ['0'],
            'peak_hours_flag': ['0']
        })
        cleaned = clean_data(stringy_df)
        self.assertIsInstance(cleaned['temperature_c'].iloc[0], (float, np.floating))

    # 5. Empty Dataset Handling
    def test_05_empty_dataset_handling(self):
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            clean_data(empty_df)

    # 6. Valid Sustainability RAG Query
    def test_06_valid_rag_query(self):
        query = "What are the recommended BEE air conditioning temperature setpoints?"
        res = self.rag.generate_grounded_response(query)
        self.assertFalse(res['is_out_of_scope'])
        self.assertGreater(len(res['retrieved_sources']), 0)
        self.assertIn("24", res['answer'])
        self.assertGreater(res['confidence_score'], 0.4)

    # 7. Unrelated Query Guardrail
    def test_07_unrelated_query_guardrail(self):
        query = "Who won the ICC Cricket World Cup in 2023?"
        res = self.rag.generate_grounded_response(query)
        self.assertTrue(res['is_out_of_scope'])
        self.assertEqual(len(res['retrieved_sources']), 0)
        self.assertIn("Responsible AI", res['answer'])

    # 8. Insufficient Info / Edge Query
    def test_08_insufficient_info_query(self):
        query = "How to bake a chocolate cake using microwave?"
        res = self.rag.generate_grounded_response(query)
        self.assertTrue(res['is_out_of_scope'])

    # 9. Energy Prediction Physical Bounds
    def test_09_prediction_physical_bounds(self):
        res = self.predictor.predict_single(
            temperature_c=34.0,
            humidity_pct=50.0,
            hour=14,
            day_of_week=2,
            month=6,
            is_weekend=0,
            occupancy_estimate=85.0
        )
        pred = res['predicted_kwh']
        self.assertGreaterEqual(pred, 20.0)
        self.assertLess(pred, 500.0)
        self.assertLessEqual(res['lower_bound_95'], pred)
        self.assertGreaterEqual(res['upper_bound_95'], pred)
        self.assertTrue(res['is_peak_hour'])

    # 10. Invalid Input Bounds in Single Predictor
    def test_10_prediction_coercion(self):
        # Pass float strings or extreme temps
        res = self.predictor.predict_single(
            temperature_c="42.5",
            humidity_pct="60",
            hour="11",
            day_of_week="1",
            month="5",
            is_weekend="0",
            occupancy_estimate="90"
        )
        self.assertIsInstance(res['predicted_kwh'], float)
        self.assertGreater(res['predicted_kwh'], 50.0)

if __name__ == '__main__':
    unittest.main()
