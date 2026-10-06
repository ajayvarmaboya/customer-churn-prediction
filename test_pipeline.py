"""
Automated Test Suite & Edge Case Scenarios for Customer Churn Prediction
Tech Stack: Python | Pytest | Pandas | Scikit-learn
"""

import os
import unittest
import numpy as np
import pandas as pd
from generate_data import generate_churn_dataset
from churn_pipeline import ChurnPipeline
from predict import ChurnPredictor

class TestCustomerChurnPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test directory and dataset before running tests."""
        cls.test_dir = "test_scratch"
        os.makedirs(cls.test_dir, exist_ok=True)
        cls.csv_path = os.path.join(cls.test_dir, "test_churn.csv")
        generate_churn_dataset(output_path=cls.csv_path, num_samples=200, random_seed=123)

    def test_1_dataset_generation(self):
        """Test dataset generator creates file with correct columns and non-empty rows."""
        self.assertTrue(os.path.exists(self.csv_path))
        df = pd.read_csv(self.csv_path)
        self.assertEqual(len(df), 200)
        self.assertIn("Churn", df.columns)
        self.assertIn("TotalCharges", df.columns)

    def test_2_data_cleaning_imputation(self):
        """Test data cleaning coerces whitespace string TotalCharges to numeric values."""
        pipeline = ChurnPipeline(data_path=self.csv_path)
        pipeline.load_data()
        clean_df = pipeline.clean_data()
        
        # Ensure TotalCharges contains no NaNs or non-numeric types
        self.assertFalse(clean_df["TotalCharges"].isna().any())
        self.assertTrue(pd.api.types.is_numeric_dtype(clean_df["TotalCharges"]))
        
        # Ensure target Churn is binary 0 or 1
        self.assertTrue(set(clean_df["Churn"].unique()).issubset({0, 1}))

    def test_3_numpy_feature_operations(self):
        """Test NumPy feature engineering functions handle division by zero safely."""
        pipeline = ChurnPipeline(data_path=self.csv_path)
        pipeline.load_data()
        clean_df = pipeline.clean_data()
        
        # Force a tenure = 0 edge case
        clean_df.loc[0, "TenureMonths"] = 0
        engineered_df = pipeline.numpy_feature_operations(clean_df)
        
        self.assertIn("AvgChargeRatio", engineered_df.columns)
        self.assertIn("LogTotalCharges", engineered_df.columns)
        self.assertFalse(np.isinf(engineered_df["AvgChargeRatio"]).any())
        self.assertFalse(np.isnan(engineered_df["AvgChargeRatio"]).any())

    def test_4_train_and_evaluate_metrics(self):
        """Test ML pipeline training produces valid classification metric scores [0.0, 1.0]."""
        pipeline = ChurnPipeline(data_path=self.csv_path, random_state=42)
        results = pipeline.train_and_evaluate()
        
        metrics = results["metrics"]
        self.assertGreaterEqual(metrics["accuracy"], 0.50)
        self.assertLessEqual(metrics["accuracy"], 1.00)
        self.assertGreaterEqual(metrics["f1_score"], 0.00)
        self.assertLessEqual(metrics["f1_score"], 1.00)
        self.assertIn("confusion_matrix", metrics)

    def test_5_high_risk_customer_prediction(self):
        """Test inference engine correctly flags high-risk churn customer (short tenure, month-to-month, high charge)."""
        predictor = ChurnPredictor(data_path=self.csv_path)
        high_risk_customer = {
            "Gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "TenureMonths": 1,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "TechSupport": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.00,
            "TotalCharges": 95.00
        }
        res = predictor.predict_single(high_risk_customer)
        self.assertEqual(res["prediction"], "CHURN")
        self.assertGreater(res["churn_probability"], 0.50)

    def test_6_low_risk_customer_prediction(self):
        """Test inference engine correctly flags low-risk customer (long tenure, two-year contract, low charge)."""
        predictor = ChurnPredictor(data_path=self.csv_path)
        low_risk_customer = {
            "Gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "TenureMonths": 60,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "TechSupport": "Yes",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 45.00,
            "TotalCharges": 2700.00
        }
        res = predictor.predict_single(low_risk_customer)
        self.assertEqual(res["prediction"], "RETAIN")
        self.assertLess(res["churn_probability"], 0.50)

if __name__ == "__main__":
    unittest.main()
