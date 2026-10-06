"""
Customer Churn Prediction - Inference Module
Tech Stack: Python | Scikit-learn | Pandas

Provides single-customer and batch prediction capabilities using the trained pipeline.
"""

import pandas as pd
import numpy as np
from churn_pipeline import ChurnPipeline

class ChurnPredictor:
    def __init__(self, data_path="data/customer_churn.csv"):
        self.pipeline = ChurnPipeline(data_path=data_path)
        self.train_results = self.pipeline.train_and_evaluate()
        self.model = self.train_results["model"]
        self.scaler = self.train_results["scaler"]
        self.feature_columns = self.train_results["feature_columns"]
        self.num_cols = self.train_results["num_cols"]

    def predict_single(self, customer_dict):
        """
        Predicts churn risk for a single customer dictionary input.
        
        Input Example:
        {
            "Gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "TenureMonths": 2,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "TechSupport": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 85.50,
            "TotalCharges": 171.00
        }
        """
        df_single = pd.DataFrame([customer_dict])

        # Apply same data cleaning
        df_single["TotalCharges"] = pd.to_numeric(df_single["TotalCharges"].astype(str).str.strip(), errors="coerce")
        df_single["TotalCharges"] = df_single["TotalCharges"].fillna(df_single["MonthlyCharges"] * df_single["TenureMonths"])
        
        # NumPy feature engineering
        df_single["AvgChargeRatio"] = np.where(
            df_single["TenureMonths"] > 0,
            df_single["TotalCharges"] / df_single["TenureMonths"],
            df_single["MonthlyCharges"]
        )
        df_single["LogTotalCharges"] = np.log1p(df_single["TotalCharges"])

        # One-Hot Encoding
        cat_cols = [c for c in self.pipeline.categorical_cols if c in df_single.columns]
        df_encoded = pd.get_dummies(df_single, columns=cat_cols)

        # Align with training feature columns (fill missing dummy columns with 0)
        for col in self.feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        df_encoded = df_encoded[self.feature_columns]

        # Scale numeric features
        df_encoded[self.num_cols] = self.scaler.transform(df_encoded[self.num_cols])

        # Predict probability
        prob = self.model.predict_proba(df_encoded)[0, 1]
        prediction = "CHURN" if prob >= 0.5 else "RETAIN"

        return {
            "prediction": prediction,
            "churn_probability": round(float(prob), 4),
            "retention_probability": round(float(1 - prob), 4)
        }

if __name__ == "__main__":
    predictor = ChurnPredictor()
    
    sample_customer = {
        "Gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "TenureMonths": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.50,
        "TotalCharges": 179.00
    }
    
    result = predictor.predict_single(sample_customer)
    print("\n" + "=" * 50)
    print("SAMPLE CUSTOMER CHURN PREDICTION")
    print("=" * 50)
    print(f"Customer Input    : Month-to-month, Fiber Optic, 2 months tenure")
    print(f"Prediction        : {result['prediction']}")
    print(f"Churn Probability : {result['churn_probability'] * 100:.2f}%")
