"""
Customer Churn Prediction Pipeline
Tech Stack: Python | Pandas | NumPy | Matplotlib | Scikit-learn
Target Audience: AI Trainer Role / Fresher Machine Learning Engineer

This script implements an end-to-end classification pipeline for predicting customer churn.
Workflow:
  1. Load & Inspect Data (Pandas)
  2. Data Cleaning & Handling Missing Values
  3. Feature Engineering & NumPy Numerical Operations
  4. Feature Preprocessing (Encoding & Scaling)
  5. Train/Test Split
  6. Model Training (Logistic Regression)
  7. Model Evaluation (Accuracy, Precision, Recall, F1-Score)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

class ChurnPipeline:
    def __init__(self, data_path="data/customer_churn.csv", random_state=42):
        self.data_path = data_path
        self.random_state = random_state
        self.raw_df = None
        self.clean_df = None
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        self.feature_names = []
        self.feature_columns = []
        self.numeric_cols = ["TenureMonths", "MonthlyCharges", "TotalCharges"]
        self.categorical_cols = [
            "Gender", "SeniorCitizen", "Partner", "Dependents",
            "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "TechSupport", "Contract",
            "PaperlessBilling", "PaymentMethod"
        ]

    def load_data(self):
        """Loads data from CSV file and performs preliminary inspection."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at '{self.data_path}'. Please run generate_data.py first.")
        
        self.raw_df = pd.read_csv(self.data_path)
        print("=" * 60)
        print("STEP 1: DATA LOADING & INSPECTION")
        print("=" * 60)
        print(f"Dataset Shape: {self.raw_df.shape[0]} rows, {self.raw_df.shape[1]} columns")
        print("\nFirst 5 rows:")
        print(self.raw_df.head(5))
        return self.raw_df

    def clean_data(self, df=None):
        """
        Cleans dataset by handling missing values, coercion of types,
        and binary encoding of target variable.
        """
        if df is None:
            df = self.raw_df.copy()
        else:
            df = df.copy()

        print("\n" + "=" * 60)
        print("STEP 2: DATA CLEANING & IMPUTATION")
        print("=" * 60)

        # Clean TotalCharges: convert whitespace string ' ' to NaN and coerce to float
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
            
            # Fill missing TotalCharges with median or estimated (MonthlyCharges * Tenure)
            missing_count = df["TotalCharges"].isna().sum()
            print(f"Found {missing_count} missing/invalid values in 'TotalCharges'.")
            if missing_count > 0:
                estimated_total = df["MonthlyCharges"] * df["TenureMonths"]
                df["TotalCharges"] = df["TotalCharges"].fillna(estimated_total)
                print("Imputed missing 'TotalCharges' using (MonthlyCharges * TenureMonths).")

        # Encode target variable Churn (Yes -> 1, No -> 0)
        if "Churn" in df.columns:
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0, 1: 1, 0: 0})

        # Drop ID column if present
        if "CustomerID" in df.columns:
            df = df.drop(columns=["CustomerID"])

        self.clean_df = df
        print(f"Cleaned Dataset Shape: {self.clean_df.shape}")
        return self.clean_df

    def numpy_feature_operations(self, df=None):
        """
        Demonstrates NumPy array operations for numerical feature analysis & transformations.
        """
        if df is None:
            df = self.clean_df

        print("\n" + "=" * 60)
        print("STEP 3: NUMPY NUMERICAL OPERATIONS & FEATURE ENGINEERING")
        print("=" * 60)

        # Convert columns to NumPy arrays
        tenure_arr = df["TenureMonths"].to_numpy()
        monthly_arr = df["MonthlyCharges"].to_numpy()
        total_arr = df["TotalCharges"].to_numpy()

        # Compute summary statistics using NumPy
        print(f"Mean Tenure (Months): {np.mean(tenure_arr):.2f} standard deviation: {np.std(tenure_arr):.2f}")
        print(f"Median Monthly Charge: ${np.median(monthly_arr):.2f}")
        print(f"Total Revenue across all customers: ${np.sum(total_arr):,.2f}")

        # NumPy feature engineering: Average charge per month active ratio
        avg_charge_ratio = np.where(tenure_arr > 0, total_arr / tenure_arr, monthly_arr)
        df = df.copy()
        df["AvgChargeRatio"] = np.round(avg_charge_ratio, 2)
        
        # Log-transform TotalCharges using NumPy to normalize right-skewed distribution
        df["LogTotalCharges"] = np.log1p(total_arr)

        return df

    def preprocess_features(self, df=None):
        """
        Preprocesses features via One-Hot Encoding for categorical features
        and StandardScaler for numerical features.
        """
        if df is None:
            df = self.clean_df

        print("\n" + "=" * 60)
        print("STEP 4: FEATURE PREPROCESSING (ENCODING & SCALING)")
        print("=" * 60)

        y = df["Churn"].values
        X_raw = df.drop(columns=["Churn"])

        # Identify categorical and numeric columns present
        num_cols = [col for col in ["TenureMonths", "MonthlyCharges", "TotalCharges", "AvgChargeRatio", "LogTotalCharges"] if col in X_raw.columns]
        cat_cols = [col for col in self.categorical_cols if col in X_raw.columns]

        # One-Hot Encoding using pandas get_dummies
        X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)
        self.feature_columns = list(X_encoded.columns)

        print(f"Features count after One-Hot Encoding: {X_encoded.shape[1]}")

        return X_encoded, y, num_cols

    def train_and_evaluate(self, test_size=0.20):
        """
        Executes complete pipeline: Train/Test Split -> Model Fit -> Metric Evaluation.
        """
        # Execute workflow steps
        self.load_data()
        cleaned_df = self.clean_data()
        engineered_df = self.numpy_feature_operations(cleaned_df)
        X_encoded, y, num_cols = self.preprocess_features(engineered_df)

        print("\n" + "=" * 60)
        print("STEP 5: TRAIN / TEST SPLIT")
        print("=" * 60)

        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        print(f"Training set: {X_train.shape[0]} samples | Testing set: {X_test.shape[0]} samples")

        # Fit StandardScaler on numeric columns in training set
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[num_cols] = self.scaler.fit_transform(X_train[num_cols])
        X_test_scaled[num_cols] = self.scaler.transform(X_test[num_cols])

        print("\n" + "=" * 60)
        print("STEP 6: LOGISTIC REGRESSION MODEL TRAINING")
        print("=" * 60)

        self.model.fit(X_train_scaled, y_train)
        print("Logistic Regression model successfully fitted!")

        print("\n" + "=" * 60)
        print("STEP 7: MODEL EVALUATION")
        print("=" * 60)

        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-Score : {f1:.4f}")

        print("\nConfusion Matrix:")
        print(pd.DataFrame(cm, index=["Actual Retained (0)", "Actual Churned (1)"],
                           columns=["Pred Retained (0)", "Pred Churned (1)"]))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

        metrics = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm.tolist()
        }

        return {
            "model": self.model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "num_cols": num_cols,
            "metrics": metrics,
            "X_test": X_test_scaled,
            "y_test": y_test,
            "y_pred": y_pred
        }

if __name__ == "__main__":
    pipeline = ChurnPipeline()
    results = pipeline.train_and_evaluate()
