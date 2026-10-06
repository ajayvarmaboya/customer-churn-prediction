# Customer Churn Prediction Project (Fresher / AI Trainer Level)

**Tech Stack**: Python | Pandas | NumPy | Matplotlib | Scikit-learn

## Overview
This repository contains a complete, beginner-friendly machine learning pipeline for predicting customer churn in a telecommunications dataset. The project demonstrates core data manipulation, exploratory data analysis (EDA), statistical operations, classification modeling (Logistic Regression), evaluation metrics, and unit testing.

## Workflow
```
Customer Churn Dataset (CSV)
        ↓
Pandas — Load & inspect
        ↓
Data Cleaning & Missing Value Imputation
        ↓
EDA — Matplotlib Visualizations
        ↓
NumPy — Numerical Operations
        ↓
Feature Preprocessing (Encoding & Scaling)
        ↓
Train/Test Split
        ↓
Logistic Regression Classifier
        ↓
Model Evaluation (Accuracy, Precision, Recall, F1)
```

## Folder Structure
```
customer_churn_prediction/
├── data/
│   └── customer_churn.csv        # Generated 1,000 customer dataset
├── eda_plots/                    # Matplotlib visual outputs
│   ├── churn_distribution.png
│   ├── tenure_vs_churn.png
│   └── correlation_matrix.png
├── generate_data.py              # Synthetic dataset generator
├── churn_pipeline.py             # Main end-to-end ML pipeline
├── visualize_eda.py              # Matplotlib EDA plotting script
├── predict.py                    # Inference script for individual customer prediction
├── test_pipeline.py              # Automated test suite (6 test scenarios)
├── requirements.txt              # Dependency requirements
├── README.md                     # Project overview
├── PROJECT_REPORT.md             # In-depth technical report
├── TESTING_REPORT.md            # Comprehensive test execution & scenario log
└── INTERVIEW_PREP.md             # AI Trainer role interview questions & answers
```

## Quick Start Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Synthetic Dataset**:
   ```bash
   python generate_data.py
   ```

3. **Run ML Training Pipeline**:
   ```bash
   python churn_pipeline.py
   ```

4. **Generate EDA Charts**:
   ```bash
   python visualize_eda.py
   ```

5. **Test Predictions on Sample Customer**:
   ```bash
   python predict.py
   ```

6. **Execute Test Suite**:
   ```bash
   python -m unittest test_pipeline.py -v
   ```
