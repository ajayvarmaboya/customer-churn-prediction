# Testing & Validation Report: Customer Churn Prediction

**Target Role**: AI Trainer / Quality Assurance ML Engineer  
**Test Suite File**: [`test_pipeline.py`](file:///C:/Users/User1/.gemini/antigravity/scratch/customer_churn_prediction/test_pipeline.py)  
**Execution Framework**: Python `unittest`  

---

## Executive Summary
Software reliability and robust edge-case handling are critical requirements for production machine learning models and AI Trainer evaluation tasks. This report outlines the automated unit testing strategy, validation matrices, boundary conditions, and test execution results for the Customer Churn Prediction pipeline.

---

## 1. Test Architecture & Coverage Matrix

The test suite covers every phase of the ML pipeline lifecycle:

```
[Dataset Generator] → [Data Cleaner] → [NumPy Engine] → [Trainer/Evaluator] → [Inference Engine]
```

### Test Case Execution Log

| Test ID | Test Case Name | Objective | Input Scenario | Status |
| :--- | :--- | :--- | :--- | :--- |
| `TEST-01` | `test_1_dataset_generation` | Verify dataset generation | Request 200 synthetic rows | **PASSED** |
| `TEST-02` | `test_2_data_cleaning_imputation` | Validate missing string handling | Whitespace string `' '` in `TotalCharges` | **PASSED** |
| `TEST-03` | `test_3_numpy_feature_operations` | Division-by-zero protection | Customer with `TenureMonths = 0` | **PASSED** |
| `TEST-04` | `test_4_train_and_evaluate_metrics` | Classification metrics validation | 80/20 train-test split | **PASSED** |
| `TEST-05` | `test_5_high_risk_customer_prediction` | High churn risk boundary test | Short tenure, Month-to-month, Fiber | **PASSED** |
| `TEST-06` | `test_6_low_risk_customer_prediction` | Low churn risk boundary test | 60 mo tenure, Two-year contract, DSL | **PASSED** |

---

## 2. Detailed Edge Case Analysis

### Edge Case 1: Missing / Blank String `TotalCharges`
- **Challenge**: Raw dataset imports often represent missing numeric values as whitespace strings `" "`, causing Pandas `.astype(float)` to throw `ValueError`.
- **Handling**: `pd.to_numeric(s.str.strip(), errors='coerce')` converts invalid strings into `NaN`, followed by fallback calculation `MonthlyCharges * TenureMonths`.
- **Validation Result**: `test_2_data_cleaning_imputation` verified zero `NaN` values remain.

### Edge Case 2: Zero Tenure Division (`TenureMonths = 0`)
- **Challenge**: Calculating ratio metrics like $\frac{\text{TotalCharges}}{\text{TenureMonths}}$ results in `inf` or `NaN` when tenure is 0.
- **Handling**: Vectorized `np.where(tenure > 0, total / tenure, monthly)` guarantees valid floats.
- **Validation Result**: `test_3_numpy_feature_operations` verified no infinite values exist in feature matrix.

### Edge Case 3: Single Customer Inference & Categorical Column Alignment
- **Challenge**: Single input dictionaries converted via `pd.get_dummies` lack columns present in full training set (e.g. `PaymentMethod_Mailed check`).
- **Handling**: Re-alignment loop forces missing dummy columns to `0` and maintains identical column ordering as training.
- **Validation Result**: `test_5` and `test_6` executed predictions without feature dimension mismatch errors.

---

## 3. How to Run the Test Suite

Run unit tests with full verbosity:
```bash
python -m unittest test_pipeline.py -v
```

Expected Output:
```text
test_1_dataset_generation (test_pipeline.TestCustomerChurnPipeline) ... ok
test_2_data_cleaning_imputation (test_pipeline.TestCustomerChurnPipeline) ... ok
test_3_numpy_feature_operations (test_pipeline.TestCustomerChurnPipeline) ... ok
test_4_train_and_evaluate_metrics (test_pipeline.TestCustomerChurnPipeline) ... ok
test_5_high_risk_customer_prediction (test_pipeline.TestCustomerChurnPipeline) ... ok
test_6_low_risk_customer_prediction (test_pipeline.TestCustomerChurnPipeline) ... ok

----------------------------------------------------------------------
Ran 6 tests in 0.854s

OK
```
