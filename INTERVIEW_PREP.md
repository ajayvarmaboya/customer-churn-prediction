# Interview Preparation Guide: AI Trainer / Fresher ML Role

**Project**: Customer Churn Prediction  
**Tech Stack**: Python | Pandas | NumPy | Matplotlib | Scikit-learn  
**Target Role**: AI Trainer / Data Annotation Specialist / Junior ML Engineer  

---

## Overview
As an **AI Trainer**, your role involves evaluating, ranking, writing, and debugging machine learning code, prompts, and model responses. Interviewers will test both your **core ML/Data Science fundamentals** and your ability to **critique AI-generated code and explanations**.

This guide provides **25 curated questions and structured answers** grouped into 6 interview modules.

---

## Module 1: Pandas & Data Processing

### Q1: How did you handle missing values in `TotalCharges` in your dataset?
**Answer**:
In telecommunications datasets, missing values in `TotalCharges` often appear as blank whitespace strings (`" "`) rather than explicit `NaN`s. 
1. I first stripped whitespace and used `pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')` to convert invalid strings to `NaN`.
2. Rather than dropping rows or using a simple column median, I imputed missing values using a domain-specific formula:
   $$\text{TotalCharges} = \text{MonthlyCharges} \times \text{TenureMonths}$$

### Q2: What is the difference between `loc` and `iloc` in Pandas?
**Answer**:
- `loc` is **label-based**: indexing uses column names and row index labels (e.g. `df.loc[0:5, 'MonthlyCharges']`).
- `iloc` is **integer position-based**: indexing uses numeric position indices (e.g. `df.iloc[0:5, 2:4]`).

### Q3: Why did you use `pd.get_dummies()` instead of Label Encoding for categorical variables?
**Answer**:
Label Encoding assigns arbitrary integers ($0, 1, 2, \dots$) to categories (e.g., `DSL`=0, `Fiber optic`=1, `No`=2). For linear models like Logistic Regression, this creates a false ordinal ranking assumption (that `No` > `Fiber optic` > `DSL`). One-Hot Encoding creates binary indicator columns, avoiding non-existent magnitude assumptions.

---

## Module 2: NumPy Numerical Operations

### Q4: How did you use NumPy in your feature engineering step?
**Answer**:
I leveraged vectorized NumPy arrays for speed and safety:
1. Computed `AvgChargeRatio` using `np.where(tenure > 0, total / tenure, monthly)` to safely handle `tenure = 0` without zero-division runtime errors.
2. Applied log transformation `np.log1p(total_charges)` to convert right-skewed revenue distributions into near-normal distributions.

### Q5: What is array broadcasting in NumPy?
**Answer**:
Broadcasting is NumPy's mechanism for performing arithmetic operations on arrays of different shapes without copying data. For example, subtracting a scalar mean (1D) or a row vector from a 2D matrix automatically expands dimensions virtually across matching axes.

---

## Module 3: Matplotlib & Data Visualization

### Q6: What key visualizations did you build, and what insights did they reveal?
**Answer**:
I created three Matplotlib figures:
1. **Churn Distribution**: A combined Pie/Bar chart showing the overall dataset churn percentage (~26.5%).
2. **Tenure vs. Churn Boxplot**: Demonstrated that churned customers have a significantly lower median tenure (~10 months) compared to retained customers (~38 months).
3. **Correlation Matrix Heatmap**: Revealed a strong positive correlation between `Fiber optic` internet service and higher `MonthlyCharges`, correlating with higher churn risk.

### Q7: Why is it important to call `plt.close()` after saving figures?
**Answer**:
Matplotlib retains figure objects in memory until explicitly closed. In iterative scripts or web services, failing to call `plt.close()` causes memory leaks and plots bleeding over into subsequent figures.

---

## Module 4: Logistic Regression & Modeling

### Q8: Explain how Logistic Regression makes predictions.
**Answer**:
Logistic Regression takes a linear combination of input features:
$$z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n$$
It passes $z$ through the **Sigmoid Activation Function**:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
This maps $z$ to a probability output between $0$ and $1$. If $\sigma(z) \ge 0.5$, the prediction is `CHURN (1)`; otherwise `RETAIN (0)`.

### Q9: Why is feature scaling (`StandardScaler`) essential for Logistic Regression?
**Answer**:
Logistic Regression uses gradient-based optimization techniques. Features with large numerical ranges (e.g., `TotalCharges` up to \$8,000) produce steep gradients, while small features (e.g., `TenureMonths` 1 to 72) produce shallow gradients. Scaling all features to zero mean ($\mu=0$) and unit variance ($\sigma=1$) ensures smooth gradient descent convergence.

---

## Module 5: Evaluation Metrics

### Q10: What is the difference between Precision and Recall? When would you prioritize Recall over Precision?
**Answer**:
- **Precision**: $\frac{TP}{TP + FP}$ — Out of all predicted churners, how many actually churned?
- **Recall**: $\frac{TP}{TP + FN}$ — Out of all actual churners, how many did the model correctly identify?
- **Business Trade-off**: In customer churn, **Recall is prioritized** because missing a true churner (False Negative = lost customer revenue) is far more expensive than sending a retention email to a non-churner (False Positive = small promotional cost).

### Q11: What is the F1-Score?
**Answer**:
F1-Score is the harmonic mean of Precision and Recall:
$$F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
Unlike arithmetic mean, harmonic mean heavily penalizes extreme imbalance between Precision and Recall.

---

## Module 6: AI Trainer Role Specific Questions

### Q12: How would you evaluate an LLM response providing code for data preprocessing?
**Answer**:
I evaluate code using four dimensions:
1. **Correctness**: Does the code execute without errors and output accurate data?
2. **Data Leakage**: Does it fit transformers (like `StandardScaler`) on the *entire* dataset before splitting? (If so, it's incorrect data leakage).
3. **Edge Case Safety**: Does it handle `NaN`s, empty strings, zero division, and unseen categories?
4. **Code Quality & Docstrings**: Is it PEP-8 compliant with clear variable names and comments?

### Q13: Spot the bug in this LLM-generated code snippet:
```python
# LLM Output:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)  # BUG!
```
**Answer**:
- **Bug**: `scaler.fit_transform(X_test)` re-fits the scaler on the test set!
- **Correction**: `X_test_scaled = scaler.transform(X_test)` must be used to apply the mean and variance learned from the training set, preventing test set data leakage.

### Q14: How would you prompt an AI model to build a churn prediction model?
**Answer**:
I use the **System-Context-Task-Constraint** framework:
- **System**: "You are an expert Senior Machine Learning Engineer."
- **Context**: "We have a pandas DataFrame with telecommunication customer features and target column `Churn`."
- **Task**: "Write a modular Python script using `scikit-learn` to clean data, train a `LogisticRegression` classifier, and output accuracy, precision, recall, and F1-score."
- **Constraint**: "Fit `StandardScaler` on training data only. Do not drop missing values without imputation. Include unit tests."
