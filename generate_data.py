import os
import numpy as np
import pandas as pd

def generate_churn_dataset(output_path="data/customer_churn.csv", num_samples=1000, random_seed=42):
    """
    Generates a realistic synthetic customer churn dataset with structured features
    and realistic missing/whitespace edge cases for data cleaning practice.
    """
    np.random.seed(random_seed)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    customer_ids = [f"{np.random.randint(1000, 9999)}-{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}" for _ in range(num_samples)]
    genders = np.random.choice(["Male", "Female"], size=num_samples)
    senior_citizens = np.random.choice([0, 1], size=num_samples, p=[0.84, 0.16])
    partners = np.random.choice(["Yes", "No"], size=num_samples, p=[0.5, 0.5])
    dependents = np.random.choice(["Yes", "No"], size=num_samples, p=[0.3, 0.7])
    
    # Tenure in months (1 to 72)
    tenure = np.random.randint(1, 73, size=num_samples)
    
    phone_service = np.random.choice(["Yes", "No"], size=num_samples, p=[0.9, 0.1])
    multiple_lines = []
    for ps in phone_service:
        if ps == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(np.random.choice(["Yes", "No"], p=[0.45, 0.55]))
            
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=num_samples, p=[0.4, 0.45, 0.15])
    
    online_security = []
    tech_support = []
    for iserv in internet_service:
        if iserv == "No":
            online_security.append("No internet service")
            tech_support.append("No internet service")
        else:
            online_security.append(np.random.choice(["Yes", "No"], p=[0.35, 0.65]))
            tech_support.append(np.random.choice(["Yes", "No"], p=[0.35, 0.65]))
            
    contracts = np.random.choice(["Month-to-month", "One year", "Two year"], size=num_samples, p=[0.55, 0.25, 0.20])
    paperless_billing = np.random.choice(["Yes", "No"], size=num_samples, p=[0.6, 0.4])
    payment_methods = np.random.choice([
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ], size=num_samples)
    
    # Calculate realistic MonthlyCharges based on services
    monthly_charges = []
    for i in range(num_samples):
        base = 20.0
        if phone_service[i] == "Yes":
            base += 15.0
        if multiple_lines[i] == "Yes":
            base += 10.0
        if internet_service[i] == "DSL":
            base += 25.0
        elif internet_service[i] == "Fiber optic":
            base += 45.0
        if online_security[i] == "Yes":
            base += 8.0
        if tech_support[i] == "Yes":
            base += 8.0
        # Add slight random variation
        monthly_charges.append(round(base + np.random.normal(0, 3.0), 2))
        
    monthly_charges = np.array(monthly_charges)
    monthly_charges = np.clip(monthly_charges, 18.25, 118.75)
    
    total_charges = []
    for i in range(num_samples):
        # 1.5% chance of missing/blank string TotalCharges (simulating new customers or data entry noise)
        if np.random.rand() < 0.015 or tenure[i] == 0:
            total_charges.append(" ")
        else:
            est_total = round(monthly_charges[i] * tenure[i] + np.random.normal(0, 10.0), 2)
            total_charges.append(str(max(0.0, est_total)))
            
    # Logistic formula to simulate churn probability based on tenure, contract, and Fiber optic
    churn_score = (
        -0.05 * tenure +
        (contracts == "Month-to-month") * 1.2 +
        (contracts == "Two year") * -1.5 +
        (internet_service == "Fiber optic") * 0.8 +
        (tech_support == "No") * 0.5 +
        (monthly_charges > 70) * 0.6 -
        0.5
    )
    churn_prob = 1 / (1 + np.exp(-churn_score))
    churn_labels = ["Yes" if np.random.rand() < p else "No" for p in churn_prob]
    
    df = pd.DataFrame({
        "CustomerID": customer_ids,
        "Gender": genders,
        "SeniorCitizen": senior_citizens,
        "Partner": partners,
        "Dependents": dependents,
        "TenureMonths": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support,
        "Contract": contracts,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_methods,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn_labels
    })
    
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Synthetic dataset generated at '{output_path}' with {len(df)} rows.")
    return df

if __name__ == "__main__":
    generate_churn_dataset()
