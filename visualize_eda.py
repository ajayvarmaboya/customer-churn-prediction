"""
Exploratory Data Analysis (EDA) & Matplotlib Visualizations
Tech Stack: Python | Pandas | Matplotlib

Generates EDA plots for Customer Churn patterns and saves them as high-resolution PNG images.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_eda_plots(data_path="data/customer_churn.csv", output_dir="eda_plots"):
    """Loads dataset, cleans total charges, and creates Matplotlib visualization charts."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    # Clean TotalCharges for plotting
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["TenureMonths"])

    # Set standard Matplotlib styling
    plt.style.use("ggplot")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10

    # -------------------------------------------------------------
    # 1. Churn Distribution (Pie + Bar Chart)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    churn_counts = df["Churn"].value_counts()
    
    colors = ["#4C72B0", "#DD8452"] # Retained, Churned
    ax1.pie(
        churn_counts, labels=churn_counts.index, autopct="%1.1f%%",
        startangle=140, colors=colors, explode=(0, 0.05),
        textprops={"fontsize": 11, "weight": "bold"}
    )
    ax1.set_title("Customer Churn Ratio", fontsize=14, pad=10, weight="bold")

    bars = ax2.bar(churn_counts.index, churn_counts.values, color=colors, width=0.5)
    ax2.set_title("Customer Churn Counts", fontsize=14, pad=10, weight="bold")
    ax2.set_ylabel("Number of Customers", fontsize=11)
    ax2.set_xlabel("Churn Status", fontsize=11)
    
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f"{height}", xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=11, weight="bold")

    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "churn_distribution.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved Churn Distribution chart to '{chart1_path}'")

    # -------------------------------------------------------------
    # 2. Tenure & Monthly Charges vs Churn (Boxplots)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    retained_tenure = df[df["Churn"] == "No"]["TenureMonths"]
    churned_tenure = df[df["Churn"] == "Yes"]["TenureMonths"]
    
    ax1.boxplot([retained_tenure, churned_tenure], tick_labels=["Retained", "Churned"], patch_artist=True,
                boxprops=dict(facecolor="#4C72B0", color="black"),
                medianprops=dict(color="yellow", linewidth=2))
    ax1.set_title("Tenure (Months) by Churn Status", fontsize=13, weight="bold")
    ax1.set_ylabel("Tenure (Months)")
    ax1.set_xlabel("Customer Status")

    retained_charges = df[df["Churn"] == "No"]["MonthlyCharges"]
    churned_charges = df[df["Churn"] == "Yes"]["MonthlyCharges"]

    ax2.boxplot([retained_charges, churned_charges], tick_labels=["Retained", "Churned"], patch_artist=True,
                boxprops=dict(facecolor="#DD8452", color="black"),
                medianprops=dict(color="yellow", linewidth=2))
    ax2.set_title("Monthly Charges ($) by Churn Status", fontsize=13, weight="bold")
    ax2.set_ylabel("Monthly Charges ($)")
    ax2.set_xlabel("Customer Status")

    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "tenure_vs_churn.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved Tenure vs Churn chart to '{chart2_path}'")

    # -------------------------------------------------------------
    # 3. Correlation Heatmap (Matplotlib imshow)
    # -------------------------------------------------------------
    numeric_df = df[["TenureMonths", "MonthlyCharges", "TotalCharges"]].copy()
    numeric_df["Churn_Binary"] = df["Churn"].map({"Yes": 1, "No": 0})
    corr_matrix = numeric_df.corr().values

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Correlation Coefficient", rotation=-90, va="bottom")

    labels = ["Tenure", "Monthly Chg", "Total Chg", "Churn Target"]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=10, weight="bold")
    ax.set_yticklabels(labels, fontsize=10, weight="bold")

    # Annotate correlation numbers inside cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=11, weight="bold")

    ax.set_title("Feature Correlation Matrix", fontsize=14, pad=12, weight="bold")
    plt.tight_layout()
    chart3_path = os.path.join(output_dir, "correlation_matrix.png")
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved Correlation Matrix chart to '{chart3_path}'")

if __name__ == "__main__":
    generate_eda_plots()
