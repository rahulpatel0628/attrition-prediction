import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path

PLOT_DIR = Path("notebooks/eda_plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


sns.set_theme(style="darkgrid", palette="muted")
COLORS = {"yes": "#E74C3C", "no": "#2ECC71", "accent": "#3498DB"}
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})


def plot_attrition_distribution(df: pd.DataFrame):
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Employee Attrition Overview", fontsize=14, fontweight="bold")

    counts = df["Attrition"].value_counts()
    axes[0].bar(["Stayed", "Left"], counts.values,
                color=[COLORS["no"], COLORS["yes"]], edgecolor="white", linewidth=1.5)
    axes[0].set_title("Attrition Count")
    axes[0].set_ylabel("Number of Employees")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")

   
    axes[1].pie(counts.values, labels=["Stayed", "Left"],
                colors=[COLORS["no"], COLORS["yes"]],
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Attrition Rate")

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "01_attrition_distribution.png")
    plt.close()
    print("Saved: 01_attrition_distribution.png")


def plot_attrition_by_department(df: pd.DataFrame):
    
    dept_attr = df.groupby("Department")["Attrition"].mean().sort_values(ascending=False) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(dept_attr.index, dept_attr.values,
                   color=[COLORS["yes"], COLORS["accent"], COLORS["no"]])
    ax.set_xlabel("Attrition Rate (%)")
    ax.set_title("Attrition Rate by Department", fontweight="bold")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())

    for bar, val in zip(bars, dept_attr.values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "02_attrition_by_department.png")
    plt.close()
    print("Saved: 02_attrition_by_department.png")


def plot_age_distribution(df: pd.DataFrame):
    
    fig, ax = plt.subplots(figsize=(10, 5))
    df[df["Attrition"] == 0]["Age"].plot.hist(ax=ax, alpha=0.6, color=COLORS["no"],
                                               label="Stayed", bins=20, edgecolor="white")
    df[df["Attrition"] == 1]["Age"].plot.hist(ax=ax, alpha=0.6, color=COLORS["yes"],
                                               label="Left", bins=20, edgecolor="white")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.set_title("Age Distribution by Attrition", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "03_age_distribution.png")
    plt.close()
    print("Saved: 03_age_distribution.png")


def plot_overtime_attrition(df: pd.DataFrame):
    
    ot_attr = df.groupby("OverTime")["Attrition"].mean() * 100
    labels = ["No Overtime", "Works Overtime"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, ot_attr.values,
                  color=[COLORS["no"], COLORS["yes"]], width=0.4, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Attrition Rate (%)")
    ax.set_title("Overtime vs Attrition Rate", fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    for bar, val in zip(bars, ot_attr.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "04_overtime_attrition.png")
    plt.close()
    print("Saved: 04_overtime_attrition.png")


def plot_salary_vs_attrition(df: pd.DataFrame):
    
    fig, ax = plt.subplots(figsize=(10, 5))
    df[df["Attrition"] == 0]["MonthlyIncome"].plot.kde(ax=ax, color=COLORS["no"], label="Stayed", linewidth=2)
    df[df["Attrition"] == 1]["MonthlyIncome"].plot.kde(ax=ax, color=COLORS["yes"], label="Left", linewidth=2)
    ax.set_xlabel("Monthly Income ($)")
    ax.set_title("Income Distribution by Attrition", fontweight="bold")
    ax.legend()
    ax.fill_between(ax.lines[0].get_xdata(), ax.lines[0].get_ydata(), alpha=0.1, color=COLORS["no"])
    ax.fill_between(ax.lines[1].get_xdata(), ax.lines[1].get_ydata(), alpha=0.1, color=COLORS["yes"])
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "05_income_distribution.png")
    plt.close()
    print("Saved: 05_income_distribution.png")


def plot_satisfaction_heatmap(df: pd.DataFrame):
    
    sat_cols = [
        "JobSatisfaction", "EnvironmentSatisfaction",
        "RelationshipSatisfaction", "WorkLifeBalance", "Attrition"
    ]
    existing = [c for c in sat_cols if c in df.columns]
    corr = df[existing].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, ax=ax, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Satisfaction Metrics Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "06_satisfaction_heatmap.png")
    plt.close()
    print("Saved: 06_satisfaction_heatmap.png")


def plot_jobrole_attrition(df: pd.DataFrame):
 
    role_attr = df.groupby("JobRole")["Attrition"].mean().sort_values(ascending=True) * 100
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(role_attr)))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(role_attr.index, role_attr.values, color=colors, edgecolor="white")
    ax.set_xlabel("Attrition Rate (%)")
    ax.set_title("Attrition Rate by Job Role", fontweight="bold")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())

    for bar, val in zip(bars, role_attr.values):
        ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "07_jobrole_attrition.png")
    plt.close()
    print("Saved: 07_jobrole_attrition.png")


def run_eda(df: pd.DataFrame):
    
    print("\n" + "="*50)
    print("STEP 4: EXPLORATORY DATA ANALYSIS")
    print("="*50)
    print(f"Output directory: {PLOT_DIR.resolve()}\n")

    plot_attrition_distribution(df)
    plot_attrition_by_department(df)
    plot_age_distribution(df)
    plot_overtime_attrition(df)
    plot_salary_vs_attrition(df)
    plot_satisfaction_heatmap(df)
    plot_jobrole_attrition(df)

    print(f"\nEDA complete → {len(list(PLOT_DIR.glob('*.png')))} plots saved")
    print("="*50 + "\n")


if __name__ == "__main__":
    from data_cleaning import clean_data
    df = clean_data("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    run_eda(df)
