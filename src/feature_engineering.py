import pandas as pd
import numpy as np


def add_years_per_promotion(df: pd.DataFrame) -> pd.DataFrame:
    df["YearsPerPromotion"] = df["YearsAtCompany"] / (df["YearsWithCurrManager"] + 1)
    return df


def add_salary_growth_gap(df: pd.DataFrame) -> pd.DataFrame:
    avg_hike = df["PercentSalaryHike"].mean()
    df["SalaryGrowthGap"] = df["PercentSalaryHike"] - avg_hike
    return df


def add_satisfaction_composite(df: pd.DataFrame) -> pd.DataFrame:
    satisfaction_cols = [
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "RelationshipSatisfaction",
        "WorkLifeBalance",
    ]
    existing = [c for c in satisfaction_cols if c in df.columns]
    df["SatisfactionComposite"] = df[existing].mean(axis=1)
    return df


def add_engagement_score(df: pd.DataFrame) -> pd.DataFrame:
    df["EngagementScore"] = (
        df.get("JobInvolvement", 2) * 0.4 +
        df.get("JobSatisfaction", 2) * 0.3 +
        df.get("WorkLifeBalance", 2) * 0.3
    )
    return df


def add_career_velocity(df: pd.DataFrame) -> pd.DataFrame:
    df["CareerVelocity"] = df["JobLevel"] / (df["YearsAtCompany"] + 1)
    return df


def add_overtime_seniority_risk(df: pd.DataFrame) -> pd.DataFrame:
    df["OvertimeSeniorityRisk"] = df["OverTime"] * df["TotalWorkingYears"]
    return df


def add_loyalty_score(df: pd.DataFrame) -> pd.DataFrame:
    df["LoyaltyScore"] = df["YearsAtCompany"] / (df["TotalWorkingYears"] + 1)
    return df


def add_distance_worklife_risk(df: pd.DataFrame) -> pd.DataFrame:
    df["DistanceWorklifeRisk"] = df["DistanceFromHome"] * (5 - df["WorkLifeBalance"])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    
    print("\n" + "="*50)
    print("STEP 5: FEATURE ENGINEERING")
    print("="*50)

    original_cols = df.shape[1]

    df = add_years_per_promotion(df)
    df = add_salary_growth_gap(df)
    df = add_satisfaction_composite(df)
    df = add_engagement_score(df)
    df = add_career_velocity(df)
    df = add_overtime_seniority_risk(df)
    df = add_loyalty_score(df)
    df = add_distance_worklife_risk(df)

    new_features = df.shape[1] - original_cols
    print(f"Created {new_features} new features:")
    new_cols = [
        "YearsPerPromotion", "SalaryGrowthGap", "SatisfactionComposite",
        "EngagementScore", "CareerVelocity", "OvertimeSeniorityRisk",
        "LoyaltyScore", "DistanceWorklifeRisk"
    ]
    for col in new_cols:
        print(f"   + {col}")

    print("="*50 + "\n")
    return df


if __name__ == "__main__":
    from data_cleaning import clean_data
    df = clean_data("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    df = engineer_features(df)
    print(df[["YearsPerPromotion", "SatisfactionComposite", "EngagementScore"]].describe())
