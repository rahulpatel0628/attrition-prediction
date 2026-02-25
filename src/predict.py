import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
MODELS_DIR = Path("models")
def load_artifacts():
    
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")

    with open(MODELS_DIR / "feature_list.json") as f:
        feature_list = json.load(f)

    return model, preprocessor, feature_list


def get_risk_label(probability: float) -> str:
   
    if probability < 0.30:
        return "Low"
    elif probability < 0.60:
        return "Medium"
    else:
        return "High"


def get_risk_actions(risk_label: str) -> list:
    
    actions = {
        "Low": [
            "Continue current engagement programs",
            "Recognize and reward performance",
            "Provide growth opportunities"
        ],
        "Medium": [
            "Schedule a 1:1 career discussion",
            "Review compensation vs market rate",
            "Assess work-life balance situation",
            "Check manager relationship health"
        ],
        "High": [
            "Immediate retention interview recommended",
            "Review salary and promotion eligibility",
            "Explore flexible work arrangements",
            "Assign mentor or career sponsor",
            "Consider role change or lateral move"
        ]
    }
    return actions.get(risk_label, [])


def predict_attrition(employee_data: dict) -> dict:
    
    model, preprocessor, feature_list = load_artifacts()

    df_input = pd.DataFrame([employee_data])

    from feature_engineering import engineer_features
    df_input = engineer_features(df_input)

    
    for col in feature_list:
        if col not in df_input.columns:
            df_input[col] = 0 

    df_input = df_input[feature_list]

 
    X = preprocessor.transform(df_input)

   
    probability = float(model.predict_proba(X)[0][1])
    prediction = int(model.predict(X)[0])
    risk_label = get_risk_label(probability)
    actions = get_risk_actions(risk_label)

    return {
        "will_attrite": bool(prediction),
        "attrition_probability": round(probability, 4),
        "risk_level": risk_label,
        "recommended_actions": actions
    }


if __name__ == "__main__":
   
    sample_employee = {
        "Age": 32,
        "BusinessTravel": "Travel_Frequently",
        "DailyRate": 800,
        "Department": "Sales",
        "DistanceFromHome": 15,
        "Education": 3,
        "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": 2,
        "Gender": "Male",
        "HourlyRate": 65,
        "JobInvolvement": 2,
        "JobLevel": 2,
        "JobRole": "Sales Representative",
        "JobSatisfaction": 2,
        "MaritalStatus": "Single",
        "MonthlyIncome": 3500,
        "MonthlyRate": 14000,
        "NumCompaniesWorked": 3,
        "OverTime": 1,
        "PercentSalaryHike": 11,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 2,
        "StockOptionLevel": 0,
        "TotalWorkingYears": 8,
        "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 2,
        "YearsAtCompany": 3,
        "YearsInCurrentRole": 2,
        "YearsSinceLastPromotion": 2,
        "YearsWithCurrManager": 1,
    }

    result = predict_attrition(sample_employee)
    print("\nPREDICTION RESULT:")
    for k, v in result.items():
        print(f"   {k}: {v}")
