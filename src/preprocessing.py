import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def get_feature_groups(df: pd.DataFrame, target: str = "Attrition"):

    df_features = df.drop(columns=[target])

    categorical_cols = df_features.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()

    print(f"Numeric features  : {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")

    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    return preprocessor


def split_data(df: pd.DataFrame, target: str = "Attrition", test_size: float = 0.2, random_state: int = 42):
   
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Train set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")
    print(f"Train attrition rate: {y_train.mean():.2%}")
    print(f"Test  attrition rate: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test

def save_preprocessor_artifacts(preprocessor: ColumnTransformer, feature_names: list):
    
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")
    print("Saved: models/preprocessor.pkl")

    with open(MODELS_DIR / "feature_list.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    print("Saved: models/feature_list.json")


def preprocess_data(df: pd.DataFrame, target: str = "Attrition"):
    
    print("\n" + "="*50)
    print("STEP 3: DATA PREPROCESSING")
    print("="*50)

    numeric_cols, categorical_cols = get_feature_groups(df, target)
    X_train, X_test, y_train, y_test = split_data(df, target)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    print(f"Transformed shapes → Train: {X_train_transformed.shape} | Test: {X_test_transformed.shape}")


    feature_names = X_train.columns.tolist()
    save_preprocessor_artifacts(preprocessor, feature_names)

    print("="*50 + "\n")

    return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor


if __name__ == "__main__":
    from data_cleaning import clean_data
    from feature_engineering import engineer_features

    df = clean_data("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    print(f"Final training shape: {X_train.shape}")