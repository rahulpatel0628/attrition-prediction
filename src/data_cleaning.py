import pandas as pd
import numpy as np

CONSTANT_COLUMNS = ["EmployeeCount", "StandardHours", "Over18"]


def load_data(filepath: str) -> pd.DataFrame:

    df = pd.read_csv(filepath)
    print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:

    cols_before = df.shape[1]
    df = df.drop(columns=[c for c in CONSTANT_COLUMNS if c in df.columns])
    print(f"Dropped {cols_before - df.shape[1]} constant columns: {CONSTANT_COLUMNS}")
    return df


def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:

    if "EmployeeNumber" in df.columns:
        df = df.drop(columns=["EmployeeNumber"])
        print("Dropped 'EmployeeNumber' (ID column)")
    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:

    if "Attrition" in df.columns:
        df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0}).astype(int)
        print("Encoded 'Attrition' → 1/0")

    if "OverTime" in df.columns:
        df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0}).astype(int)
        print("Encoded 'OverTime' → 1/0")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    
    before = df.shape[0]
    df = df.drop_duplicates()
    dropped = before - df.shape[0]
    print(f"Removed {dropped} duplicate rows")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]

    if null_cols.empty:
        print("No missing values found")
        return df

    print(f"Found missing values in {len(null_cols)} columns — fixing...")

    for col in null_cols.index:
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    print("Missing values handled")
    return df


def clean_data(filepath: str) -> pd.DataFrame:
    
    print("\n" + "="*50)
    print("STEP 2: DATA CLEANING")
    print("="*50)

    df = load_data(filepath)
    df = drop_constant_columns(df)
    df = drop_id_columns(df)
    df = fix_dtypes(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)

    print(f"\n Cleaning complete → {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Target distribution:\n{df['Attrition'].value_counts().to_string()}")
    print("="*50 + "\n")

    return df


if __name__ == "__main__":
    df = clean_data("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    print(df.head())
    print(df.dtypes)
