import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    classification_report,
)
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    GridSearchCV,
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

PLOT_DIR = Path("notebooks/eda_plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CV_FOLDS = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(model, X_test, y_test, name=""):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Model": name,
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "F1-Score": f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Accuracy": accuracy_score(y_test, y_pred),
    }


def print_metrics(m):
    print(
        f"{m['Model']:<25} | "
        f"AUC={m['ROC-AUC']:.4f} | "
        f"F1={m['F1-Score']:.4f} | "
        f"P={m['Precision']:.4f} | "
        f"R={m['Recall']:.4f}"
    )

def train_base_learners(X_train, X_test, y_train, y_test):

    base_models = {
        "Bagging": BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=6),
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
        ),

        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=150,
            learning_rate=0.1,
            random_state=42,
        ),
    }

    trained = {}
    for name, model in base_models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        print_metrics(evaluate_model(model, X_test, y_test, name))

    return trained

def tune_xgboost_gridsearch(X_train, y_train, X_test, y_test):

    print("\nTUNING XGBOOST USING GRIDSEARCHCV")

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
        verbosity=0,
    )

    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.7, 0.9],
        "scale_pos_weight": [1, 3, 5],
    }

    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=CV_FOLDS,
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X_train, y_train)

    best_xgb = grid.best_estimator_

    print("Best Params:", grid.best_params_)
    print_metrics(evaluate_model(best_xgb, X_test, y_test, "XGBoost (GridSearch)"))

    return best_xgb

def build_ensembles(xgb, X_train, y_train, X_test, y_test):

    voting = VotingClassifier(
        estimators=[
            ("xgb", xgb),
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
            ("gb", GradientBoostingClassifier(n_estimators=200)),
        ],
        voting="soft",
        weights=[2, 1, 1],
        n_jobs=-1,
    )

    voting.fit(X_train, y_train)
    print_metrics(evaluate_model(voting, X_test, y_test, "Voting Ensemble"))

    stacking = StackingClassifier(
        estimators=[
            ("xgb", xgb),
            ("rf", RandomForestClassifier(n_estimators=150)),
            ("gb", GradientBoostingClassifier(n_estimators=150)),
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        n_jobs=-1,
    )

    stacking.fit(X_train, y_train)
    print_metrics(evaluate_model(stacking, X_test, y_test, "Stacking Ensemble"))

    return voting, stacking

def save_best_model(models, X_test, y_test):

    scores = {}
    for name, model in models.items():
        scores[name] = evaluate_model(model, X_test, y_test, name)

    best_name = max(scores, key=lambda x: scores[x]["ROC-AUC"])
    best_model = models[best_name]

    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")

    print(f"\n BEST MODEL: {best_name}")
    return best_model

if __name__ == "__main__":

    from data_cleaning import clean_data
    from feature_engineering import engineer_features
    from preprocessing import preprocess_data

    df = clean_data("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    df = engineer_features(df)

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    base_models = train_base_learners(X_train, X_test, y_train, y_test)
    xgb_best = tune_xgboost_gridsearch(X_train, y_train, X_test, y_test)

    voting, stacking = build_ensembles(
        xgb_best, X_train, y_train, X_test, y_test
    )

    all_models = {
        **base_models,
        "XGBoost": xgb_best,
        "Voting": voting,
        "Stacking": stacking,
    }

    best_model = save_best_model(all_models, X_test, y_test)
    print("\nTraining Complete ")