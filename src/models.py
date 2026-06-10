import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay,
)
from sklearn.model_selection import cross_validate, StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
from src.config import RANDOM_STATE, CV_FOLDS


def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, scale_pos_weight=1.4,
            use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_STATE, verbosity=0,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=RANDOM_STATE, verbose=-1,
        ),
    }


def evaluate_model(model, X_test, y_test, threshold=0.5) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_prob),
    }


def cross_validate_all(models: dict, X, y) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        row = {"model": name}
        for metric in scoring:
            key = f"test_{metric}"
            row[metric + "_mean"] = scores[key].mean()
            row[metric + "_std"]  = scores[key].std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def plot_confusion_matrix(model, X_test, y_test, title="", ax=None):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Fallo"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    if ax:
        ax.set_title(title)
    return cm


def plot_roc_curves(models: dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", label="Aleatorio")
    ax.set_title("Curvas ROC — Comparación de modelos")
    ax.legend(loc="lower right")
    return fig
