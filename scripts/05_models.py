"""Fase 5 — Comparación y entrenamiento de modelos predictivos.

Entrada:  data/processed/data_engineered.csv
Salidas:  models/best_model.pkl  models/scaler.pkl  models/feature_names.pkl
          models/<nombre_modelo>.pkl  (uno por modelo)
"""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    f1_score, precision_score, recall_score,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
)

from src.config import (TARGET, FIGURES_DIR, MODELS_DIR, PROCESSED_DIR,
                        RANDOM_STATE, CV_FOLDS)
from src.features import ALL_FEATURES
from src.models import get_models, evaluate_model, cross_validate_all, plot_roc_curves

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
sns.set_theme(style="whitegrid")


def plot_cv_comparison(cv_results):
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = ["accuracy_mean", "precision_mean", "recall_mean", "f1_mean", "roc_auc_mean"]
    labels  = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    colors  = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c", "#9b59b6"]
    x = np.arange(len(cv_results))
    w = 0.15
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        ax.bar(x + i * w, cv_results[metric].values, w, label=label, color=color, alpha=0.85)
    ax.set_xticks(x + w * 2)
    ax.set_xticklabels(cv_results.index, rotation=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Puntuación")
    ax.set_title("Comparación de modelos — Validación cruzada 5-fold", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_cv_comparison.png", bbox_inches="tight")
    plt.close()


def plot_confusion_matrices(trained_models, X_test, y_test):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (name, model) in zip(axes.flatten(), trained_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["Normal", "Fallo"]).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(name, fontweight="bold")
    plt.suptitle("Matrices de confusión — Test Set", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_confusion_matrices.png", bbox_inches="tight")
    plt.close()


def plot_threshold_analysis(model, X_test, y_test, name):
    y_prob = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.1, 0.91, 0.05)
    rows = []
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        rows.append({"threshold": t,
                     "precision": precision_score(y_test, yp, zero_division=0),
                     "recall":    recall_score(y_test, yp, zero_division=0),
                     "f1":        f1_score(y_test, yp, zero_division=0)})
    tdf = pd.DataFrame(rows)
    best_t = tdf.loc[tdf["f1"].idxmax(), "threshold"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tdf["threshold"], tdf["precision"], "b-o", label="Precision", markersize=4)
    ax.plot(tdf["threshold"], tdf["recall"],    "r-o", label="Recall",    markersize=4)
    ax.plot(tdf["threshold"], tdf["f1"],        "g-o", label="F1",        markersize=4)
    ax.axvline(0.5, color="gray",   linestyle="--", label="Umbral por defecto (0.5)")
    ax.axvline(best_t, color="purple", linestyle="--", label=f"Mejor F1 (t={best_t:.2f})")
    ax.set_title(f"Análisis de umbral — {name}", fontweight="bold")
    ax.set_xlabel("Umbral de decisión")
    ax.set_ylabel("Puntuación")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_threshold_analysis.png", bbox_inches="tight")
    plt.close()
    return best_t


def plot_learning_curve(model, X_train, y_train, name):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    sizes, tr_sc, val_sc = learning_curve(
        model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(sizes, tr_sc.mean(1) - tr_sc.std(1), tr_sc.mean(1) + tr_sc.std(1),
                    alpha=0.2, color="#3498db")
    ax.fill_between(sizes, val_sc.mean(1) - val_sc.std(1), val_sc.mean(1) + val_sc.std(1),
                    alpha=0.2, color="#e74c3c")
    ax.plot(sizes, tr_sc.mean(1),  "b-o", label="Train F1")
    ax.plot(sizes, val_sc.mean(1), "r-o", label="Val F1 (CV)")
    ax.set_title(f"Curva de aprendizaje — {name}", fontweight="bold")
    ax.set_xlabel("Tamaño del train set")
    ax.set_ylabel("F1-score")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_learning_curve.png", bbox_inches="tight")
    plt.close()


def main():
    t0 = time.time()
    print("=" * 55)
    print("FASE 5 — Comparación y Entrenamiento de Modelos")
    print("=" * 55)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR / "data_engineered.csv")
    X = df[ALL_FEATURES]
    y = df[TARGET]
    print(f"Dataset: {X.shape[0]} muestras × {X.shape[1]} características")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    models = get_models()

    print("\n--- Validación cruzada (5-fold) ---")
    cv_results = cross_validate_all(models, X_train_sc, y_train)
    print(cv_results[["f1_mean", "f1_std", "roc_auc_mean", "precision_mean", "recall_mean"]].round(4).to_string())

    print("\n[1/5] Gráfico de comparación CV...")
    plot_cv_comparison(cv_results)

    print("\n--- Entrenamiento final ---")
    trained_models = {}
    test_results = {}

    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        metrics = evaluate_model(model, X_test_sc, y_test)
        test_results[name] = metrics
        trained_models[name] = model
        print(f"  {name}: F1={metrics['f1']:.4f}  AUC={metrics['roc_auc']:.4f}")

    test_df = pd.DataFrame(test_results).T
    print("\n--- Métricas en Test Set ---")
    print(test_df.round(4).sort_values("f1", ascending=False).to_string())

    best_name = test_df["f1"].idxmax()
    best_model = trained_models[best_name]
    print(f"\nModelo seleccionado: {best_name}")

    print("\n[2/5] Matrices de confusión...")
    plot_confusion_matrices(trained_models, X_test_sc, y_test)

    print("[3/5] Curvas ROC...")
    fig = plot_roc_curves(trained_models, X_test_sc, y_test)
    fig.savefig(FIGURES_DIR / "05_roc_curves.png", bbox_inches="tight")
    plt.close()

    print("[4/5] Análisis de umbral...")
    best_t = plot_threshold_analysis(best_model, X_test_sc, y_test, best_name)
    print(f"  Mejor umbral F1: {best_t:.2f}")

    print("[5/5] Curva de aprendizaje...")
    plot_learning_curve(best_model, X_train_sc, y_train, best_name)

    # Guardar modelos
    joblib.dump(best_model,   MODELS_DIR / "best_model.pkl")
    joblib.dump(scaler,       MODELS_DIR / "scaler.pkl")
    joblib.dump(list(ALL_FEATURES), MODELS_DIR / "feature_names.pkl")
    for name, model in trained_models.items():
        safe = name.replace(" ", "_").lower()
        joblib.dump(model, MODELS_DIR / f"{safe}.pkl")

    print(f"\nModelos guardados en: {MODELS_DIR}")
    print(f"Fase 5 completada en {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
