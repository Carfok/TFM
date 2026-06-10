"""Fase 6 — Interpretación de Resultados (SHAP, Permutation Importance, PDP).

Entrada:  data/processed/data_engineered.csv  +  models/
Salida:   reports/figures/06_*.png
"""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

from src.config import (TARGET, FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE)
from src.features import ALL_FEATURES

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
sns.set_theme(style="whitegrid")


def plot_rf_importance(model, feature_cols):
    imp = pd.DataFrame({"feature": feature_cols,
                        "importance": model.feature_importances_}
                       ).sort_values("importance", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(imp)))
    ax.barh(imp["feature"], imp["importance"], color=colors)
    ax.set_title("Feature Importance — Random Forest", fontweight="bold")
    ax.set_xlabel("Importancia (Gini)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_rf_feature_importance.png", bbox_inches="tight")
    plt.close()
    return imp


def plot_permutation_importance(result, feature_cols):
    perm = pd.DataFrame({"feature": feature_cols,
                         "mean": result.importances_mean,
                         "std":  result.importances_std}
                        ).sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(perm["feature"], perm["mean"], xerr=perm["std"],
            color="#3498db", alpha=0.8, error_kw=dict(ecolor="gray", capsize=3))
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Permutation Importance (caída en F1)", fontweight="bold")
    ax.set_xlabel("Caída media en F1")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_permutation_importance.png", bbox_inches="tight")
    plt.close()
    return perm


def plot_shap_summary(shap_vals, X_test, feature_cols):
    shap.summary_plot(shap_vals, X_test, feature_names=feature_cols,
                      plot_type="dot", show=False, max_display=18)
    plt.title("SHAP Summary — Impacto en predicción de fallo", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_shap_summary.png", bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_vals, X_test, feature_names=feature_cols,
                      plot_type="bar", show=False, max_display=18)
    plt.title("SHAP — Importancia media global", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_shap_bar.png", bbox_inches="tight")
    plt.close()


def plot_shap_dependence(shap_vals, X_test_sc, feature_cols, top_features):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    X_df = pd.DataFrame(X_test_sc, columns=feature_cols)
    for ax, feat in zip(axes.flatten(), top_features):
        idx = list(feature_cols).index(feat)
        xv = X_df[feat].values
        sv = shap_vals[:, idx]
        sc = ax.scatter(xv, sv, c=sv, cmap="coolwarm", s=12, alpha=0.7, linewidths=0)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xlabel(feat)
        ax.set_ylabel("SHAP value")
        ax.set_title(f"SHAP Dependence — {feat}", fontweight="bold")
        plt.colorbar(sc, ax=ax, label="SHAP value")
    plt.suptitle("SHAP Dependence Plots — Top 4 características",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_shap_dependence.png", bbox_inches="tight")
    plt.close()


def plot_importance_comparison(feature_cols, rf_imp, perm_imp, shap_vals):
    cmp = pd.DataFrame({
        "feature":      feature_cols,
        "RF":           rf_imp.set_index("feature")["importance"].reindex(feature_cols).values,
        "Permutation":  perm_imp.set_index("feature")["mean"].reindex(feature_cols).clip(0).values,
        "SHAP":         np.abs(shap_vals).mean(0),
    })
    for col in ["RF", "Permutation", "SHAP"]:
        mx = cmp[col].max()
        cmp[col] = cmp[col] / mx if mx > 0 else cmp[col]
    cmp["mean_rank"] = cmp[["RF", "Permutation", "SHAP"]].mean(1)
    cmp = cmp.sort_values("mean_rank", ascending=False)

    fig, ax = plt.subplots(figsize=(11, 8))
    x = np.arange(len(cmp))
    w = 0.28
    ax.barh(x,           cmp["RF"],          w, label="RF Importance",        color="#3498db", alpha=0.8)
    ax.barh(x + w,       cmp["Permutation"], w, label="Permutation Importance", color="#e74c3c", alpha=0.8)
    ax.barh(x + 2*w,     cmp["SHAP"],        w, label="SHAP Mean |value|",     color="#2ecc71", alpha=0.8)
    ax.set_yticks(x + w)
    ax.set_yticklabels(cmp["feature"])
    ax.set_title("Comparación RF vs Permutation vs SHAP (normalizadas)", fontweight="bold")
    ax.set_xlabel("Importancia normalizada (0–1)")
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_importance_comparison.png", bbox_inches="tight")
    plt.close()


def plot_pdp(model, X_test_sc, feature_cols, top2_idx):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    PartialDependenceDisplay.from_estimator(
        model, X_test_sc, features=top2_idx,
        feature_names=feature_cols, ax=axes, kind="average"
    )
    for ax, idx in zip(axes, top2_idx):
        ax.set_title(f"PDP — {feature_cols[idx]}", fontweight="bold")
    plt.suptitle("Partial Dependence Plots — Efecto marginal de variables principales",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_pdp.png", bbox_inches="tight")
    plt.close()


def main():
    t0 = time.time()
    print("=" * 55)
    print("FASE 6 — Interpretación de Resultados")
    print("=" * 55)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR / "data_engineered.csv")
    feature_cols = list(ALL_FEATURES)
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    scaler    = joblib.load(MODELS_DIR / "scaler.pkl")
    model     = joblib.load(MODELS_DIR / "best_model.pkl")
    rf_model  = joblib.load(MODELS_DIR / "random_forest.pkl")

    X_train_sc = scaler.transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print("\n--- Feature Importance intrínseca (Random Forest) ---")
    rf_imp = plot_rf_importance(rf_model, feature_cols)
    print(rf_imp.head(10).to_string(index=False))

    print("\n--- Permutation Importance ---")
    perm_result = permutation_importance(
        model, X_test_sc, y_test,
        n_repeats=30, random_state=RANDOM_STATE, scoring="f1", n_jobs=-1
    )
    perm_imp = plot_permutation_importance(perm_result, feature_cols)
    print(perm_imp.head(10).to_string(index=False))

    print("\n--- SHAP Values ---")
    explainer  = shap.TreeExplainer(model)
    shap_vals_raw = explainer.shap_values(X_test_sc)
    if isinstance(shap_vals_raw, list):
        shap_vals = shap_vals_raw[1]
    elif hasattr(shap_vals_raw, 'ndim') and shap_vals_raw.ndim == 3:
        shap_vals = shap_vals_raw[:, :, 1]
    else:
        shap_vals = shap_vals_raw

    print("[1/5] SHAP summary plots...")
    plot_shap_summary(shap_vals, pd.DataFrame(X_test_sc, columns=feature_cols), feature_cols)

    top4 = perm_imp["feature"].head(4).tolist()
    print("[2/5] SHAP dependence plots...")
    plot_shap_dependence(shap_vals, X_test_sc, feature_cols, top4)

    print("[3/5] PDP plots...")
    top2_idx = [feature_cols.index(f) for f in perm_imp["feature"].head(2).tolist()]
    plot_pdp(model, X_test_sc, feature_cols, top2_idx)

    print("[4/5] Comparación de importancias...")
    plot_importance_comparison(feature_cols, rf_imp, perm_imp, shap_vals)

    print("[5/5] Análisis de errores...")
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    error_counts = pd.Series({
        "Verdadero Positivo":  int(((y_pred == 1) & (y_test.values == 1)).sum()),
        "Verdadero Negativo":  int(((y_pred == 0) & (y_test.values == 0)).sum()),
        "Falso Positivo":      int(((y_pred == 1) & (y_test.values == 0)).sum()),
        "Falso Negativo":      int(((y_pred == 0) & (y_test.values == 1)).sum()),
    })
    print(error_counts.to_string())

    print(f"\nFase 6 completada en {time.time()-t0:.1f}s — figuras en {FIGURES_DIR}")


if __name__ == "__main__":
    main()
