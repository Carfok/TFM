"""Fase 3 — Ingeniería de Características.

Salida: data/processed/data_engineered.csv
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from matplotlib.patches import Patch

from src.config import (RAW_DATA, SENSOR_FEATURES, TARGET, FIGURES_DIR,
                        PROCESSED_DIR, RANDOM_STATE, CV_FOLDS)
from src.preprocessing import load_data
from src.features import engineer_features, ENGINEERED_FEATURES, ALL_FEATURES

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
sns.set_theme(style="whitegrid")


def plot_engineered_distributions(df_eng):
    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    for i, feat in enumerate(ENGINEERED_FEATURES):
        ax = axes[i // 3, i % 3]
        for cls, color, label in [(0, "#3498db", "Normal"), (1, "#e74c3c", "Fallo")]:
            ax.hist(df_eng[df_eng[TARGET] == cls][feat], bins=20, alpha=0.6,
                    color=color, label=label, density=True, edgecolor="white")
        ax.set_title(feat, fontweight="bold")
        ax.set_xlabel("Valor")
        ax.legend(fontsize=9)
    plt.suptitle("Características ingenierizadas por clase", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_engineered_distributions.png", bbox_inches="tight")
    plt.close()


def plot_temp_bucket_fail_rate(df_eng):
    fail_rate = (df_eng.groupby("temp_bucket", observed=True)[TARGET]
                 .agg(["mean", "count"]).rename(columns={"mean": "tasa", "count": "n"}))
    fail_rate["tasa_pct"] = fail_rate["tasa"] * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    bars = ax.bar(fail_rate.index, fail_rate["tasa_pct"],
                  color=colors[:len(fail_rate)], edgecolor="white")
    for bar, (_, row) in zip(bars, fail_rate.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{row['tasa_pct']:.1f}%\n(n={int(row['n'])})", ha="center", fontsize=10)
    ax.set_title("Tasa de fallo por rango de temperatura", fontweight="bold")
    ax.set_xlabel("Bucket de temperatura")
    ax.set_ylabel("Tasa de fallo (%)")
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_temp_bucket_fail_rate.png", bbox_inches="tight")
    plt.close()


def plot_mutual_info(df_eng, y):
    X_all = df_eng[ALL_FEATURES]
    mi = mutual_info_classif(X_all, y, random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({"feature": ALL_FEATURES, "mi": mi}).sort_values("mi", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#e74c3c" if f in ENGINEERED_FEATURES else "#3498db" for f in mi_df["feature"]]
    ax.barh(mi_df["feature"], mi_df["mi"], color=colors)
    ax.set_title("Información mutua con 'fail'", fontweight="bold")
    ax.set_xlabel("Información mutua")
    legend_elements = [Patch(facecolor="#3498db", label="Original"),
                       Patch(facecolor="#e74c3c", label="Ingenierizada")]
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_mutual_info.png", bbox_inches="tight")
    plt.close()
    return mi_df


def compare_feature_sets(df, df_eng, y):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                random_state=RANDOM_STATE, n_jobs=-1)
    orig = cross_val_score(rf, df[SENSOR_FEATURES], y, cv=cv, scoring="f1")
    engi = cross_val_score(rf, df_eng[ALL_FEATURES], y, cv=cv, scoring="f1")
    return orig, engi


def main():
    t0 = time.time()
    print("=" * 55)
    print("FASE 3 — Ingeniería de Características")
    print("=" * 55)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    df_eng = engineer_features(df)
    y = df_eng[TARGET]

    print(f"\nCaracterísticas originales: {len(SENSOR_FEATURES)}")
    print(f"Características tras ingeniería: {len(ALL_FEATURES)}")
    print(f"\nNuevas características: {ENGINEERED_FEATURES}")

    print("\n--- Estadísticas nuevas características ---")
    print(df_eng[ENGINEERED_FEATURES].describe().round(3).to_string())

    print("\n--- Tasa de fallo por bucket de temperatura ---")
    fail_rate = (df_eng.groupby("temp_bucket", observed=True)[TARGET]
                 .agg(["mean", "count"]).rename(columns={"mean": "tasa", "count": "n"}))
    fail_rate["tasa_pct"] = (fail_rate["tasa"] * 100).round(1)
    print(fail_rate.to_string())

    print("\n[1/3] Distribuciones de características ingenierizadas...")
    plot_engineered_distributions(df_eng)

    print("[2/3] Tasa de fallo por temperatura...")
    plot_temp_bucket_fail_rate(df_eng)

    print("[3/3] Información mutua...")
    mi_df = plot_mutual_info(df_eng, y)
    print("\nTop 10 características por información mutua:")
    print(mi_df.head(10).to_string(index=False))

    print("\n--- Comparación F1 CV (RandomForest) ---")
    orig_scores, eng_scores = compare_feature_sets(df, df_eng, y)
    print(f"  Originales:       {orig_scores.mean():.4f} ± {orig_scores.std():.4f}")
    print(f"  + Ingenierizadas: {eng_scores.mean():.4f} ± {eng_scores.std():.4f}")
    delta = eng_scores.mean() - orig_scores.mean()
    print(f"  Mejora: {delta:+.4f}")

    output = PROCESSED_DIR / "data_engineered.csv"
    df_eng.to_csv(output, index=False)
    print(f"\nDataset guardado: {output}")
    print(f"Fase 3 completada en {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
