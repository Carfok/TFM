"""Fase 2 — Calidad y Preparación de los Datos."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.config import RAW_DATA, SENSOR_FEATURES, TARGET, FIGURES_DIR
from src.preprocessing import (load_data, check_missing, check_duplicates,
                                detect_outliers_iqr, detect_outliers_zscore)

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
sns.set_theme(style="whitegrid")

EXPECTED_RANGES = {
    "footfall": (0, 7300), "tempMode": (0, 7), "AQ": (1, 7),
    "USS": (1, 7), "CS": (1, 7), "VOC": (0, 6),
    "RP": (19, 91), "IP": (1, 7), "Temperature": (1, 24), "fail": (0, 1),
}


def plot_outlier_boxplots(df):
    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    for i, col in enumerate(SENSOR_FEATURES):
        ax = axes[i // 3, i % 3]
        ax.boxplot(df[col], patch_artist=True,
                   boxprops=dict(facecolor="#3498db", alpha=0.6),
                   medianprops=dict(color="#e74c3c", linewidth=2),
                   flierprops=dict(marker="o", color="#e74c3c", markersize=4, alpha=0.7))
        ax.set_title(col, fontweight="bold")
        ax.set_ylabel("Valor")
    plt.suptitle("Boxplots — Detección de outliers (IQR)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_outlier_boxplots.png", bbox_inches="tight")
    plt.close()


def plot_footfall_analysis(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].hist(df["footfall"], bins=40, color="#3498db", edgecolor="white", alpha=0.8)
    axes[0].set_title("Distribución de footfall", fontweight="bold")
    axes[0].set_xlabel("Valor")
    axes[0].set_ylabel("Frecuencia")

    axes[1].boxplot(df["footfall"], patch_artist=True,
                    boxprops=dict(facecolor="#3498db", alpha=0.6),
                    medianprops=dict(color="#e74c3c", linewidth=2),
                    flierprops=dict(marker="o", color="#e74c3c", markersize=4))
    axes[1].set_title("Boxplot footfall", fontweight="bold")

    stats.probplot(df["footfall"], dist="norm", plot=axes[2])
    axes[2].set_title("Q-Q Plot footfall", fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_footfall_analysis.png", bbox_inches="tight")
    plt.close()


def validate_ranges(df):
    results = []
    for col, (lo, hi) in EXPECTED_RANGES.items():
        out = ((df[col] < lo) | (df[col] > hi)).sum()
        results.append({
            "columna": col,
            "rango_esperado": f"[{lo}, {hi}]",
            "min_observado": df[col].min(),
            "max_observado": df[col].max(),
            "fuera_de_rango": out,
            "ok": out == 0,
        })
    return pd.DataFrame(results)


def main():
    t0 = time.time()
    print("=" * 55)
    print("FASE 2 — Calidad y Preparación de los Datos")
    print("=" * 55)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    print("\n--- Valores nulos ---")
    missing = check_missing(df)
    if missing.empty:
        print("✓ Sin valores nulos.")
    else:
        print(missing.to_string())

    print("\n--- Duplicados ---")
    dup = check_duplicates(df)
    if dup["n_duplicates"] == 0:
        print("✓ Sin registros duplicados.")
    else:
        print(f"⚠ {dup['n_duplicates']} duplicados ({dup['pct']:.2f}%)")

    print("\n--- Outliers IQR ---")
    iqr_df = detect_outliers_iqr(df)
    print(iqr_df[["feature", "lower_fence", "upper_fence", "n_outliers", "pct_outliers"]].to_string(index=False))

    print("\n--- Outliers Z-Score (umbral 3σ) ---")
    z_df = detect_outliers_zscore(df)
    print(z_df[["feature", "n_outliers", "pct_outliers"]].to_string(index=False))

    print("\n--- Análisis footfall ---")
    print(f"  Skewness: {df['footfall'].skew():.3f}")
    print(f"  Curtosis: {df['footfall'].kurt():.3f}")
    print(f"  Percentil 95: {df['footfall'].quantile(0.95):.0f}")
    print(f"  Valores > 2000: {(df['footfall'] > 2000).sum()}")

    print("\n--- Validación de rangos ---")
    ranges_df = validate_ranges(df)
    for _, row in ranges_df.iterrows():
        status = "✓" if row["ok"] else "⚠"
        print(f"  {status} {row['columna']:12s} rango {row['rango_esperado']:12s} "
              f"obs [{row['min_observado']}, {row['max_observado']}]  "
              f"fuera: {row['fuera_de_rango']}")

    print("\n[1/2] Boxplots de outliers...")
    plot_outlier_boxplots(df)
    print("[2/2] Análisis footfall...")
    plot_footfall_analysis(df)

    print(f"\nFase 2 completada en {time.time()-t0:.1f}s — figuras en {FIGURES_DIR}")


if __name__ == "__main__":
    main()
