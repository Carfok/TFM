"""Fase 1 — Análisis Exploratorio de Datos (EDA)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

from src.config import RAW_DATA, SENSOR_FEATURES, TARGET, FIGURES_DIR
from src.preprocessing import load_data

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
sns.set_theme(style="whitegrid", palette="muted")


def plot_class_distribution(df):
    fail_counts = df[TARGET].value_counts().sort_index()
    fail_pct = fail_counts / len(df) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bars = axes[0].bar(["Normal (0)", "Fallo (1)"], fail_counts.values,
                       color=["#2ecc71", "#e74c3c"], edgecolor="white")
    for bar, count, pct in zip(bars, fail_counts.values, fail_pct.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f"{count}\n({pct:.1f}%)", ha="center", va="bottom", fontweight="bold")
    axes[0].set_title("Distribución de la variable objetivo", fontweight="bold")
    axes[0].set_ylabel("Muestras")
    axes[0].set_ylim(0, fail_counts.max() * 1.25)

    axes[1].pie(fail_counts.values, labels=["Normal", "Fallo"],
                colors=["#2ecc71", "#e74c3c"], autopct="%1.1f%%",
                startangle=90, textprops={"fontsize": 12})
    axes[1].set_title("Proporción de clases", fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_class_distribution.png", bbox_inches="tight")
    plt.close()


def plot_sensor_distributions(df):
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for i, col in enumerate(SENSOR_FEATURES):
        ax = axes[i // 3, i % 3]
        for cls, color, label in [(0, "#3498db", "Normal"), (1, "#e74c3c", "Fallo")]:
            ax.hist(df[df[TARGET] == cls][col], bins=20, alpha=0.6, color=color,
                    label=label, density=True, edgecolor="white")
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("Valor")
        ax.set_ylabel("Densidad")
        ax.legend(fontsize=9)
    plt.suptitle("Distribución de sensores por clase", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_sensor_distributions.png", bbox_inches="tight")
    plt.close()


def plot_boxplots(df):
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for i, col in enumerate(SENSOR_FEATURES):
        ax = axes[i // 3, i % 3]
        df.boxplot(column=col, by=TARGET, ax=ax,
                   boxprops=dict(color="#2c3e50"),
                   medianprops=dict(color="#e74c3c", linewidth=2),
                   whiskerprops=dict(color="#2c3e50"),
                   capprops=dict(color="#2c3e50"))
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("Fallo")
        ax.set_ylabel("Valor")
    plt.suptitle("Boxplots por clase", fontsize=14, fontweight="bold")
    fig.suptitle("")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_boxplots_by_class.png", bbox_inches="tight")
    plt.close()


def plot_correlation_matrix(df):
    corr = df[SENSOR_FEATURES + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=-1, vmax=1, center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Matriz de correlaciones (Pearson)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_correlation_matrix.png", bbox_inches="tight")
    plt.close()


def plot_correlation_barplot(df):
    corr = df[SENSOR_FEATURES + [TARGET]].corr()
    corr_with_fail = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    corr_with_fail.plot(kind="barh", ax=ax,
                        color=["#e74c3c" if v > 0 else "#3498db" for v in corr_with_fail])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Correlación de cada sensor con 'fail'", fontweight="bold")
    ax.set_xlabel("Correlación de Pearson")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_correlation_with_fail.png", bbox_inches="tight")
    plt.close()
    return corr_with_fail


def plot_pairplot(df):
    selected = ["CS", "RP", "Temperature", "IP", TARGET]
    pair_df = df[selected].copy()
    pair_df[TARGET] = pair_df[TARGET].map({0: "Normal", 1: "Fallo"})
    g = sns.pairplot(pair_df, hue=TARGET,
                     palette={"Normal": "#3498db", "Fallo": "#e74c3c"},
                     plot_kws={"alpha": 0.5, "s": 20}, diag_kind="kde")
    g.fig.suptitle("Pair Plot — Variables clave", y=1.02, fontweight="bold")
    g.fig.savefig(FIGURES_DIR / "01_pairplot.png", bbox_inches="tight")
    plt.close()


def run_significance_tests(df):
    results = []
    for col in SENSOR_FEATURES:
        g0 = df[df[TARGET] == 0][col]
        g1 = df[df[TARGET] == 1][col]
        _, p = mannwhitneyu(g0, g1, alternative="two-sided")
        results.append({"sensor": col, "p_value": round(p, 6),
                        "significativo": p < 0.05})
    return pd.DataFrame(results).sort_values("p_value")


def main():
    t0 = time.time()
    print("=" * 55)
    print("FASE 1 — Análisis Exploratorio de Datos")
    print("=" * 55)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    print(f"Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")

    print("\n--- Información general ---")
    print(df.describe().round(2).to_string())

    n_fail = df[TARGET].sum()
    print(f"\nDistribución objetivo: {int(n_fail)} fallos ({n_fail/len(df)*100:.1f}%) "
          f"/ {len(df)-int(n_fail)} normales ({(1-n_fail/len(df))*100:.1f}%)")

    print("\n[1/6] Distribución de clases...")
    plot_class_distribution(df)

    print("[2/6] Distribuciones de sensores...")
    plot_sensor_distributions(df)

    print("[3/6] Boxplots por clase...")
    plot_boxplots(df)

    print("[4/6] Matriz de correlaciones...")
    plot_correlation_matrix(df)

    print("[5/6] Correlación con fail...")
    corr = plot_correlation_barplot(df)
    print(corr.round(3).to_string())

    print("[6/6] Pair plot...")
    plot_pairplot(df)

    print("\n--- Tests de significancia (Mann-Whitney U) ---")
    sig = run_significance_tests(df)
    print(sig.to_string(index=False))

    print(f"\nFase 1 completada en {time.time()-t0:.1f}s — figuras en {FIGURES_DIR}")


if __name__ == "__main__":
    main()
