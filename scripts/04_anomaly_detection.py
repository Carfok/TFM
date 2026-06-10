"""Fase 4 — Detección de Anomalías (Isolation Forest, LOF, One-Class SVM).

Entrada:  data/processed/data_engineered.csv
Salida:   data/processed/data_with_anomalies.csv
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from src.config import (SENSOR_FEATURES, TARGET, FIGURES_DIR, PROCESSED_DIR, RANDOM_STATE)
from src.preprocessing import load_data
from src.features import engineer_features, ALL_FEATURES

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
sns.set_theme(style="whitegrid")

CONTAMINATION = 0.10


def run_detectors(X_scaled, X_normal):
    iso = IsolationForest(n_estimators=200, contamination=CONTAMINATION,
                          random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_scaled)
    iso_pred  = (iso.predict(X_scaled) == -1).astype(int)
    iso_score = -iso.score_samples(X_scaled)

    lof = LocalOutlierFactor(n_neighbors=20, contamination=CONTAMINATION, n_jobs=-1)
    lof_pred_raw = lof.fit_predict(X_scaled)
    lof_pred  = (lof_pred_raw == -1).astype(int)
    lof_score = -lof.negative_outlier_factor_

    ocsvm = OneClassSVM(kernel="rbf", nu=CONTAMINATION, gamma="scale")
    ocsvm.fit(X_normal)
    ocsvm_pred  = (ocsvm.predict(X_scaled) == -1).astype(int)
    ocsvm_score = -ocsvm.decision_function(X_scaled)

    return (iso_pred, iso_score), (lof_pred, lof_score), (ocsvm_pred, ocsvm_score)


def plot_pca_anomalies(X_scaled, y_true, preds_dict):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plots = [("Etiquetas reales (fail)", y_true)] + list(preds_dict.items())
    for ax, (title, labels) in zip(axes.flatten(), plots):
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels,
                        cmap="RdYlGn_r", alpha=0.5, s=10, edgecolors="none")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        plt.colorbar(sc, ax=ax, label="0=Normal / 1=Anomalía")
    plt.suptitle("PCA — Anomalías detectadas", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_pca_anomalies.png", bbox_inches="tight")
    plt.close()


def plot_confusion_grids(y_true, preds_dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (method, pred) in zip(axes.flatten(), preds_dict.items()):
        cm = confusion_matrix(y_true, pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Normal", "Anómalo"],
                    yticklabels=["Normal (real)", "Fallo (real)"])
        ax.set_title(method, fontweight="bold")
        ax.set_xlabel("Predicción anomalía")
        ax.set_ylabel("Etiqueta real")
    plt.suptitle("Matrices de confusión: anomalía vs fallo real", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_anomaly_confusion_matrices.png", bbox_inches="tight")
    plt.close()


def overlap_metrics(y_true, pred, name):
    overlap  = ((pred == 1) & (y_true == 1)).sum()
    prec     = overlap / pred.sum() if pred.sum() > 0 else 0
    recall   = overlap / y_true.sum() if y_true.sum() > 0 else 0
    f1       = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
    return {"método": name, "anomalías": pred.sum(), "∩ fallos": overlap,
            "precisión": f"{prec*100:.1f}%", "recall": f"{recall*100:.1f}%", "f1": f"{f1:.3f}"}


def main():
    t0 = time.time()
    print("=" * 55)
    print("FASE 4 — Detección de Anomalías")
    print("=" * 55)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    eng_path = PROCESSED_DIR / "data_engineered.csv"
    df = pd.read_csv(eng_path)
    print(f"Dataset cargado: {df.shape[0]} muestras × {df.shape[1]} columnas")

    X = df[ALL_FEATURES].values
    y_true = df[TARGET].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_normal = X_scaled[y_true == 0]

    print(f"Tasa de fallos: {y_true.mean()*100:.1f}% | Contaminación objetivo: {CONTAMINATION*100:.0f}%")

    print("\n--- Entrenando detectores ---")
    (iso_pred, iso_sc), (lof_pred, lof_sc), (ocs_pred, ocs_sc) = run_detectors(X_scaled, X_normal)

    consensus = ((iso_pred + lof_pred + ocs_pred) >= 2).astype(int)

    preds_dict = {
        "Isolation Forest": iso_pred,
        "LOF":              lof_pred,
        "One-Class SVM":    ocs_pred,
        "Consenso (≥2/3)":  consensus,
    }

    print("\n--- Resumen de anomalías detectadas ---")
    rows = [overlap_metrics(y_true, pred, name) for name, pred in preds_dict.items()]
    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))

    print("\n[1/2] Matrices de confusión...")
    plot_confusion_grids(y_true, preds_dict)

    print("[2/2] Proyección PCA...")
    plot_pca_anomalies(X_scaled, y_true, preds_dict)

    df_out = df.copy()
    df_out["iso_anomaly"]    = iso_pred
    df_out["lof_anomaly"]    = lof_pred
    df_out["ocsvm_anomaly"]  = ocs_pred
    df_out["consensus_anomaly"] = consensus
    df_out["iso_score"]      = iso_sc
    df_out["lof_score"]      = lof_sc
    df_out["ocsvm_score"]    = ocs_sc

    output = PROCESSED_DIR / "data_with_anomalies.csv"
    df_out.to_csv(output, index=False)
    print(f"\nDataset con anomalías guardado: {output}")
    print(f"Fase 4 completada en {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
