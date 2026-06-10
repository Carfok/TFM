"""Fase 7 — Monitorización del rendimiento del modelo y sistema de alertas.

Implementa:
  1. Seguimiento de métricas del modelo en producción
  2. Comparación e interpretación de todos los modelos entrenados
  3. Detección de data drift (cambio en la distribución de los datos)
  4. Sistema de alertas automáticas por umbral
  5. Generación de informe JSON completo (métricas + interpretación + drift + alertas)

Entrada:  data/processed/data_engineered.csv  + models/
Salidas:  reports/monitoring_report.json
          reports/figures/07_drift_distributions.png
          reports/figures/07_alert_dashboard.png
"""
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import shap
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix

from src.config import (
    TARGET, SENSOR_FEATURES, FIGURES_DIR, MODELS_DIR, PROCESSED_DIR,
    RANDOM_STATE, REPORTS_DIR,
)
from src.features import ALL_FEATURES

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest":       "random_forest.pkl",
    "XGBoost":             "xgboost.pkl",
    "LightGBM":            "lightgbm.pkl",
}

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
sns.set_theme(style="whitegrid")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Umbrales de alerta ────────────────────────────────────────────────────────
ALERT_THRESHOLDS = {
    "f1_min":             0.75,   # F1 mínimo aceptable
    "roc_auc_min":        0.80,   # AUC mínimo aceptable
    "recall_min":         0.70,   # Recall mínimo (fallos no detectados)
    "drift_pvalue_min":   0.05,   # p-valor mínimo KS test (< → drift)
    "failure_rate_max":   0.60,   # Tasa de fallos máxima antes de alertar
    "high_risk_pct_max":  0.30,   # % registros alto riesgo antes de alertar
}


# ── Alertas ───────────────────────────────────────────────────────────────────
class AlertSystem:
    def __init__(self):
        self.alerts: list[dict] = []

    def check(self, name: str, value: float, threshold: float, mode: str = "min", critical: bool = False):
        """Registra una alerta si el valor supera el umbral."""
        triggered = (value < threshold) if mode == "min" else (value > threshold)
        level = "CRITICA" if (triggered and critical) else ("ALERTA" if triggered else "OK")
        self.alerts.append({
            "metric": name,
            "value": round(float(value), 4),
            "threshold": threshold,
            "mode": mode,
            "status": level,
            "timestamp": datetime.now().isoformat(),
        })
        icon = "[CRIT]" if level == "CRITICA" else ("[WARN]" if level == "ALERTA" else "[ OK ]")
        op = ">=" if mode == "min" else "<="
        print(f"  {icon} {name}: {value:.4f} (umbral {op} {threshold}) -> {level}")
        return triggered

    def summary(self) -> dict:
        n_ok = sum(1 for a in self.alerts if a["status"] == "OK")
        n_warn = sum(1 for a in self.alerts if a["status"] == "ALERTA")
        n_crit = sum(1 for a in self.alerts if a["status"] == "CRITICA")
        return {"ok": n_ok, "warnings": n_warn, "critical": n_crit, "total": len(self.alerts)}


# ── Interpretación comparativa de todos los modelos ──────────────────────────
def interpret_all_models(scaler, df: pd.DataFrame) -> dict:
    """
    Carga todos los modelos entrenados y genera:
      - Métricas comparativas en test
      - Feature importances (modelos de árbol)
      - SHAP mean |values| del mejor modelo
      - Permutation importance del mejor modelo
      - Matriz de confusión por modelo
      - Conclusiones automáticas
    """
    print("\n[0/4] Interpretación comparativa de modelos")

    X = df[ALL_FEATURES]
    y = df[TARGET]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_test_sc = scaler.transform(X_test)
    feature_cols = list(ALL_FEATURES)

    models_loaded = {}
    for name, fname in MODEL_FILES.items():
        path = MODELS_DIR / fname
        if path.exists():
            models_loaded[name] = joblib.load(path)

    if not models_loaded:
        print("  [AVISO] No se encontraron modelos entrenados.")
        return {}

    # ── Métricas y matrices de confusión por modelo ───────────────────────────
    per_model_metrics = {}
    per_model_cm = {}
    for name, model in models_loaded.items():
        y_prob = model.predict_proba(X_test_sc)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred).tolist()
        per_model_metrics[name] = {
            "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "tp": cm[1][1], "tn": cm[0][0], "fp": cm[0][1], "fn": cm[1][0],
        }
        per_model_cm[name] = cm
        print(f"  {name}: F1={per_model_metrics[name]['f1']:.4f}  AUC={per_model_metrics[name]['roc_auc']:.4f}")

    # ── Ranking de modelos ────────────────────────────────────────────────────
    ranking = sorted(per_model_metrics.items(), key=lambda x: x[1]["f1"], reverse=True)
    best_name = ranking[0][0]
    best_model = models_loaded[best_name]
    print(f"  Mejor modelo (F1): {best_name}")

    # ── Feature importances intrínsecas (árboles) ─────────────────────────────
    tree_importances = {}
    for name, model in models_loaded.items():
        if hasattr(model, "feature_importances_"):
            imp = dict(zip(feature_cols,
                           [round(float(v), 6) for v in model.feature_importances_]))
            tree_importances[name] = dict(
                sorted(imp.items(), key=lambda x: x[1], reverse=True)
            )

    # ── SHAP values del mejor modelo ──────────────────────────────────────────
    shap_importance = {}
    try:
        explainer = shap.TreeExplainer(best_model)
        shap_raw = explainer.shap_values(X_test_sc)
        if isinstance(shap_raw, list):
            sv = shap_raw[1]
        elif hasattr(shap_raw, "ndim") and shap_raw.ndim == 3:
            sv = shap_raw[:, :, 1]
        else:
            sv = shap_raw
        mean_abs = np.abs(sv).mean(axis=0)
        shap_importance = dict(
            sorted(zip(feature_cols, [round(float(v), 6) for v in mean_abs]),
                   key=lambda x: x[1], reverse=True)
        )
        print(f"  SHAP calculado para {best_name}")
    except Exception as e:
        print(f"  [AVISO] SHAP no disponible: {e}")

    # ── Permutation importance del mejor modelo ───────────────────────────────
    perm_importance = {}
    try:
        perm = permutation_importance(
            best_model, X_test_sc, y_test,
            n_repeats=15, random_state=RANDOM_STATE, scoring="f1", n_jobs=-1
        )
        perm_importance = dict(
            sorted(zip(feature_cols, [round(float(v), 6) for v in perm.importances_mean]),
                   key=lambda x: x[1], reverse=True)
        )
        print(f"  Permutation importance calculada para {best_name}")
    except Exception as e:
        print(f"  [AVISO] Permutation importance no disponible: {e}")

    # ── Conclusiones automáticas ──────────────────────────────────────────────
    top3_shap = list(shap_importance.keys())[:3] if shap_importance else []
    top3_perm = list(perm_importance.keys())[:3] if perm_importance else []
    top3_common = [f for f in top3_shap if f in top3_perm]

    f1_spread = ranking[0][1]["f1"] - ranking[-1][1]["f1"]
    conclusions = {
        "best_model": best_name,
        "worst_model": ranking[-1][0],
        "f1_spread_between_models": round(f1_spread, 4),
        "top3_shap_features": top3_shap,
        "top3_permutation_features": top3_perm,
        "features_consistently_important": top3_common,
        "interpretation": (
            f"El modelo {best_name} obtiene el mejor F1 ({ranking[0][1]['f1']:.4f}). "
            f"La diferencia entre el mejor y peor modelo es de {f1_spread:.4f} puntos. "
            + (f"Las características más relevantes según SHAP y permutación son: {', '.join(top3_common)}."
               if top3_common else "No hay consenso claro entre métodos de importancia.")
        ),
    }

    return {
        "best_model": best_name,
        "model_ranking": [{"model": n, **m} for n, m in ranking],
        "per_model_metrics": per_model_metrics,
        "confusion_matrices": per_model_cm,
        "feature_importances": {
            "tree_based": tree_importances,
            "shap_mean_abs": {best_name: shap_importance},
            "permutation": {best_name: perm_importance},
        },
        "conclusions": conclusions,
    }


# ── 1. Evaluación de métricas del modelo ─────────────────────────────────────
def evaluate_model_metrics(model, scaler, df: pd.DataFrame, alerts: AlertSystem) -> dict:
    print("\n[1/4] Métricas del modelo en datos actuales")
    X = df[ALL_FEATURES]
    y = df[TARGET]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    X_test_sc = scaler.transform(X_test)
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "f1":       f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":  roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":   recall_score(y_test, y_pred, zero_division=0),
    }

    alerts.check("F1-score",  metrics["f1"],      ALERT_THRESHOLDS["f1_min"],      "min", critical=True)
    alerts.check("ROC-AUC",   metrics["roc_auc"], ALERT_THRESHOLDS["roc_auc_min"], "min")
    alerts.check("Recall",    metrics["recall"],  ALERT_THRESHOLDS["recall_min"],  "min", critical=True)

    return metrics


# ── 2. Detección de data drift ────────────────────────────────────────────────
def detect_data_drift(df_ref: pd.DataFrame, df_new: pd.DataFrame, alerts: AlertSystem) -> dict:
    """
    Compara la distribución de cada sensor entre el dataset de referencia
    y el dataset actual usando el test KS (Kolmogorov-Smirnov).
    """
    print("\n[2/4] Detección de data drift (KS test)")
    drift_results = {}

    for feat in SENSOR_FEATURES:
        stat, pvalue = stats.ks_2samp(df_ref[feat].dropna(), df_new[feat].dropna())
        drift = pvalue < ALERT_THRESHOLDS["drift_pvalue_min"]
        drift_results[feat] = {
            "ks_statistic": round(float(stat), 4),
            "p_value": round(float(pvalue), 4),
            "drift_detected": bool(drift),
        }
        icon = "[DRIFT]" if drift else "[  OK ]"
        print(f"  {feat:15s} | KS={stat:.4f}  p={pvalue:.4f}  -> {icon}")

    n_drift = sum(1 for v in drift_results.values() if v["drift_detected"])
    if n_drift > 0:
        alerts.alerts.append({
            "metric": "data_drift",
            "value": n_drift,
            "threshold": 0,
            "mode": "max",
            "status": "ALERTA" if n_drift < 3 else "CRITICA",
            "detail": f"{n_drift} feature(s) con drift detectado",
            "timestamp": datetime.now().isoformat(),
        })
        print(f"  [WARN] {n_drift} features con drift detectado")

    return drift_results


# ── 3. Análisis de distribución de predicciones ───────────────────────────────
def analyze_prediction_distribution(model, scaler, df: pd.DataFrame, alerts: AlertSystem) -> dict:
    print("\n[3/4] Distribución de predicciones en producción")
    X_sc = scaler.transform(df[ALL_FEATURES])
    probs = model.predict_proba(X_sc)[:, 1]
    preds = (probs >= 0.5).astype(int)

    failure_rate = float(preds.mean())
    high_risk_pct = float((probs > 0.7).mean())
    avg_prob = float(probs.mean())

    print(f"  Tasa de fallos predicha:   {failure_rate*100:.1f}%")
    print(f"  Registros de alto riesgo:  {high_risk_pct*100:.1f}%")
    print(f"  Probabilidad media:        {avg_prob:.4f}")

    alerts.check("Tasa de fallos predicha",
                 failure_rate, ALERT_THRESHOLDS["failure_rate_max"], "max")
    alerts.check("% alto riesgo (>70%)",
                 high_risk_pct, ALERT_THRESHOLDS["high_risk_pct_max"], "max")

    return {
        "failure_rate": round(failure_rate, 4),
        "high_risk_pct": round(high_risk_pct, 4),
        "avg_failure_prob": round(avg_prob, 4),
        "probs": probs,
    }


# ── 4. Visualizaciones de monitorización ─────────────────────────────────────
def plot_drift_report(df_ref: pd.DataFrame, df_new: pd.DataFrame, drift_results: dict):
    """Histogramas comparativos por feature para visualizar drift."""
    features_to_plot = SENSOR_FEATURES[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Detección de Data Drift — Distribuciones ref. vs actual", fontsize=13, fontweight="bold")

    for ax, feat in zip(axes.flatten(), features_to_plot):
        ax.hist(df_ref[feat].dropna(), bins=20, alpha=0.6, label="Referencia", color="#3498db", density=True)
        ax.hist(df_new[feat].dropna(), bins=20, alpha=0.6, label="Actual", color="#e74c3c", density=True)
        dr = drift_results[feat]
        drift_label = "[DRIFT]" if dr["drift_detected"] else "[ OK ]"
        ax.set_title(f"{feat}  |  KS={dr['ks_statistic']:.3f}  p={dr['p_value']:.3f}  {drift_label}",
                     fontsize=9)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "07_drift_distributions.png", bbox_inches="tight")
    plt.close()
    print("  -> 07_drift_distributions.png guardado")


def plot_alert_dashboard(metrics: dict, pred_stats: dict, drift_results: dict, alerts: AlertSystem):
    """Panel resumen de alertas y métricas clave."""
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle(f"Dashboard de Monitorización — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                 fontsize=14, fontweight="bold")

    # 1. KPIs de métricas del modelo
    ax1 = fig.add_subplot(gs[0, 0])
    metric_names = list(metrics.keys())
    metric_vals = list(metrics.values())
    colors = ["#2ecc71" if v >= 0.75 else "#e74c3c" for v in metric_vals]
    bars = ax1.barh(metric_names, metric_vals, color=colors)
    ax1.set_xlim(0, 1.1)
    ax1.axvline(0.75, color="gray", linestyle="--", alpha=0.7, label="Umbral mín.")
    ax1.set_title("Métricas del modelo", fontweight="bold")
    ax1.legend(fontsize=8)
    for bar, val in zip(bars, metric_vals):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.3f}",
                 va="center", fontsize=9)

    # 2. Resumen de alertas
    ax2 = fig.add_subplot(gs[0, 1])
    summary = alerts.summary()
    wedges, texts, autotexts = ax2.pie(
        [summary["ok"], summary["warnings"], summary["critical"]],
        labels=["OK", "Alertas", "Críticas"],
        colors=["#2ecc71", "#f1c40f", "#e74c3c"],
        autopct="%1.0f%%",
        startangle=90,
    )
    ax2.set_title("Estado de alertas", fontweight="bold")

    # 3. Distribución de predicciones
    ax3 = fig.add_subplot(gs[0, 2])
    probs = pred_stats["probs"]
    ax3.hist(probs, bins=20, color="#3498db", edgecolor="white", alpha=0.85)
    ax3.axvline(0.5, color="#e74c3c", linestyle="--", label="Umbral 0.5")
    ax3.axvline(0.7, color="#e67e22", linestyle="--", label="Alto riesgo 0.7")
    ax3.set_xlabel("Probabilidad de fallo")
    ax3.set_ylabel("Frecuencia")
    ax3.set_title("Distribución de probabilidades", fontweight="bold")
    ax3.legend(fontsize=8)

    # 4. Drift: estadístico KS por feature
    ax4 = fig.add_subplot(gs[1, :2])
    feats = list(drift_results.keys())
    ks_stats = [v["ks_statistic"] for v in drift_results.values()]
    bar_colors = ["#e74c3c" if v["drift_detected"] else "#2ecc71" for v in drift_results.values()]
    ax4.bar(feats, ks_stats, color=bar_colors)
    ax4.axhline(0.1, color="gray", linestyle="--", alpha=0.7, label="Umbral KS referencia")
    ax4.set_title("Estadístico KS por feature (rojo = drift)", fontweight="bold")
    ax4.set_ylabel("KS statistic")
    ax4.tick_params(axis="x", rotation=30)
    ax4.legend(fontsize=8)

    # 5. Tabla de alertas activas
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    active_alerts = [a for a in alerts.alerts if a["status"] != "OK"]
    if active_alerts:
        table_data = [[a["metric"][:20], f"{a['value']:.3f}", a["status"]] for a in active_alerts[:6]]
        tbl = ax5.table(
            cellText=table_data,
            colLabels=["Métrica", "Valor", "Estado"],
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", fontweight="bold")
            elif table_data[r-1][2] == "CRITICA":
                cell.set_facecolor("#fdedec")
            elif table_data[r-1][2] == "ALERTA":
                cell.set_facecolor("#fef9e7")
        ax5.set_title("Alertas activas", fontweight="bold")
    else:
        ax5.text(0.5, 0.5, "✓ Sin alertas activas", ha="center", va="center",
                 fontsize=13, color="#2ecc71", fontweight="bold")
        ax5.set_title("Alertas activas", fontweight="bold")

    plt.savefig(FIGURES_DIR / "07_alert_dashboard.png", bbox_inches="tight")
    plt.close()
    print("  -> 07_alert_dashboard.png guardado")


# ── 5. Informe JSON ───────────────────────────────────────────────────────────
def save_report(metrics: dict, pred_stats: dict, drift_results: dict,
                alerts: AlertSystem, interpretation: dict):
    report = {
        "timestamp": datetime.now().isoformat(),
        "best_model_metrics": {k: round(v, 4) for k, v in metrics.items()},
        "prediction_distribution": {
            k: v for k, v in pred_stats.items() if k != "probs"
        },
        "model_interpretation": interpretation,
        "data_drift": drift_results,
        "alerts": alerts.alerts,
        "alert_summary": alerts.summary(),
        "thresholds_used": ALERT_THRESHOLDS,
    }
    out_path = REPORTS_DIR / "monitoring_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  -> Informe JSON guardado: {out_path}")
    return report


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 55)
    print("FASE 7 — Monitorización y Sistema de Alertas")
    print("=" * 55)

    # Carga de artefactos
    try:
        model = joblib.load(MODELS_DIR / "best_model.pkl")
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    except FileNotFoundError:
        print("ERROR: Modelos no encontrados. Ejecuta primero scripts/05_models.py")
        sys.exit(1)

    df = pd.read_csv(PROCESSED_DIR / "data_engineered.csv")
    print(f"Dataset cargado: {len(df)} muestras")

    # Simulación de datos de producción: particionamos el dataset
    # En producción esto vendría de un stream o batch nuevo
    df_ref = df.iloc[:int(len(df) * 0.6)].copy()  # 60% = datos de referencia (entrenamiento)
    df_new = df.iloc[int(len(df) * 0.6):].copy()  # 40% = datos "en producción"

    print(f"Dataset referencia: {len(df_ref)} muestras | Producción: {len(df_new)} muestras")

    alerts = AlertSystem()

    interpretation = interpret_all_models(scaler, df)
    metrics = evaluate_model_metrics(model, scaler, df_new, alerts)
    drift_results = detect_data_drift(df_ref, df_new, alerts)
    pred_stats = analyze_prediction_distribution(model, scaler, df_new, alerts)

    # Visualizaciones
    print("\n[4/4] Generando visualizaciones")
    plot_drift_report(df_ref, df_new, drift_results)
    plot_alert_dashboard(metrics, pred_stats, drift_results, alerts)

    # Informe
    report = save_report(metrics, pred_stats, drift_results, alerts, interpretation)

    # Resumen final
    summary = alerts.summary()
    print("\n" + "=" * 55)
    print("RESUMEN DE MONITORIZACIÓN")
    print("=" * 55)
    print(f"  Checks OK:      {summary['ok']}")
    print(f"  Alertas:        {summary['warnings']}")
    print(f"  Críticas:       {summary['critical']}")
    print(f"  F1 actual:      {metrics['f1']:.4f}")
    print(f"  ROC-AUC actual: {metrics['roc_auc']:.4f}")
    drift_count = sum(1 for v in drift_results.values() if v["drift_detected"])
    print(f"  Features con drift: {drift_count}/{len(drift_results)}")

    if summary["critical"] > 0:
        print("\n[CRIT] ACCION REQUERIDA: hay alertas criticas. Considera reentrenar el modelo.")
    elif summary["warnings"] > 0:
        print("\n[WARN] Atencion: hay alertas activas. Monitoriza la evolucion.")
    else:
        print("\n[ OK ] Sistema en estado nominal. Sin alertas activas.")

    print(f"\nFase 7 completada en {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
