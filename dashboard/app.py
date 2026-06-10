"""
Dashboard — Sistema Inteligente de Monitorización de Fallos Industriales
Ejecutar con: streamlit run dashboard/app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings("ignore")

from src.config import (RAW_DATA, SENSOR_FEATURES, TARGET, SENSOR_DESCRIPTIONS,
                        HEALTH_LEVELS, HEALTH_COLORS, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE)
from src.preprocessing import load_data
from src.features import engineer_features, ALL_FEATURES, ENGINEERED_FEATURES
from src.health_index import add_health_index, classify_health_level, get_health_color

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Monitor Industrial ML",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        border-radius: 12px; padding: 20px; color: white; text-align: center;
    }
    .health-badge {
        border-radius: 8px; padding: 6px 14px; font-weight: bold;
        display: inline-block; font-size: 14px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px 6px 0 0; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_full_data():
    eng_path = PROCESSED_DIR / "data_engineered.csv"
    if eng_path.exists():
        df = pd.read_csv(eng_path)
    else:
        df = engineer_features(load_data())
    return df


@st.cache_resource
def load_models():
    try:
        model  = joblib.load(MODELS_DIR / "best_model.pkl")
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        return model, scaler, True
    except FileNotFoundError:
        return None, None, False


df = load_full_data()
model, scaler, models_available = load_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/maintenance.png", width=70)
    st.title("Monitor Industrial ML")
    st.caption("TFM — Predicción de Fallos en Equipos")
    st.divider()
    page = st.radio(
        "Navegación",
        ["Resumen General", "Explorador de Datos", "Detección de Anomalías",
         "Comparación de Modelos", "Rendimiento del Modelo", "Índice de Salud"],
        index=0,
    )
    st.divider()
    st.caption(f"Dataset: {len(df)} muestras")
    fail_rate = df[TARGET].mean() * 100
    st.metric("Tasa de fallos global", f"{fail_rate:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: RESUMEN GENERAL
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Resumen General":
    st.title("⚙️ Monitor de Fallos — Resumen General")
    st.caption("Vista global del estado de los equipos y alertas principales")

    # KPIs
    total = len(df)
    n_fail = df[TARGET].sum()
    n_ok   = total - n_fail

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de registros", f"{total:,}")
    with col2:
        st.metric("Fallos detectados", f"{int(n_fail):,}", delta=f"{n_fail/total*100:.1f}%", delta_color="inverse")
    with col3:
        st.metric("Funcionamiento correcto", f"{int(n_ok):,}")
    with col4:
        if models_available:
            X_all = scaler.transform(df[ALL_FEATURES])
            probs = model.predict_proba(X_all)[:, 1]
            high_risk = (probs > 0.7).sum()
            st.metric("Alertas alto riesgo (>70%)", f"{high_risk:,}", delta_color="inverse")
        else:
            st.metric("Modelos", "No entrenados", delta_color="off")

    st.divider()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Distribución de clases")
        fig = px.pie(
            values=[int(n_ok), int(n_fail)],
            names=["Normal", "Fallo"],
            color_discrete_sequence=["#2ecc71", "#e74c3c"],
            hole=0.45,
        )
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig.update_layout(height=320, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Distribución de sensores clave")
        sensor_sel = st.selectbox("Sensor", SENSOR_FEATURES, index=SENSOR_FEATURES.index("CS"))
        fig = px.histogram(
            df, x=sensor_sel, color=TARGET,
            color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
            barmode="overlay", opacity=0.7,
            labels={sensor_sel: SENSOR_DESCRIPTIONS.get(sensor_sel, sensor_sel), TARGET: "Estado"},
            nbins=25,
        )
        fig.update_layout(height=320, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Estadísticas descriptivas por clase")
    class_stats = df.groupby(TARGET)[SENSOR_FEATURES].mean().round(3).T
    class_stats.columns = ["Normal (media)", "Fallo (media)"]
    class_stats["Diferencia"] = (class_stats["Fallo (media)"] - class_stats["Normal (media)"]).round(3)
    class_stats["Δ %"] = (class_stats["Diferencia"] / (class_stats["Normal (media)"] + 1e-9) * 100).round(1)
    st.dataframe(class_stats.style.background_gradient(subset=["Δ %"], cmap="RdYlGn_r"), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: EXPLORADOR DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Explorador de Datos":
    st.title("🔍 Explorador de Datos")

    tab1, tab2, tab3 = st.tabs(["Distribuciones", "Correlaciones", "Tabla de datos"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            x_feat = st.selectbox("Eje X", SENSOR_FEATURES, index=0)
        with col2:
            y_feat = st.selectbox("Eje Y", SENSOR_FEATURES, index=6)

        fig = px.scatter(
            df, x=x_feat, y=y_feat, color=df[TARGET].map({0: "Normal", 1: "Fallo"}),
            color_discrete_map={"Normal": "#2ecc71", "Fallo": "#e74c3c"},
            opacity=0.6, size_max=6,
            labels={x_feat: SENSOR_DESCRIPTIONS.get(x_feat, x_feat),
                    y_feat: SENSOR_DESCRIPTIONS.get(y_feat, y_feat)},
            title=f"{x_feat} vs {y_feat}",
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Distribución individual por sensor")
        feat_dist = st.selectbox("Sensor", SENSOR_FEATURES, key="dist_sensor")
        fig2 = px.box(df, x=TARGET, y=feat_dist, color=df[TARGET].map({0: "Normal", 1: "Fallo"}),
                      color_discrete_map={"Normal": "#2ecc71", "Fallo": "#e74c3c"},
                      points="outliers",
                      labels={feat_dist: SENSOR_DESCRIPTIONS.get(feat_dist, feat_dist), TARGET: "Clase"})
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("Matriz de correlaciones")
        corr = df[SENSOR_FEATURES + [TARGET]].corr().round(3)
        fig_corr = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdYlGn",
            zmin=-1, zmax=1, aspect="auto",
            title="Correlación de Pearson",
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab3:
        st.subheader("Datos del dataset")
        fail_filter = st.selectbox("Filtrar por clase", ["Todos", "Normal (0)", "Fallo (1)"])
        filtered = df.copy()
        if fail_filter == "Normal (0)":
            filtered = df[df[TARGET] == 0]
        elif fail_filter == "Fallo (1)":
            filtered = df[df[TARGET] == 1]
        st.dataframe(filtered[SENSOR_FEATURES + [TARGET]].reset_index(drop=True),
                     use_container_width=True, height=400)
        st.caption(f"Mostrando {len(filtered):,} registros")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: DETECCIÓN DE ANOMALÍAS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Detección de Anomalías":
    st.title("🚨 Detección de Anomalías")

    anom_path = PROCESSED_DIR / "data_with_anomalies.csv"
    if not anom_path.exists():
        st.info("Ejecuta primero el notebook `04_anomaly_detection.py` para generar los resultados.")
        st.stop()

    df_anom = pd.read_csv(anom_path)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Isolation Forest", f"{df_anom['iso_anomaly'].sum()}", "anomalías")
    with col2:
        st.metric("LOF", f"{df_anom['lof_anomaly'].sum()}", "anomalías")
    with col3:
        st.metric("One-Class SVM", f"{df_anom['ocsvm_anomaly'].sum()}", "anomalías")
    with col4:
        st.metric("Consenso (≥2/3)", f"{df_anom['consensus_anomaly'].sum()}", "anomalías")

    st.divider()

    method = st.selectbox(
        "Método de detección",
        ["iso_anomaly", "lof_anomaly", "ocsvm_anomaly", "consensus_anomaly"],
        format_func=lambda x: {"iso_anomaly": "Isolation Forest", "lof_anomaly": "LOF",
                               "ocsvm_anomaly": "One-Class SVM", "consensus_anomaly": "Consenso"}[x]
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Distribución de anomalías")
        anom_counts = df_anom[method].value_counts().rename({0: "Normal", 1: "Anomalía"})
        fig = px.pie(values=anom_counts.values, names=anom_counts.index,
                     color_discrete_map={"Normal": "#2ecc71", "Anomalía": "#e74c3c"},
                     hole=0.4)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Solapamiento con fallos reales")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(df_anom[TARGET], df_anom[method])
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                           x=["Normal", "Anomalía"], y=["Normal (real)", "Fallo (real)"],
                           title="Anomalía detectada vs Fail real")
        fig_cm.update_layout(height=300)
        st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("Puntuaciones de anomalía por sensor")
    score_col = method.replace("anomaly", "score")
    if score_col in df_anom.columns:
        feat_anom = st.selectbox("Sensor", SENSOR_FEATURES, key="anom_feat")
        fig_sc = px.scatter(
            df_anom, x=feat_anom, y=score_col,
            color=df_anom[method].map({0: "Normal", 1: "Anomalía"}),
            color_discrete_map={"Normal": "#3498db", "Anomalía": "#e74c3c"},
            opacity=0.6,
            labels={feat_anom: SENSOR_DESCRIPTIONS.get(feat_anom, feat_anom),
                    score_col: "Puntuación de anomalía"},
            title=f"Puntuación de anomalía vs {feat_anom}",
        )
        fig_sc.update_layout(height=380)
        st.plotly_chart(fig_sc, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: COMPARACIÓN DE MODELOS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Comparación de Modelos":
    st.title("🏆 Comparación de Modelos")
    st.caption("Análisis comparativo de los 4 modelos entrenados: métricas, curvas ROC, matrices de confusión e importancia de características.")

    if not models_available:
        st.warning("Modelos no encontrados. Ejecuta primero `scripts/05_models.py`.")
        st.stop()

    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
    from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                                 f1_score, precision_score, recall_score, roc_auc_score)

    @st.cache_data
    def load_all_models_and_evaluate():
        model_files = {
            "Logistic Regression": "logistic_regression.pkl",
            "Random Forest":       "random_forest.pkl",
            "XGBoost":             "xgboost.pkl",
            "LightGBM":            "lightgbm.pkl",
        }
        loaded = {}
        for name, fname in model_files.items():
            p = MODELS_DIR / fname
            if p.exists():
                loaded[name] = joblib.load(p)

        X = df[ALL_FEATURES]
        y = df[TARGET]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
        )
        X_te_sc = scaler.transform(X_te)

        metrics_rows = []
        roc_data = {}
        cm_data = {}
        for name, m in loaded.items():
            y_prob = m.predict_proba(X_te_sc)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            metrics_rows.append({
                "Modelo": name,
                "F1":        round(f1_score(y_te, y_pred, zero_division=0), 4),
                "ROC-AUC":   round(roc_auc_score(y_te, y_prob), 4),
                "Precision": round(precision_score(y_te, y_pred, zero_division=0), 4),
                "Recall":    round(recall_score(y_te, y_pred, zero_division=0), 4),
            })
            fpr, tpr, _ = roc_curve(y_te, y_prob)
            roc_data[name] = (fpr.tolist(), tpr.tolist(), round(auc(fpr, tpr), 4))
            cm_data[name] = confusion_matrix(y_te, y_pred).tolist()

        metrics_df = pd.DataFrame(metrics_rows).set_index("Modelo")
        return loaded, metrics_df, roc_data, cm_data, X_te_sc, y_te

    loaded_models, metrics_df, roc_data, cm_data, X_te_sc, y_te = load_all_models_and_evaluate()

    # ── Tabla resumen de métricas ──────────────────────────────────────────────
    st.subheader("Métricas comparativas en Test Set (80/20)")
    best_model_name = metrics_df["F1"].idxmax()

    styled = (
        metrics_df.style
        .format("{:.4f}")
        .background_gradient(cmap="RdYlGn", subset=["F1", "ROC-AUC", "Precision", "Recall"])
        .set_properties(**{"font-weight": "bold"}, subset=pd.IndexSlice[[best_model_name], :])
    )
    st.dataframe(styled, use_container_width=True)
    st.caption(f"Mejor modelo: **{best_model_name}** (F1 = {metrics_df.loc[best_model_name, 'F1']:.4f})")

    st.divider()

    # ── Gráfico de barras de métricas ──────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Comparación de métricas")
        metric_sel = st.selectbox("Métrica", ["F1", "ROC-AUC", "Precision", "Recall"])
        colors = ["#e74c3c" if n == best_model_name else "#3498db"
                  for n in metrics_df.index]
        fig_bar = px.bar(
            metrics_df.reset_index(), x="Modelo", y=metric_sel,
            color="Modelo",
            color_discrete_sequence=colors,
            text=metric_sel,
            title=f"{metric_sel} por modelo",
        )
        fig_bar.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_bar.update_layout(height=380, showlegend=False, yaxis_range=[0, 1.1],
                              xaxis_tickangle=-15)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.subheader("Radar de métricas")
        categories = ["F1", "ROC-AUC", "Precision", "Recall"]
        fig_radar = go.Figure()
        palette = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
        for (name, row), color in zip(metrics_df.iterrows(), palette):
            vals = [row[c] for c in categories] + [row[categories[0]]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=categories + [categories[0]],
                fill="toself", name=name, line_color=color, opacity=0.7,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=380, showlegend=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # ── Curvas ROC ─────────────────────────────────────────────────────────────
    st.subheader("Curvas ROC")
    fig_roc = go.Figure()
    palette_roc = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    for (name, (fpr, tpr, roc_auc)), color in zip(roc_data.items(), palette_roc):
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, name=f"{name} (AUC={roc_auc:.3f})",
            line=dict(color=color, width=2),
        ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], name="Aleatorio",
        line=dict(color="gray", dash="dash"), showlegend=True,
    ))
    fig_roc.update_layout(
        xaxis_title="Tasa de Falsos Positivos",
        yaxis_title="Tasa de Verdaderos Positivos",
        height=420, title="Curvas ROC — Comparación de modelos",
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    st.divider()

    # ── Matrices de confusión ──────────────────────────────────────────────────
    st.subheader("Matrices de confusión")
    cm_cols = st.columns(len(cm_data))
    for col, (name, cm) in zip(cm_cols, cm_data.items()):
        with col:
            fig_cm = px.imshow(
                cm, text_auto=True, color_continuous_scale="Blues",
                x=["Normal", "Fallo"], y=["Normal", "Fallo"],
                title=name,
            )
            fig_cm.update_layout(height=280, margin=dict(t=40, b=10, l=10, r=10),
                                 coloraxis_showscale=False)
            st.plotly_chart(fig_cm, use_container_width=True)
            tp, fp = cm[1][1], cm[0][1]
            fn, tn = cm[1][0], cm[0][0]
            st.caption(f"TP={tp} FP={fp} FN={fn} TN={tn}")

    st.divider()

    # ── Feature importance por modelo (árboles) ────────────────────────────────
    st.subheader("Importancia de características por modelo")
    tree_models = {n: m for n, m in loaded_models.items()
                   if hasattr(m, "feature_importances_")}
    if tree_models:
        imp_sel = st.selectbox("Modelo", list(tree_models.keys()))
        imp_df = pd.DataFrame({
            "feature":    ALL_FEATURES,
            "importance": tree_models[imp_sel].feature_importances_,
        }).sort_values("importance", ascending=True).tail(15)
        fig_imp = px.bar(
            imp_df, x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale="RdYlGn",
            title=f"Feature Importance — {imp_sel}",
        )
        fig_imp.update_layout(height=480, coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Importancia intrínseca no disponible para Logistic Regression.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5: RENDIMIENTO DEL MODELO
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Rendimiento del Modelo":
    st.title("📊 Rendimiento del Modelo")

    if not models_available:
        st.warning("Modelos no encontrados. Ejecuta primero el notebook `05_models.py`.")
        st.stop()

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 roc_curve, auc, precision_recall_curve)
    from src.models import get_models

    X_all_sc = scaler.transform(df[ALL_FEATURES])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all_sc, df[TARGET], test_size=0.2, random_state=RANDOM_STATE, stratify=df[TARGET]
    )
    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    report = classification_report(y_te, y_pred, target_names=["Normal", "Fallo"], output_dict=True)
    report_df = pd.DataFrame(report).T

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Métricas de clasificación")
        st.dataframe(report_df.style.format("{:.3f}").background_gradient(cmap="RdYlGn", subset=["f1-score"]),
                     use_container_width=True)

    with col2:
        st.subheader("Matriz de confusión")
        cm = confusion_matrix(y_te, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                           x=["Normal", "Fallo"], y=["Normal (real)", "Fallo (real)"])
        fig_cm.update_layout(height=320)
        st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("Curva ROC")
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    roc_auc = auc(fpr, tpr)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC = {roc_auc:.3f}", line=dict(color="#e74c3c", width=2)))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Aleatorio", line=dict(color="gray", dash="dash")))
    fig_roc.update_layout(xaxis_title="FPR", yaxis_title="TPR", height=380,
                          title="Curva ROC — Test Set")
    st.plotly_chart(fig_roc, use_container_width=True)

    st.subheader("Importancia de características")
    if hasattr(model, "feature_importances_"):
        feat_imp = pd.DataFrame({
            "feature": ALL_FEATURES,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=True)
        fig_imp = px.bar(feat_imp, x="importance", y="feature", orientation="h",
                         color="importance", color_continuous_scale="RdYlGn",
                         title="Feature Importance del modelo final")
        fig_imp.update_layout(height=550)
        st.plotly_chart(fig_imp, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6: ÍNDICE DE SALUD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Índice de Salud":
    st.title("❤️ Índice de Salud de la Máquina")
    st.caption("Clasificación del estado operativo en 5 niveles de riesgo")

    if not models_available:
        st.warning("Modelos no encontrados. Ejecuta primero el notebook `05_models.py`.")
        st.stop()

    X_all_sc = scaler.transform(df[ALL_FEATURES])
    probs = model.predict_proba(X_all_sc)[:, 1]
    df_health = add_health_index(df.copy(), probs)

    # Distribución por nivel de salud
    health_counts = df_health["health_level"].value_counts()
    level_order = list(HEALTH_LEVELS.keys())
    health_counts = health_counts.reindex([l for l in level_order if l in health_counts.index])

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Distribución por nivel de salud")
        fig = px.bar(
            x=health_counts.index, y=health_counts.values,
            color=health_counts.index,
            color_discrete_map=HEALTH_COLORS,
            labels={"x": "Nivel de salud", "y": "N.º de registros"},
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Tasa de fallos reales por nivel de salud")
        fail_by_health = df_health.groupby("health_level")[TARGET].mean() * 100
        fail_by_health = fail_by_health.reindex([l for l in level_order if l in fail_by_health.index])
        fig2 = px.bar(
            x=fail_by_health.index, y=fail_by_health.values,
            color=fail_by_health.index,
            color_discrete_map=HEALTH_COLORS,
            labels={"x": "Nivel de salud", "y": "Tasa de fallos reales (%)"},
        )
        fig2.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Índice de salud vs probabilidad de fallo")
    fig3 = px.scatter(
        df_health, x="health_index", y=probs * 100,
        color="health_level", color_discrete_map=HEALTH_COLORS,
        opacity=0.6, size_max=6,
        labels={"health_index": "Índice de salud (0-100)", "y": "Probabilidad de fallo (%)"},
        title="Relación Índice de Salud ↔ Probabilidad de fallo",
    )
    fig3.update_layout(height=420)
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Leyenda de niveles de salud")
    for level, (lo, hi) in HEALTH_LEVELS.items():
        color = HEALTH_COLORS[level]
        count = (df_health["health_level"] == level).sum()
        st.markdown(
            f'<span class="health-badge" style="background-color:{color};color:white;">'
            f'{level}</span> &nbsp; Puntuación {lo}–{hi} &nbsp; | &nbsp; {count} registros',
            unsafe_allow_html=True,
        )
        st.write("")
