PYTHON     := python
STREAMLIT  := streamlit

# ── Rutas de entrada / salida ──────────────────────────────────────────────────
DATA_RAW   := data/raw/data.csv
DATA_ENG   := data/processed/data_engineered.csv
DATA_ANOM  := data/processed/data_with_anomalies.csv
BEST_MODEL := models/best_model.pkl

SENTINEL_1 := reports/figures/01_class_distribution.png
SENTINEL_2 := reports/figures/02_footfall_analysis.png
SENTINEL_6 := reports/figures/06_shap_summary.png

# ── Targets principales ────────────────────────────────────────────────────────
.PHONY: all install dashboard clean help api monitoring spark

## all: ejecuta todas las fases en orden
all: phase0 phase1 phase2 phase3 phase4 phase5 phase6 phase7

## install: instala las dependencias de Python
install:
	pip install -r requirements.txt

## dashboard: lanza el dashboard Streamlit en :8501
dashboard: $(BEST_MODEL) $(DATA_ANOM)
	$(STREAMLIT) run dashboard/app.py

## api: lanza la API REST con FastAPI/uvicorn en :8000
api: $(BEST_MODEL)
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

## monitoring: ejecuta la fase 7 (monitorizacion y alertas)
monitoring: $(BEST_MODEL)
	$(PYTHON) scripts/07_monitoring.py

## spark: ejecuta la fase 0 (EDA con PySpark)
spark:
	$(PYTHON) scripts/00_spark_eda.py

## clean: elimina todos los artefactos generados
clean:
	rm -rf data/processed
	rm -rf reports/figures
	find models -name "*.pkl" -delete
	find models -name "*.joblib" -delete
	@echo "Limpieza completada."

## help: muestra esta ayuda
help:
	@echo ""
	@echo "Uso: make <target>"
	@echo ""
	@echo "  all          Ejecuta todas las fases (0-7) en orden"
	@echo "  install      Instala dependencias (pip install -r requirements.txt)"
	@echo "  phase0       Big Data EDA con PySpark"
	@echo "  phase1       Analisis exploratorio de datos"
	@echo "  phase2       Calidad de datos"
	@echo "  phase3       Ingenieria de caracteristicas  ->  data_engineered.csv"
	@echo "  phase4       Deteccion de anomalias         ->  data_with_anomalies.csv"
	@echo "  phase5       Entrenamiento de modelos       ->  models/"
	@echo "  phase6       Interpretacion (SHAP)          ->  reports/figures/06_*"
	@echo "  phase7       Monitorizacion y alertas       ->  reports/monitoring_report.json"
	@echo "  dashboard    Lanza Streamlit (requiere phase3 + phase4 + phase5)"
	@echo "  api          Lanza la API REST FastAPI en :8000"
	@echo "  monitoring   Ejecuta analisis de drift y alertas"
	@echo "  spark        Ejecuta EDA con PySpark (requiere pyspark)"
	@echo "  clean        Elimina todos los artefactos generados"
	@echo ""

# ── Fases individuales ─────────────────────────────────────────────────────────

## phase0: Big Data EDA con PySpark
.PHONY: phase0
phase0:
	@echo "──────────────────────────────────────────"
	@echo " FASE 0 — Big Data EDA (PySpark)"
	@echo "──────────────────────────────────────────"
	$(PYTHON) scripts/00_spark_eda.py

## phase1: Analisis exploratorio de datos
.PHONY: phase1
phase1: $(SENTINEL_1)

$(SENTINEL_1): $(DATA_RAW)
	@echo "──────────────────────────────────────────"
	@echo " FASE 1 — EDA"
	@echo "──────────────────────────────────────────"
	$(PYTHON) scripts/01_eda.py

# ──────────────────────────────────────────────────────────────────────────────

## phase2: Calidad y preparacion de los datos
.PHONY: phase2
phase2: $(SENTINEL_2)

$(SENTINEL_2): $(DATA_RAW)
	@echo "──────────────────────────────────────────"
	@echo " FASE 2 — Calidad de datos"
	@echo "──────────────────────────────────────────"
	$(PYTHON) scripts/02_data_quality.py

# ──────────────────────────────────────────────────────────────────────────────

## phase3: Ingenieria de caracteristicas
.PHONY: phase3
phase3: $(DATA_ENG)

$(DATA_ENG): $(DATA_RAW)
	@echo "──────────────────────────────────────────"
	@echo " FASE 3 — Ingenieria de caracteristicas"
	@echo "──────────────────────────────────────────"
	$(PYTHON) scripts/03_feature_engineering.py

# ──────────────────────────────────────────────────────────────────────────────

## phase4: Deteccion de anomalias
.PHONY: phase4
phase4: $(DATA_ANOM)

$(DATA_ANOM): $(DATA_ENG)
	@echo "──────────────────────────────────────────"
	@echo " FASE 4 — Deteccion de anomalias"
	@echo "──────────────────────────────────────────"
	$(PYTHON) scripts/04_anomaly_detection.py

# ──────────────────────────────────────────────────────────────────────────────

## phase5: Entrenamiento y comparacion de modelos
.PHONY: phase5
phase5: $(BEST_MODEL)

$(BEST_MODEL): $(DATA_ENG)
	@echo "──────────────────────────────────────────"
	@echo " FASE 5 — Entrenamiento de modelos"
	@echo "──────────────────────────────────────────"
	$(PYTHON) scripts/05_models.py

# ──────────────────────────────────────────────────────────────────────────────

## phase6: Interpretacion (SHAP, Permutation, PDP)
.PHONY: phase6
phase6: $(SENTINEL_6)

$(SENTINEL_6): $(BEST_MODEL) $(DATA_ENG)
	@echo "──────────────────────────────────────────"
	@echo " FASE 6 — Interpretacion"
	@echo "──────────────────────────────────────────"
	$(PYTHON) scripts/06_interpretation.py

# ──────────────────────────────────────────────────────────────────────────────

## phase7: Monitorizacion del modelo y sistema de alertas
.PHONY: phase7
phase7: $(BEST_MODEL)
	@echo "──────────────────────────────────────────"
	@echo " FASE 7 — Monitorizacion y Alertas"
	@echo "──────────────────────────────────────────"
	$(PYTHON) scripts/07_monitoring.py
