# Sistema Inteligente de Monitorizacion y Prediccion de Fallos en Equipos Industriales

Trabajo de Fin de Master (TFM) — Sistemas Inteligentes  
Autor: Carfok

---

## Descripcion del proyecto

Este proyecto implementa un sistema de machine learning completo para la monitorizacion y prediccion de fallos en equipos industriales a partir de datos de sensores. El sistema cubre el ciclo completo de un proyecto de ciencia de datos: desde el analisis exploratorio hasta el despliegue en produccion mediante contenedores Docker y una API REST.

El objetivo principal es anticipar fallos en maquinaria industrial utilizando lecturas de sensores en tiempo real, clasificando cada registro en dos estados (normal o fallo) y calculando un indice de salud del equipo en cinco niveles de riesgo.

---

## Dataset

- **Fichero:** `data/raw/data.csv`
- **Muestras:** 944 registros
- **Variables de entrada (sensores):**

| Variable    | Descripcion                          | Tipo       |
|-------------|--------------------------------------|------------|
| footfall    | Trafico de personas u objetos        | Continua   |
| tempMode    | Modo termico de operacion (0-7)      | Ordinal    |
| AQ          | Indice de calidad del aire (1-7)     | Ordinal    |
| USS         | Sensor ultrasonico (1-7)             | Ordinal    |
| CS          | Sensor de corriente electrica (1-7)  | Ordinal    |
| VOC         | Compuestos organicos volatiles (1-7) | Ordinal    |
| RP          | Posicion rotacional / RPM            | Continua   |
| IP          | Presion de entrada (1-7)             | Ordinal    |
| Temperature | Temperatura de operacion (C)         | Continua   |

- **Variable objetivo:** `fail` (0 = normal, 1 = fallo)
- **Balance de clases:** 58.4 % normal / 41.6 % fallo

---

## Arquitectura del sistema

```
data/raw/                  Dataset original
    |
scripts/00_spark_eda.py    Fase 0: EDA con Apache Spark (Big Data)
scripts/01_eda.py          Fase 1: Analisis exploratorio de datos
scripts/02_data_quality.py Fase 2: Calidad y preparacion de datos
scripts/03_feature_eng.py  Fase 3: Ingenieria de caracteristicas
scripts/04_anomaly.py      Fase 4: Deteccion de anomalias
scripts/05_models.py       Fase 5: Entrenamiento y comparacion de modelos
scripts/06_interpret.py    Fase 6: Interpretabilidad (SHAP, Permutation, PDP)
scripts/07_monitoring.py   Fase 7: Monitorizacion y sistema de alertas
    |
models/                    Modelos entrenados (.pkl)
data/processed/            Datos preprocesados
reports/                   Figuras y JSON de monitorizacion
    |
dashboard/app.py           Dashboard interactivo (Streamlit)
api/main.py                API REST (FastAPI)
docker-compose.yml         Despliegue en contenedores
```

---

## Estructura del repositorio

```
TFM/
├── api/
│   └── main.py                  API REST con FastAPI
├── dashboard/
│   └── app.py                   Dashboard interactivo con Streamlit
├── data/
│   └── raw/
│       └── data.csv             Dataset original
├── models/                      Modelos entrenados (generados en ejecucion)
├── reports/
│   └── figures/                 Graficos generados por cada fase
├── scripts/
│   ├── 00_spark_eda.py          Fase 0: EDA con PySpark
│   ├── 01_eda.py                Fase 1: Analisis exploratorio
│   ├── 02_data_quality.py       Fase 2: Calidad de datos
│   ├── 03_feature_engineering.py Fase 3: Ingenieria de caracteristicas
│   ├── 04_anomaly_detection.py  Fase 4: Deteccion de anomalias
│   ├── 05_models.py             Fase 5: Entrenamiento de modelos
│   ├── 06_interpretation.py     Fase 6: Interpretabilidad
│   └── 07_monitoring.py         Fase 7: Monitorizacion y alertas
├── src/
│   ├── config.py                Rutas y constantes globales
│   ├── features.py              Ingenieria de caracteristicas reutilizable
│   ├── health_index.py          Calculo del Machine Health Index
│   ├── models.py                Definicion y evaluacion de modelos
│   └── preprocessing.py        Carga y preprocesado de datos
├── Dockerfile.api               Imagen Docker para la API
├── Dockerfile.dashboard         Imagen Docker para el dashboard
├── docker-compose.yml           Orquestacion de servicios
├── Makefile                     Automatizacion de tareas
├── requirements.txt             Dependencias Python
└── README.md                    Este fichero
```

---

## Requisitos del sistema

- Python 3.10 o superior
- pip
- Docker Desktop (solo para despliegue en contenedores)
- Java 8 o superior (solo si se usa PySpark en la Fase 0)

---

## Instalacion local

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd TFM
```

### 2. Crear un entorno virtual (recomendado)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

Para usar PySpark (Fase 0), instalar adicionalmente:

```bash
pip install pyspark
```

---

## Ejecucion del pipeline completo

Las fases deben ejecutarse en orden. Cada fase genera los artefactos que necesita la siguiente.

### Fase 0 — EDA con Apache Spark (opcional, requiere pyspark)

```bash
python scripts/00_spark_eda.py
```

Demuestra el procesamiento del dataset con la API de Spark. Si PySpark no esta instalado, ejecuta automaticamente un modo de simulacion con pandas para mostrar los mismos resultados.

### Fase 1 — Analisis exploratorio de datos

```bash
python scripts/01_eda.py
```

Genera graficos de distribucion, correlaciones, boxplots por clase y analisis estadistico. Salida: `reports/figures/01_*.png`.

### Fase 2 — Calidad y preparacion de datos

```bash
python scripts/02_data_quality.py
```

Detecta valores nulos, duplicados y outliers mediante IQR y Z-score. Salida: `reports/figures/02_*.png`.

### Fase 3 — Ingenieria de caracteristicas

```bash
python scripts/03_feature_engineering.py
```

Crea 9 caracteristicas derivadas (indice de carga, estres ambiental, interacciones de sensores, etc.). Salida: `data/processed/data_engineered.csv`.

### Fase 4 — Deteccion de anomalias

```bash
python scripts/04_anomaly_detection.py
```

Aplica tres algoritmos no supervisados: Isolation Forest, Local Outlier Factor y One-Class SVM. Genera etiquetas de anomalia por consenso. Salida: `data/processed/data_with_anomalies.csv`.

### Fase 5 — Entrenamiento y comparacion de modelos

```bash
python scripts/05_models.py
```

Entrena y compara cuatro modelos mediante validacion cruzada estratificada de 5 pliegues: Logistic Regression, Random Forest, XGBoost y LightGBM. Selecciona el mejor modelo por F1-score. Salida: `models/*.pkl`.

### Fase 6 — Interpretabilidad

```bash
python scripts/06_interpretation.py
```

Calcula SHAP values, Permutation Importance y Partial Dependence Plots para el mejor modelo. Salida: `reports/figures/06_*.png`.

### Fase 7 — Monitorizacion y sistema de alertas

```bash
python scripts/07_monitoring.py
```

Evalua las metricas del modelo sobre datos de produccion simulados, detecta data drift mediante el test de Kolmogorov-Smirnov y genera alertas automaticas. Salida:

- `reports/figures/07_drift_distributions.png`
- `reports/figures/07_alert_dashboard.png`
- `reports/monitoring_report.json`

El fichero `monitoring_report.json` contiene las siguientes secciones:

```
timestamp                  Fecha y hora de la ejecucion
best_model_metrics         F1, AUC, Precision, Recall del mejor modelo en produccion
prediction_distribution    Tasa de fallos y probabilidad media predicha
model_interpretation
  model_ranking            Todos los modelos ordenados por F1
  per_model_metrics        Metricas individuales y matrices de confusion por modelo
  feature_importances      Importancias intrinsecas (RF, XGBoost, LightGBM), SHAP y Permutation
  conclusions              Texto interpretativo automatico con el mejor modelo y features clave
data_drift                 Estadistico KS y p-valor por sensor
alerts                     Lista de alertas generadas con estado OK / ALERTA / CRITICA
alert_summary              Conteo de alertas por nivel
thresholds_used            Umbrales configurados para cada comprobacion
```

### Ejecucion completa con Make

```bash
make all
```

Ejecuta las fases 0 a 7 en orden. Otros targets disponibles:

```bash
make install       # Instala dependencias
make dashboard     # Lanza el dashboard Streamlit en :8501
make api           # Lanza la API REST en :8000
make monitoring    # Ejecuta la fase 7 (monitorizacion)
make spark         # Ejecuta la fase 0 (PySpark)
make clean         # Elimina todos los artefactos generados
make help          # Muestra todos los targets disponibles
```

---

## Dashboard interactivo

```bash
streamlit run dashboard/app.py
```

Acceso: http://localhost:8501

El dashboard incluye seis secciones:

| Seccion              | Contenido                                                        |
|----------------------|------------------------------------------------------------------|
| Resumen General      | KPIs globales, distribucion de clases, estadisticas por sensor   |
| Explorador de Datos  | Scatter plots, boxplots, matriz de correlacion, tabla filtrable  |
| Deteccion Anomalias  | Resultados de Isolation Forest, LOF y One-Class SVM              |
| Comparacion Modelos  | Metricas, curvas ROC, matrices de confusion, feature importances |
| Rendimiento Modelo   | Metricas del mejor modelo, curva ROC, informe de clasificacion   |
| Indice de Salud      | Distribucion del Machine Health Index en 5 niveles de riesgo     |

---

## API REST

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Documentacion interactiva: http://localhost:8000/docs

### Endpoints disponibles

| Metodo | Ruta              | Descripcion                                        |
|--------|-------------------|----------------------------------------------------|
| GET    | /health           | Estado del servicio y tiempo de actividad          |
| GET    | /model/info       | Metadatos del modelo cargado                       |
| POST   | /predict          | Prediccion para una lectura de sensores            |
| POST   | /predict/batch    | Prediccion por lotes (maximo 500 lecturas)         |
| POST   | /predict/explain  | Prediccion con valores de caracteristicas derivadas|

### Ejemplo de llamada

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "footfall": 12.5,
    "tempMode": 3,
    "AQ": 4,
    "USS": 2,
    "CS": 5,
    "VOC": 3,
    "RP": 45.2,
    "IP": 4,
    "Temperature": 18.0
  }'
```

Respuesta:

```json
{
  "failure_probability": 0.445,
  "prediction": 0,
  "prediction_label": "Normal",
  "health_index": 45.5,
  "health_level": "Riesgo moderado",
  "risk_category": "MODERADO"
}
```

---

## Despliegue con Docker

Requiere Docker Desktop en ejecucion.

### Construir e iniciar todos los servicios

```bash
docker compose up --build -d
```

### Servicios disponibles

| Servicio  | URL                      | Descripcion              |
|-----------|--------------------------|--------------------------|
| API REST  | http://localhost:8000    | FastAPI + uvicorn        |
| Swagger   | http://localhost:8000/docs | Documentacion interactiva|
| Dashboard | http://localhost:8501    | Streamlit                |

### Detener los servicios

```bash
docker compose down
```

---

## Modelos implementados

| Modelo              | Tipo                  | Parametros relevantes                    |
|---------------------|-----------------------|------------------------------------------|
| Logistic Regression | Lineal                | max_iter=1000, class_weight=balanced     |
| Random Forest       | Ensemble (bagging)    | n_estimators=200, class_weight=balanced  |
| XGBoost             | Ensemble (boosting)   | n_estimators=200, scale_pos_weight=1.4   |
| LightGBM            | Ensemble (boosting)   | n_estimators=200, class_weight=balanced  |

La seleccion del mejor modelo se basa en el F1-score sobre el conjunto de test (particion 80/20 estratificada).

---

## Machine Health Index

El indice de salud es un valor compuesto (0-100, mayor es mejor) que combina la probabilidad de fallo del modelo con penalizaciones por estres de sensores y condiciones termicas extremas. Se clasifica en cinco niveles:

| Nivel           | Rango    |
|-----------------|----------|
| Saludable       | 80 - 100 |
| Bajo riesgo     | 60 - 80  |
| Riesgo moderado | 40 - 60  |
| Riesgo alto     | 20 - 40  |
| Critico         | 0  - 20  |

---

## Dependencias principales

| Libreria      | Version minima | Uso                                      |
|---------------|----------------|------------------------------------------|
| pandas        | 2.0.0          | Manipulacion de datos                    |
| numpy         | 1.24.0         | Operaciones numericas                    |
| scikit-learn  | 1.3.0          | Modelos, preprocesado, evaluacion        |
| xgboost       | 1.7.0          | Modelo XGBoost                           |
| lightgbm      | 4.0.0          | Modelo LightGBM                          |
| shap          | 0.42.0          | Interpretabilidad SHAP                   |
| streamlit     | 1.28.0         | Dashboard interactivo                    |
| fastapi       | 0.104.0        | API REST                                 |
| uvicorn       | 0.24.0         | Servidor ASGI para FastAPI               |
| plotly        | 5.15.0         | Visualizaciones interactivas             |
| scipy         | 1.10.0         | Tests estadisticos (KS, Mann-Whitney)    |
| pyspark       | 3.5.0          | Big Data (opcional)                      |

---

## Notas de reproducibilidad

- La semilla aleatoria global es `RANDOM_STATE = 42`, definida en `src/config.py`.
- Todas las particiones train/test usan estratificacion por la variable objetivo.
- Los modelos entrenados se serializan con `joblib` en `models/`. El fichero `best_model.pkl` apunta siempre al modelo con mejor F1 en el ultimo entrenamiento.
- Los datos procesados en `data/processed/` y los modelos en `models/*.pkl` no se incluyen en el repositorio (excluidos en `.gitignore`). Deben generarse ejecutando el pipeline completo.
