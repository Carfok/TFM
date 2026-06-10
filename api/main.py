"""
API REST — Sistema de Predicción de Fallos Industriales
Ejecutar con: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
Documentación: http://localhost:8000/docs
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from src.config import MODELS_DIR, SENSOR_FEATURES, HEALTH_LEVELS, HEALTH_COLORS
from src.features import engineer_features, ALL_FEATURES
from src.health_index import add_health_index

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Monitor Industrial ML — API REST",
    description=(
        "API para predicción de fallos en equipos industriales mediante Machine Learning. "
        "Expone el modelo entrenado (Fase 5 del TFM) como servicio HTTP."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Carga de modelos al arrancar ──────────────────────────────────────────────
_model = None
_scaler = None
_feature_names = None
_load_time = None


def _load_artifacts():
    global _model, _scaler, _feature_names, _load_time
    try:
        _model = joblib.load(MODELS_DIR / "best_model.pkl")
        _scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        _feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
        _load_time = time.time()
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Modelos no encontrados en {MODELS_DIR}. "
            "Ejecuta primero scripts/05_models.py."
        ) from e


@app.on_event("startup")
async def startup_event():
    _load_artifacts()


# ── Schemas ───────────────────────────────────────────────────────────────────
class SensorReading(BaseModel):
    footfall: float = Field(..., ge=0, description="Tráfico de personas/objetos")
    tempMode: int = Field(..., ge=0, le=7, description="Modo térmico de operación (0-7)")
    AQ: int = Field(..., ge=1, le=7, description="Índice de calidad del aire (1-7)")
    USS: int = Field(..., ge=1, le=7, description="Sensor ultrasónico (1-7)")
    CS: int = Field(..., ge=1, le=7, description="Sensor de corriente eléctrica (1-7)")
    VOC: int = Field(..., ge=1, le=7, description="Compuestos orgánicos volátiles (1-7)")
    RP: float = Field(..., ge=0, description="Posición rotacional / RPM")
    IP: int = Field(..., ge=1, le=7, description="Presión de entrada (1-7)")
    Temperature: float = Field(..., ge=0, le=40, description="Temperatura de operación (°C)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "footfall": 12.5,
                "tempMode": 3,
                "AQ": 4,
                "USS": 2,
                "CS": 5,
                "VOC": 3,
                "RP": 45.2,
                "IP": 4,
                "Temperature": 18.0,
            }
        }
    }


class PredictionResponse(BaseModel):
    failure_probability: float = Field(..., description="Probabilidad de fallo (0.0 – 1.0)")
    prediction: int = Field(..., description="0 = Normal, 1 = Fallo")
    prediction_label: str
    health_index: float = Field(..., description="Índice de salud (0-100, mayor es mejor)")
    health_level: str = Field(..., description="Nivel de salud textual")
    risk_category: str = Field(..., description="Categoría de riesgo: BAJO / MODERADO / ALTO / CRITICO")


class BatchRequest(BaseModel):
    readings: list[SensorReading] = Field(..., min_items=1, max_items=500)


class BatchResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]
    summary: dict


class ModelInfo(BaseModel):
    model_type: str
    feature_count: int
    features: list[str]
    health_levels: dict
    uptime_seconds: float


# ── Helpers ───────────────────────────────────────────────────────────────────
def _reading_to_df(reading: SensorReading) -> pd.DataFrame:
    return pd.DataFrame([reading.dict()])


def _predict_df(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Returns (probabilities, predictions)."""
    df_eng = engineer_features(df)
    for col in ALL_FEATURES:
        if col not in df_eng.columns:
            df_eng[col] = 0
    X = _scaler.transform(df_eng[ALL_FEATURES])
    probs = _model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return probs, preds


def _risk_category(prob: float) -> str:
    if prob < 0.30:
        return "BAJO"
    elif prob < 0.55:
        return "MODERADO"
    elif prob < 0.75:
        return "ALTO"
    return "CRITICO"


def _build_prediction_response(reading_df: pd.DataFrame, prob: float, pred: int) -> PredictionResponse:
    hi_df = add_health_index(reading_df, np.array([prob]))
    health_score = float(hi_df["health_index"].iloc[0])
    health_level = str(hi_df["health_level"].iloc[0])
    return PredictionResponse(
        failure_probability=round(float(prob), 4),
        prediction=int(pred),
        prediction_label="Fallo" if pred == 1 else "Normal",
        health_index=round(health_score, 2),
        health_level=health_level,
        risk_category=_risk_category(prob),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Monitor Industrial ML — API REST",
        "version": "1.0.0",
        "docs": "/docs",
        "health_check": "/health",
    }


@app.get("/health", tags=["Info"])
def health_check():
    """Comprueba que la API y los modelos están disponibles."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelos no cargados")
    return {
        "status": "ok",
        "model_loaded": True,
        "uptime_seconds": round(time.time() - _load_time, 1),
    }


@app.get("/model/info", response_model=ModelInfo, tags=["Modelo"])
def model_info():
    """Devuelve metadatos del modelo cargado."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    return ModelInfo(
        model_type=type(_model).__name__,
        feature_count=len(_feature_names),
        features=_feature_names,
        health_levels={k: {"min": v[0], "max": v[1]} for k, v in HEALTH_LEVELS.items()},
        uptime_seconds=round(time.time() - _load_time, 1),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
def predict(reading: SensorReading):
    """
    Predice la probabilidad de fallo para una única lectura de sensores.

    Devuelve:
    - `failure_probability`: probabilidad de fallo (0–1)
    - `prediction`: 0 (normal) o 1 (fallo)
    - `health_index`: índice de salud del equipo (0–100)
    - `health_level`: categoría de salud textual
    - `risk_category`: BAJO / MODERADO / ALTO / CRITICO
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    try:
        df = _reading_to_df(reading)
        probs, preds = _predict_df(df)
        return _build_prediction_response(engineer_features(df), probs[0], preds[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse, tags=["Predicción"])
def predict_batch(request: BatchRequest):
    """
    Predicción en lote para múltiples lecturas (máx. 500).

    Devuelve predicciones individuales + resumen agregado.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    try:
        df = pd.DataFrame([r.dict() for r in request.readings])
        probs, preds = _predict_df(df)
        df_eng = engineer_features(df)

        responses = []
        for i in range(len(df)):
            row_df = df_eng.iloc[[i]].reset_index(drop=True)
            responses.append(_build_prediction_response(row_df, probs[i], preds[i]))

        n_fail = int(preds.sum())
        summary = {
            "total": len(preds),
            "failures_predicted": n_fail,
            "normal_predicted": len(preds) - n_fail,
            "failure_rate_pct": round(n_fail / len(preds) * 100, 2),
            "avg_failure_probability": round(float(probs.mean()), 4),
            "max_failure_probability": round(float(probs.max()), 4),
            "high_risk_count": int((probs > 0.7).sum()),
        }
        return BatchResponse(count=len(responses), predictions=responses, summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/explain", tags=["Predicción"])
def predict_explain(reading: SensorReading):
    """
    Predicción con explicabilidad básica: devuelve los valores engineered
    de cada característica junto con la predicción.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    try:
        df = _reading_to_df(reading)
        probs, preds = _predict_df(df)
        df_eng = engineer_features(df)

        response = _build_prediction_response(df_eng, probs[0], preds[0])

        feature_values = {
            col: round(float(df_eng[col].iloc[0]), 4)
            for col in ALL_FEATURES
            if col in df_eng.columns
        }

        return {
            **response.dict(),
            "engineered_features": feature_values,
            "raw_sensors": reading.dict(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
