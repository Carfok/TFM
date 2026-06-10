from pathlib import Path

ROOT = Path(__file__).parent.parent

DATA_DIR = ROOT / "data"
RAW_DATA = DATA_DIR / "raw" / "data.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Column definitions
SENSOR_FEATURES = ["footfall", "tempMode", "AQ", "USS", "CS", "VOC", "RP", "IP", "Temperature"]
TARGET = "fail"

# Ordinal-scale sensors (discrete 1-7 range)
ORDINAL_FEATURES = ["tempMode", "AQ", "USS", "CS", "VOC", "IP"]

# Continuous sensors
CONTINUOUS_FEATURES = ["footfall", "RP", "Temperature"]

# Sensor descriptions (for dashboard labels)
SENSOR_DESCRIPTIONS = {
    "footfall":    "Tráfico de personas/objetos",
    "tempMode":    "Modo térmico de operación",
    "AQ":          "Índice de calidad del aire",
    "USS":         "Sensor ultrasónico",
    "CS":          "Sensor de corriente eléctrica",
    "VOC":         "Compuestos orgánicos volátiles",
    "RP":          "Posición rotacional / RPM",
    "IP":          "Presión de entrada",
    "Temperature": "Temperatura de operación",
}

# Machine Health Index thresholds (score 0-100, higher = healthier)
HEALTH_LEVELS = {
    "Saludable":       (80, 100),
    "Bajo riesgo":     (60, 80),
    "Riesgo moderado": (40, 60),
    "Riesgo alto":     (20, 40),
    "Crítico":         (0,  20),
}

HEALTH_COLORS = {
    "Saludable":       "#2ecc71",
    "Bajo riesgo":     "#f1c40f",
    "Riesgo moderado": "#e67e22",
    "Riesgo alto":     "#e74c3c",
    "Crítico":         "#8e44ad",
}

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
