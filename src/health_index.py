import pandas as pd
import numpy as np
from src.config import HEALTH_LEVELS, HEALTH_COLORS


def compute_health_index(df: pd.DataFrame, fail_proba: np.ndarray) -> np.ndarray:
    """
    Composite Machine Health Index (0-100, higher = healthier).

    Combines model failure probability with sensor stress signals
    so operators get a single actionable number.
    """
    # Base score: invert failure probability (0=certain failure, 100=certainly healthy)
    base = (1 - fail_proba) * 100

    # Penalty: high sensor stress score pushes index down
    if "sensor_stress_score" in df.columns:
        stress_norm = df["sensor_stress_score"].values / df["sensor_stress_score"].max()
        penalty = stress_norm * 10
    else:
        penalty = np.zeros(len(df))

    # Penalty: thermal extremes
    if "thermal_extreme" in df.columns:
        penalty += df["thermal_extreme"].values * 5

    health = np.clip(base - penalty, 0, 100)
    return health


def classify_health_level(score: float) -> str:
    for level, (lo, hi) in HEALTH_LEVELS.items():
        if lo <= score <= hi:
            return level
    return "Crítico"


def get_health_color(score: float) -> str:
    return HEALTH_COLORS.get(classify_health_level(score), "#8e44ad")


def add_health_index(df: pd.DataFrame, fail_proba: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df["health_index"] = compute_health_index(df, fail_proba)
    df["health_level"] = df["health_index"].apply(classify_health_level)
    df["health_color"] = df["health_index"].apply(get_health_color)
    return df
