import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Temperatura agrupada en cuartiles operativos
    df["temp_bucket"] = pd.cut(
        df["Temperature"],
        bins=[0, 6, 12, 18, 24],
        labels=["Fría", "Normal", "Cálida", "Caliente"],
        include_lowest=True,
    )
    df["temp_bucket_code"] = df["temp_bucket"].cat.codes

    # 2. Índice de carga operativa: combinación de corriente (CS) y RPM (RP)
    df["load_index"] = (df["CS"] * df["RP"]) / (df["CS"].max() * df["RP"].max())

    # 3. Ratio presión/temperatura (estrés mecánico-térmico)
    df["pressure_temp_ratio"] = df["IP"] / (df["Temperature"] + 1)

    # 4. Estrés ambiental: combinación de VOC y calidad del aire
    df["env_stress"] = (df["VOC"] + df["AQ"]) / 2

    # 5. Puntuación de estrés global del sensor (media ponderada sensores ordinales)
    df["sensor_stress_score"] = (
        df["AQ"] * 0.15
        + df["USS"] * 0.10
        + df["CS"] * 0.25
        + df["VOC"] * 0.15
        + df["IP"] * 0.20
        + df["tempMode"] * 0.15
    )

    # 6. Indicador de alto tráfico (footfall > percentil 75)
    footfall_p75 = df["footfall"].quantile(0.75)
    df["high_footfall"] = (df["footfall"] > footfall_p75).astype(int)

    # 7. RPM normalizado a rango 0-1
    df["rp_normalized"] = (df["RP"] - df["RP"].min()) / (df["RP"].max() - df["RP"].min())

    # 8. Interacción corriente × presión (indicador de sobrecarga eléctrica-mecánica)
    df["cs_ip_interaction"] = df["CS"] * df["IP"]

    # 9. Indicador modo térmico extremo (modos 0 o 7 = extremos)
    df["thermal_extreme"] = df["tempMode"].apply(lambda x: 1 if x in [0, 7] else 0)

    return df


ENGINEERED_FEATURES = [
    "temp_bucket_code",
    "load_index",
    "pressure_temp_ratio",
    "env_stress",
    "sensor_stress_score",
    "high_footfall",
    "rp_normalized",
    "cs_ip_interaction",
    "thermal_extreme",
]

ALL_FEATURES = [
    "footfall", "tempMode", "AQ", "USS", "CS", "VOC", "RP", "IP", "Temperature",
] + ENGINEERED_FEATURES
