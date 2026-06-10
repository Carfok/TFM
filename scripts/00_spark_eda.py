"""Fase 0 — Big Data: EDA con Apache Spark (PySpark).

Demuestra el uso de PySpark para el procesamiento de datos industriales a escala.
Aunque el dataset es pequeño (944 filas), el pipeline es escalable a millones de registros.

Instalación:  pip install pyspark
Ejecutar con: python scripts/00_spark_eda.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType, IntegerType
    from pyspark.ml.feature import VectorAssembler, StandardScaler as SparkScaler
    from pyspark.ml.stat import Correlation
    from pyspark.ml import Pipeline
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    print("[AVISO] PySpark no instalado. Ejecuta: pip install pyspark")

import pandas as pd
from src.config import RAW_DATA, SENSOR_FEATURES, TARGET, PROCESSED_DIR, FIGURES_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def create_spark_session() -> "SparkSession":
    return (
        SparkSession.builder
        .appName("TFM_Industrial_Fault_Prediction")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def load_and_inspect(spark, path: str):
    """Carga el CSV con Spark e imprime estadísticas básicas."""
    df = spark.read.csv(path, header=True, inferSchema=True)

    print(f"\nEsquema del dataset:")
    df.printSchema()

    print(f"\nDimensiones: {df.count()} filas × {len(df.columns)} columnas")
    print(f"Columnas: {df.columns}")

    print(f"\nPrimeras 5 filas:")
    df.show(5, truncate=False)

    return df


def descriptive_statistics(df, features: list):
    """Estadísticas descriptivas con Spark."""
    print("\n" + "=" * 55)
    print("ESTADÍSTICAS DESCRIPTIVAS (Spark describe)")
    print("=" * 55)
    df.select(features + [TARGET]).describe().show(truncate=False)

    print("\nDistribución de la variable objetivo:")
    df.groupBy(TARGET).count().withColumn(
        "porcentaje",
        F.round(F.col("count") / df.count() * 100, 2)
    ).show()


def missing_values_analysis(df, features: list):
    """Detección de valores nulos con Spark."""
    print("\n" + "=" * 55)
    print("VALORES NULOS POR COLUMNA")
    print("=" * 55)
    null_counts = df.select(
        [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in features + [TARGET]]
    )
    null_counts.show(truncate=False)


def class_statistics(df, features: list):
    """Estadísticas por clase (normal vs fallo)."""
    print("\n" + "=" * 55)
    print("MEDIAS POR CLASE (normal=0, fallo=1)")
    print("=" * 55)
    df.groupBy(TARGET).agg(
        *[F.round(F.mean(F.col(c)), 3).alias(f"avg_{c}") for c in features]
    ).orderBy(TARGET).show(truncate=False)


def outlier_detection(df, features: list):
    """Detección de outliers con IQR mediante Spark."""
    print("\n" + "=" * 55)
    print("DETECCIÓN DE OUTLIERS (IQR)")
    print("=" * 55)
    for feat in features[:5]:  # Solo los 5 primeros para no saturar el output
        quantiles = df.approxQuantile(feat, [0.25, 0.75], 0.01)
        if len(quantiles) == 2:
            q1, q3 = quantiles
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            n_outliers = df.filter(
                (F.col(feat) < lower) | (F.col(feat) > upper)
            ).count()
            print(f"  {feat:15s} | Q1={q1:.2f} Q3={q3:.2f} IQR={iqr:.2f} | "
                  f"Outliers: {n_outliers} ({n_outliers/df.count()*100:.1f}%)")


def spark_ml_pipeline(df, features: list):
    """Pipeline de preprocesado con Spark ML."""
    print("\n" + "=" * 55)
    print("PIPELINE SPARK ML — VECTOR ASSEMBLER + SCALER")
    print("=" * 55)

    # Cast a Double para Spark ML
    for feat in features:
        df = df.withColumn(feat, F.col(feat).cast(DoubleType()))

    assembler = VectorAssembler(inputCols=features, outputCol="features_raw", handleInvalid="skip")
    scaler = SparkScaler(inputCol="features_raw", outputCol="features_scaled")
    pipeline = Pipeline(stages=[assembler, scaler])

    pipeline_model = pipeline.fit(df)
    df_transformed = pipeline_model.transform(df)

    print("Pipeline completado. Muestra de vectores escalados:")
    df_transformed.select(TARGET, "features_scaled").show(3, truncate=True)

    return df_transformed


def correlation_matrix(df_transformed):
    """Matriz de correlaciones con Spark ML."""
    print("\n" + "=" * 55)
    print("MATRIZ DE CORRELACIONES (Pearson — Spark ML)")
    print("=" * 55)
    r1 = Correlation.corr(df_transformed, "features_scaled").head()
    corr_array = r1[0].toArray()
    print(f"Dimensión de la matriz: {corr_array.shape}")
    print("Top correlaciones calculadas con Spark (ver heatmap en EDA estándar).")


def save_spark_output(df, spark):
    """Exporta el dataframe procesado a CSV via Spark."""
    out_path = str(PROCESSED_DIR / "spark_output")
    df.coalesce(1).write.mode("overwrite").option("header", "true").csv(out_path)
    print(f"\nOutput Spark guardado en: {out_path}/")


def main():
    if not SPARK_AVAILABLE:
        print("\nEjecución en modo SIMULADO (PySpark no instalado).")
        print("Para usar PySpark real: pip install pyspark")
        _simulate_spark_output()
        return

    print("=" * 55)
    print("FASE 0 — Big Data EDA con Apache Spark")
    print("=" * 55)

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    print(f"Spark version: {spark.version}")
    print(f"Cores disponibles: {spark.sparkContext.defaultParallelism}")

    df = load_and_inspect(spark, str(RAW_DATA))
    descriptive_statistics(df, SENSOR_FEATURES)
    missing_values_analysis(df, SENSOR_FEATURES)
    class_statistics(df, SENSOR_FEATURES)
    outlier_detection(df, SENSOR_FEATURES)

    df_transformed = spark_ml_pipeline(df, SENSOR_FEATURES)
    correlation_matrix(df_transformed)
    save_spark_output(df, spark)

    print("\nFase 0 (Spark EDA) completada.")
    print("Nota: el mismo pipeline escala a datasets de millones de registros")
    print("      en un clúster Spark/Databricks sin cambios de código.")

    spark.stop()


def _simulate_spark_output():
    """Simula el output de Spark con pandas para mostrar los resultados."""
    print("\n--- Simulación de output Spark con pandas ---\n")
    df = pd.read_csv(RAW_DATA)

    print(f"Dataset: {len(df)} filas × {len(df.columns)} columnas")
    print(f"Columnas: {list(df.columns)}\n")

    print("Estadísticas descriptivas:")
    print(df[SENSOR_FEATURES].describe().round(3).to_string())

    print("\nDistribución de la variable objetivo:")
    vc = df[TARGET].value_counts()
    for k, v in vc.items():
        print(f"  {k}: {v} ({v/len(df)*100:.1f}%)")

    print("\nMedias por clase:")
    print(df.groupby(TARGET)[SENSOR_FEATURES].mean().round(3).to_string())

    print("\nDetección de outliers IQR (5 primeras features):")
    for feat in SENSOR_FEATURES[:5]:
        q1, q3 = df[feat].quantile([0.25, 0.75])
        iqr = q3 - q1
        n_out = ((df[feat] < q1 - 1.5*iqr) | (df[feat] > q3 + 1.5*iqr)).sum()
        print(f"  {feat:15s} | Q1={q1:.2f} Q3={q3:.2f} | Outliers: {n_out}")

    print("\nPipeline Spark ML (VectorAssembler + StandardScaler): SIMULADO OK")
    print("Para ejecutar en Spark real: pip install pyspark")


if __name__ == "__main__":
    main()
