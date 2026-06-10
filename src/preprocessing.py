import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats
from src.config import RAW_DATA, TARGET, SENSOR_FEATURES, RANDOM_STATE, TEST_SIZE


def load_data(path=None) -> pd.DataFrame:
    path = path or RAW_DATA
    df = pd.read_csv(path)
    return df


def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum()
    pct = missing / len(df) * 100
    return pd.DataFrame({"missing": missing, "pct": pct}).query("missing > 0")


def check_duplicates(df: pd.DataFrame) -> dict:
    n_dup = df.duplicated().sum()
    return {"n_duplicates": n_dup, "pct": n_dup / len(df) * 100}


def detect_outliers_iqr(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    cols = cols or SENSOR_FEATURES
    results = []
    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        results.append({
            "feature": col,
            "q1": q1, "q3": q3, "iqr": iqr,
            "lower_fence": lower, "upper_fence": upper,
            "n_outliers": n_out,
            "pct_outliers": n_out / len(df) * 100,
        })
    return pd.DataFrame(results)


def detect_outliers_zscore(df: pd.DataFrame, cols=None, threshold=3.0) -> pd.DataFrame:
    cols = cols or SENSOR_FEATURES
    results = []
    for col in cols:
        z = np.abs(stats.zscore(df[col]))
        n_out = (z > threshold).sum()
        results.append({
            "feature": col,
            "threshold": threshold,
            "n_outliers": n_out,
            "pct_outliers": n_out / len(df) * 100,
        })
    return pd.DataFrame(results)


def get_train_test_split(df: pd.DataFrame, features=None):
    features = features or SENSOR_FEATURES
    X = df[features]
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def scale_features(X_train, X_test, scaler_type="standard"):
    scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    return X_train_sc, X_test_sc, scaler
