
import os
import sys
import time
import warnings
import platform

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# DEPENDENCIAS
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    tf.get_logger().setLevel("ERROR")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except ImportError:
    tf = None

# RUTAS DEL PROYECTO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "2-Data", "shahrekord", "94_95_96_97_all.csv")
EXPERIMENT_DIR = os.path.join(BASE_DIR, "3-Experiment")
FIGURES_DIR = os.path.join(BASE_DIR, "4-Figures")

RESULTS_PATH = os.path.join(EXPERIMENT_DIR, "results.csv")
PREDICTIONS_PATH = os.path.join(EXPERIMENT_DIR, "predictions.csv")
FEATURE_IMPORTANCE_PATH = os.path.join(EXPERIMENT_DIR, "feature_importance.csv")
SCENARIOS_PATH = os.path.join(EXPERIMENT_DIR, "feature_scenarios_results.csv")
HIGH_DEMAND_PATH = os.path.join(EXPERIMENT_DIR, "high_demand_results.csv")
CONFIG_PATH = os.path.join(EXPERIMENT_DIR, "model_configuration.csv")
SUMMARY_PATH = os.path.join(EXPERIMENT_DIR, "experiment_summary.txt")


RANDOM_STATE = 42
LAGS = [0, 1, 2, 3, 24, 48, 168]
LSTM_WINDOW = 24
SVR_MAX_SAMPLES = 5000
TRAIN_YEARS = [1, 2]
VALID_YEARS = [3]
TEST_YEARS = [4]

PERMUTATION_SAMPLES = 1500

HIGH_DEMAND_PERCENTILE = 90

FIG3_DISPLAY_STEPS = 336

# Número de variables mostradas en Fig6
FIG6_TOP_N = 15

# Optimización de hiperparámetros
RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_leaf": [1, 2, 4],
    "min_samples_split": [2, 5],
}
XGB_PARAM_GRID = {
    "n_estimators": [200, 300, 400],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}
SVR_PARAM_GRID = {
    "svr__C": [1, 10, 100],
    "svr__epsilon": [0.05, 0.1, 0.2],
    "svr__gamma": ["scale", "auto"],
}
RF_N_ITER = 12
XGB_N_ITER = 12
SVR_N_ITER = 10

# Número de ejecuciones por modelo para reportar media ± desviación estándar
# de los tiempos de entrenamiento y de predicción (reduce el efecto de las
# variaciones del entorno de ejecución).
N_RUNS = 3

# Búsqueda reducida de configuraciones para la LSTM (unidades y dropout).
LSTM_TUNING_GRID = [
    {"units": 32, "dropout": 0.2},
    {"units": 32, "dropout": 0.3},
    {"units": 64, "dropout": 0.2},
    {"units": 64, "dropout": 0.3},
]

# Ruta del reporte de optimización de hiperparámetros
TUNING_PATH = os.path.join(EXPERIMENT_DIR, "hyperparameter_tuning.csv")

# UTILIDADES
def print_section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-6
    return 100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }


def set_reproducible():
    import random
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    if tf is not None:
        tf.random.set_seed(RANDOM_STATE)


def get_cpu_name():
    """Obtiene el nombre comercial del procesador (p. ej. 'AMD Ryzen 5 5500U')."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).Name"],
            text=True, timeout=30)
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if lines:
            return lines[0]
    except Exception:
        pass
    return platform.processor()


def get_ram_gb():
    """Memoria física total en GB."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB)"],
            text=True, timeout=30)
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if lines:
            return lines[0]
    except Exception:
        pass
    return "N/A"

# CARGA Y LIMPIEZA DEL DATASET
def load_dataset(path):
    print_section("CARGA DEL DATASET SHAHREKORD")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No se encontró el dataset en:\n{path}")

    df = pd.read_csv(path)

    print(f"Archivo: {path}")
    print(f"Filas:   {len(df):,}")
    print(f"Columnas ({len(df.columns)}):")
    for c in df.columns:
        print(f"    - {c!r}")

    rename = {
        "YEAR": "year",
        "Weekday": "weekday",
        "Holiday": "holiday",
        "Day of Month": "day_of_month",
        "Month Num": "month",
        "Hour": "hour_raw",
        "Demand": "demand",
        "Load": "load_type",
        "Min Tem": "min_tem",
        "Max Tem": "max_tem",
        "avr tem": "avr_tem",
        "min hum ": "min_hum",
        "max hum": "max_hum",
        "avr hum": "avr_hum",
        "rain": "rain",
        "snow ": "snow",
        "T precipitation": "total_precipitation",
        "snow level": "snow_level",
        "W D 03": "wind_dir_03",
        "W S 03": "wind_speed_03",
        "W D 09": "wind_dir_09",
        "W S 09": "wind_speed_09",
        "W D 15": "wind_dir_15",
        "W S 15": "wind_speed_15",
        "max W D ": "max_wind_dir",
        "max W S ": "max_wind_speed",
    }
    df = df.rename(columns=rename)

    # Valores faltantes
    n_nan = int(df.isnull().sum().sum())
    if n_nan > 0:
        print(f"\nValores faltantes detectados: {n_nan}. Se imputan por "
              "interpolación lineal temporal.")
        df = df.interpolate(method="linear", limit_direction="both")

    # Duplicados
    n_dup = int(df.duplicated().sum())
    if n_dup > 0:
        print(f"Duplicados detectados: {n_dup}. Se eliminan.")
        df = df.drop_duplicates()

    df = df.sort_values(["year", "month", "day_of_month", "hour_raw"]).reset_index(drop=True)
    n = len(df)
    df["datetime"] = pd.date_range(start="2015-03-03 00:00:00", periods=n, freq="h")

    df["hour"] = df["datetime"].dt.hour

    print(f"\nPeriodo (índice sintético): {df['datetime'].min()} -> {df['datetime'].max()}")
    print(f"Frecuencia: horaria (24 observaciones/día)")
    print(f"Rango de demanda: {df['demand'].min():.2f} - {df['demand'].max():.2f} MW")
    print(f"Media de demanda: {df['demand'].mean():.2f} MW")
    print(f"Observaciones por año:\n{df.groupby('year').size().to_string()}")

    return df

# INGENIERÍA DE CARACTERÍSTICAS
def build_temporal_features(df):
    """Características temporales/calendario (Grupo C y parte de D)."""
    df = df.copy()

    df["day_of_week"] = df["weekday"].astype(int)
    # Representación cíclica de la hora (periodo 24)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    # Representación cíclica del día de la semana (periodo 7)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    # Representación cíclica del mes (periodo 12)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    # Día del año (aprox. secuencial)
    df["day_of_year"] = df["datetime"].dt.dayofyear
    # Indicador de fin de semana (Irán: jueves=6 y viernes=7)
    df["is_weekend"] = (df["day_of_week"] >= 6).astype(int)

    return df


def build_lag_features(df):
    """
    Crea lags y medias móviles (Grupo A) y el objetivo.

    target(t) = demanda(t+1). Los lags y medias móviles usan solo
    información disponible en t (pasado), sin fuga de información.
    """
    df = df.copy()

    for lag in LAGS:
        df[f"lag_{lag}"] = df["demand"].shift(lag)

    # Medias móviles causales (ventana hacia atrás que incluye el instante t).
    df["rolling_mean_3"] = df["demand"].rolling(3).mean()
    df["rolling_mean_24"] = df["demand"].rolling(24).mean()

    # Objetivo: demanda de la siguiente hora
    df["target"] = df["demand"].shift(-1)

    return df


# ============================================================
# GRUPOS DE INDICADORES
# ============================================================
GROUP_A = [
    "lag_0", "lag_1", "lag_2", "lag_3", "lag_24", "lag_48", "lag_168",
    "rolling_mean_3", "rolling_mean_24",
]

GROUP_B = [
    "min_tem", "max_tem", "avr_tem",
    "min_hum", "max_hum", "avr_hum",
    "rain", "snow", "total_precipitation", "snow_level",
    "wind_dir_03", "wind_speed_03",
    "wind_dir_09", "wind_speed_09",
    "wind_dir_15", "wind_speed_15",
    "max_wind_dir", "max_wind_speed",
]

GROUP_C = [
    "hour", "hour_sin", "hour_cos",
    "day_of_week", "dow_sin", "dow_cos",
    "month", "month_sin", "month_cos",
    "day_of_month", "day_of_year",
    "is_weekend", "load_type",
]

GROUP_D = ["holiday"]

SCENARIOS = {
    "A": {"label": "Historical", "features": GROUP_A},
    "B": {"label": "Historical + Temporal", "features": GROUP_A + GROUP_C},
    "C": {"label": "Historical + Meteorological", "features": GROUP_A + GROUP_B},
    "D": {"label": "Full model", "features": GROUP_A + GROUP_B + GROUP_C + GROUP_D},
}

# Características por paso de la LSTM (equivalentes a los grupos)
LSTM_GROUP_A = ["demand"]
LSTM_GROUP_B = GROUP_B
LSTM_GROUP_C = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos", "is_weekend", "holiday", "load_type",
]
LSTM_SCENARIO_FEATURES = {
    "A": LSTM_GROUP_A,
    "B": LSTM_GROUP_A + LSTM_GROUP_C,
    "C": LSTM_GROUP_A + LSTM_GROUP_B,
    "D": LSTM_GROUP_A + LSTM_GROUP_B + LSTM_GROUP_C,
}


# ============================================================
# MODELOS TABULARES
# ============================================================
def train_random_forest(X_train, y_train, X_test, params):
    model = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **params,
    )
    start = time.perf_counter()
    model.fit(X_train, y_train)
    t_train = time.perf_counter() - start
    start = time.perf_counter()
    preds = model.predict(X_test)
    t_pred = time.perf_counter() - start
    return model, preds, t_train, t_pred


def train_svr(X_train, y_train, X_test, params):
    y_train = np.asarray(y_train, dtype=float).reshape(-1, 1)

    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train).ravel()

    n = len(X_train_scaled)
    if n > SVR_MAX_SAMPLES:
        stride = int(np.ceil(n / SVR_MAX_SAMPLES))
        idx = np.arange(0, n, stride)
        X_train_scaled = X_train_scaled[idx]
        y_train_scaled = y_train_scaled[idx]

    model = SVR(kernel="rbf", **params)

    start = time.perf_counter()
    model.fit(X_train_scaled, y_train_scaled)
    t_train = time.perf_counter() - start

    start = time.perf_counter()
    preds_scaled = model.predict(X_test_scaled)
    t_pred = time.perf_counter() - start

    preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    return model, preds, t_train, t_pred


def train_xgboost(X_train, y_train, X_test, params):
    if XGBRegressor is None:
        raise RuntimeError("XGBoost no está instalado.")
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **params,
    )
    start = time.perf_counter()
    model.fit(X_train, y_train)
    t_train = time.perf_counter() - start
    start = time.perf_counter()
    preds = model.predict(X_test)
    t_pred = time.perf_counter() - start
    return model, preds, t_train, t_pred


# LSTM
def build_sequences(hourly_df, positions, feature_cols, window):
    vals = hourly_df[feature_cols].to_numpy(dtype=np.float32)
    X = np.stack([vals[p - window + 1: p + 1] for p in positions], axis=0)
    return X


def train_lstm(hourly_df, positions, y, feature_cols, window,
               train_mask, val_mask, test_mask, params=None):
    if tf is None:
        raise RuntimeError("TensorFlow no está instalado.")

    if params is None:
        params = {"units": 64, "dropout": 0.2, "dense": 32}

    X = build_sequences(hourly_df, positions, feature_cols, window)
    n_samples, w, n_feat = X.shape

    X_train = X[train_mask]
    X_val = X[val_mask]
    X_test = X[test_mask]

    y = np.asarray(y, dtype=float)
    y_train = y[train_mask].reshape(-1, 1)
    y_val = y[val_mask].reshape(-1, 1)
    y_test = y[test_mask]

    feature_scaler = StandardScaler()
    feature_scaler.fit(X_train.reshape(-1, n_feat))
    X_train_s = feature_scaler.transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape)
    X_val_s = feature_scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape)
    X_test_s = feature_scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)

    target_scaler = StandardScaler()
    y_train_s = target_scaler.fit_transform(y_train)
    y_val_s = target_scaler.transform(y_val)

    set_reproducible()

    model = Sequential([
        LSTM(params["units"], input_shape=(w, n_feat), return_sequences=False),
        Dropout(params["dropout"]),
        Dense(params["dense"], activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    start = time.perf_counter()
    model.fit(
        X_train_s, y_train_s,
        validation_data=(X_val_s, y_val_s),
        epochs=100,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=0,
    )
    t_train = time.perf_counter() - start

    start = time.perf_counter()
    preds_s = model.predict(X_test_s, verbose=0)
    t_pred = time.perf_counter() - start

    preds = target_scaler.inverse_transform(preds_s).ravel()
    return model, preds, t_train, t_pred, n_feat


# OPTIMIZACIÓN DE HIPERPARÁMETROS
DEFAULT_PARAMS = {
    "Random Forest": {
        "n_estimators": 200, "max_depth": 20,
        "min_samples_split": 2, "min_samples_leaf": 1,
    },
    "SVR": {"C": 10.0, "epsilon": 0.1, "gamma": "scale"},
    "XGBoost": {
        "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
    },
    "LSTM": {"units": 64, "dropout": 0.2, "dense": 32},
}


def _predefined_split(n_train, n_val):
    test_fold = np.concatenate([
        -np.ones(n_train, dtype=int),
        np.zeros(n_val, dtype=int),
    ])
    return PredefinedSplit(test_fold)


def _subsample(X, y, max_samples):
    n = len(X)
    if n > max_samples:
        stride = int(np.ceil(n / max_samples))
        idx = np.arange(0, n, stride)
        return X[idx], y[idx]
    return X, y


def tune_random_forest(X_train, y_train, X_val, y_val):
    ps = _predefined_split(len(y_train), len(y_val))
    est = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    search = RandomizedSearchCV(
        est, RF_PARAM_GRID, n_iter=RF_N_ITER,
        scoring="neg_root_mean_squared_error", cv=ps,
        random_state=RANDOM_STATE, n_jobs=1, refit=False,
    )
    search.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
    return dict(search.best_params_), -float(search.best_score_)


def tune_xgboost(X_train, y_train, X_val, y_val):
    if XGBRegressor is None:
        raise RuntimeError("XGBoost no está instalado.")
    ps = _predefined_split(len(y_train), len(y_val))
    est = XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE,
                       n_jobs=-1)
    search = RandomizedSearchCV(
        est, XGB_PARAM_GRID, n_iter=XGB_N_ITER,
        scoring="neg_root_mean_squared_error", cv=ps,
        random_state=RANDOM_STATE, n_jobs=1, refit=False,
    )
    search.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
    return dict(search.best_params_), -float(search.best_score_)


def tune_svr(X_train, y_train, X_val, y_val):
    y_train = np.asarray(y_train, dtype=float).reshape(-1, 1)
    y_val = np.asarray(y_val, dtype=float).reshape(-1, 1)

    feature_scaler = StandardScaler().fit(X_train)
    X_train_s = feature_scaler.transform(X_train)
    X_val_s = feature_scaler.transform(X_val)

    target_scaler = StandardScaler().fit(y_train)
    y_train_s = target_scaler.transform(y_train).ravel()
    y_val_s = target_scaler.transform(y_val).ravel()

    X_train_s, y_train_s = _subsample(X_train_s, y_train_s, SVR_MAX_SAMPLES)

    best_params, best_rmse = None, np.inf
    for C in SVR_PARAM_GRID["svr__C"]:
        for eps in SVR_PARAM_GRID["svr__epsilon"]:
            for gamma in SVR_PARAM_GRID["svr__gamma"]:
                model = SVR(kernel="rbf", C=C, epsilon=eps, gamma=gamma)
                model.fit(X_train_s, y_train_s)
                preds_s = model.predict(X_val_s)
                preds_mw = target_scaler.inverse_transform(
                    preds_s.reshape(-1, 1)).ravel()
                rmse_mw = float(np.sqrt(mean_squared_error(y_val.ravel(), preds_mw)))
                if rmse_mw < best_rmse:
                    best_rmse = rmse_mw
                    best_params = {"C": C, "epsilon": eps, "gamma": gamma}
    return best_params, best_rmse


def tune_lstm(hourly_df, positions, y, feature_cols, window,
              train_mask, val_mask):
    if tf is None:
        raise RuntimeError("TensorFlow no está instalado.")

    X = build_sequences(hourly_df, positions, feature_cols, window)
    n_samples, w, n_feat = X.shape
    X_train = X[train_mask]
    X_val = X[val_mask]

    y = np.asarray(y, dtype=float)
    y_train = y[train_mask].reshape(-1, 1)
    y_val = y[val_mask].reshape(-1, 1)

    feature_scaler = StandardScaler().fit(X_train.reshape(-1, n_feat))
    X_train_s = feature_scaler.transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape)
    X_val_s = feature_scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape)

    target_scaler = StandardScaler().fit(y_train)
    y_train_s = target_scaler.transform(y_train)
    y_val_s = target_scaler.transform(y_val)

    best_params, best_rmse = None, np.inf
    for cfg in LSTM_TUNING_GRID:
        set_reproducible()
        if tf is not None:
            tf.keras.backend.clear_session()
        model = Sequential([
            LSTM(cfg["units"], input_shape=(w, n_feat), return_sequences=False),
            Dropout(cfg["dropout"]),
            Dense(32, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True)
        model.fit(X_train_s, y_train_s,
                  validation_data=(X_val_s, y_val_s),
                  epochs=100, batch_size=64,
                  callbacks=[early_stopping], verbose=0)
        preds_s = model.predict(X_val_s, verbose=0)
        preds_mw = target_scaler.inverse_transform(preds_s).ravel()
        rmse_mw = float(np.sqrt(mean_squared_error(y_val.ravel(), preds_mw)))
        if rmse_mw < best_rmse:
            best_rmse = rmse_mw
            best_params = {"units": cfg["units"], "dropout": cfg["dropout"],
                           "dense": 32}
    return best_params, best_rmse


# IMPORTANCIA DE INDICADORES
def compute_feature_importance(rf_model, xgb_model, X_test, y_test, feature_names):
    rows = []

    # Importancia nativa de Random Forest
    rf_imp = rf_model.feature_importances_
    for feat, imp in sorted(zip(feature_names, rf_imp), key=lambda x: -x[1]):
        rows.append({"model": "Random Forest", "method": "native",
                     "feature": feat, "importance": float(imp)})

    # Importancia nativa de XGBoost
    if xgb_model is not None:
        xgb_imp = xgb_model.feature_importances_
        for feat, imp in sorted(zip(feature_names, xgb_imp), key=lambda x: -x[1]):
            rows.append({"model": "XGBoost", "method": "native",
                         "feature": feat, "importance": float(imp)})

    # Permutation importance (submuestra del test)
    n = min(PERMUTATION_SAMPLES, len(X_test))
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X_test), n, replace=False)
    X_sub = X_test[idx]
    y_sub = np.asarray(y_test)[idx]

    for name, model in [("Random Forest", rf_model), ("XGBoost", xgb_model)]:
        if model is None:
            continue
        perm = permutation_importance(
            model, X_sub, y_sub, n_repeats=10,
            random_state=RANDOM_STATE, n_jobs=-1, scoring="neg_mean_absolute_error"
        )
        for feat, imp in sorted(zip(feature_names, perm.importances_mean),
                                key=lambda x: -x[1]):
            rows.append({"model": name, "method": "permutation",
                         "feature": feat, "importance": float(imp)})

    df = pd.DataFrame(rows)

    # Rank dentro de cada (model, method)
    df["rank"] = df.groupby(["model", "method"])["importance"].rank(
        ascending=False, method="first"
    ).astype(int)
    df = df.sort_values(["model", "method", "rank"]).reset_index(drop=True)
    return df


# FIGURAS
def setup_plot_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "figure.dpi": 100,
        "savefig.dpi": 300,
    })


GRAY = "#000000"
GRAYS = ["#000000", "#555555", "#999999", "#cccccc"]
MODEL_ORDER = ["Random Forest", "SVR", "XGBoost", "LSTM"]
MODEL_MARKERS = {"Random Forest": "o", "SVR": "s", "XGBoost": "^", "LSTM": "D"}
MODEL_LINESTYLES = {"Random Forest": "-", "SVR": "--", "XGBoost": "-.", "LSTM": ":"}


def fig1_methodology():
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.axis("off")
    boxes = [
        "Shahrekord\ndataset\n(hourly, MW)",
        "Preprocessing\n(cleaning, calendar,\nlags, rolling means)",
        "Indicator groups\nA history · B weather\nC temporal · D special",
        "Scenarios\nA · B · C · D",
        "Models\nRF · SVR · XGB · LSTM",
        "Evaluation\nMAE · RMSE · R² · MAPE\nimportance · high demand",
    ]
    n = len(boxes)
    x_positions = np.linspace(0.03, 0.97, n)
    for x, label in zip(x_positions, boxes):
        ax.text(x, 0.5, label, transform=ax.transAxes, ha="center", va="center",
                fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor=GRAY, linewidth=1.0))
    for i in range(n - 1):
        ax.annotate("", xy=(x_positions[i + 1] - 0.045, 0.5),
                    xytext=(x_positions[i] + 0.045, 0.5),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", lw=1.4, color=GRAY))
    ax.set_title("Experimental methodology", fontsize=14)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "Fig1_methodology.png")
    ensure_dir(path)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figura creada: {path}")


def fig2_demand(df):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(df["datetime"], df["demand"], linewidth=0.6, color="black")
    ax.set_title("Hourly electricity demand (Shahrekord, 2015-2019)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand (MW)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "Fig2_electricity_demand.png")
    ensure_dir(path)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figura creada: {path}")


def fig3_predictions(pred_df, models):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    display = pred_df.iloc[-FIG3_DISPLAY_STEPS:].copy()
    x = pd.to_datetime(display["timestamp"])
    ax.plot(x, display["actual"], label="Actual",
            linewidth=1.8, color="black")
    for i, m in enumerate(models):
        if m in display.columns:
            ax.plot(x, display[m], label=m,
                    linewidth=1.0, color=GRAYS[(i + 1) % len(GRAYS)],
                    linestyle=MODEL_LINESTYLES.get(m, "-"))
    ax.set_title("Actual vs. predicted demand (test set, representative window)")
    ax.set_xlabel("Date and time")
    ax.set_ylabel("Demand (MW)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "Fig3_predictions.png")
    ensure_dir(path)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figura creada: {path}")


def fig4_model_comparison(results):
    results = results.sort_values("RMSE").reset_index(drop=True)
    models = results["model"].tolist()
    mae = results["MAE"].tolist()
    rmse = results["RMSE"].tolist()
    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w / 2, mae, w, label="MAE", color="#555555", edgecolor="black")
    b2 = ax.bar(x + w / 2, rmse, w, label="RMSE", color="#cccccc", edgecolor="black")
    ax.set_title("Error comparison among predictive models (full model, test set)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Error (MW)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "Fig4_model_comparison.png")
    ensure_dir(path)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figura creada: {path}")


def fig5_complexity(results):
    fig, ax = plt.subplots(figsize=(9, 5))
    for _, row in results.iterrows():
        ax.scatter(row["training_time"], row["RMSE"], s=120,
                   facecolor="white", edgecolor="black", linewidth=1.5)
        ax.annotate(row["model"], (row["training_time"], row["RMSE"]),
                    xytext=(8, 8), textcoords="offset points", fontsize=10)
    ax.set_title("Predictive accuracy vs. computational cost")
    ax.set_xlabel("Training time (s, log scale)")
    ax.set_ylabel("RMSE (MW)")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "Fig5_accuracy_vs_complexity.png")
    ensure_dir(path)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figura creada: {path}")


def fig6_importance(feat_df):
    perm = feat_df[feat_df["method"] == "permutation"]
    if perm.empty:
        return
    models = [m for m in ["Random Forest", "XGBoost"] if (perm["model"] == m).any()]
    fig, axes = plt.subplots(len(models), 1, figsize=(9, 3.4 * len(models)))
    if len(models) == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        sub = (perm[perm["model"] == m]
               .sort_values("importance", ascending=False)
               .head(FIG6_TOP_N).iloc[::-1])
        bars = ax.barh(sub["feature"], sub["importance"],
                       color="#555555", edgecolor="black")
        ax.set_title(f"Permutation importance - {m}")
        ax.set_xlabel("Mean importance (MAE increase, MW)")
        ax.grid(axis="x", alpha=0.3)
        for b, v in zip(bars, sub["importance"]):
            ax.text(v, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                    va="center", ha="left", fontsize=8)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "Fig6_feature_importance.png")
    ensure_dir(path)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figura creada: {path}")


def fig7_high_demand(high_df):
    high_df = high_df.set_index("model").reindex(
        [m for m in MODEL_ORDER if m in high_df["model"].values]
    ).reset_index()
    models = high_df["model"].tolist()
    mae_n = high_df["MAE_normal"].tolist()
    mae_h = high_df["MAE_high_demand"].tolist()
    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w / 2, mae_n, w, label="MAE normal", color="#555555",
                edgecolor="black")
    b2 = ax.bar(x + w / 2, mae_h, w, label="MAE high demand", color="#cccccc",
                edgecolor="black")
    ax.set_title("Error under normal vs. high-demand periods (test set)")
    ax.set_xlabel("Model")
    ax.set_ylabel("MAE (MW)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    max_h = max(max(mae_n), max(mae_h))
    ax.set_ylim(0, max_h * 1.15)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "Fig7_high_demand_error.png")
    ensure_dir(path)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figura creada: {path}")


def fig8_scenarios(scen_df):
    pivot = scen_df.pivot(index="model", columns="scenario", values="MAE")
    models = [m for m in MODEL_ORDER if m in pivot.index]
    scenarios = pivot.columns.tolist()
    x = np.arange(len(scenarios))
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(models):
        offset = (i - (len(models) - 1) / 2) * w
        vals = pivot.loc[m].values
        bars = ax.bar(x + offset, vals, w, label=m, color=GRAYS[i % len(GRAYS)],
                      edgecolor="black", linewidth=0.6)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_title("MAE across experimental scenarios")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("MAE (MW)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s} ({SCENARIOS[s]['label']})" for s in scenarios],
                       rotation=15, fontsize=9)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, pivot.values.max() * 1.15)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "Fig8_feature_scenario_comparison.png")
    ensure_dir(path)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figura creada: {path}")


def analyze_meteorology(data):
    """Analiza la calidad y la correlación de las variables meteorológicas
    (Grupo B) con la demanda y con las variables temporales, para explicar
    por qué su incorporación no aporta una mejora predictiva adicional."""
    temporal_refs = ["hour_sin", "hour_cos", "month_sin", "month_cos",
                     "day_of_year", "day_of_week"]
    cols = ["target", "lag_0"] + GROUP_B + [c for c in temporal_refs
                                             if c in data.columns]
    sub = data[cols].dropna()
    corr = sub.corr(method="pearson")
    rows = []
    for v in GROUP_B:
        row = {"variable": v, "corr_target": corr.loc[v, "target"],
               "corr_lag0": corr.loc[v, "lag_0"]}
        for t in temporal_refs:
            if t in corr.index:
                row[f"corr_{t}"] = corr.loc[v, t]
        rows.append(row)
    met_df = pd.DataFrame(rows).sort_values(
        "corr_target", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

    path = os.path.join(EXPERIMENT_DIR, "meteorological_analysis.csv")
    ensure_dir(path)
    met_df.to_csv(path, index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_pos = np.arange(len(met_df))
    ax.barh(y_pos, met_df["corr_target"], height=0.55, color="#999999",
            edgecolor="black", label="Correlación con la demanda (t+1)")
    ax.scatter(met_df["corr_lag0"], y_pos, color="black", marker="o", s=60,
               zorder=3, label="Correlación con la demanda actual (lag_0)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(met_df["variable"], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coeficiente de correlación de Pearson")
    ax.set_title("Correlación de las variables meteorológicas con la demanda "
                 "(t+1) y con la demanda actual (lag_0)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "Fig9_weather_correlation.png")
    ensure_dir(fig_path)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Análisis meteorológico guardado en: {path}")
    print(f"Figura creada: {fig_path}")
    return met_df


def rolling_origin_validation(data, X_all, y, best_params, hourly, positions,
                              window):
    """Validación de origen móvil (rolling origin) con 3 ventanas de prueba
    semestrales sucesivas para comprobar la estabilidad de las métricas."""
    year = data["year"].to_numpy()
    month = data["month"].to_numpy()
    D_features = SCENARIOS["D"]["features"]
    lstm_features = LSTM_SCENARIO_FEATURES["D"]

    def half_mask(yy, h):
        if h == 1:
            return (year == yy) & (month <= 6)
        return (year == yy) & (month >= 7)

    folds = [
        {"name": "Ventana 1", "train": (year <= 2),
         "val": half_mask(3, 1), "test": half_mask(3, 2)},
        {"name": "Ventana 2", "train": (year <= 2) | half_mask(3, 1),
         "val": half_mask(3, 2), "test": half_mask(4, 1)},
        {"name": "Ventana 3", "train": (year <= 3),
         "val": half_mask(4, 1), "test": half_mask(4, 2)},
    ]

    rows = []
    for fold in folds:
        tr = fold["train"]
        va = fold["val"]
        te = fold["test"]
        X_tr = X_all[D_features].to_numpy()[tr]
        X_te = X_all[D_features].to_numpy()[te]
        y_tr = y[tr].to_numpy()
        y_te = y[te].to_numpy()
        for model in ["Random Forest", "SVR", "XGBoost", "LSTM"]:
            try:
                if model == "Random Forest":
                    _, preds, _, _ = train_random_forest(
                        X_tr, y_tr, X_te, best_params["Random Forest"])
                elif model == "SVR":
                    _, preds, _, _ = train_svr(
                        X_tr, y_tr, X_te, best_params["SVR"])
                elif model == "XGBoost":
                    _, preds, _, _ = train_xgboost(
                        X_tr, y_tr, X_te, best_params["XGBoost"])
                else:
                    _, preds, _, _, _ = train_lstm(
                        hourly, positions, y.to_numpy(), lstm_features, window,
                        tr, va, te, best_params["LSTM"])
                met = calc_metrics(y_te, preds)
                rows.append({"fold": fold["name"], "model": model,
                             "test_samples": int(te.sum()),
                             "MAE": met["MAE"], "RMSE": met["RMSE"],
                             "R2": met["R2"], "MAPE": met["MAPE"]})
            except Exception as e:
                print(f"  [ERROR] {fold['name']} - {model}: {e}")

    tv_df = pd.DataFrame(rows)
    path = os.path.join(EXPERIMENT_DIR, "temporal_validation.csv")
    ensure_dir(path)
    tv_df.to_csv(path, index=False)

    summary_rows = []
    for model in ["Random Forest", "SVR", "XGBoost", "LSTM"]:
        sub = tv_df[tv_df["model"] == model]
        summary_rows.append({
            "model": model,
            "MAE_mean": sub["MAE"].mean(), "MAE_std": sub["MAE"].std(),
            "RMSE_mean": sub["RMSE"].mean(), "RMSE_std": sub["RMSE"].std(),
            "R2_mean": sub["R2"].mean(), "R2_std": sub["R2"].std(),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(EXPERIMENT_DIR, "temporal_validation_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    fold_names = [f["name"] for f in folds]
    x = np.arange(len(fold_names))
    for i, m in enumerate(["Random Forest", "SVR", "XGBoost", "LSTM"]):
        vals = tv_df[tv_df["model"] == m]["MAE"].values
        ax.plot(x, vals, marker=MODEL_MARKERS[m], linestyle=MODEL_LINESTYLES[m],
                color=GRAYS[i % len(GRAYS)], label=m, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(fold_names)
    ax.set_xlabel("Ventana de prueba (origen móvil)")
    ax.set_ylabel("MAE (MW)")
    ax.set_title("Estabilidad del MAE entre ventanas de prueba sucesivas")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "Fig10_temporal_validation.png")
    ensure_dir(fig_path)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Validación temporal guardada en: {path}")
    print(f"Resumen (media ± desv. estándar) en: {summary_path}")
    print(f"Figura creada: {fig_path}")
    return tv_df, summary_df


# EJECUCIÓN PRINCIPAL
def main():
    print("=" * 72)
    print("COMPARATIVE ML / DL EXPERIMENT - SHORT-TERM LOAD FORECASTING")
    print("Shahrekord Electricity Demand Dataset")
    print("=" * 72)

    set_reproducible()
    setup_plot_style()

    # 1. Carga
    df = load_dataset(DATA_PATH)
    n_original_cols = df.shape[1]

    # 2. Características temporales y lags
    df = build_temporal_features(df)
    data = build_lag_features(df)

    # 3. Eliminar filas con NaN (introducidos por lags y target)
    data = data.dropna(subset=["target"] + GROUP_A).reset_index(drop=True)
    # Ordenar por datetime tras el reset
    data = data.sort_values("datetime").reset_index(drop=True)

    feature_names_all = (GROUP_A + GROUP_B + GROUP_C + GROUP_D)
    X_all = data[feature_names_all].astype(float)
    y = data["target"].astype(float)

    # 4. División temporal por años
    year_arr = data["year"].to_numpy()
    train_mask = np.isin(year_arr, TRAIN_YEARS)
    val_mask = np.isin(year_arr, VALID_YEARS)
    test_mask = np.isin(year_arr, TEST_YEARS)

    print_section("DIVISIÓN TEMPORAL (AÑOS COMPLETOS)")
    print(f"Train      (años {TRAIN_YEARS}): {train_mask.sum():,}")
    print(f"Validation (años {VALID_YEARS}): {val_mask.sum():,}")
    print(f"Test       (años {TEST_YEARS}): {test_mask.sum():,}")

    # 5. Umbral de alta demanda (SOLO train)
    high_threshold = float(np.percentile(y[train_mask], HIGH_DEMAND_PERCENTILE))
    print(f"\nUmbral de alta demanda (P{HIGH_DEMAND_PERCENTILE} de train): "
          f"{high_threshold:.2f} MW")

    # Preparación de secuencias LSTM (posiciones en el df horario)
    hourly = df.set_index("datetime", drop=False).sort_index()
    positions = hourly.index.get_indexer(data["datetime"])

    # 5b. Optimización de hiperparámetros (independiente por escenario)
    print_section("OPTIMIZACIÓN DE HIPERPARÁMETROS (POR ESCENARIO)")
    print("Búsqueda controlada: train (años 1-2) + validación (año 3).")
    print("El conjunto de prueba (año 4) NO participa en la selección.")
    print("Cada escenario se optimiza de forma independiente para que la")
    print("comparación entre escenarios aísle el efecto de cada grupo de")
    print("variables sin asumir que la configuración óptima del modelo")
    print("completo es igualmente adecuada para el resto de escenarios.")

    # 6. Ejecución de escenarios: optimización + entrenamiento + evaluación
    scenario_rows = []
    scenario_models = {}   # scenario -> {model: preds(test)}
    best_params_by_scenario = {}
    tuning_rows = []

    for scen_name, scen in SCENARIOS.items():
        print_section(f"ESCENARIO {scen_name} - {scen['label']} "
                      f"({len(scen['features'])} features)")

        X = X_all[scen["features"]].to_numpy()
        X_train = X[train_mask]
        X_val = X[val_mask]
        X_test = X[test_mask]
        y_train = y[train_mask].to_numpy()
        y_val = y[val_mask].to_numpy()
        y_test = y[test_mask].to_numpy()

        params = {k: dict(v) for k, v in DEFAULT_PARAMS.items()}

        # ---- Optimización de hiperparámetros (por escenario) ----
        try:
            params["Random Forest"], val_rmse_rf = tune_random_forest(
                X_train, y_train, X_val, y_val)
            tuning_rows.append({"scenario": scen_name, "model": "Random Forest",
                                "method": "RandomizedSearchCV",
                                "best_params": str(params["Random Forest"]),
                                "val_RMSE_MW": round(val_rmse_rf, 4)})
            print(f"  [tuning] Random Forest: {params['Random Forest']} "
                  f"(val RMSE={val_rmse_rf:.2f} MW)")
        except Exception as e:
            print(f"  [ERROR] tuning Random Forest: {e}")

        try:
            params["SVR"], val_rmse_svr = tune_svr(
                X_train, y_train, X_val, y_val)
            tuning_rows.append({"scenario": scen_name, "model": "SVR",
                                "method": "Grid search (C/epsilon/gamma)",
                                "best_params": str(params["SVR"]),
                                "val_RMSE_MW": round(val_rmse_svr, 4)})
            print(f"  [tuning] SVR: {params['SVR']} (val RMSE={val_rmse_svr:.2f} MW)")
        except Exception as e:
            print(f"  [ERROR] tuning SVR: {e}")

        try:
            params["XGBoost"], val_rmse_xgb = tune_xgboost(
                X_train, y_train, X_val, y_val)
            tuning_rows.append({"scenario": scen_name, "model": "XGBoost",
                                "method": "RandomizedSearchCV",
                                "best_params": str(params["XGBoost"]),
                                "val_RMSE_MW": round(val_rmse_xgb, 4)})
            print(f"  [tuning] XGBoost: {params['XGBoost']} "
                  f"(val RMSE={val_rmse_xgb:.2f} MW)")
        except Exception as e:
            print(f"  [ERROR] tuning XGBoost: {e}")

        try:
            params["LSTM"], val_rmse_lstm = tune_lstm(
                hourly, positions, y.to_numpy(),
                LSTM_SCENARIO_FEATURES[scen_name], LSTM_WINDOW,
                train_mask, val_mask)
            tuning_rows.append({"scenario": scen_name, "model": "LSTM",
                                "method": "Grid (units/dropout)",
                                "best_params": str(params["LSTM"]),
                                "val_RMSE_MW": round(val_rmse_lstm, 4)})
            print(f"  [tuning] LSTM: {params['LSTM']} (val RMSE={val_rmse_lstm:.2f} MW)")
        except Exception as e:
            print(f"  [ERROR] tuning LSTM: {e}")

        best_params_by_scenario[scen_name] = params
        scenario_models[scen_name] = {}

        # ---- Entrenamiento y evaluación ----
        try:
            m, preds, tt, tp = train_random_forest(
                X_train, y_train, X_test, params["Random Forest"])
            met = calc_metrics(y_test, preds)
            scenario_rows.append({"scenario": scen_name, "model": "Random Forest",
                                  **met, "training_time": tt, "prediction_time": tp})
            scenario_models[scen_name]["Random Forest"] = preds
            print(f"  Random Forest: MAE={met['MAE']:.2f} RMSE={met['RMSE']:.2f} "
                  f"R2={met['R2']:.3f}")
            if scen_name == "D":
                rf_full, rf_preds_full = m, preds
        except Exception as e:
            print(f"  [ERROR] Random Forest: {e}")

        try:
            m, preds, tt, tp = train_svr(X_train, y_train, X_test,
                                         params["SVR"])
            met = calc_metrics(y_test, preds)
            scenario_rows.append({"scenario": scen_name, "model": "SVR",
                                  **met, "training_time": tt, "prediction_time": tp})
            scenario_models[scen_name]["SVR"] = preds
            print(f"  SVR: MAE={met['MAE']:.2f} RMSE={met['RMSE']:.2f} "
                  f"R2={met['R2']:.3f}")
        except Exception as e:
            print(f"  [ERROR] SVR: {e}")

        try:
            m, preds, tt, tp = train_xgboost(X_train, y_train, X_test,
                                             params["XGBoost"])
            met = calc_metrics(y_test, preds)
            scenario_rows.append({"scenario": scen_name, "model": "XGBoost",
                                  **met, "training_time": tt, "prediction_time": tp})
            scenario_models[scen_name]["XGBoost"] = preds
            print(f"  XGBoost: MAE={met['MAE']:.2f} RMSE={met['RMSE']:.2f} "
                  f"R2={met['R2']:.3f}")
            if scen_name == "D":
                xgb_full, xgb_preds_full = m, preds
        except Exception as e:
            print(f"  [ERROR] XGBoost: {e}")

        try:
            m, preds, tt, tp, n_feat = train_lstm(
                hourly, positions, y.to_numpy(),
                LSTM_SCENARIO_FEATURES[scen_name], LSTM_WINDOW,
                train_mask, val_mask, test_mask, params["LSTM"]
            )
            met = calc_metrics(y_test, preds)
            scenario_rows.append({"scenario": scen_name, "model": "LSTM",
                                  **met, "training_time": tt, "prediction_time": tp})
            scenario_models[scen_name]["LSTM"] = preds
            print(f"  LSTM: MAE={met['MAE']:.2f} RMSE={met['RMSE']:.2f} "
                  f"R2={met['R2']:.3f}")
        except Exception as e:
            print(f"  [ERROR] LSTM: {e}")

    best_params = best_params_by_scenario["D"]

    tuning_df = pd.DataFrame(tuning_rows)
    ensure_dir(TUNING_PATH)
    tuning_df.to_csv(TUNING_PATH, index=False)
    print(f"\nReporte de optimización guardado en: {TUNING_PATH}")

    # 7. Resultados de escenarios
    scen_df = pd.DataFrame(scenario_rows)
    scen_df = scen_df[["scenario", "model", "MAE", "RMSE", "R2", "MAPE",
                       "training_time", "prediction_time"]]
    ensure_dir(SCENARIOS_PATH)
    scen_df.to_csv(SCENARIOS_PATH, index=False)
    print_section("RESULTADOS POR ESCENARIO")
    print(scen_df.to_string(index=False))

    # 8. results.csv = escenario completo (D)
    results_df = scen_df[scen_df["scenario"] == "D"].copy()
    results_df = results_df.sort_values("RMSE").reset_index(drop=True)
    results_df.insert(0, "rank", range(1, len(results_df) + 1))
    ensure_dir(RESULTS_PATH)
    results_df.to_csv(RESULTS_PATH, index=False)
    print_section("RESULTADOS PRINCIPALES (ESCENARIO D)")
    print(results_df.to_string(index=False))

    # 9. predictions.csv (escenario D)
    test_dt = data.loc[test_mask, "datetime"]
    pred_df = pd.DataFrame({"timestamp": test_dt.values,
                            "actual": y[test_mask].to_numpy()})
    for m, p in scenario_models.get("D", {}).items():
        pred_df[m] = p
    ensure_dir(PREDICTIONS_PATH)
    pred_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"\nPredicciones guardadas en: {PREDICTIONS_PATH}")

    # 10. Importancia de indicadores (escenario D)
    try:
        X_test_D = X_all[SCENARIOS["D"]["features"]].to_numpy()[test_mask]
        feat_df = compute_feature_importance(
            rf_full, xgb_full, X_test_D, y[test_mask].to_numpy(),
            SCENARIOS["D"]["features"]
        )
        ensure_dir(FEATURE_IMPORTANCE_PATH)
        feat_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
        print_section("IMPORTANCIA DE INDICADORES (TOP 10 - PERMUTATION)")
        perm = feat_df[feat_df["method"] == "permutation"]
        top10 = (perm.groupby("feature")["importance"].mean()
                 .sort_values(ascending=False).head(10))
        print(top10.to_string())
    except Exception as e:
        feat_df = None
        print(f"[ERROR] Importancia de variables: {e}")

    # 11. Análisis de alta demanda (escenario D)
    high_rows = []
    y_test_arr = y[test_mask].to_numpy()
    high_mask = y_test_arr >= high_threshold
    for m, p in scenario_models.get("D", {}).items():
        met_n = calc_metrics(y_test_arr[~high_mask], p[~high_mask])
        met_h = calc_metrics(y_test_arr[high_mask], p[high_mask])
        high_rows.append({
            "model": m,
            "MAE_normal": met_n["MAE"], "RMSE_normal": met_n["RMSE"],
            "MAPE_normal": met_n["MAPE"],
            "MAE_high_demand": met_h["MAE"], "RMSE_high_demand": met_h["RMSE"],
            "MAPE_high_demand": met_h["MAPE"],
            "high_demand_threshold_MW": high_threshold,
            "n_normal": int((~high_mask).sum()),
            "n_high_demand": int(high_mask.sum()),
        })
    high_df = pd.DataFrame(high_rows)
    ensure_dir(HIGH_DEMAND_PATH)
    high_df.to_csv(HIGH_DEMAND_PATH, index=False)
    print_section("ANÁLISIS DE ALTA DEMANDA")
    print(high_df.to_string(index=False))

    # 11b. Baseline de persistencia (naive): Load(t+1) = Load(t)
    baseline_path = os.path.join(EXPERIMENT_DIR, "baseline_results.csv")
    try:
        y_true_b = data["target"].to_numpy()[test_mask]
        y_prev_b = data["demand"].to_numpy()[test_mask]  # lag_0 = demanda(t)
        mae_b = float(mean_absolute_error(y_true_b, y_prev_b))
        rmse_b = float(np.sqrt(mean_squared_error(y_true_b, y_prev_b)))
        r2_b = float(r2_score(y_true_b, y_prev_b))
        mape_b = float(mape(y_true_b, y_prev_b))
        pd.DataFrame([{
            "model": "Persistence (naive)", "MAE": mae_b, "RMSE": rmse_b,
            "R2": r2_b, "MAPE": mape_b, "test_samples": int(test_mask.sum()),
            "description": "predict Load(t+1) = Load(t), test set (year 4)",
        }]).to_csv(baseline_path, index=False)
        print_section("BASELINE DE PERSISTENCIA (NAIVE)")
        print(f"MAE={mae_b:.2f} MW  RMSE={rmse_b:.2f} MW  "
              f"R2={r2_b:.4f}  MAPE={mape_b:.2f}%")
    except Exception as e:
        print(f"[ERROR] Baseline de persistencia: {e}")

    # 11c. Análisis de variables meteorológicas (calidad y correlación)
    print_section("ANÁLISIS DE VARIABLES METEOROLÓGICAS")
    try:
        met_df = analyze_meteorology(data)
        print(met_df.head(10).to_string(index=False))
    except Exception as e:
        met_df = None
        print(f"[ERROR] Análisis meteorológico: {e}")

    # 11d. Validación temporal de origen móvil (ventanas de prueba sucesivas)
    print_section("VALIDACIÓN TEMPORAL (ORIGEN MÓVIL)")
    try:
        tv_df, tv_summary = rolling_origin_validation(
            data, X_all, y, best_params, hourly, positions, LSTM_WINDOW)
        print(tv_summary.to_string(index=False))
    except Exception as e:
        tv_df = None
        tv_summary = None
        print(f"[ERROR] Validación temporal: {e}")

    # 11e. Múltiples ejecuciones para tiempos (media ± desv. estándar)
    print_section("TIEMPOS DE EJECUCIÓN (MEDIA ± DESV. ESTÁNDAR)")
    try:
        X_tD = X_all[SCENARIOS["D"]["features"]].to_numpy()
        X_trD = X_tD[train_mask]
        X_teD = X_tD[test_mask]
        y_trD = y[train_mask].to_numpy()
        y_teD = y[test_mask].to_numpy()
        timing_rows = []
        for run in range(N_RUNS):
            for model in ["Random Forest", "SVR", "XGBoost", "LSTM"]:
                if model == "Random Forest":
                    _, _, tt, tp = train_random_forest(
                        X_trD, y_trD, X_teD, best_params["Random Forest"])
                elif model == "SVR":
                    _, _, tt, tp = train_svr(
                        X_trD, y_trD, X_teD, best_params["SVR"])
                elif model == "XGBoost":
                    _, _, tt, tp = train_xgboost(
                        X_trD, y_trD, X_teD, best_params["XGBoost"])
                else:
                    _, _, tt, tp, _ = train_lstm(
                        hourly, positions, y.to_numpy(),
                        LSTM_SCENARIO_FEATURES["D"], LSTM_WINDOW,
                        train_mask, val_mask, test_mask, best_params["LSTM"])
                timing_rows.append({"run": run + 1, "model": model,
                                    "training_time": tt, "prediction_time": tp})
        timing_df = pd.DataFrame(timing_rows)
        timing_summary = (timing_df.groupby("model")
                          .agg(tt_mean=("training_time", "mean"),
                               tt_std=("training_time", "std"),
                               tp_mean=("prediction_time", "mean"),
                               tp_std=("prediction_time", "std"))
                          .reset_index())
        timing_path = os.path.join(EXPERIMENT_DIR, "timing_results.csv")
        ensure_dir(timing_path)
        timing_df.to_csv(timing_path, index=False)
        timing_summary_path = os.path.join(EXPERIMENT_DIR, "timing_summary.csv")
        timing_summary.to_csv(timing_summary_path, index=False)
        print(timing_summary.to_string(index=False))
    except Exception as e:
        timing_df = None
        timing_summary = None
        print(f"[ERROR] Tiempos de ejecución: {e}")

    # 12. model_configuration.csv
    config_rows = []
    d_full = scen_df[(scen_df["scenario"] == "D")].set_index("model")
    hyper = {
        "Random Forest": ", ".join(f"{k}={v}" for k, v in best_params["Random Forest"].items()),
        "SVR": "kernel=rbf, " + ", ".join(f"{k}={v}" for k, v in best_params["SVR"].items())
               + f", max_train_samples={SVR_MAX_SAMPLES}",
        "XGBoost": ", ".join(f"{k}={v}" for k, v in best_params["XGBoost"].items()),
        "LSTM": (f"LSTM({best_params['LSTM']['units']})"
                 f"+Dropout({best_params['LSTM']['dropout']})"
                 f"+Dense({best_params['LSTM']['dense']})+Dense(1), "
                 "adam, mse, epochs<=100, early_stopping(patience=10)"),
    }
    for m in ["Random Forest", "SVR", "XGBoost", "LSTM"]:
        if m in d_full.index:
            seq_len = LSTM_WINDOW if m == "LSTM" else "N/A"
            n_feat = len(LSTM_SCENARIO_FEATURES["D"]) if m == "LSTM" else len(SCENARIOS["D"]["features"])
            config_rows.append({
                "model": m,
                "hyperparameters": hyper[m],
                "input_features": n_feat,
                "input_feature_list": "; ".join(
                    LSTM_SCENARIO_FEATURES["D"] if m == "LSTM" else SCENARIOS["D"]["features"]
                ),
                "sequence_length": seq_len,
                "scaler": "StandardScaler (fit on train only)",
                "training_samples": int(train_mask.sum()),
                "validation_samples": int(val_mask.sum()),
                "test_samples": int(test_mask.sum()),
                "training_time": float(d_full.loc[m, "training_time"]),
                "prediction_time": float(d_full.loc[m, "prediction_time"]),
            })
    config_df = pd.DataFrame(config_rows)
    ensure_dir(CONFIG_PATH)
    config_df.to_csv(CONFIG_PATH, index=False)

    # 13. Figuras
    print_section("GENERACIÓN DE FIGURAS")
    fig1_methodology()
    fig2_demand(df)
    fig3_predictions(pred_df, list(scenario_models.get("D", {}).keys()))
    fig4_model_comparison(results_df)
    fig5_complexity(results_df)
    if feat_df is not None:
        fig6_importance(feat_df)
    fig7_high_demand(high_df)
    fig8_scenarios(scen_df)

    # 14. experiment_summary.txt
    write_summary(df, data, feat_df, results_df, scen_df, high_df,
                  train_mask, val_mask, test_mask, high_threshold,
                  n_original_cols, best_params, tv_summary, timing_summary)

    print_section("EXPERIMENTO FINALIZADO")
    print(f"Mejor modelo (escenario D, RMSE): {results_df.iloc[0]['model']}")
    print("\nArchivos generados en 3-Experiment/:")
    for f in ["results.csv", "predictions.csv", "feature_importance.csv",
              "feature_scenarios_results.csv", "high_demand_results.csv",
              "model_configuration.csv", "hyperparameter_tuning.csv",
              "meteorological_analysis.csv", "temporal_validation.csv",
              "temporal_validation_summary.csv", "timing_results.csv",
              "timing_summary.csv", "experiment_summary.txt"]:
        print(f"  - {f}")
    print("\nFiguras generadas en 4-Figures/:")
    for f in ["Fig1_methodology.png", "Fig2_electricity_demand.png",
              "Fig3_predictions.png", "Fig4_model_comparison.png",
              "Fig5_accuracy_vs_complexity.png", "Fig6_feature_importance.png",
              "Fig7_high_demand_error.png", "Fig8_feature_scenario_comparison.png",
              "Fig9_weather_correlation.png", "Fig10_temporal_validation.png"]:
        print(f"  - {f}")


def write_summary(df, data, feat_df, results_df, scen_df, high_df,
                  train_mask, val_mask, test_mask, high_threshold,
                  n_original_cols, best_params, tv_summary=None,
                  timing_summary=None):
    lines = []
    lines.append("EXPERIMENT SUMMARY - REPRODUCIBILITY REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Title: Evaluación comparativa de modelos de aprendizaje")
    lines.append("       automático y aprendizaje profundo para la predicción")
    lines.append("       de la demanda eléctrica a corto plazo: estudio de")
    lines.append("       caso de Shahrekord.")
    lines.append("")
    lines.append("DATASET")
    lines.append("-" * 60)
    lines.append(f"  Name:        Shahrekord Electricity Demand Dataset")
    lines.append(f"  File:        94_95_96_97_all.csv")
    lines.append(f"  Location:    Shahrekord, Iran")
    lines.append(f"  Resolution:  1 hour")
    lines.append(f"  Rows:        {len(df):,}")
    lines.append(f"  Columns:     {n_original_cols} (original)")
    lines.append(f"  Period:      2015-03-03 00:00 -> 2019-03-02 23:00 (synthetic)")
    lines.append(f"  Target:      Demand (MW)")
    lines.append(f"  Missing:     0")
    lines.append(f"  Duplicates:  0")
    lines.append("")
    lines.append("PROBLEM FORMULATION")
    lines.append("-" * 60)
    lines.append("  y(t) = Load(t+1)  (next-hour demand)")
    lines.append("  No future information used in predictors.")
    lines.append("")
    lines.append("INDICATOR GROUPS")
    lines.append("-" * 60)
    lines.append(f"  A History: {len(GROUP_A)} features (lags + rolling means)")
    lines.append(f"  B Weather: {len(GROUP_B)} features")
    lines.append(f"  C Temporal: {len(GROUP_C)} features")
    lines.append(f"  D Special: {len(GROUP_D)} features")
    lines.append("")
    lines.append("SCENARIOS")
    lines.append("-" * 60)
    for s, sc in SCENARIOS.items():
        lines.append(f"  {s}: {sc['label']} ({len(sc['features'])} features)")
    lines.append("")
    lines.append("TEMPORAL SPLIT (complete years)")
    lines.append("-" * 60)
    lines.append(f"  Train:       {int(train_mask.sum()):,} samples (years {TRAIN_YEARS})")
    lines.append(f"  Validation:  {int(val_mask.sum()):,} samples (years {VALID_YEARS})")
    lines.append(f"  Test:        {int(test_mask.sum()):,} samples (years {TEST_YEARS})")
    lines.append("")
    lines.append("MODELS")
    lines.append("-" * 60)
    lines.append("  Random Forest, SVR (RBF), XGBoost, LSTM (window=24).")
    lines.append("  Hyperparameters tuned independently per scenario (A/B/C/D).")
    lines.append(f"  LSTM window (sequence length): {LSTM_WINDOW} hours")
    lines.append("")
    lines.append("HYPERPARAMETER OPTIMIZATION (per scenario, validation year 3)")
    lines.append("-" * 60)
    lines.append("  Scenario D (full model) selected hyperparameters:")
    for k, v in best_params.items():
        lines.append(f"    {k}: {v}")
    lines.append("")
    lines.append("TEMPORAL VALIDATION (rolling origin, 3 six-month test windows)")
    lines.append("-" * 60)
    if tv_summary is not None and not tv_summary.empty:
        for _, r in tv_summary.iterrows():
            lines.append(f"  {r['model']:<14} MAE={r['MAE_mean']:.2f}+/-"
                         f"{r['MAE_std']:.2f}  RMSE={r['RMSE_mean']:.2f}+/-"
                         f"{r['RMSE_std']:.2f}  R2={r['R2_mean']:.4f}+/-"
                         f"{r['R2_std']:.4f}")
    lines.append("")
    lines.append("EXECUTION TIMES (scenario D, N runs)")
    lines.append("-" * 60)
    if timing_summary is not None and not timing_summary.empty:
        for _, r in timing_summary.iterrows():
            lines.append(f"  {r['model']:<14} train={r['tt_mean']:.2f}+/-"
                         f"{r['tt_std']:.2f}s  pred={r['tp_mean']:.3f}+/-"
                         f"{r['tp_std']:.3f}s")
    lines.append("")
    lines.append("HIGH-DEMAND THRESHOLD")
    lines.append("-" * 60)
    lines.append(f"  Percentile {HIGH_DEMAND_PERCENTILE} of TRAIN demand = {high_threshold:.2f} MW")
    lines.append("")
    lines.append("RESULTS (full model, scenario D, test set)")
    lines.append("-" * 60)
    for _, r in results_df.iterrows():
        lines.append(f"  {r['model']:<14} MAE={r['MAE']:.2f}  RMSE={r['RMSE']:.2f}  "
                     f"R2={r['R2']:.4f}  MAPE={r['MAPE']:.2f}%")
    lines.append("")
    lines.append("PERSISTENCE BASELINE (naive, test set)")
    lines.append("-" * 60)
    y_true_b = data["target"].to_numpy()[test_mask]
    y_prev_b = data["demand"].to_numpy()[test_mask]
    mae_b = float(np.mean(np.abs(y_true_b - y_prev_b)))
    lines.append(f"  Predict Load(t+1) = Load(t):  MAE = {mae_b:.2f} MW")
    lines.append("")
    lines.append("ENVIRONMENT")
    lines.append("-" * 60)
    lines.append(f"  Python:       {platform.python_version()}")
    lines.append(f"  OS:           {platform.system()} {platform.release()}")
    lines.append(f"  CPU:          {get_cpu_name()}")
    lines.append(f"  RAM:          {get_ram_gb()} GB")
    lines.append(f"  GPU:          None (CPU only)")
    lines.append(f"  N runs (times): {N_RUNS}")
    lines.append(f"  numpy:        {np.__version__}")
    lines.append(f"  pandas:       {pd.__version__}")
    import sklearn
    lines.append(f"  scikit-learn: {sklearn.__version__}")
    if XGBRegressor is not None:
        import xgboost
        lines.append(f"  xgboost:      {xgboost.__version__}")
    if tf is not None:
        lines.append(f"  tensorflow:   {tf.__version__}")
    import matplotlib as mpl
    lines.append(f"  matplotlib:   {mpl.__version__}")
    lines.append(f"  Random seed:  {RANDOM_STATE}")
    lines.append("")

    ensure_dir(SUMMARY_PATH)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Resumen guardado en: {SUMMARY_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        import traceback
        print()
        print("=" * 72)
        print("ERROR DURANTE EL EXPERIMENTO")
        print("=" * 72)
        traceback.print_exc()
        sys.exit(1)
