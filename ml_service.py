"""
IndustrAI — ML Service (FastAPI)
Принимает сырые показания сенсоров → возвращает risk от Random Forest.

POST /predict
{temperature, vibration, pressure, load_factor, humidity,
 temp_pressure_ratio, vibration_load_interaction}
→ {risk: float, confidence: float, features_used: list}
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import os
import warnings

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Пути к артефактам (можно переопределить через env) ────────
MODEL_PATH    = os.environ.get('MODEL_PATH',    'models/incident_random_forest_model.pkl')
SCALER_PATH   = os.environ.get('SCALER_PATH',   'models/scaler.pkl')
FEATURES_PATH = os.environ.get('FEATURES_PATH', 'models/model_features.pkl')

# ── Параметры масштабирования (fallback если scaler.pkl недоступен) ──
SCALER_MEAN  = [70.5418, 3.0592, 99.8955, 0.7004, 40.0525]
SCALER_SCALE = [6.0869,  0.5256,  8.0476, 0.0991, 10.1011]

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(application: FastAPI):
    load_artifacts()
    yield

app = FastAPI(title="IndustrAI ML Service", version="1.0.0", lifespan=lifespan)

# ── Загрузка модели ───────────────────────────────────────────
model = None
scaler = None
feature_names = None

def fix_sklearn_compat(rf_model):
    """Совместимость sklearn 1.3 → 1.5+: патчим атрибуты деревьев."""
    for est in rf_model.estimators_:
        if not hasattr(est, 'monotonic_cst'):
            est.monotonic_cst = None
        # sklearn 1.4+ добавил _n_features (может отсутствовать в старых pickle)
        if not hasattr(est, '_n_features'):
            est._n_features = est.n_features_in_
        # sklearn 1.5+ переименовал n_features_ → n_features_in_
        if not hasattr(est, 'n_features_in_') and hasattr(est, 'n_features_'):
            est.n_features_in_ = est.n_features_
    return rf_model

def load_artifacts():
    global model, scaler, feature_names
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings('ignore', category=InconsistentVersionWarning)

    # ── Model ──
    print(f"[ML] Loading model from {MODEL_PATH} ...")
    raw = joblib.load(MODEL_PATH)
    print(f"[ML] Loaded type: {type(raw).__name__}")
    if hasattr(raw, 'estimators_'):
        model = fix_sklearn_compat(raw)
    elif isinstance(raw, (list, tuple)) and hasattr(raw[0], 'estimators_'):
        model = fix_sklearn_compat(raw[0])
    else:
        raise ValueError(f"Expected RandomForestClassifier, got {type(raw)}")
    print(f"[ML] Model OK: {model.n_estimators} trees, classes={model.classes_}")

    # ── Scaler ──
    print(f"[ML] Loading scaler from {SCALER_PATH} ...")
    raw_sc = joblib.load(SCALER_PATH)
    print(f"[ML] Scaler type: {type(raw_sc).__name__}")
    if hasattr(raw_sc, 'mean_'):
        scaler = raw_sc
        print(f"[ML] Scaler OK: mean={scaler.mean_.tolist()}")
    else:
        scaler = None
        print(f"[ML] WARNING: scaler.pkl is not a StandardScaler — using hardcoded constants")

    # ── Feature names ──
    print(f"[ML] Loading features from {FEATURES_PATH} ...")
    raw_feat = joblib.load(FEATURES_PATH)
    if isinstance(raw_feat, list):
        feature_names = raw_feat
    elif hasattr(raw_feat, 'tolist'):
        feature_names = raw_feat.tolist()
    else:
        feature_names = list(raw_feat)
    print(f"[ML] Features ({len(feature_names)}): {feature_names}")

# ── Схемы запрос / ответ ──────────────────────────────────────
class SensorReading(BaseModel):
    temperature:               float = Field(..., ge=0,   le=200,  description="°C")
    vibration:                 float = Field(..., ge=0,   le=50,   description="mm/s")
    pressure:                  float = Field(..., ge=0,   le=300,  description="кПа")
    load_factor:               float = Field(..., ge=0,   le=5,    description="о.е.")
    humidity:                  float = Field(..., ge=0,   le=100,  description="%")
    # Производные признаки — если переданы клиентом, используем их.
    # Если нет — вычисляем на сервере.
    temp_pressure_ratio:       float | None = None
    vibration_load_interaction: float | None = None

class PredictResponse(BaseModel):
    risk:          float          # вероятность инцидента [0..1]
    confidence:    float          # уверенность: max(p_0, p_1)
    features_used: list[str]
    feature_values: dict[str, float]

# ── Вычисление признаков ──────────────────────────────────────
def build_feature_vector(data: SensorReading):
    """Масштабирует сенсоры и вычисляет производные признаки.
    Возвращает (DataFrame для модели, dict значений признаков).
    """
    import pandas as pd
    t, v, p, l, h = (
        data.temperature, data.vibration, data.pressure,
        data.load_factor, data.humidity
    )

    # Масштабирование — передаём DataFrame с именами колонок
    # (scaler был обучен с feature_names_in_=['temperature','vibration',...])
    if scaler is not None:
        raw_df = pd.DataFrame(
            [[t, v, p, l, h]],
            columns=['temperature', 'vibration', 'pressure', 'load_factor', 'humidity']
        )
        scaled = scaler.transform(raw_df)[0]
        ts, vs, ps, ls, hs = scaled
    else:
        ts = (t - SCALER_MEAN[0]) / SCALER_SCALE[0]
        vs = (v - SCALER_MEAN[1]) / SCALER_SCALE[1]
        ps = (p - SCALER_MEAN[2]) / SCALER_SCALE[2]
        ls = (l - SCALER_MEAN[3]) / SCALER_SCALE[3]
        hs = (h - SCALER_MEAN[4]) / SCALER_SCALE[4]

    # Производные: берём из запроса или вычисляем
    tpr = data.temp_pressure_ratio if data.temp_pressure_ratio is not None \
          else (round(float(ts / ps), 6) if ps != 0 else 0.0)
    vli = data.vibration_load_interaction if data.vibration_load_interaction is not None \
          else (round(float(vs / ls), 6) if ls != 0 else 0.0)

    feature_map = {
        'temperature_scaled':         float(ts),
        'vibration_scaled':           float(vs),
        'pressure_scaled':            float(ps),
        'load_factor_scaled':         float(ls),
        'humidity_scaled':            float(hs),
        'temp_pressure_ratio':        tpr,
        'vibration_load_interaction': vli,
    }

    # DataFrame с именами колонок — модель обучена с feature_names_in_
    names = feature_names if feature_names else list(feature_map.keys())
    import pandas as pd
    feat_df = pd.DataFrame([[feature_map[n] for n in names]], columns=names)
    return feat_df, feature_map

# ── Endpoints ─────────────────────────────────────────────────
# startup handled by lifespan above

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "features": feature_names,
    }

@app.post("/predict", response_model=PredictResponse)
def predict(data: SensorReading):
    if model is None:
        raise HTTPException(503, "Model not loaded")

    vector, fmap = build_feature_vector(data)

    # Инференс
    proba = model.predict_proba(vector)[0]          # [p_class0, p_class1]
    risk  = float(proba[1] / proba.sum())           # нормализованная вероятность
    confidence = float(max(proba))

    return PredictResponse(
        risk=round(risk, 6),
        confidence=round(confidence, 4),
        features_used=feature_names or list(fmap.keys()),
        feature_values={k: round(v, 6) for k, v in fmap.items()},
    )

@app.post("/predict/batch")
def predict_batch(readings: list[SensorReading]):
    """Пакетный инференс для нескольких строк сразу."""
    if model is None:
        raise HTTPException(503, "Model not loaded")
    results = []
    for r in readings:
        vector, fmap = build_feature_vector(r)
        proba = model.predict_proba(vector)[0]
        risk  = float(proba[1] / proba.sum())
        results.append({"risk": round(risk, 6), "confidence": round(float(max(proba)), 4)})
    return results

if __name__ == '__main__':
    import uvicorn
    host = os.environ.get('ML_HOST', '0.0.0.0')
    port = int(os.environ.get('ML_PORT', 8000))
    print(f"[ML] Starting on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
