"""
IndustrAI — Dashboard Backend (Flask + SocketIO)

Режимы работы (ML_MODE env):
  live   — вызывает ML Service по HTTP для каждого тика (production-like)
  cached — читает predictions.csv (быстрый демо-режим, без ML сервиса)
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import os
import time
import pandas as pd
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

# ── Конфигурация ──────────────────────────────────────────────
DATA_CSV     = os.environ.get('DATA_CSV',  'data/equipment_simulation_with_anomalies_25000.csv')
ML_MODE      = os.environ.get('ML_MODE',   'live')    # 'live' | 'cached'
ML_URL       = os.environ.get('ML_URL',    'http://ml:8000')   # адрес ML Service
# Fallback: если ML_MODE=cached или ML Service недоступен
PREDICTIONS_CSV = os.environ.get('PREDICTIONS_CSV', 'data/predictions.csv')
HOST         = os.environ.get('HOST', '0.0.0.0')
PORT         = int(os.environ.get('PORT', 5000))

SCALER_MEAN  = [70.5418, 3.0592, 99.8955, 0.7004, 40.0525]
SCALER_SCALE = [6.0869,  0.5256,  8.0476, 0.0991, 10.1011]
SPEED_MAP    = {1: 1.0, 2: 0.5, 5: 0.2}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'industrai-secret'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet')

# ── Данные ────────────────────────────────────────────────────
sim_data    = []          # [{temperature, vibration, pressure, load_factor, humidity}, ...]
predictions = []          # [float, ...] — только в cached режиме
total_rows  = 0
effective_mode = ML_MODE  # может переключиться на 'cached' если ML недоступен

def load_data():
    global sim_data, predictions, total_rows, effective_mode

    print(f"[Dashboard] Loading {DATA_CSV} ...")
    df = pd.read_csv(DATA_CSV)
    sensor_cols = ['temperature', 'vibration', 'pressure', 'load_factor', 'humidity']

    if effective_mode == 'cached':
        print(f"[Dashboard] Mode: CACHED — loading {PREDICTIONS_CSV} ...")
        pred_df  = pd.read_csv(PREDICTIONS_CSV)
        pred_col = 'risk' if 'risk' in pred_df.columns else pred_df.columns[0]
        preds    = pred_df[pred_col].tolist()
        n = min(len(df), len(preds))
        sim_data    = [{c: float(row[c]) for c in sensor_cols} for row in df.iloc[:n].to_dict('records')]
        predictions = [float(p) for p in preds[:n]]
        total_rows  = n
        print(f"[Dashboard] Ready (cached): {total_rows} rows, "
              f"{sum(1 for p in predictions if p >= 0.5)} incidents pre-loaded")
    else:
        # live mode — грузим только телеметрию, риск будет считать ML Service
        n = len(df)
        sim_data   = [{c: float(row[c]) for c in sensor_cols} for row in df.to_dict('records')]
        total_rows = n
        print(f"[Dashboard] Ready (live): {total_rows} rows — risk via ML Service at {ML_URL}")
        _check_ml_service()

def _check_ml_service():
    """Проверяем доступность ML Service при старте; если нет — падаем в cached."""
    global effective_mode
    import urllib.request, urllib.error
    try:
        with urllib.request.urlopen(f"{ML_URL}/health", timeout=3) as r:
            import json
            body = json.loads(r.read())
            if body.get('model_loaded'):
                print(f"[Dashboard] ML Service OK — model loaded, features: {body.get('features')}")
                return
    except Exception as e:
        pass
    print(f"[Dashboard] WARNING: ML Service not reachable at {ML_URL}. Falling back to cached mode.")
    effective_mode = 'cached'
    # Перезагрузить данные в cached режиме
    global predictions
    try:
        pred_df  = pd.read_csv(PREDICTIONS_CSV)
        pred_col = 'risk' if 'risk' in pred_df.columns else pred_df.columns[0]
        predictions = [float(p) for p in pred_df[pred_col].tolist()[:total_rows]]
        print(f"[Dashboard] Cached predictions loaded as fallback.")
    except Exception as e:
        print(f"[Dashboard] ERROR: could not load fallback predictions: {e}")

# ── ML inference (вызов HTTP в eventlet-корутине) ─────────────
def call_ml_service(row: dict, derived: dict) -> float:
    """POST к ML Service, возвращает risk float. При ошибке возвращает -1."""
    import urllib.request, urllib.error, json
    payload = json.dumps({**row, **derived}).encode()
    req = urllib.request.Request(
        f"{ML_URL}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            body = json.loads(resp.read())
            return float(body['risk'])
    except Exception as e:
        print(f"[Dashboard] ML call failed: {e}")
        return -1.0   # сигнал об ошибке

# ── Производные признаки ──────────────────────────────────────
def compute_derived(row: dict) -> dict:
    ts = (row['temperature'] - SCALER_MEAN[0]) / SCALER_SCALE[0]
    vs = (row['vibration']   - SCALER_MEAN[1]) / SCALER_SCALE[1]
    ps = (row['pressure']    - SCALER_MEAN[2]) / SCALER_SCALE[2]
    ls = (row['load_factor'] - SCALER_MEAN[3]) / SCALER_SCALE[3]
    return {
        'temp_pressure_ratio':        round(ts / ps, 4) if ps != 0 else 0.0,
        'vibration_load_interaction': round(vs / ls, 4) if ls != 0 else 0.0,
    }

# ── Сессии ────────────────────────────────────────────────────
sessions = {}

def run_simulation(sid: str):
    state = sessions.get(sid)
    if not state or state.get('running'):
        return
    state['running'] = True

    # Счётчик ошибок ML — после 5 подряд переключаем на cached
    ml_errors = 0

    try:
        while True:
            state = sessions.get(sid)
            if state is None:
                break
            if not state['playing']:
                socketio.sleep(0.05)
                continue

            idx = state['index']
            if idx >= total_rows:
                socketio.emit('sim_done', {'total': total_rows}, to=sid)
                state['playing'] = False
                break

            row     = sim_data[idx]
            derived = compute_derived(row)

            # ── Получаем risk ──
            if state.get('mode', effective_mode) == 'live' and effective_mode == 'live':
                risk = call_ml_service(row, derived)
                if risk < 0:
                    ml_errors += 1
                    if ml_errors >= 5:
                        print(f"[Dashboard] {ml_errors} ML errors for {sid}, using cached")
                        # fallback: берём из predictions если есть
                        risk = predictions[idx] if idx < len(predictions) else 0.0
                else:
                    ml_errors = 0
            else:
                risk = predictions[idx] if idx < len(predictions) else 0.0

            payload = {
                'index': idx,
                'total': total_rows,
                **row,
                'risk':  round(risk, 4),
                **derived,
                'ml_mode': state.get('mode', effective_mode),
            }
            socketio.emit('tick', payload, to=sid)
            state['index'] += 1
            socketio.sleep(state['speed'])
    finally:
        state = sessions.get(sid)
        if state:
            state['running'] = False

# ── HTTP ──────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html',
                           total_rows=total_rows,
                           scaler_mean=SCALER_MEAN,
                           scaler_scale=SCALER_SCALE,
                           ml_mode=effective_mode,
                           ml_url=ML_URL)

@app.route('/api/status')
def api_status():
    return {
        'total_rows':    total_rows,
        'loaded':        total_rows > 0,
        'ml_mode':       effective_mode,
        'ml_url':        ML_URL,
    }

# ── WebSocket ─────────────────────────────────────────────────
@socketio.on('connect')
def on_connect():
    sid = request.sid
    sessions[sid] = {
        'index': 0, 'speed': 1.0,
        'playing': False, 'running': False,
        'mode': effective_mode,
    }
    emit('ready', {'total': total_rows, 'ml_mode': effective_mode})
    print(f"[WS] connect {sid} (ml_mode={effective_mode})")

@socketio.on('disconnect')
def on_disconnect():
    sessions.pop(request.sid, None)
    print(f"[WS] disconnect {request.sid}")

@socketio.on('start')
def on_start():
    sid = request.sid
    if sid not in sessions:
        return
    sessions[sid]['playing'] = True
    socketio.start_background_task(run_simulation, sid)

@socketio.on('pause')
def on_pause():
    if request.sid in sessions:
        sessions[request.sid]['playing'] = False

@socketio.on('resume')
def on_resume():
    sid = request.sid
    if sid not in sessions:
        return
    sessions[sid]['playing'] = True
    if not sessions[sid].get('running'):
        socketio.start_background_task(run_simulation, sid)

@socketio.on('reset')
def on_reset():
    sid = request.sid
    if sid in sessions:
        sessions[sid].update({'index': 0, 'playing': False})
        emit('reset_ok', {'total': total_rows})

@socketio.on('set_speed')
def on_set_speed(data):
    sid = request.sid
    if sid in sessions:
        sessions[sid]['speed'] = SPEED_MAP.get(int(data.get('multiplier', 1)), 1.0)

@socketio.on('start_after_reset')
def on_start_after_reset():
    sid = request.sid
    if sid in sessions:
        sessions[sid]['playing'] = True
        if not sessions[sid].get('running'):
            socketio.start_background_task(run_simulation, sid)

if __name__ == '__main__':
    load_data()
    print(f"[Dashboard] http://{HOST}:{PORT}  mode={effective_mode}")
    socketio.run(app, host=HOST, port=PORT, debug=False, allow_unsafe_werkzeug=True)
