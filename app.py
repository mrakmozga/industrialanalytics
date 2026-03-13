"""
IndustrAI — Flask + SocketIO backend
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import os
import pandas as pd
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

DATA_CSV        = os.environ.get('DATA_CSV',        'data/equipment_simulation_with_anomalies_25000.csv')
PREDICTIONS_CSV = os.environ.get('PREDICTIONS_CSV', 'data/predictions.csv')
HOST            = os.environ.get('HOST', '0.0.0.0')
PORT            = int(os.environ.get('PORT', 5000))

SCALER_MEAN  = [70.5418, 3.0592, 99.8955, 0.7004, 40.0525]
SCALER_SCALE = [6.0869,  0.5256,  8.0476, 0.0991, 10.1011]
SPEED_MAP    = {1: 1.0, 2: 0.5, 5: 0.2}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'industrai-secret'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet')

sim_data    = []
predictions = []
total_rows  = 0

def load_data():
    global sim_data, predictions, total_rows
    print(f"[IndustrAI] Loading {DATA_CSV} ...")
    df = pd.read_csv(DATA_CSV)
    print(f"[IndustrAI] Loading {PREDICTIONS_CSV} ...")
    pred_df  = pd.read_csv(PREDICTIONS_CSV)
    pred_col = 'risk' if 'risk' in pred_df.columns else pred_df.columns[0]
    preds    = pred_df[pred_col].tolist()

    sensor_cols = ['temperature', 'vibration', 'pressure', 'load_factor', 'humidity']
    n = min(len(df), len(preds))
    sim_data    = [{c: float(row[c]) for c in sensor_cols} for row in df.iloc[:n].to_dict('records')]
    predictions = [float(p) for p in preds[:n]]
    total_rows  = n
    print(f"[IndustrAI] Ready: {total_rows} rows, {sum(1 for p in predictions if p>=0.5)} incidents")

sessions = {}

def compute_derived(row):
    ts = (row['temperature'] - SCALER_MEAN[0]) / SCALER_SCALE[0]
    vs = (row['vibration']   - SCALER_MEAN[1]) / SCALER_SCALE[1]
    ps = (row['pressure']    - SCALER_MEAN[2]) / SCALER_SCALE[2]
    ls = (row['load_factor'] - SCALER_MEAN[3]) / SCALER_SCALE[3]
    return {
        'temp_pressure_ratio':        round(ts / ps, 4) if ps != 0 else 0.0,
        'vibration_load_interaction': round(vs / ls, 4) if ls != 0 else 0.0,
    }

def run_simulation(sid):
    # Guard: mark thread as running, exit if another thread already running
    state = sessions.get(sid)
    if not state:
        return
    if state.get('running'):
        return
    state['running'] = True

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
            row  = sim_data[idx]
            risk = predictions[idx]
            payload = {
                'index': idx,
                'total': total_rows,
                **row,
                'risk': round(risk, 4),
                **compute_derived(row),
            }
            socketio.emit('tick', payload, to=sid)
            state['index'] += 1
            socketio.sleep(state['speed'])
    finally:
        state = sessions.get(sid)
        if state:
            state['running'] = False

@app.route('/')
def index():
    return render_template('index.html',
                           total_rows=total_rows,
                           scaler_mean=SCALER_MEAN,
                           scaler_scale=SCALER_SCALE)

@app.route('/api/status')
def api_status():
    return {'total_rows': total_rows, 'loaded': total_rows > 0}

@socketio.on('connect')
def on_connect():
    sid = request.sid
    sessions[sid] = {'index': 0, 'speed': 1.0, 'playing': False, 'running': False}
    emit('ready', {'total': total_rows})
    print(f"[WS] connect {sid}")

@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    sessions.pop(sid, None)
    print(f"[WS] disconnect {sid}")

@socketio.on('start')
def on_start():
    sid = request.sid
    if sid not in sessions:
        return
    sessions[sid]['playing'] = True
    socketio.start_background_task(run_simulation, sid)

@socketio.on('pause')
def on_pause():
    sid = request.sid
    if sid in sessions:
        sessions[sid]['playing'] = False

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
        mult = int(data.get('multiplier', 1))
        sessions[sid]['speed'] = SPEED_MAP.get(mult, 1.0)

@socketio.on('start_after_reset')
def on_start_after_reset():
    sid = request.sid
    if sid in sessions:
        sessions[sid]['playing'] = True
        if not sessions[sid].get('running'):
            socketio.start_background_task(run_simulation, sid)

if __name__ == '__main__':
    load_data()
    print(f"[IndustrAI] http://{HOST}:{PORT}")
    socketio.run(app, host=HOST, port=PORT, debug=False, allow_unsafe_werkzeug=True)
