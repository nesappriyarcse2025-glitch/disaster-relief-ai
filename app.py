"""
app.py
Disaster Relief AI — Flask backend that loads the trained XGBoost model
and serves severity/priority predictions over a simple JSON API.

Run this from inside your activated venv, AFTER running train_model.py:
    python app.py

Then test it in a second Command Prompt window:
    curl http://127.0.0.1:5000/health
"""

import json
import os
import secrets

import joblib
import numpy as np
import pandas as pd
import shap
from flask import Flask, jsonify, request, session
from flask_cors import CORS
from xgboost import XGBRegressor

import notify
import gdacs_monitor
import auth

MODEL_DIR = "model"

app = Flask(__name__)
# CORS with credentials=True + explicit origins is required for session
# cookies (login) to work when the dashboard HTML is opened as a local
# file or from a different origin than the API.
CORS(app, supports_credentials=True)

# Secret key signs the session cookie. Set your own fixed value via the
# DR_SECRET_KEY environment variable in production (e.g. on Render) so
# sessions survive restarts; otherwise a random one is generated each run
# (fine for local testing — it just means everyone gets logged out on restart).
app.secret_key = os.environ.get("DR_SECRET_KEY", secrets.token_hex(32))

# ---- Load model + encoders once at startup ----
model = XGBRegressor()
model.load_model(os.path.join(MODEL_DIR, "xgb_severity_model.json"))

artifacts = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
encoder = artifacts["encoder"]
thresholds = artifacts["thresholds"]

with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
    schema = json.load(f)

FEATURE_COLS = schema["feature_cols"]
CATEGORICAL_COLS = schema["categorical_cols"]
NUMERIC_COLS = schema["numeric_cols"]

# ---- SHAP explainer (TreeExplainer is fast + exact for XGBoost) ----
explainer = shap.TreeExplainer(model)


def tier_of(score: float) -> str:
    if score >= thresholds["q90"]:
        return "Critical"
    if score >= thresholds["q75"]:
        return "High"
    if score >= thresholds["q50"]:
        return "Moderate"
    return "Low"


def explain_row(df_row: pd.DataFrame, top_n: int = 5):
    """Returns the top_n most influential features for a single-row DataFrame,
    as a list of {feature, shap_value, direction} sorted by absolute impact."""
    shap_values = explainer.shap_values(df_row[FEATURE_COLS])
    values = shap_values[0]  # single row

    contributions = list(zip(FEATURE_COLS, values))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    return [
        {
            "feature": feat,
            "shap_value": round(float(val), 3),
            "direction": "increases_severity" if val > 0 else "decreases_severity",
        }
        for feat, val in contributions[:top_n]
    ]


def compute_prediction(raw_features: dict) -> dict:
    """Shared prediction path used by /predict AND the GDACS auto-monitor,
    so both go through identical preprocessing/encoding/scoring logic."""
    row = {col: raw_features.get(col, None) for col in FEATURE_COLS}
    df = pd.DataFrame([row])

    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("Unknown").astype(str)
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(0)

    df[CATEGORICAL_COLS] = encoder.transform(df[CATEGORICAL_COLS])

    score = float(model.predict(df[FEATURE_COLS])[0])
    score = max(0.0, min(100.0, score))
    tier = tier_of(score)
    top_features = explain_row(df, top_n=5)

    return {
        "severity_score": round(score, 2),
        "priority_tier": tier,
        "thresholds": thresholds,
        "explanation": top_features,
    }


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True, "shap_ready": True})


@app.route("/signup", methods=["POST"])
def signup():
    """Self-service account creation — anyone can sign up, no approval needed."""
    payload = request.get_json(force=True)
    ok, result = auth.create_account(
        username=payload.get("username", ""),
        name=payload.get("name", ""),
        password=payload.get("password", ""),
        role=payload.get("role", "official"),
    )
    if not ok:
        return jsonify({"error": result}), 400

    # log them in immediately after signup, for convenience
    session["username"] = result["username"]
    session["name"] = result["name"]
    session["role"] = result["role"]
    return jsonify({"created": True, "logged_in": True, "user": result})


@app.route("/login", methods=["POST"])
def login():
    payload = request.get_json(force=True)
    username = payload.get("username", "")
    password = payload.get("password", "")

    user = auth.verify_login(username, password)
    if user is None:
        return jsonify({"error": "Invalid username or password"}), 401

    session["username"] = user["username"]
    session["name"] = user["name"]
    session["role"] = user["role"]
    return jsonify({"logged_in": True, "user": user})


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"logged_out": True})


@app.route("/me", methods=["GET"])
def me():
    if "username" not in session:
        return jsonify({"logged_in": False}), 401
    return jsonify({
        "logged_in": True,
        "user": {"username": session["username"], "name": session["name"], "role": session["role"]},
    })


@app.route("/predict", methods=["POST"])
@auth.login_required
def predict():
    """
    Expects JSON body with the raw feature values, e.g.:
    {
        "Disaster Type": "Flood",
        "Disaster Subtype": "Riverine flood",
        "Country": "India",
        "Region": "Asia",
        "Subregion": "Southern Asia",
        "Start Month": 8,
        "Magnitude": null,
        "Magnitude Scale": "Km2",
        "OFDA/BHA Response": "No",
        "Appeal": "No",
        "Declaration": "Yes"
    }
    Missing fields are filled the same way they were at training time.

    Response now also includes "explanation": the top 5 features driving
    this specific prediction, via SHAP (TreeExplainer).
    """
    payload = request.get_json(force=True)
    result = compute_prediction(payload)

    event_name = payload.get("event_name", payload.get("Country", "Unspecified location"))
    top_reason = result["explanation"][0]["feature"] if result["explanation"] else ""
    notify.send_alert(event_name, result["severity_score"], result["priority_tier"], top_reason)

    return jsonify(result)


@app.route("/predict_batch", methods=["POST"])
@auth.login_required
def predict_batch():
    """Same as /predict but expects {"events": [ {...}, {...} ]} and returns a list."""
    payload = request.get_json(force=True)
    events = payload.get("events", [])
    if not events:
        return jsonify({"error": "no events provided"}), 400

    rows = []
    for ev in events:
        row = {col: ev.get(col, None) for col in FEATURE_COLS}
        rows.append(row)
    df = pd.DataFrame(rows)

    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("Unknown").astype(str)
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(0)

    df[CATEGORICAL_COLS] = encoder.transform(df[CATEGORICAL_COLS])
    scores = model.predict(df[FEATURE_COLS])
    scores = np.clip(scores, 0, 100)

    results = [
        {"severity_score": round(float(s), 2), "priority_tier": tier_of(float(s))}
        for s in scores
    ]
    return jsonify({"results": results})


@app.route("/monitor/status", methods=["GET"])
@auth.login_required
def monitor_status():
    """Shows what the GDACS auto-monitor has seen so far — useful for a demo/viva,
    so you can show it's actually polling a real feed, not faking activity."""
    return jsonify(gdacs_monitor.get_status())


# ---- Start the GDACS monitor ----
# Runs at import time (not just under `python app.py`) so it also starts when
# gunicorn imports this file on Render. The WERKZEUG_RUN_MAIN check only
# matters for local `python app.py` with Flask's debug auto-reloader, which
# would otherwise start two monitor threads.
# ---- Start the GDACS monitor ----
# Runs once at import time (also works under gunicorn on Render, which just
# imports this module once per worker — no __main__ block runs there).
gdacs_monitor.start(compute_prediction, notify.send_alert, poll_seconds=300)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # use_reloader=False: Flask's debug auto-reloader runs this file in TWO
    # processes (a watcher + a worker), which would start the monitor thread
    # twice. Disabling the reloader keeps debug tracebacks but avoids that.
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=port)
