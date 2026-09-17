"""
Disaster Relief AI — Backend
Flask + SQLite auth (signup / login / session) + prediction endpoint.

Run:
    pip install -r requirements.txt
    python app.py

Then open http://127.0.0.1:5000 in your browser (or on your phone if on
the same Wi-Fi: http://<your-computer-ip>:5000).

WHERE TO PLUG IN YOUR REAL MODEL
---------------------------------
Look for the block marked  >>> YOUR MODEL CODE GOES HERE <<<  inside the
predict() function below. Replace the placeholder scoring logic with your
trained XGBoost model + SHAP explainer. Everything else (auth, sessions,
serving the frontend) is already wired up and does not need to change.
"""

from flask import Flask, request, jsonify, session, render_template
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import sqlite3
import os
import secrets

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# In a real deployment, set SECRET_KEY as an environment variable instead of
# hardcoding it. For a college project demo, this fallback is fine.
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_HTTPONLY"] = True

DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'responder',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Not authenticated"}), 401
        return f(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    # Serves the dashboard shell. The page itself checks /me on load and
    # shows the login/signup screen or the dashboard accordingly.
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------
@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json(force=True, silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if len(username) < 3:
        return jsonify({"error": "Username must be at least 3 characters."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, generate_password_hash(password)),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        return jsonify({"error": "That username is already taken."}), 409
    finally:
        conn.close()

    return jsonify({"ok": True, "message": "Account created. You can now log in."})


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(force=True, silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    conn = get_db()
    row = conn.execute(
        "SELECT id, password_hash, role FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()

    if not row or not check_password_hash(row["password_hash"], password):
        return jsonify({"error": "Incorrect username or password."}), 401

    session.clear()
    session["user_id"] = row["id"]
    session["username"] = username
    session.permanent = True

    return jsonify({"ok": True, "username": username, "role": row["role"]})


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/me", methods=["GET"])
def me():
    if "user_id" not in session:
        return jsonify({"authenticated": False})
    return jsonify({"authenticated": True, "username": session.get("username")})


# ---------------------------------------------------------------------------
# Prediction endpoint (protected — must be logged in)
# ---------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    features = request.get_json(force=True, silent=True) or {}

    # >>> YOUR MODEL CODE GOES HERE <<<
    # Replace this placeholder block with your trained XGBoost model and
    # SHAP explainer. It should return the same shape as below:
    #   severity_score : number 0-100
    #   priority_tier  : one of "Critical" | "High" | "Moderate" | "Low"
    #   explanation    : list of {feature, shap_value, direction}
    #
    # Example of loading a saved model once at startup (put this near the
    # top of the file, outside any route function):
    #   import joblib, shap
    #   model = joblib.load("model.pkl")
    #   explainer = shap.TreeExplainer(model)
    #
    # Then inside predict():
    #   X = preprocess(features)              # your existing preprocessing
    #   score = float(model.predict_proba(X)[0][1]) * 100
    #   shap_values = explainer.shap_values(X)
    #   explanation = [...]                    # build from shap_values

    score = _placeholder_score(features)
    tier = _tier_for_score(score)
    explanation = _placeholder_explanation(features, score)

    return jsonify(
        {
            "severity_score": round(score, 1),
            "priority_tier": tier,
            "explanation": explanation,
        }
    )


def _placeholder_score(features):
    """Deterministic stand-in so the dashboard works end-to-end before your
    real model is wired in. Swap this out."""
    base = 35
    dtype = (features.get("Disaster Type") or "").lower()
    subtype = (features.get("Disaster Subtype") or "").lower()
    if "flash" in subtype:
        base += 30
    elif "flood" in dtype:
        base += 15
    if "cyclone" in subtype or "storm" in dtype:
        base += 18
    if "landslide" in dtype:
        base += 12
    if features.get("Declaration") == "Yes":
        base += 10
    if features.get("Appeal") == "Yes":
        base += 8
    return max(5, min(97, base))


def _tier_for_score(score):
    if score >= 70:
        return "Critical"
    if score >= 50:
        return "High"
    if score >= 30:
        return "Moderate"
    return "Low"


def _placeholder_explanation(features, score):
    return [
        {"feature": "Disaster Subtype", "shap_value": 0.42, "direction": "increases_severity"},
        {"feature": "Declaration status", "shap_value": 0.21, "direction": "increases_severity"},
        {"feature": "OFDA/BHA Response", "shap_value": 0.15, "direction": "decreases_severity"},
        {"feature": "Start Month (monsoon)", "shap_value": 0.11, "direction": "increases_severity"},
    ]


if __name__ == "__main__":
    # host="0.0.0.0" lets you open this from your phone on the same Wi-Fi
    # at http://<your-computer's-LAN-IP>:5000
    app.run(host="0.0.0.0", port=5000, debug=True)
