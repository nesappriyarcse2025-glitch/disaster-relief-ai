"""
auth.py
Disaster Relief AI — login system with separate accounts per official.

Passwords are NEVER stored as plain text — only a salted hash
(werkzeug's generate_password_hash), same technique used by Flask itself.

Accounts live in users.json. Add/manage accounts with create_user.py
(don't edit users.json by hand — let the script hash the password for you).
"""

import json
import os
from functools import wraps

from flask import jsonify, session
from werkzeug.security import check_password_hash, generate_password_hash

USERS_FILE = "users.json"


def _load_users():
    if not os.path.exists(USERS_FILE):
        return []
    with open(USERS_FILE) as f:
        return json.load(f)


def _save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def create_account(username: str, name: str, password: str, role: str = "official"):
    """Self-service signup — anyone can create an account, no approval needed.
    Returns (True, user_dict) on success, or (False, error_message) on failure."""
    username = (username or "").strip()
    name = (name or "").strip() or username
    if not username or not password:
        return False, "Username and password are required."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    users = _load_users()
    if any(u["username"] == username for u in users):
        return False, "That username is already taken."

    users.append({
        "username": username,
        "name": name,
        "role": role,
        "password_hash": generate_password_hash(password),
    })
    _save_users(users)
    return True, {"username": username, "name": name, "role": role}


def verify_login(username: str, password: str):
    """Returns the user dict (without password_hash) if credentials are
    correct, else None."""
    users = _load_users()
    for u in users:
        if u["username"] == username:
            if check_password_hash(u["password_hash"], password):
                return {"username": u["username"], "name": u["name"], "role": u["role"]}
            return None
    return None


def login_required(fn):
    """Decorator: blocks the route unless someone is logged in via session."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "username" not in session:
            return jsonify({"error": "Not logged in"}), 401
        return fn(*args, **kwargs)
    return wrapper
