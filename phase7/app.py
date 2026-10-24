"""
app.py -- Phase 7: Register / Login + database for SafeRoute
----------------------------------------------------------------
Adds user accounts to the existing SafeRoute app:
  - a Register page collecting first name, last name, mobile number,
    email, password, permanent address, nationality, and THREE
    emergency contact numbers (each checked to be different from the
    user's own mobile number, and from each other).
  - a Login page (email or mobile + password).
  - a SQLite database (saferoute.db, created automatically on first
    run) that all of this is actually saved into.

This single Flask server now serves EVERYTHING on one port:
  - the existing map app (phase6/app/index.html and its data files),
    unchanged, at  http://localhost:5000/
  - the new pages at  /register  and  /login
  - JSON APIs at  /api/register  and  /api/login  that the pages call

WHY A BACKEND AT ALL: the existing app is a static page with no server
logic -- browser JavaScript cannot safely talk to a database directly
(there'd be nothing stopping anyone from reading or editing every
user's data straight out of the browser). Flask is the backend that
sits between the register/login pages and the database, checks the
submitted data, and is the only thing that actually touches the
database file.

HOW TO RUN:
    cd phase7
    pip install flask
    python app.py
Then open http://localhost:5000/ in your browser (you no longer need
to separately run "python -m http.server" inside phase6/app -- this
replaces that for all of phase6's pages too; frontend_data fetches
still resolve the same way since this serves index.html from the same
relative location).
"""

import json
import os
import re
import sqlite3
from datetime import datetime, timezone

from flask import Flask, g, jsonify, redirect, render_template, request, send_from_directory, session

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAP_APP_DIR = os.path.join(BASE_DIR, "..", "phase6", "app")
DB_PATH = os.path.join(BASE_DIR, "saferoute.db")

app = Flask(__name__, static_folder=MAP_APP_DIR, static_url_path="")
app.secret_key = "dev-only-secret-change-this-before-any-real-deployment"

MOBILE_RE = re.compile(r"^[6-9]\d{9}$")  # 10-digit Indian mobile, same rule for user + emergency contacts
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ---------------------------------------------------------------- database

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            mobile TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            address TEXT NOT NULL,
            nationality TEXT NOT NULL,
            emergency_contact_1 TEXT NOT NULL,
            emergency_contact_2 TEXT NOT NULL,
            emergency_contact_3 TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            accuracy REAL,
            city TEXT,
            client_timestamp TEXT,
            recorded_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sos_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            message TEXT NOT NULL,
            contacts_total INTEGER NOT NULL,
            contacts_sent INTEGER NOT NULL,
            delivery_note TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    # Upgrade an existing database in place (older accounts keep working):
    # add the emergency e-mail columns, and the e-mail delivery columns on
    # the alert log, if they aren't there yet.
    def add_missing_columns(table, columns):
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, decl in columns:
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")

    add_missing_columns("users", [
        ("emergency_email_1", "TEXT"),
        ("emergency_email_2", "TEXT"),
        ("emergency_email_3", "TEXT"),
    ])
    add_missing_columns("sos_alerts", [
        ("emails_total", "INTEGER"),
        ("emails_sent", "INTEGER"),
        ("email_note", "TEXT"),
    ])
    conn.commit()
    conn.close()


# ---------------------------------------------------------------- config

def load_config():
    """Reads phase7/sms_config.json (kept private -- holds passwords/tokens)."""
    cfg_path = os.path.join(BASE_DIR, "sms_config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


# ---------------------------------------------------------------- email sending

def send_email(to_address, subject, body):
    """
    Sends one e-mail through an SMTP server configured in sms_config.json
    (smtp_host, smtp_port, smtp_user, smtp_password, email_from) or the
    SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASSWORD / EMAIL_FROM
    environment variables. Returns (ok, note). Never reports success
    unless the SMTP server actually accepted the message.
    """
    import smtplib
    from email.message import EmailMessage

    cfg = load_config()
    host = os.environ.get("SMTP_HOST") or cfg.get("smtp_host")
    port = int(os.environ.get("SMTP_PORT") or cfg.get("smtp_port") or 587)
    user = os.environ.get("SMTP_USER") or cfg.get("smtp_user")
    password = os.environ.get("SMTP_PASSWORD") or cfg.get("smtp_password")
    sender = os.environ.get("EMAIL_FROM") or cfg.get("email_from") or user
    if not (host and sender):
        return False, "Email is not configured (fill in the smtp_* fields in phase7/sms_config.json)."

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_address
    msg.set_content(body)

    try:
        if port == 465:
            server = smtplib.SMTP_SSL(host, port, timeout=15)
        else:
            server = smtplib.SMTP(host, port, timeout=15)
        with server:
            server.ehlo()
            if port != 465 and server.has_extn("starttls"):
                server.starttls()
                server.ehlo()
            if user and password:
                server.login(user, password)
            server.send_message(msg)
        return True, "sent"
    except smtplib.SMTPAuthenticationError:
        return False, f"to {to_address}: the email server rejected the login (for Gmail, use an App Password, not your normal password)"
    except Exception as e:
        return False, f"to {to_address}: could not send email ({e})"


# ---------------------------------------------------------------- SMS sending

def send_sms(to_number_10_digit, body):
    """
    Sends one SMS through Twilio, IF credentials are configured via the
    environment variables TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN and
    TWILIO_FROM_NUMBER. Returns (ok, note). Without them, nothing is
    sent and the note says so -- the app never pretends a message was
    delivered when it wasn't.
    """
    # Credentials come from environment variables if set, otherwise from
    # a local file phase7/sms_config.json (easier on Windows than
    # re-typing "set ..." every time). Keep that file private.
    cfg = load_config()
    sid = os.environ.get("TWILIO_ACCOUNT_SID") or cfg.get("account_sid")
    token = os.environ.get("TWILIO_AUTH_TOKEN") or cfg.get("auth_token")
    from_number = os.environ.get("TWILIO_FROM_NUMBER") or cfg.get("from_number")
    if not (sid and token and from_number):
        return False, "SMS provider is not configured (fill in phase7/sms_config.json, or set the TWILIO_* environment variables)."

    import base64
    import urllib.error
    import urllib.parse
    import urllib.request

    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    payload = urllib.parse.urlencode({
        "To": "+91" + to_number_10_digit,
        "From": from_number,
        "Body": body,
    }).encode()
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Authorization", "Basic " + base64.b64encode(f"{sid}:{token}".encode()).decode())
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200 <= resp.status < 300, "sent"
    except urllib.error.HTTPError as e:
        # Twilio explains the refusal in the response body -- show it, so
        # the real cause (unverified number, trial restriction, wrong
        # From number...) is visible instead of just "HTTP 400".
        detail = ""
        try:
            err = json.loads(e.read().decode())
            detail = f" -- Twilio error {err.get('code')}: {err.get('message')}"
        except Exception:
            pass
        return False, f"to {to_number_10_digit}: provider rejected the message (HTTP {e.code}){detail}"
    except Exception as e:
        return False, f"could not reach the SMS provider ({e})"


# ---------------------------------------------------------------- validation

def validate_registration(data):
    """Returns a list of human-readable error strings (empty = all good)."""
    errors = []

    first_name = (data.get("first_name") or "").strip()
    last_name = (data.get("last_name") or "").strip()
    mobile = (data.get("mobile") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    address = (data.get("address") or "").strip()
    nationality = (data.get("nationality") or "").strip()
    emergency = [
        (data.get("emergency_contact_1") or "").strip(),
        (data.get("emergency_contact_2") or "").strip(),
        (data.get("emergency_contact_3") or "").strip(),
    ]
    emergency_emails = [
        (data.get("emergency_email_1") or "").strip().lower(),
        (data.get("emergency_email_2") or "").strip().lower(),
        (data.get("emergency_email_3") or "").strip().lower(),
    ]

    if not first_name:
        errors.append("First name is required.")
    if not last_name:
        errors.append("Last name is required.")
    if not address:
        errors.append("Permanent address is required.")
    if not nationality:
        errors.append("Nationality is required.")

    if not MOBILE_RE.match(mobile):
        errors.append("Mobile number must be a valid 10-digit Indian mobile number.")

    if not EMAIL_RE.match(email):
        errors.append("Email address is not valid.")

    if len(password) < 8:
        errors.append("Password must be at least 8 characters.")

    for i, contact in enumerate(emergency, start=1):
        if not MOBILE_RE.match(contact):
            errors.append(f"Emergency contact {i} must be a valid 10-digit mobile number.")

    # Emergency contacts must differ from the user's own mobile number --
    # explicitly required. Also require the three emergency contacts to
    # be distinct from EACH OTHER, since three "emergency contacts" that
    # are secretly the same number defeats the point of having three.
    if mobile and mobile in emergency:
        errors.append("An emergency contact number cannot be the same as your own mobile number.")
    if len(set(c for c in emergency if c)) < len([c for c in emergency if c]):
        errors.append("The three emergency contact numbers must all be different from each other.")

    # Each emergency contact also needs an e-mail address, which receives
    # the SOS e-mail. Same rules as the numbers: valid, not the user's own
    # e-mail, and all three different from each other.
    for i, addr in enumerate(emergency_emails, start=1):
        if not EMAIL_RE.match(addr):
            errors.append(f"Emergency contact {i} needs a valid email address.")
    if email and email in emergency_emails:
        errors.append("An emergency contact email cannot be the same as your own email.")
    if len(set(a for a in emergency_emails if a)) < len([a for a in emergency_emails if a]):
        errors.append("The three emergency contact emails must all be different from each other.")

    return errors, {
        "emergency_email_1": emergency_emails[0],
        "emergency_email_2": emergency_emails[1],
        "emergency_email_3": emergency_emails[2],
        "first_name": first_name,
        "last_name": last_name,
        "mobile": mobile,
        "email": email,
        "password": password,
        "address": address,
        "nationality": nationality,
        "emergency_contact_1": emergency[0],
        "emergency_contact_2": emergency[1],
        "emergency_contact_3": emergency[2],
    }


# ---------------------------------------------------------------- pages

@app.route("/")
def map_app():
    # Gate the map behind login: visiting localhost:5000 with no active
    # session lands on the Register page (which has a "Sign in" link for
    # existing users); only a logged-in session reaches the map itself.
    if "user_id" not in session:
        return redirect("/register")
    return send_from_directory(app.static_folder, "index.html")


@app.route("/register")
def register_page():
    if "user_id" in session:
        return redirect("/")
    return render_template("register.html")


@app.route("/login")
def login_page():
    if "user_id" in session:
        return redirect("/")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


# ---------------------------------------------------------------- APIs

@app.route("/api/register", methods=["POST"])
def api_register():
    from werkzeug.security import generate_password_hash

    data = request.get_json(silent=True) or request.form
    errors, clean = validate_registration(data)
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    db = get_db()
    existing = db.execute(
        "SELECT id FROM users WHERE mobile = ? OR email = ?", (clean["mobile"], clean["email"])
    ).fetchone()
    if existing:
        return jsonify({"ok": False, "errors": ["An account with this mobile number or email already exists."]}), 409

    db.execute(
        """
        INSERT INTO users (first_name, last_name, mobile, email, password_hash,
                            address, nationality,
                            emergency_contact_1, emergency_contact_2, emergency_contact_3,
                            emergency_email_1, emergency_email_2, emergency_email_3,
                            created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            clean["first_name"], clean["last_name"], clean["mobile"], clean["email"],
            generate_password_hash(clean["password"]),
            clean["address"], clean["nationality"],
            clean["emergency_contact_1"], clean["emergency_contact_2"], clean["emergency_contact_3"],
            clean["emergency_email_1"], clean["emergency_email_2"], clean["emergency_email_3"],
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    db.commit()

    # Auto-login right after registering, so the user lands straight on
    # the map instead of having to separately log in with the password
    # they just set.
    new_user = db.execute("SELECT id, first_name FROM users WHERE mobile = ?", (clean["mobile"],)).fetchone()
    session["user_id"] = new_user["id"]
    session["first_name"] = new_user["first_name"]
    return jsonify({"ok": True, "redirect": "/"})


@app.route("/api/login", methods=["POST"])
def api_login():
    from werkzeug.security import check_password_hash

    data = request.get_json(silent=True) or request.form
    identifier = (data.get("identifier") or "").strip().lower()
    password = data.get("password") or ""

    if not identifier or not password:
        return jsonify({"ok": False, "errors": ["Enter your email/mobile and password."]}), 400

    db = get_db()
    user = db.execute(
        "SELECT * FROM users WHERE email = ? OR mobile = ?", (identifier, identifier)
    ).fetchone()

    if user is None or not check_password_hash(user["password_hash"], password):
        return jsonify({"ok": False, "errors": ["Incorrect email/mobile or password."]}), 401

    session["user_id"] = user["id"]
    session["first_name"] = user["first_name"]
    return jsonify({"ok": True, "redirect": "/"})


@app.route("/api/location", methods=["POST"])
def api_save_location():
    """
    Called by index.html's saveLastRecordedLocation() every time a real
    GPS fix (Online Mode) succeeds. Logs it against the signed-in user,
    so each account builds up its own location history in the database
    -- not just the single most-recent fix the browser's localStorage
    already kept. Requires an active login session; otherwise there is
    no user to attach the location to, so it's silently ignored (the
    frontend already has it safely in localStorage either way).
    """
    if "user_id" not in session:
        return jsonify({"ok": False, "errors": ["Not logged in."]}), 401

    data = request.get_json(silent=True) or {}
    try:
        latitude = float(data.get("latitude"))
        longitude = float(data.get("longitude"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "errors": ["Missing or invalid latitude/longitude."]}), 400

    accuracy = data.get("accuracy")
    try:
        accuracy = float(accuracy) if accuracy is not None else None
    except (TypeError, ValueError):
        accuracy = None

    db = get_db()
    db.execute(
        """
        INSERT INTO locations (user_id, latitude, longitude, accuracy, city, client_timestamp, recorded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session["user_id"], latitude, longitude, accuracy,
            data.get("city"), data.get("timestamp"),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    db.commit()
    return jsonify({"ok": True})


@app.route("/api/me")
def api_me():
    if "user_id" not in session:
        return jsonify({"ok": False}), 401
    return jsonify({"ok": True, "first_name": session.get("first_name")})


@app.route("/api/sos", methods=["POST"])
def api_sos():
    """
    SOS button: texts all three of the signed-in user's emergency
    contacts, with a Google Maps link to the user's location, and logs
    the alert in the sos_alerts table.
    """
    if "user_id" not in session:
        return jsonify({"ok": False, "errors": ["Not logged in."]}), 401

    data = request.get_json(silent=True) or {}
    try:
        latitude = float(data.get("latitude"))
        longitude = float(data.get("longitude"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "errors": ["No valid location to send. Switch to Online Mode once to record your location."]}), 400

    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    contacts = [user["emergency_contact_1"], user["emergency_contact_2"], user["emergency_contact_3"]]

    maps_link = f"https://www.google.com/maps?q={latitude:.6f},{longitude:.6f}"
    message = (
        f"SOS ALERT from {user['first_name']} {user['last_name']} (SafeRoute). "
        f"They may be in danger and needs help. Their location: {maps_link} "
        f"Call them on {user['mobile']}."
    )

    # E-mail addresses of the three emergency contacts (accounts created
    # before this feature existed have none -- those are simply skipped).
    emails = [e for e in (user["emergency_email_1"], user["emergency_email_2"], user["emergency_email_3"]) if e]
    email_subject = f"SOS ALERT: {user['first_name']} {user['last_name']} needs help"
    email_body = (
        f"This is an automatic SOS alert from SafeRoute.\n\n"
        f"{user['first_name']} {user['last_name']} pressed the SOS button and may be in danger.\n\n"
        f"Their location: {maps_link}\n"
        f"(latitude {latitude:.6f}, longitude {longitude:.6f})\n\n"
        f"Phone: {user['mobile']}\n"
        f"Time (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"Please call them now. If you cannot reach them, contact the local police.\n"
    )

    # Send everything at once (all emails + all SMS attempts in parallel)
    # so the alert goes out immediately instead of waiting one by one.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as pool:
        sms_futures = [pool.submit(send_sms, n, message) for n in contacts]
        email_futures = [pool.submit(send_email, a, email_subject, email_body) for a in emails]
        sms_results = [f.result() for f in sms_futures]
        email_results = [f.result() for f in email_futures]

    sent = sum(1 for ok, _ in sms_results if ok)
    note = "; ".join(sorted({n for ok, n in sms_results if not ok}))
    emails_sent = sum(1 for ok, _ in email_results if ok)
    email_note = "; ".join(sorted({n for ok, n in email_results if not ok}))
    if not emails:
        email_note = "No emergency email addresses are saved on this account."

    db.execute(
        "INSERT INTO sos_alerts (user_id, latitude, longitude, message, contacts_total, contacts_sent, delivery_note, "
        "emails_total, emails_sent, email_note, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (user["id"], latitude, longitude, message, len(contacts), sent, note or None,
         len(emails), emails_sent, email_note or None,
         datetime.now(timezone.utc).isoformat()),
    )
    db.commit()
    # message + contacts are returned too, so the page can offer one-tap
    # "Send via WhatsApp / SMS" links from the user's own phone when the
    # automatic SMS provider didn't deliver.
    return jsonify({"ok": True, "sent": sent, "total": len(contacts), "note": note,
                    "message": message, "contacts": contacts,
                    "emails_sent": emails_sent, "emails_total": len(emails), "email_note": email_note})


@app.route("/api/location/latest")
def api_latest_location():
    """Returns the signed-in user's most recently saved location, if any."""
    if "user_id" not in session:
        return jsonify({"ok": False, "errors": ["Not logged in."]}), 401

    db = get_db()
    row = db.execute(
        "SELECT latitude, longitude, accuracy, city, client_timestamp, recorded_at "
        "FROM locations WHERE user_id = ? ORDER BY id DESC LIMIT 1",
        (session["user_id"],),
    ).fetchone()
    return jsonify({"ok": True, "location": dict(row) if row else None})


if __name__ == "__main__":
    init_db()
    print(f"Database ready at {DB_PATH}")
    print("Open http://localhost:5000/  (map app)")
    print("Open http://localhost:5000/register  (create an account)")
    print("Open http://localhost:5000/login  (sign in)")
    app.run(debug=True, port=5000)