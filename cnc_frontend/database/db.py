import sqlite3
import json
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).resolve().parent / "cnc.sqlite3"

@contextmanager
def connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with connection() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS machines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            rtsp_url TEXT NOT NULL,
            max_persons INTEGER NOT NULL CHECK(max_persons >= 1),
            multiple_limit_seconds INTEGER NOT NULL DEFAULT 120,
            absence_limit_seconds INTEGER NOT NULL DEFAULT 300,
            active INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """)
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(machines)")}
        if "zone_limits" not in columns:
            conn.execute("ALTER TABLE machines ADD COLUMN zone_limits TEXT NOT NULL DEFAULT '{}'")
        legacy_machines = conn.execute(
            """SELECT id, max_persons FROM machines
               WHERE zone_limits IS NULL OR zone_limits = '' OR zone_limits = '{}'"""
        ).fetchall()
        for machine in legacy_machines:
            conn.execute(
                "UPDATE machines SET zone_limits = ? WHERE id = ?",
                (json.dumps({"zone_1": machine["max_persons"]}), machine["id"]),
            )
        conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            details TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(machine_id) REFERENCES machines(id)
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS camera_credentials (
            host TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            password TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """)

def list_machines():
    with connection() as conn:
        machines = [dict(row) for row in conn.execute("SELECT * FROM machines ORDER BY id DESC")]
    for machine in machines:
        try:
            machine["zone_limits"] = json.loads(machine["zone_limits"] or "{}")
        except json.JSONDecodeError as exc:
            print(f"Invalid zone_limits for machine {machine['id']}: {exc}")
            machine["zone_limits"] = {}
    return machines

def get_machine(machine_id: int):
    with connection() as conn:
        row = conn.execute("SELECT * FROM machines WHERE id = ?", (machine_id,)).fetchone()
        machine = dict(row) if row else None
    if machine:
        try:
            machine["zone_limits"] = json.loads(machine["zone_limits"] or "{}")
        except json.JSONDecodeError:
            machine["zone_limits"] = {}
    return machine

def create_machine(
    name, rtsp_url, max_persons, multiple_limit_seconds,
    absence_limit_seconds, zone_limits=None,
):
    with connection() as conn:
        cur = conn.execute(
            """INSERT INTO machines
            (name, rtsp_url, max_persons, multiple_limit_seconds, absence_limit_seconds, zone_limits)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (name, rtsp_url, max_persons, multiple_limit_seconds,
             absence_limit_seconds, json.dumps(zone_limits or {"zone_1": max_persons})),
        )
        return cur.lastrowid

def set_active(machine_id, active):
    with connection() as conn:
        conn.execute("UPDATE machines SET active = ? WHERE id = ?", (int(active), machine_id))

def delete_machine(machine_id):
    with connection() as conn:
        conn.execute("DELETE FROM events WHERE machine_id = ?", (machine_id,))
        result = conn.execute("DELETE FROM machines WHERE id = ?", (machine_id,))
        return result.rowcount > 0

def update_machine_settings(machine_id, max_persons, multiple_limit_seconds, absence_limit_seconds,zone_limits,):
    with connection() as conn:
        conn.execute(
            """UPDATE machines
               SET max_persons = ?, multiple_limit_seconds = ?, absence_limit_seconds = ?, zone_limits = ?
               WHERE id = ?""",
            (
                max_persons,
                multiple_limit_seconds,
                absence_limit_seconds,
                json.dumps(zone_limits),
                machine_id,
            ),
        )

def get_camera_credentials(host):
    with connection() as conn:
        row = conn.execute(
            "SELECT username, password FROM camera_credentials WHERE host = ?",
            (host,),
        ).fetchone()
    return dict(row) if row else None

def save_camera_credentials(host, username, password):
    with connection() as conn:
        conn.execute(
            """INSERT INTO camera_credentials (host, username, password)
               VALUES (?, ?, ?)
               ON CONFLICT(host) DO UPDATE SET
               username = excluded.username,
               password = excluded.password,
               updated_at = CURRENT_TIMESTAMP""",
            (host, username, password),
        )

def add_event(machine_id, event_type, details):
    with connection() as conn:
        conn.execute(
            "INSERT INTO events (machine_id, event_type, details) VALUES (?, ?, ?)",
            (machine_id, event_type, details),
        )

def recent_events(limit=50):
    with connection() as conn:
        return [dict(row) for row in conn.execute(
            """SELECT events.*, machines.name AS machine_name
               FROM events JOIN machines ON machines.id = events.machine_id
               ORDER BY events.id DESC LIMIT ?""", (limit,)
        )]
