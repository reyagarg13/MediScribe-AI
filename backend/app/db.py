import os
import sqlite3
from contextlib import contextmanager
from typing import Generator, Optional

# Firestore config detection
FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH")
FIREBASE_CREDENTIALS_JSON = os.getenv("FIREBASE_CREDENTIALS_JSON")

# Optional psycopg2 import; only required when using a real Postgres DB via DATABASE_URL
DATABASE_URL = os.getenv("DATABASE_URL")

PG_AVAILABLE = False
psycopg2 = None
if DATABASE_URL:
    try:
        import psycopg2
        PG_AVAILABLE = True
    except Exception:
        PG_AVAILABLE = False

# Try to import firestore wrapper if firebase envs provided
USE_FIRESTORE = bool(FIREBASE_CREDENTIALS_PATH or FIREBASE_CREDENTIALS_JSON)
firestore_db = None
if USE_FIRESTORE:
    try:
        from . import firestore_db
    except Exception:
        firestore_db = None


@contextmanager
def get_db() -> Generator:
    """
    Yields a DB connection object. If DATABASE_URL (Postgres) is set and psycopg2 is
    available, a psycopg2 connection will be returned. Otherwise a sqlite3
    connection to a local file is used as a fallback.
    """
    # If firestore configuration present and available, yield a special token
    if USE_FIRESTORE and firestore_db is not None:
        # Firestore uses a different API; we'll return the module itself for callers to use
        yield firestore_db
        return

    if DATABASE_URL and PG_AVAILABLE:
        conn = psycopg2.connect(DATABASE_URL)
        try:
            yield conn
        finally:
            conn.close()
    else:
        # default to sqlite file for development/fallback
        db_path = os.getenv("SQLITE_PATH", "mediscribe.db")
        conn = sqlite3.connect(db_path)
        try:
            yield conn
        finally:
            conn.close()


def init_db():
    """
    Initialize tables. Uses Postgres-compatible DDL when connected to Postgres,
    otherwise uses sqlite DDL.
    """
    # Initialize Firestore if configured
    if USE_FIRESTORE and firestore_db is not None:
        try:
            firestore_db.init_firestore()
        except Exception as e:
            print("Failed to init Firestore:", e)
        return

    if DATABASE_URL and PG_AVAILABLE:
        # Postgres setup
        with get_db() as conn:
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id SERIAL PRIMARY KEY,
                    doctor_name TEXT,
                    patient_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS transcriptions (
                    id SERIAL PRIMARY KEY,
                    session_id INTEGER,
                    audio_path TEXT,
                    transcription TEXT,
                    summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                )
                """
            )
            try:
                cur.close()
            except Exception:
                pass
    else:
        # SQLite setup (existing behaviour)
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doctor_name TEXT,
                    patient_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transcriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    audio_path TEXT,
                    transcription TEXT,
                    summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                )
            ''')
            conn.commit()


if __name__ == "__main__":
    init_db()
