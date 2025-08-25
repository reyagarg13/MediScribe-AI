from fastapi import FastAPI
from dotenv import load_dotenv
import os
from fastapi import UploadFile, File
from .speech import transcribe_audio
from .nlp import summarize_text
from .db import get_db, init_db
from .image_analysis import router as image_router
from fastapi.middleware.cors import CORSMiddleware


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

app = FastAPI()

# optional gemini adapter
gemini_available = False
try:
    from . import gemini
    gemini_available = True
except Exception:
    gemini_available = False

init_db()
app.include_router(image_router)
from fastapi import HTTPException
from typing import Optional
@app.post("/sessions")
def create_session(doctor_name: str, patient_name: str):
    with get_db() as conn:
        # If conn is the firestore module (firestore_db wrapper), use its API
        if hasattr(conn, "create_session"):
            session_id = conn.create_session(doctor_name, patient_name)
            return {"session_id": session_id}

        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (doctor_name, patient_name) VALUES (?, ?)",
            (doctor_name, patient_name)
        )
        conn.commit()
        session_id = cursor.lastrowid
    return {"session_id": session_id}

@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    with get_db() as conn:
        if hasattr(conn, "get_session"):
            s = conn.get_session(session_id)
            if not s:
                raise HTTPException(status_code=404, detail="Session not found")
            return s

        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        session = cursor.fetchone()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "id": session[0],
            "doctor_name": session[1],
            "patient_name": session[2],
            "created_at": session[3]
        }

@app.get("/sessions")
def list_sessions():
    with get_db() as conn:
        if hasattr(conn, "list_sessions"):
            return conn.list_sessions()

        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions ORDER BY created_at DESC")
        sessions = cursor.fetchall()
        return [
            {
                "id": s[0],
                "doctor_name": s[1],
                "patient_name": s[2],
                "created_at": s[3]
            } for s in sessions
        ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "AI Medical Scribe Backend is running!"}


@app.get("/health")
def health():
    # Detect DB mode
    db_mode = "sqlite"
    try:
        from . import db as _db
        if hasattr(_db, "USE_FIRESTORE") and _db.USE_FIRESTORE:
            db_mode = "firestore"
        elif _db.DATABASE_URL:
            db_mode = "postgres"
    except Exception:
        pass

    return {
        "ok": True,
        "db_mode": db_mode,
        "firestore_configured": bool(os.getenv("FIREBASE_CREDENTIALS_PATH") or os.getenv("FIREBASE_CREDENTIALS_JSON")),
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY") and os.getenv("GEMINI_API_URL")),
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), session_id: Optional[int] = None):
    audio_path = "temp.wav"
    with open(audio_path, "wb") as f:
        f.write(await file.read())
    text = transcribe_audio(audio_path)
    summary = summarize_text(text)
    # If Gemini configured, enrich the result
    gemini_result = None
    if gemini_available and gemini.available():
        try:
            gemini_result = gemini.analyze_text(text, params={"mode": "clinical_summary"})
            # Optionally, you can extract a more specific summary from gemini_result
        except Exception as e:
            print("Gemini analysis failed:", e)
    # Store in DB if session_id provided
    if session_id:
        with get_db() as conn:
            if hasattr(conn, "add_transcription"):
                try:
                    conn.add_transcription(str(session_id), audio_path, text, summary)
                except Exception as e:
                    # log but don't fail the request
                    print("Firestore add_transcription failed:", e)
            else:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO transcriptions (session_id, audio_path, transcription, summary) VALUES (?, ?, ?, ?)",
                    (session_id, audio_path, text, summary)
                )
                conn.commit()
    resp = {"transcription": text, "summary": summary}
    if gemini_result:
        resp["gemini"] = gemini_result
    return resp


# List all transcriptions for a session
@app.get("/sessions/{session_id}/transcriptions")
def list_transcriptions(session_id: int):
    with get_db() as conn:
        if hasattr(conn, "list_transcriptions"):
            return conn.list_transcriptions(str(session_id))

        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, audio_path, transcription, summary, created_at FROM transcriptions WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,)
        )
        rows = cursor.fetchall()
        return [
            {
                "id": r[0],
                "audio_path": r[1],
                "transcription": r[2],
                "summary": r[3],
                "created_at": r[4]
            } for r in rows
        ]

