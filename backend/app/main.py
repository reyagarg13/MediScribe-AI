from fastapi import FastAPI
from fastapi import UploadFile, File
from .speech import transcribe_audio
from .nlp import summarize_text
from .db import get_db, init_db
from .image_analysis import router as image_router
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
init_db()
app.include_router(image_router)
from fastapi import HTTPException
from typing import Optional
@app.post("/sessions")
def create_session(doctor_name: str, patient_name: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (doctor_name, patient_name) VALUES (?, ?)",
            (doctor_name, patient_name)
        )
        conn.commit()
        session_id = cursor.lastrowid
    return {"session_id": session_id}

@app.get("/sessions/{session_id}")
def get_session(session_id: int):
    with get_db() as conn:
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
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "AI Medical Scribe Backend is running!"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), session_id: Optional[int] = None):
    audio_path = "temp.wav"
    with open(audio_path, "wb") as f:
        f.write(await file.read())
    text = transcribe_audio(audio_path)
    summary = summarize_text(text)
    # Store in DB if session_id provided
    if session_id:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO transcriptions (session_id, audio_path, transcription, summary) VALUES (?, ?, ?, ?)",
                (session_id, audio_path, text, summary)
            )
            conn.commit()
    return {"transcription": text, "summary": summary}


# List all transcriptions for a session
@app.get("/sessions/{session_id}/transcriptions")
def list_transcriptions(session_id: int):
    with get_db() as conn:
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

