import os
from typing import Optional, Dict, Any, List

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except Exception:
    firebase_admin = None
    credentials = None
    firestore = None
    FIREBASE_AVAILABLE = False


def init_firestore():
    """Initialize Firebase app using credentials provided via env or path."""
    if not FIREBASE_AVAILABLE:
        raise RuntimeError("firebase-admin not installed")

    # Prefer a path to a service account JSON file
    cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
    cred_json = os.getenv("FIREBASE_CREDENTIALS_JSON")

    if cred_path and os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)
    elif cred_json:
        # credentials.Certificate can accept a dict
        import json
        cred = credentials.Certificate(json.loads(cred_json))
    else:
        # Try application default credentials
        cred = None

    if not firebase_admin._apps:
        if cred:
            firebase_admin.initialize_app(cred)
        else:
            firebase_admin.initialize_app()


def get_firestore_client():
    if not FIREBASE_AVAILABLE:
        raise RuntimeError("firebase-admin not installed")
    return firestore.client()


def create_session(doctor_name: str, patient_name: str) -> str:
    db = get_firestore_client()
    doc_ref = db.collection("sessions").document()
    doc_ref.set({
        "doctor_name": doctor_name,
        "patient_name": patient_name,
        "created_at": firestore.SERVER_TIMESTAMP,
    })
    return doc_ref.id


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    db = get_firestore_client()
    doc = db.collection("sessions").document(session_id).get()
    if not doc.exists:
        return None
    data = doc.to_dict()
    data["id"] = doc.id
    return data


def list_sessions(limit: int = 50) -> List[Dict[str, Any]]:
    db = get_firestore_client()
    docs = db.collection("sessions").order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit).stream()
    result = []
    for d in docs:
        item = d.to_dict()
        item["id"] = d.id
        result.append(item)
    return result


def add_transcription(session_id: str, audio_path: str, transcription: str, summary: str):
    db = get_firestore_client()
    doc_ref = db.collection("sessions").document(session_id).collection("transcriptions").document()
    doc_ref.set({
        "audio_path": audio_path,
        "transcription": transcription,
        "summary": summary,
        "created_at": firestore.SERVER_TIMESTAMP,
    })
    return doc_ref.id


def list_transcriptions(session_id: str, limit: int = 100):
    db = get_firestore_client()
    docs = db.collection("sessions").document(session_id).collection("transcriptions").order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit).stream()
    res = []
    for d in docs:
        item = d.to_dict()
        item["id"] = d.id
        res.append(item)
    return res
