from pydantic import BaseModel
from typing import Optional, List

class SessionCreate(BaseModel):
	doctor_name: str
	patient_name: str

class SessionOut(BaseModel):
	id: int
	doctor_name: str
	patient_name: str
	created_at: str

class TranscriptionOut(BaseModel):
	id: int
	audio_path: str
	transcription: str
	summary: str
	created_at: str
