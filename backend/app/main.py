from fastapi import FastAPI
from fastapi import UploadFile, File
from .speech import transcribe_audio
from .nlp import summarize_text
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

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
async def transcribe(file: UploadFile = File(...)):
    with open("temp.wav", "wb") as f:
        f.write(await file.read())
    text = transcribe_audio("temp.wav")
    return {"transcription": text}

@app.post("/summarize")
async def summarize(data: dict):
    return {"summary": summarize_text(data["transcription"])}

