from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from pathlib import Path
import uvicorn
import shutil, uuid, os
from .infer import infer

app = FastAPI(title='Mediscribe AI API')

UPLOAD_DIR = Path('data/uploads')
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

class OCRResponse(BaseModel):
    text: str
    filename: str

@app.post('/upload/', response_model=OCRResponse)
async def upload_image(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1]
    fname = f"{uuid.uuid4().hex}{ext}"
    out_path = UPLOAD_DIR / fname
    with open(out_path, 'wb') as f:
        content = await file.read()
        f.write(content)
    # perform inference (synchronous)
    try:
        text = infer(str(out_path), model_dir='models/trocr_mediscribe_final')
    except Exception as e:
        text = f'ERROR: {e}'
    return {'text': text, 'filename': fname}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
