# Mediscribe AI - Repo (Scaffold)

This repository scaffold contains scripts and configuration to:
- generate pseudo labels using EasyOCR
- preprocess images
- fine-tune TrOCR (Microsoft) for handwritten prescription OCR
- run inference via a FastAPI server
- utilities for evaluation and parsing

## Quick Start (local)
1. Create virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. Put your prescription images into `data/images/`.

3. Generate pseudo-labels:
   ```
   python src/generate_pseudo_labels.py --images_dir data/images --out data/pseudo_labels.jsonl
   ```

4. Train (example):
   ```
   python src/train_trocr.py --train_data data/pseudo_labels.jsonl --output_dir models/trocr_mediscribe
   ```

5. Run API server:
   ```
   uvicorn src.api:app --host 0.0.0.0 --port 8000
   ```

## Notes
- This scaffold is ready-to-edit. You may need to adjust batch sizes and fp16 flags to fit your RTX 4060 (8GB VRAM).
- For production deployment, build the Docker image and run the container.
