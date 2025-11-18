# TrOCR Prescription OCR Training Guide

This guide provides step-by-step instructions for training a custom TrOCR model on the Kaggle prescription dataset for your MediScribe AI project.

## 🎯 Overview

You're building a custom prescription OCR model that will replace the basic "Image Analysis" functionality in your app. This model will be specifically trained on illegible medical prescriptions to achieve much better accuracy than generic OCR.

## 🏗️ Setup

### 1. Install Dependencies

```bash
cd /mnt/c/Users/pallav/Desktop/Python/MediScribe-AI/mediscribe_repo
pip install -r requirements.txt
```


### 2. Download Kaggle Dataset

1. Go to https://www.kaggle.com/datasets/mehaksingal/illegible-medical-prescription-images-dataset
2. Download the dataset
3. Extract to `data/images/` folder:

```bash
mkdir -p data/images
# Extract your downloaded dataset here
# You should have: data/images/image_001.jpg, data/images/image_002.jpg, etc.
```

## 📊 Data Preparation

### Step 1: Generate Pseudo-Labels with EasyOCR

This creates initial labels for training (noisy but useful):

```bash
python src/generate_pseudo_labels.py \
    --images_dir data/images \
    --out data/pseudo_labels.jsonl
```

Expected output format:
```json
{"image_path": "data/images/img_001.jpg", "text": "Amoxicillin 500mg twice daily", "conf": 0.85}
{"image_path": "data/images/img_002.jpg", "text": "Paracetamol 1 tablet 3x day", "conf": 0.72}
```

### Step 2: Manual Annotation (Recommended)

Create a high-quality test set by manually correcting some examples:

1. Sample 200-500 images for manual annotation
2. Use a simple text editor or Label Studio
3. Save as `data/gold_labels.jsonl` with same format

### Step 3: Split Dataset

```bash
python -c "
from src.dataset import split_dataset
split_dataset('data/pseudo_labels.jsonl', train_ratio=0.8)
"
```

This creates:
- `data/pseudo_labels_train.jsonl` (80% for training)
- `data/pseudo_labels_val.jsonl` (20% for validation)

## 🚀 Training

### Quick Start (RTX 4060 Optimized)

```bash
python src/train_trocr.py \
    --train_data data/pseudo_labels_train.jsonl \
    --val_data data/pseudo_labels_val.jsonl \
    --output_dir models/trocr_prescription_v1 \
    --epochs 5 \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --weighted_loss \
    --early_stopping_patience 3
```

### Advanced Training with Monitoring

```bash
python src/train_trocr.py \
    --train_data data/pseudo_labels_train.jsonl \
    --val_data data/pseudo_labels_val.jsonl \
    --output_dir models/trocr_prescription_v1 \
    --epochs 8 \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --weighted_loss \
    --wandb_project "mediscribe-ocr" \
    --experiment_name "trocr-prescription-v1"
```

### Expected Training Time on RTX 4060
- **Small dataset (1K images)**: ~30 minutes per epoch
- **Medium dataset (5K images)**: ~2 hours per epoch  
- **Large dataset (10K+ images)**: ~4+ hours per epoch

## 📈 Evaluation

### Evaluate on Test Set

```bash
python src/evaluate.py \
    --model_dir models/trocr_prescription_v1 \
    --test_file data/gold_labels.jsonl \
    --output_file results/evaluation_v1.json
```

### Target Metrics
- **Character Error Rate (CER)**: < 20% (initial target), < 10% (goal)
- **Word Error Rate (WER)**: < 40% (initial target), < 25% (goal)
- **Medication F1**: > 0.75
- **Processing Speed**: < 2 seconds per image on RTX 4060

## 🔧 Inference

### Single Image

```bash
python src/infer.py \
    --model_dir models/trocr_prescription_v1 \
    --image path/to/prescription.jpg
```

### Batch Processing

```bash
python src/infer.py \
    --model_dir models/trocr_prescription_v1 \
    --images_dir data/test_images \
    --output results/batch_results.json
```

### Programmatic Usage

```python
from src.infer import PrescriptionOCRModel

# Load model
model = PrescriptionOCRModel("models/trocr_prescription_v1")

# Single prediction
result = model.predict_single("prescription.jpg")
print(f"Text: {result['text']}")
print(f"Processing time: {result['processing_time']:.3f}s")

# Batch prediction
results = model.predict_batch(["img1.jpg", "img2.jpg", "img3.jpg"])
```

## 🔗 Integration with MediScribe AI

### Option 1: Replace Image Analysis Endpoint

Modify `/backend/app/image_analysis.py` to use your trained model:

```python
from mediscribe_repo.src.infer import PrescriptionOCRModel

# Global model instance
custom_ocr_model = None

def load_custom_ocr_model():
    global custom_ocr_model
    if custom_ocr_model is None:
        model_path = "mediscribe_repo/models/trocr_prescription_v1"
        custom_ocr_model = PrescriptionOCRModel(model_path)
    return custom_ocr_model

@router.post("/analyze-image")
async def analyze_image_custom(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Use your custom model
    model = load_custom_ocr_model()
    
    # Convert bytes to PIL Image
    from PIL import Image
    import io
    image = Image.open(io.BytesIO(contents))
    
    # Run prediction
    result = model.predict_single(image)
    
    return JSONResponse(content={
        "method": "custom_trocr",
        "extracted_text": result["text"],
        "processing_time": result["processing_time"],
        "model_info": result["model_info"]
    })
```

### Option 2: Add New Endpoint

Add a new endpoint specifically for your custom model:

```python
@router.post("/custom-prescription-ocr")
async def custom_prescription_ocr(file: UploadFile = File(...)):
    model = load_custom_ocr_model()
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    result = model.predict_single(image)
    return JSONResponse(content=result)
```

## 🎯 Performance Optimization

### For RTX 4060 (16GB RAM)

1. **Batch Size**: Keep at 2 (optimal for 8GB VRAM)
2. **Gradient Accumulation**: Use 4 steps (effective batch size 8)
3. **Mixed Precision**: Always enable FP16
4. **Gradient Checkpointing**: Enabled to save VRAM
5. **Image Size**: 384x384 (good balance of quality vs speed)

### Memory Monitoring

```bash
# Monitor GPU usage during training
watch -n 1 nvidia-smi

# Monitor system RAM
htop
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch_size to 1
   - Reduce image_size to 256
   - Enable gradient_checkpointing

2. **Slow Training**
   - Ensure FP16 is enabled
   - Check GPU utilization with nvidia-smi
   - Reduce num_workers if CPU bottleneck

3. **Poor Results**
   - Check pseudo-label quality
   - Add more manual annotations
   - Increase training epochs
   - Try different learning rates

4. **Import Errors**
   - Ensure all dependencies installed
   - Check Python path includes src/

### Performance Expectations

On RTX 4060:
- **Training Speed**: ~1.5-2 samples/second
- **Inference Speed**: ~0.5-1 second per image
- **Memory Usage**: ~6-7GB VRAM during training

## 📁 File Structure

```
mediscribe_repo/
├── data/
│   ├── images/              # Kaggle dataset images
│   ├── pseudo_labels.jsonl  # EasyOCR generated labels
│   ├── gold_labels.jsonl    # Manual annotations
│   └── splits/              # Train/val splits
├── models/
│   └── trocr_prescription_v1/  # Trained model
├── src/
│   ├── dataset.py           # Dataset classes
│   ├── train_trocr.py       # Training script
│   ├── evaluate.py          # Evaluation metrics
│   ├── infer.py             # Inference wrapper
│   └── generate_pseudo_labels.py
└── results/
    └── evaluation_v1.json   # Evaluation results
```

## 🎉 Next Steps

1. **Start with pseudo-labeling** the Kaggle dataset
2. **Train initial model** with default settings
3. **Evaluate performance** and identify weak points
4. **Create gold labels** for difficult cases
5. **Re-train with improved data**
6. **Integrate with your main application**
7. **Monitor performance** in production

Good luck with your prescription OCR model! 🏥✨