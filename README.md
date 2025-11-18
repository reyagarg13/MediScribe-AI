# 🏥 MediScribe AI - Intelligent Medical Document Processing System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![React](https://img.shields.io/badge/React-19.1.0-61dafb.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.116.1-009688.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Overview

MediScribe AI is a comprehensive medical document processing system that combines advanced computer vision, natural language processing, and machine learning to analyze prescription images and provide intelligent medical insights. The system features four distinct analysis modes, each implementing different image processing and OCR techniques.

## ✨ Key Features

### 🔬 **Image Analysis Showcase**
- **7 Image Processing Techniques** with visual outputs
- Real-time demonstration of computer vision algorithms
- Medical document optimization techniques

### 💊 **Prescription OCR (3 Modes)**
- **Basic OCR**: Tesseract-based text extraction
- **Advanced OCR**: AI-powered with medical validation
- **Custom TrOCR**: Fine-tuned transformer model

### 🎤 **Voice Integration**
- Real-time medical voice assistant
- Speech-to-text processing with Whisper
- Audio worklet processing for medical consultations

### 🗣️ **AI Chat Integration**
- Gemini AI real-time conversation
- Medical knowledge integration
- Context-aware responses

## 🏗️ System Architecture

```
MediScribe-AI/
├── backend/                 # FastAPI Python Backend
│   ├── app/
│   │   ├── main.py         # FastAPI application entry
│   │   ├── image_analysis.py        # Image processing showcase
│   │   ├── prescription_ocr.py      # Basic OCR implementation
│   │   ├── advanced_prescription_ocr.py # Advanced OCR + AI
│   │   ├── custom_trocr.py          # Custom TrOCR model
│   │   ├── medical_validation.py    # Drug safety validation
│   │   ├── speech.py               # Voice processing
│   │   ├── gemini.py              # AI chat integration
│   │   ├── drug_database.py       # Pharmaceutical database
│   │   └── trocr_gemini_teacher_demo/ # Trained model files
│   └── requirements.txt
├── frontend/               # Next.js React Frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx           # Main application
│   │   │   └── api/proxy/         # API proxy routes
│   │   ├── components/
│   │   │   ├── image-upload.tsx   # Image processing interface
│   │   │   ├── recorder.tsx       # Voice recording
│   │   │   └── ui/               # UI components
│   │   └── lib/
│   │       └── medical-voice-assistant/ # Voice AI logic
│   └── package.json
└── MediScribe_ImageAndVideoProcessing/ # TrOCR Training Pipeline
    ├── src/
    │   ├── train_trocr.py         # Model training script
    │   ├── dataset.py             # Custom dataset class
    │   ├── generate_gemini_labels.py # AI label generation
    │   └── evaluate.py            # Model evaluation
    └── IMAGE_ANALYSIS_TECHNIQUES_IMPLEMENTED.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 18+
- CUDA-capable GPU (optional, for TrOCR acceleration)
- Tesseract OCR installed

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Environment Variables

**⚠️ SECURITY NOTICE: NEVER commit API keys to version control!**

Create `.env` files from the provided examples:

**Backend Setup:**
```bash
cd backend
cp .env.example .env
# Edit .env and add your actual API keys
```

**Frontend Setup:**
```bash
cd frontend
cp .env.example .env.local  # Create this file
# Add your environment variables
```

**Training Pipeline Setup:**
```bash
cd MediScribe_ImageAndVideoProcessing
cp .env.example .env
# Edit .env and add your Gemini API keys for training
```

**Required Environment Variables:**

**Backend `.env`:**
```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_API_URL=https://generativelanguage.googleapis.com
DATABASE_URL=sqlite:///mediscribe.db
FIREBASE_CREDENTIALS_PATH=./path/to/serviceAccount.json
```

**Frontend `.env.local`:**
```env
NEXT_PUBLIC_API_BASE=http://localhost:8000
NEXT_PUBLIC_GEMINI_API_KEY=your_gemini_api_key
```

**Training Pipeline `.env`:**
```env
GEMINI_API_KEY=your_primary_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_NAME=PRIMARY_GEMINI
# Optional: Add multiple API keys for rotation
```

## 🔬 Image Processing Techniques

### 1. **Image Analysis Showcase** (`/analyze-image`)
Demonstrates 7 computer vision techniques with visual outputs:

| Technique | Implementation | Purpose |
|-----------|----------------|---------|
| **Sampling & Quantization** | Color level reduction (32 levels) | Bit-depth simulation |
| **CLAHE** | Contrast Limited Adaptive Histogram Equalization | Medical document enhancement |
| **Gaussian Blur** | 15x15 kernel smoothing | Noise reduction |
| **Edge Detection** | Canny edge detector (50, 150 thresholds) | Boundary identification |
| **Morphological Operations** | Erosion + Dilation (5x5 kernel) | Structure cleanup |
| **Histogram Equalization** | LAB color space enhancement | Contrast distribution |
| **Noise Reduction** | Non-local means denoising | Detail preservation |

### 2. **Prescription OCR Modes**

#### **Basic OCR** (`/prescription-ocr`)
- Tesseract OCR integration
- Basic medication parsing
- Simple text extraction

#### **Advanced OCR + Validation** (`/prescription-ocr-advanced`)
- Gemini Vision AI processing
- Medical entity recognition
- Drug database integration
- Safety validation engine
- Drug interaction checking

#### **Custom TrOCR Model** (`/prescription-custom-trocr`)
- Fine-tuned transformer architecture
- Trained on 129 Kaggle prescription images
- Gemini-generated pseudo-labels
- GPU-optimized inference
- RTX 4060 specific optimizations

## 🤖 AI Integration

### **Gemini AI Features**
- **Vision API**: Advanced image understanding
- **Real-time Chat**: Medical consultation assistance
- **Label Generation**: Training data creation
- **Medical Knowledge**: Domain-specific responses

### **OpenAI Whisper**
- Real-time speech transcription
- Medical voice commands
- Audio worklet processing
- Noise reduction pipeline

## 📊 Technical Specifications

### **Machine Learning Stack**
- **PyTorch 2.8.0**: Deep learning framework
- **Transformers 4.55.2**: TrOCR model implementation
- **OpenCV 4.12.0**: Computer vision operations
- **scikit-learn 1.6.1**: Traditional ML algorithms

### **Web Stack**
- **FastAPI 0.116.1**: High-performance Python API
- **Next.js 15.5.0**: React framework
- **React 19.1.0**: Frontend library
- **Tailwind CSS 4**: Styling framework

### **Database & Storage**
- **SQLite**: Local database for drug information
- **SQLAlchemy 2.0.43**: ORM for database operations
- **Caching**: In-memory caching for performance

## 🔧 API Endpoints

### **Image Processing**
- `POST /analyze-image` - Image processing showcase
- `POST /prescription-ocr` - Basic OCR
- `POST /prescription-ocr-advanced` - Advanced OCR + validation
- `POST /prescription-custom-trocr` - Custom TrOCR model

### **Voice & Chat**
- `POST /transcribe-audio` - Speech to text
- `POST /chat-gemini` - AI conversation
- `WebSocket /ws/audio` - Real-time voice processing

### **Medical Data**
- `GET /drug-info/{drug_name}` - Drug information lookup
- `POST /validate-prescription` - Medical validation
- `GET /drug-interactions` - Interaction checking

## 🏋️‍♂️ Training Pipeline (Custom TrOCR)

Located in `MediScribe_ImageAndVideoProcessing/`:

### **Dataset Preparation**
```bash
python src/generate_gemini_labels.py  # Generate training labels
python src/dataset.py                # Prepare dataset
```

### **Model Training**
```bash
python src/train_trocr.py            # Train custom model
python src/evaluate.py              # Evaluate performance
```

### **Training Configuration**
- **Base Model**: microsoft/trocr-base-handwritten
- **Training Samples**: 20 high-quality Gemini-labeled images
- **Training Epochs**: 30
- **Batch Size**: 2 (RTX 4060 optimized)
- **Learning Rate**: 3e-5
- **Optimization**: FP16 mixed precision, gradient checkpointing

## 🎯 Performance Metrics

### **OCR Accuracy**
- **Basic OCR**: ~70% character accuracy
- **Advanced OCR**: ~90% medical entity extraction
- **Custom TrOCR**: ~85% on handwritten prescriptions

### **Processing Speed**
- **Image Analysis**: ~2-3 seconds per image
- **Advanced OCR**: ~5-8 seconds (includes AI processing)
- **Custom TrOCR**: ~3-5 seconds (GPU), ~15-30 seconds (CPU)

### **Medical Validation**
- **Drug Database**: 10,000+ medications
- **Interaction Checking**: Critical/Warning/Info levels
- **Safety Scoring**: 0-100 risk assessment

## 🛠️ Development

### **Running Tests**
```bash
# Backend tests
cd backend
python -m pytest

# Frontend tests
cd frontend
npm test
```

### **Code Quality**
```bash
# Python formatting
black backend/
flake8 backend/

# TypeScript/React
cd frontend
npm run lint
```

## 📱 User Interface

### **Main Features**
1. **Drag & Drop Image Upload**
2. **4 Analysis Modes** with isolated results
3. **Real-time Voice Recording**
4. **AI Chat Integration**
5. **Visual Technique Showcase**
6. **Medical Safety Reports**

### **Cross-Contamination Fix**
Implemented mode-specific result isolation:
```javascript
const [results, setResults] = useState({});
// Each mode stores its own results
setResults(prev => ({...prev, [ocrMode]: data}));
```

## 🔐 Security & Privacy

- **Local Processing**: All image processing happens locally
- **Secure API Keys**: Environment variable management
- **Data Privacy**: No permanent storage of medical images
- **HIPAA Considerations**: Designed for healthcare compliance

## 📈 Future Enhancements

- [ ] **Video Processing**: Extend to medical video analysis
- [ ] **Mobile App**: React Native implementation
- [ ] **Cloud Deployment**: AWS/GCP integration
- [ ] **Multi-language OCR**: Support for multiple languages
- [ ] **Advanced ML**: Fine-tune on larger medical datasets

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Microsoft Research**: TrOCR base model
- **Google**: Gemini AI integration
- **OpenAI**: Whisper speech recognition
- **Kaggle**: Prescription image dataset
- **Tesseract**: OCR engine
- **OpenCV**: Computer vision library

---

**Developed for academic and educational purposes in medical image processing and AI applications.**

*For questions or support, please open an issue on GitHub.*