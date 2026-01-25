# Model Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test Engine (Before Downloading Models)

```bash
python scripts/test_engine.py
```

This will test the engine with dummy models and show you what's working.

### 3. Download Models

#### Option A: Minimal Setup (Recommended for Development)

```bash
python scripts/download_models.py --mode minimal
```

Downloads: HuBERT Base (~360MB)

#### Option B: Full Setup (Recommended for Production)

```bash
python scripts/download_models.py --mode all
```

Downloads: XLS-R 300M + HuBERT Base (~1.6GB)

#### Option C: XLS-R Only (Best for Multilingual)

```bash
python scripts/download_models.py --mode xls-r
```

Downloads: XLS-R 300M (~1.2GB)

### 4. Test with Downloaded Models

```bash
python scripts/test_engine.py
```

### 5. Start the API

```bash
python src/main.py
```

## What Changed

The detection engine has been completely updated to work with the new model structure:

### Old Architecture

- Single models per language (tamil_voice_detector.h5, etc.)
- MFCC features as input
- Direct classification

### New Architecture

- **Foundation Models**: XLS-R, HuBERT, Wav2Vec2 for feature extraction
- **Language Classifiers**: Lightweight models that work on foundation model embeddings
- **Two-stage process**: Foundation model → Embeddings → Classifier → Result

### Benefits

- **Better accuracy**: Foundation models provide richer representations
- **Multilingual support**: XLS-R works across all your languages
- **Smaller classifiers**: Only need to train small models on top of embeddings
- **Flexibility**: Can switch foundation models without retraining classifiers

### 6. Verify Installation

```bash
python -c "from src.detection.engine import DetectionEngine; print('Models ready!')"
```

## Model Recommendations by Use Case

### Development & Testing

- **Model**: HuBERT Base
- **Size**: 360MB
- **Languages**: Universal (works with all 5 languages)
- **Performance**: Good for prototyping

### Production (Multilingual Focus)

- **Model**: XLS-R 300M
- **Size**: 1.2GB
- **Languages**: Optimized for 128 languages including yours
- **Performance**: Excellent accuracy

### Production (English Focus)

- **Model**: Wav2Vec2 Base
- **Size**: 360MB
- **Languages**: English-optimized
- **Performance**: Excellent for English

## Manual Download (Alternative)

If the script fails, you can download manually:

### Using Hugging Face Hub

```python
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# For XLS-R 300M
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xls-r-300m")

model.save_pretrained("models/foundation/xls-r-300m/")
processor.save_pretrained("models/foundation/xls-r-300m/")
```

### Using Git LFS

```bash
# Install git-lfs first
git lfs install

# Clone model repository
git clone https://huggingface.co/facebook/wav2vec2-xls-r-300m models/foundation/xls-r-300m/
```

## Directory Structure After Setup

```
models/
├── foundation/
│   ├── xls-r-300m/          # XLS-R 300M model files
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   └── preprocessor_config.json
│   └── hubert-base/         # HuBERT Base model files
│       ├── pytorch_model.bin
│       └── config.json
├── classifiers/             # Your custom classifiers (to be trained)
│   ├── tamil_classifier.h5
│   ├── english_classifier.h5
│   ├── hindi_classifier.h5
│   ├── malayalam_classifier.h5
│   └── telugu_classifier.h5
└── preprocessing/           # Feature scalers and preprocessors
    └── feature_scalers/
```

## Training Your Own Classifiers

Once you have the foundation models, you'll need to train language-specific classifiers:

1. **Collect Data**: Human speech + AI-generated samples for each language
2. **Extract Features**: Use foundation models to get embeddings
3. **Train Classifiers**: Binary classification (Human vs AI)
4. **Save Models**: Store in `models/classifiers/`

## Troubleshooting

### Common Issues

1. **Out of Memory**: Use smaller models (HuBERT Base instead of XLS-R)
2. **Slow Download**: Use `--models-dir` to specify faster storage
3. **Import Errors**: Ensure all requirements are installed

### Performance Tips

1. **GPU Acceleration**: Install CUDA for faster inference
2. **Model Caching**: Models are cached after first load
3. **Batch Processing**: Process multiple files together

## Next Steps

1. Run the API: `python src/main.py`
2. Test with sample audio files
3. Train custom classifiers with your data
4. Deploy to production

For more details, see `docs/MODEL_RECOMMENDATIONS.md`
