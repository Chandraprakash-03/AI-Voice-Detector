# Model Recommendations for AI Voice Detection API

## Primary Architecture (Recommended)

### 1. Feature Extraction Layer

**XLS-R 300M** (`facebook/wav2vec2-xls-r-300m`)

- **Why**: Best balance of performance and size for multilingual tasks
- **Languages**: Supports Tamil, English, Hindi, Malayalam, Telugu
- **Size**: ~1.2GB
- **Use**: Extract rich speech representations

### 2. Classification Head

**Custom CNN-LSTM Architecture** (per language)

- **Input**: XLS-R embeddings (1024-dim)
- **Architecture**:
  - CNN layers for local pattern detection
  - LSTM layers for temporal modeling
  - Dense layers for classification
- **Output**: Binary classification (Human/AI-generated)

## Alternative Architectures

### Option A: Lightweight Setup

```
HuBERT Base (95M) + Simple MLP
- Faster inference
- Lower memory usage
- Good for development/testing
```

### Option B: High-Performance Setup

```
XLS-R 1B + Transformer Classifier
- Best accuracy
- Higher computational requirements
- Production-ready for high-stakes applications
```

### Option C: Hybrid Approach

```
XLS-R 300M + Language-specific fine-tuned heads
- Good balance of accuracy and efficiency
- Specialized performance per language
- Recommended for production
```

## Model Files Structure

```
models/
├── foundation/
│   ├── xls-r-300m/
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   └── preprocessor_config.json
│   └── hubert-base/
│       ├── pytorch_model.bin
│       └── config.json
├── classifiers/
│   ├── tamil_classifier.h5
│   ├── english_classifier.h5
│   ├── hindi_classifier.h5
│   ├── malayalam_classifier.h5
│   └── telugu_classifier.h5
└── preprocessing/
    ├── feature_scalers/
    └── tokenizers/
```

## Download Instructions

### Using Hugging Face Hub

```python
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# Download XLS-R 300M
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xls-r-300m")

# Save locally
model.save_pretrained("models/foundation/xls-r-300m/")
processor.save_pretrained("models/foundation/xls-r-300m/")
```

### Using Git LFS

```bash
# Install git-lfs
git lfs install

# Clone model repository
git clone https://huggingface.co/facebook/wav2vec2-xls-r-300m models/foundation/xls-r-300m/
```

## Performance Expectations

| Model       | Languages | Size  | Inference Time | Accuracy  |
| ----------- | --------- | ----- | -------------- | --------- |
| HuBERT Base | Universal | 360MB | ~100ms         | Good      |
| XLS-R 300M  | 128 langs | 1.2GB | ~200ms         | Excellent |
| XLS-R 1B    | 128 langs | 3.9GB | ~500ms         | Best      |

## Training Data Requirements

### For Custom Classifiers

- **Human Speech**: 10-50 hours per language
- **AI-Generated**: 10-50 hours per language
- **Sources**: TTS systems, voice cloning, deepfake audio
- **Balance**: Equal amounts of human/AI samples

### Data Sources

- **Human**: Common Voice, LibriSpeech, language-specific corpora
- **AI-Generated**:
  - Tacotron 2 outputs
  - WaveNet samples
  - Real-time voice cloning samples
  - Commercial TTS (Google, Amazon, Azure)

## Implementation Priority

1. **Phase 1**: XLS-R 300M + Simple MLP classifiers
2. **Phase 2**: Add language-specific fine-tuning
3. **Phase 3**: Implement ensemble methods
4. **Phase 4**: Add real-time optimization

## Memory and Compute Requirements

### Development

- **RAM**: 8GB minimum
- **GPU**: Optional (CPU inference ~200ms)
- **Storage**: 5GB for all models

### Production

- **RAM**: 16GB recommended
- **GPU**: RTX 3080 or equivalent for <50ms inference
- **Storage**: 10GB with model versioning
