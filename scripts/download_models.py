#!/usr/bin/env python3
"""
Model Download Script for AI Voice Detection API

This script downloads the recommended models for the AI voice detection system.
Run this script to set up your model directory with the necessary files.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from transformers import (
        AutoModel, 
        AutoProcessor,
        Wav2Vec2Model, 
        Wav2Vec2Processor,
        HubertModel,
        Wav2Vec2FeatureExtractor
    )
    from huggingface_hub import hf_hub_download
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not installed. Install with: pip install transformers")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDownloader:
    """Downloads and organizes models for the AI voice detection API."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

        os.environ["HF_HOME"] = str(self.models_dir / ".hf_cache")
        
        # Create subdirectories
        (self.models_dir / "foundation").mkdir(exist_ok=True)
        (self.models_dir / "classifiers").mkdir(exist_ok=True)
        (self.models_dir / "explanation").mkdir(exist_ok=True)
        (self.models_dir / "preprocessing").mkdir(exist_ok=True)
        (self.models_dir / "preprocessing" / "feature_scalers").mkdir(exist_ok=True)
    
    def download_xls_r_300m(self) -> bool:
        try:
            model_name = "facebook/wav2vec2-xls-r-300m"
            save_path = self.models_dir / "foundation" / "xls-r-300m"

            logger.info(f"Downloading {model_name}...")
            logger.info(f"Saving model to: {save_path}")

            model = AutoModel.from_pretrained(model_name)
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

            model.save_pretrained(str(save_path))
            feature_extractor.save_pretrained(str(save_path))

            logger.info(f"XLS-R 300M saved to {save_path}")
            return True

        except Exception as e:
            logger.error(
                f"Failed to download XLS-R 300M: {e}",
                exc_info=True
            )
            return False
    
    def download_hubert_base(self) -> bool:
        """Download HuBERT Base model (lightweight option)."""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("transformers library required for model download")
            return False
        
        try:
            model_name = "facebook/hubert-base-ls960"
            save_path = self.models_dir / "foundation" / "hubert-base"
            
            logger.info(f"Downloading {model_name}...")
            logger.info(f"Saving model to: {save_path} ({type(save_path)})")
            # Download model and feature extractor
            model = HubertModel.from_pretrained(model_name)
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            
            # logger.info(f"Saving model to: {save_path} ({type(save_path)})")
            # Save locally
            model.save_pretrained(str(save_path))
            feature_extractor.save_pretrained(str(save_path))
            
            logger.info(f"HuBERT Base saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download HuBERT Base: {e}")
            return False
    
    def download_wav2vec2_base(self) -> bool:
        """Download Wav2Vec2 Base model (English-focused)."""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("transformers library required for model download")
            return False
        
        try:
            model_name = "facebook/wav2vec2-base-960h"
            save_path = self.models_dir / "foundation" / "wav2vec2-base"
            
            logger.info(f"Downloading {model_name}...")
            logger.info(f"Saving model to: {save_path} ({type(save_path)})")
            # Download model and processor
            model = Wav2Vec2Model.from_pretrained(model_name)
            processor = Wav2Vec2Processor.from_pretrained(model_name)
            
            
            # Save locally
            model.save_pretrained(str(save_path))
            processor.save_pretrained(str(save_path))
            
            logger.info(f"Wav2Vec2 Base saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download Wav2Vec2 Base: {e}")
            return False
    
    def download_explanation_models(self) -> bool:
        """Download explanation generation models."""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("transformers library required for explanation models")
            return False
        
        try:
            explanation_dir = self.models_dir / "explanation"
            
            # Download DistilGPT2 (recommended small model)
            model_name = "distilgpt2"
            save_path = explanation_dir / "distilgpt2"
            
            logger.info(f"Downloading {model_name} for explanation generation...")
            
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Download model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Save locally
            tokenizer.save_pretrained(save_path)
            model.save_pretrained(save_path)
            
            logger.info(f"DistilGPT2 explanation model saved to {save_path}")
            
            # Also download FLAN-T5 small as alternative
            model_name = "google/flan-t5-small"
            save_path = explanation_dir / "flan-t5-small"
            
            logger.info(f"Downloading {model_name} for explanation generation...")
            
            from transformers import AutoModelForSeq2SeqLM
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            tokenizer.save_pretrained(save_path)
            model.save_pretrained(save_path)
            
            logger.info(f"FLAN-T5 small explanation model saved to {save_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download explanation models: {e}")
            return False
    
    def create_dummy_classifiers(self) -> bool:
        """Create dummy classifier files for development."""
        try:
            languages = ["tamil", "english", "hindi", "malayalam", "telugu"]
            classifiers_dir = self.models_dir / "classifiers"
            
            for lang in languages:
                classifier_path = classifiers_dir / f"{lang}_classifier.h5"
                
                # Create empty file as placeholder
                classifier_path.touch()
                logger.info(f"Created placeholder: {classifier_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create dummy classifiers: {e}")
            return False
    
    def create_model_info(self) -> bool:
        """Create model information file."""
        try:
            info_content = """# Model Information

## Downloaded Models

### Foundation Models
- XLS-R 300M: Multilingual speech representations (1.2GB)
- HuBERT Base: Self-supervised speech model (360MB)
- Wav2Vec2 Base: English speech representations (360MB)

### Explanation Models
- DistilGPT2: Lightweight GPT-2 for explanation generation (82MB)
- FLAN-T5 Small: Instruction-tuned T5 for explanations (77MB)

### Classifiers
- Language-specific binary classifiers for AI detection
- Input: Foundation model embeddings
- Output: Human/AI-generated probability

## Usage

1. Load foundation model for feature extraction
2. Extract embeddings from audio
3. Pass embeddings to language-specific classifier
4. Get AI detection result
5. Generate natural explanation using small LM

## Model Sizes
- Total storage: ~2-5GB depending on models downloaded
- Runtime memory: ~3-5GB for inference (including explanation generation)
- GPU memory: ~2-3GB (optional, for faster inference)

## AI-Powered Explanations

The system now uses small language models to generate natural, contextual explanations:
- **DistilGPT2**: Fast, lightweight GPT-2 variant for natural text generation
- **FLAN-T5 Small**: Instruction-tuned model for structured explanations
- **Fallback**: Hardcoded explanations when models are unavailable

Benefits:
- More natural and varied explanations
- Context-aware descriptions
- Adapts to different confidence levels and languages
- Professional, technical language appropriate for the domain
"""
            
            info_path = self.models_dir / "README.md"
            with open(info_path, "w") as f:
                f.write(info_content)
            
            logger.info(f"Model info created: {info_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create model info: {e}")
            return False
    
    def download_all_recommended(self) -> Dict[str, bool]:
        """Download all recommended models."""
        results = {}
        
        logger.info("Starting model downloads...")
        
        # Download foundation models
        results["xls_r_300m"] = self.download_xls_r_300m()
        results["hubert_base"] = self.download_hubert_base()
        
        # Download explanation models
        results["explanation_models"] = self.download_explanation_models()
        
        # Create placeholders
        results["dummy_classifiers"] = self.create_dummy_classifiers()
        results["model_info"] = self.create_model_info()
        
        return results
    
    def download_minimal(self) -> Dict[str, bool]:
        """Download minimal set for development."""
        results = {}
        
        logger.info("Starting minimal model download...")
        
        # Just HuBERT for development
        results["hubert_base"] = self.download_hubert_base()
        
        # Download lightweight explanation model
        results["explanation_models"] = self.download_explanation_models()
        
        results["dummy_classifiers"] = self.create_dummy_classifiers()
        results["model_info"] = self.create_model_info()
        
        return results


def main():
    """Main function to run model downloads."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download models for AI voice detection")
    parser.add_argument(
        "--mode", 
        choices=["all", "minimal", "xls-r", "hubert", "explanation"], 
        default="minimal",
        help="Download mode: all (all models), minimal (HuBERT + explanation), xls-r (XLS-R only), hubert (HuBERT only), explanation (explanation models only)"
    )
    parser.add_argument(
        "--models-dir", 
        default="models",
        help="Directory to save models (default: models)"
    )
    
    args = parser.parse_args()
    
    if not TRANSFORMERS_AVAILABLE:
        print("Error: transformers library not installed")
        print("Install with: pip install transformers torch")
        sys.exit(1)
    
    downloader = ModelDownloader(args.models_dir)
    
    if args.mode == "all":
        results = downloader.download_all_recommended()
    elif args.mode == "minimal":
        results = downloader.download_minimal()
    elif args.mode == "xls-r":
        results = {"xls_r_300m": downloader.download_xls_r_300m()}
    elif args.mode == "hubert":
        results = {"hubert_base": downloader.download_hubert_base()}
    elif args.mode == "explanation":
        results = {"explanation_models": downloader.download_explanation_models()}
    
    # Print results
    print("\n" + "="*50)
    print("DOWNLOAD RESULTS")
    print("="*50)
    
    for model, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{model:20} {status}")
    
    successful = sum(results.values())
    total = len(results)
    print(f"\nCompleted: {successful}/{total} downloads successful")
    
    if successful > 0:
        print(f"\nModels saved to: {Path(args.models_dir).absolute()}")
        print("You can now run your AI voice detection API!")


if __name__ == "__main__":
    main()