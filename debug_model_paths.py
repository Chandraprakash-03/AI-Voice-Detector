#!/usr/bin/env python3
"""
Debug script to check model paths and loading
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import get_settings
from src.detection.engine import DetectionEngine

def debug_model_paths():
    """Debug model path construction and loading"""
    
    print("🔍 Debugging Model Paths")
    print("=" * 50)
    
    # Get settings
    settings = get_settings()
    
    print(f"Models base path: {settings.ml.models_base_path}")
    print(f"Foundation model setting: {settings.ml.foundation_model}")
    print(f"Foundation models path: {settings.ml.foundation_models_path}")
    
    # Check what models exist
    models_base = Path(settings.ml.models_base_path)
    foundation_dir = models_base / "foundation"
    
    print(f"\nFoundation directory: {foundation_dir}")
    print(f"Foundation directory exists: {foundation_dir.exists()}")
    
    if foundation_dir.exists():
        print("Available foundation models:")
        for item in foundation_dir.iterdir():
            if item.is_dir():
                print(f"  - {item.name}: {list(item.glob('*.json'))}")
    
    # Test DetectionEngine initialization
    print(f"\n🚀 Testing DetectionEngine initialization...")
    
    try:
        engine = DetectionEngine(settings)
        print(f"✅ Engine initialized successfully")
        print(f"Selected foundation model: {engine.foundation_model_name}")
        print(f"Foundation config: {engine.foundation_config}")
        
        # Check the constructed path
        foundation_model_path = engine.model_base_path / engine.foundation_config["path"]
        print(f"Constructed path: {foundation_model_path}")
        print(f"Path exists: {foundation_model_path.exists()}")
        
        if foundation_model_path.exists():
            print("Files in model directory:")
            for file in foundation_model_path.iterdir():
                print(f"  - {file.name}")
        
        # Test model loading directly
        print(f"\n🧪 Testing direct model loading...")
        try:
            from transformers import HubertModel, Wav2Vec2FeatureExtractor
            
            # Test HuBERT loading
            hubert_path = models_base / "foundation" / "hubert-base"
            print(f"Testing HuBERT loading from: {hubert_path}")
            
            if hubert_path.exists():
                model = HubertModel.from_pretrained(str(hubert_path))
                processor = Wav2Vec2FeatureExtractor.from_pretrained(str(hubert_path))
                print("✅ HuBERT loaded successfully")
                print(f"Model config: {model.config}")
            else:
                print("❌ HuBERT path doesn't exist")
                
        except Exception as e:
            print(f"❌ Direct model loading failed: {e}")
        
    except Exception as e:
        print(f"❌ Engine initialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model_paths()