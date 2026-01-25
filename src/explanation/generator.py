"""
AI-powered explanation generator for voice detection results.

This module uses small language models to generate natural, contextual explanations
for AI vs human voice detection results instead of hardcoded templates.
"""

import logging
import time
import warnings
from typing import Dict, Any, Optional, List
from pathlib import Path

try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        pipeline,
        set_seed
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers library not available for explanation generation")

from src.audio.processor import AudioFeatures
from src.detection.engine import DetectionResult

logger = logging.getLogger(__name__)

class ExplanationGenerator:
    """
    Generates natural language explanations for voice detection results using small LMs.
    """
    
    # Available explanation models (small, efficient models)
    EXPLANATION_MODELS = {
        "distilgpt2": {
            "model_name": "distilgpt2",
            "type": "causal",
            "size": "82MB",
            "recommended": True,
            "description": "Fast, lightweight GPT-2 variant"
        },
        "flan-t5-small": {
            "model_name": "google/flan-t5-small",
            "type": "seq2seq", 
            "size": "77MB",
            "recommended": True,
            "description": "Instruction-tuned T5 model"
        },
        "gpt2": {
            "model_name": "gpt2",
            "type": "causal",
            "size": "124MB", 
            "recommended": False,
            "description": "Original GPT-2 small"
        }
    }
    
    def __init__(self, model_name: str = "auto", cache_dir: Optional[str] = None):
        """
        Initialize the explanation generator.
        
        Args:
            model_name: Model to use ("auto", "distilgpt2", "flan-t5-small", "gpt2")
            cache_dir: Directory to cache models
        """
        self.model_name = self._select_model(model_name)
        self.cache_dir = cache_dir or "models/explanation"
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        # Create cache directory
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Set random seed for reproducible explanations
        set_seed(42)
        
        logger.info(f"Explanation generator initialized with model: {self.model_name}")
    
    def _select_model(self, model_name: str) -> str:
        """Select the best available explanation model."""
        if model_name == "auto":
            # Auto-select based on availability and recommendation
            for name, config in self.EXPLANATION_MODELS.items():
                if config["recommended"]:
                    logger.info(f"Auto-selected explanation model: {name}")
                    return name
            return "distilgpt2"  # Default fallback
        
        if model_name not in self.EXPLANATION_MODELS:
            available = list(self.EXPLANATION_MODELS.keys())
            raise ValueError(f"Unknown explanation model: {model_name}. Available: {available}")
        
        return model_name
    
    def _load_model(self):
        """Load the explanation model and tokenizer."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("transformers not available, using fallback explanations")
            return
        
        if self.model is not None:
            return  # Already loaded
        
        try:
            config = self.EXPLANATION_MODELS[self.model_name]
            model_name = config["model_name"]
            
            logger.info(f"Loading explanation model: {model_name}")
            
            if config["type"] == "causal":
                # For GPT-style models
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    cache_dir=self.cache_dir
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                
                # Add padding token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Create text generation pipeline
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if torch.cuda.is_available() else -1,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                
            elif config["type"] == "seq2seq":
                # For T5-style models
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir
                )
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                
                # Create text2text generation pipeline
                self.pipeline = pipeline(
                    "text2text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if torch.cuda.is_available() else -1,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            
            logger.info(f"Explanation model loaded successfully: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load explanation model {self.model_name}: {e}")
            self.model = None
            self.tokenizer = None
            self.pipeline = None
    
    def _create_prompt(self, result: DetectionResult, features: AudioFeatures, 
                      language: str) -> str:
        """Create a prompt for the language model."""
        
        # Extract key information
        classification = result.classification.replace("_", " ").lower()
        confidence = result.confidence_score
        duration = features.duration
        model_version = result.model_version
        
        # Determine confidence level
        if confidence > 0.8:
            confidence_level = "high"
        elif confidence > 0.6:
            confidence_level = "moderate"
        else:
            confidence_level = "low"
        
        # Create structured prompt
        if self.EXPLANATION_MODELS[self.model_name]["type"] == "seq2seq":
            # For T5-style models (instruction format)
            prompt = (
                f"Explain why this {language} audio sample was classified as {classification} "
                f"with {confidence_level} confidence ({confidence:.1%}). "
                f"The audio is {duration:.1f} seconds long and was analyzed using {model_version}. "
                f"Provide a clear, technical explanation in 2-3 sentences."
            )
        else:
            # For GPT-style models (completion format)
            prompt = (
                f"Audio Analysis Report:\n"
                f"Language: {language}\n"
                f"Classification: {classification}\n"
                f"Confidence: {confidence:.1%} ({confidence_level})\n"
                f"Duration: {duration:.1f} seconds\n"
                f"Model: {model_version}\n\n"
                f"Explanation: This audio sample was classified as {classification} because"
            )
        
        return prompt
    
    def _post_process_explanation(self, generated_text: str, prompt: str) -> str:
        """Clean and post-process the generated explanation."""
        try:
            # For causal models, remove the prompt from the generated text
            if self.EXPLANATION_MODELS[self.model_name]["type"] == "causal":
                if generated_text.startswith(prompt):
                    explanation = generated_text[len(prompt):].strip()
                else:
                    explanation = generated_text.strip()
            else:
                explanation = generated_text.strip()
            
            # Clean up the explanation
            explanation = explanation.replace("\n", " ").strip()
            
            # Remove incomplete sentences at the end
            sentences = explanation.split(". ")
            if len(sentences) > 1 and not sentences[-1].endswith(('.', '!', '?')):
                explanation = ". ".join(sentences[:-1]) + "."
            
            # Ensure it's not too long
            if len(explanation) > 300:
                explanation = explanation[:297] + "..."
            
            # Ensure it's not too short or empty
            if len(explanation) < 20:
                return self._fallback_explanation(prompt)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error post-processing explanation: {e}")
            return self._fallback_explanation(prompt)
    
    def _fallback_explanation(self, prompt: str) -> str:
        """Generate a fallback explanation when the model fails."""
        # Extract basic info from prompt for fallback
        if "human" in prompt.lower():
            return "The audio exhibits characteristics consistent with natural human speech patterns."
        else:
            return "The audio shows indicators suggesting it may be artificially generated."
    
    def generate_explanation(self, result: DetectionResult, features: AudioFeatures, 
                           language: str) -> str:
        """
        Generate a natural language explanation for the detection result.
        
        Args:
            result: Detection result from the engine
            features: Original audio features
            language: Target language
            
        Returns:
            Natural language explanation string
        """
        start_time = time.time()
        
        try:
            # Load model if not already loaded
            self._load_model()
            
            # If model loading failed, use fallback
            if self.pipeline is None:
                logger.warning("Using fallback explanation - model not available")
                return self._generate_fallback_explanation(result, features, language)
            
            # Create prompt
            prompt = self._create_prompt(result, features, language)
            
            # Generate explanation
            logger.debug(f"Generating explanation with prompt: {prompt[:100]}...")
            
            if self.EXPLANATION_MODELS[self.model_name]["type"] == "seq2seq":
                # T5-style generation
                outputs = self.pipeline(
                    prompt,
                    max_length=150,
                    min_length=30,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1
                )
                generated_text = outputs[0]["generated_text"]
            else:
                # GPT-style generation
                outputs = self.pipeline(
                    prompt,
                    max_new_tokens=100,
                    min_length=len(prompt) + 30,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                generated_text = outputs[0]["generated_text"]
            
            # Post-process the explanation
            explanation = self._post_process_explanation(generated_text, prompt)
            
            generation_time = time.time() - start_time
            logger.info(f"Generated explanation in {generation_time:.3f}s using {self.model_name}")
            
            return explanation
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return self._generate_fallback_explanation(result, features, language)
    
    def _generate_fallback_explanation(self, result: DetectionResult, features: AudioFeatures, 
                                     language: str) -> str:
        """Generate a simple fallback explanation when the model is unavailable."""
        classification = result.classification.replace("_", " ").lower()
        confidence = result.confidence_score
        duration = features.duration
        
        if classification == "human":
            if confidence > 0.8:
                return (
                    f"The {duration:.1f}-second {language} audio sample exhibits strong "
                    f"characteristics of natural human speech with {confidence:.1%} confidence."
                )
            else:
                return (
                    f"The {duration:.1f}-second {language} audio sample appears to be human "
                    f"speech with {confidence:.1%} confidence, though some features are ambiguous."
                )
        else:
            if confidence > 0.8:
                return (
                    f"The {duration:.1f}-second {language} audio sample shows strong indicators "
                    f"of AI-generated speech with {confidence:.1%} confidence."
                )
            else:
                return (
                    f"The {duration:.1f}-second {language} audio sample appears to be "
                    f"AI-generated with {confidence:.1%} confidence, though classification is uncertain."
                )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current explanation model."""
        config = self.EXPLANATION_MODELS[self.model_name]
        
        return {
            "model_name": self.model_name,
            "model_config": config,
            "is_loaded": self.model is not None,
            "cache_dir": self.cache_dir,
            "transformers_available": TRANSFORMERS_AVAILABLE
        }
    
    def clear_cache(self):
        """Clear the model from memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            del self.pipeline
            self.model = None
            self.tokenizer = None
            self.pipeline = None
            
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Explanation model cache cleared")


# Global explanation generator instance
_explanation_generator = None

def get_explanation_generator(model_name: str = "auto") -> ExplanationGenerator:
    """Get a global explanation generator instance."""
    global _explanation_generator
    
    if _explanation_generator is None or _explanation_generator.model_name != model_name:
        _explanation_generator = ExplanationGenerator(model_name)
    
    return _explanation_generator