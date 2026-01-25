"""
Audio processing module for the AI-Generated Voice Detection API.

This module handles base64 decoding, MP3 validation, and audio feature extraction
for voice detection analysis.
"""

import base64
import io
import logging
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
from pydantic import BaseModel

from src.exceptions import (
    AudioProcessingError,
    Base64DecodingError,
    MP3ValidationError,
    FeatureExtractionError,
    InvalidAudioFormatError
)

logger = logging.getLogger(__name__)


class AudioFeatures(BaseModel):
    """Data model for extracted audio features."""
    
    mfcc: List[List[float]]  # 13 x N matrix (13 MFCC coefficients over N frames)
    spectral: List[List[float]]  # Spectral features matrix
    temporal: List[float]  # Temporal characteristics vector
    duration: float  # Audio duration in seconds
    sample_rate: int  # Audio sample rate


class AudioProcessor:
    """
    Handles audio processing operations including base64 decoding,
    MP3 validation, and feature extraction.
    """
    
    def __init__(self, settings=None):
        """
        Initialize the AudioProcessor.
        
        Args:
            settings: Application settings object containing configuration
        """
        from src.config import get_settings
        
        if settings is None:
            settings = get_settings()
        
        self.settings = settings
        self.target_sample_rate = settings.ml.sample_rate
        self.n_mfcc = settings.ml.n_mfcc
        self.hop_length = settings.ml.hop_length
        self.n_fft = settings.ml.n_fft
        self.max_duration = settings.ml.max_audio_duration
        self.min_duration = settings.ml.min_audio_duration
        
    def decode_base64_audio(self, base64_data: str) -> bytes:
        """
        Decode base64-encoded audio data.
        
        Args:
            base64_data: Base64-encoded audio string
            
        Returns:
            Decoded audio bytes
            
        Raises:
            Base64DecodingError: If base64 decoding fails
        """
        try:
            # Remove any whitespace and validate base64 format
            cleaned_data = base64_data.strip()
            if not cleaned_data:
                raise Base64DecodingError("Base64 audio data is empty")
                
            # Decode base64 data
            audio_bytes = base64.b64decode(cleaned_data)
            
            if len(audio_bytes) == 0:
                raise Base64DecodingError("Decoded audio data is empty")
                
            return audio_bytes
            
        except Base64DecodingError:
            raise
        except Exception as e:
            logger.error(f"Base64 decoding failed: {str(e)}")
            raise Base64DecodingError(f"Invalid base64 encoding: {str(e)}")
    
    def validate_mp3_format(self, audio_bytes: bytes) -> bool:
        """
        Validate that the audio data is in MP3 format.
        
        Args:
            audio_bytes: Raw audio bytes
            
        Returns:
            True if valid MP3 format
            
        Raises:
            MP3ValidationError: If MP3 validation fails
        """
        try:
            # Check MP3 header signature
            if len(audio_bytes) < 3:
                raise MP3ValidationError("Audio data too short to be valid MP3")
                
            # MP3 files typically start with ID3 tag or MP3 frame header
            # ID3v2 starts with "ID3"
            # MP3 frame header starts with sync bits (0xFF 0xFB, 0xFF 0xFA, etc.)
            if audio_bytes[:3] == b'ID3':
                return True
            elif len(audio_bytes) >= 2 and audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0:
                return True
            else:
                raise MP3ValidationError("Invalid MP3 format: missing MP3 header signature")
                
        except MP3ValidationError:
            raise
        except Exception as e:
            logger.error(f"MP3 validation failed: {str(e)}")
            raise MP3ValidationError(f"MP3 format validation error: {str(e)}")
    
    def load_audio_from_bytes(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """
        Load audio data from bytes using librosa.
        
        Args:
            audio_bytes: Raw audio bytes
            
        Returns:
            Tuple of (audio_data, sample_rate)
            
        Raises:
            AudioProcessingError: If audio loading fails
        """
        try:
            # Create a BytesIO object from the audio bytes
            audio_buffer = io.BytesIO(audio_bytes)
            
            # Load audio using librosa
            audio_data, sample_rate = librosa.load(
                audio_buffer, 
                sr=self.target_sample_rate,
                mono=True  # Convert to mono
            )
            
            if len(audio_data) == 0:
                raise AudioProcessingError("Loaded audio data is empty")
                
            return audio_data, sample_rate
            
        except Exception as e:
            if isinstance(e, AudioProcessingError):
                raise
            logger.error(f"Audio loading failed: {str(e)}")
            raise AudioProcessingError(f"Failed to load audio data: {str(e)}")
    
    def extract_mfcc_features(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract MFCC (Mel-frequency cepstral coefficients) features.
        
        Args:
            audio_data: Audio time series
            sample_rate: Sample rate of audio
            
        Returns:
            MFCC feature matrix (13 x N)
        """
        try:
            # Extract 13 MFCC coefficients
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=sample_rate,
                n_mfcc=13,
                n_fft=2048,
                hop_length=512
            )
            
            return mfcc
            
        except Exception as e:
            logger.error(f"MFCC extraction failed: {str(e)}")
            raise FeatureExtractionError(f"Failed to extract MFCC features: {str(e)}", feature_type="mfcc")
    
    def extract_spectral_features(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract spectral features from audio.
        
        Args:
            audio_data: Audio time series
            sample_rate: Sample rate of audio
            
        Returns:
            Spectral feature matrix
        """
        try:
            # Extract various spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
            
            # Combine spectral features
            spectral_features = np.vstack([
                spectral_centroids,
                spectral_rolloff,
                spectral_bandwidth,
                zero_crossing_rate
            ])
            
            return spectral_features
            
        except Exception as e:
            logger.error(f"Spectral feature extraction failed: {str(e)}")
            raise FeatureExtractionError(f"Failed to extract spectral features: {str(e)}", feature_type="spectral")
    
    def extract_temporal_features(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Extract temporal characteristics from audio.
        
        Args:
            audio_data: Audio time series
            sample_rate: Sample rate of audio
            
        Returns:
            Temporal feature vector
        """
        try:
            # Calculate temporal features
            rms_energy = librosa.feature.rms(y=audio_data)
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
            
            # Statistical measures
            mean_energy = np.mean(rms_energy)
            std_energy = np.std(rms_energy)
            max_energy = np.max(rms_energy)
            
            # Combine temporal features
            temporal_features = np.array([
                tempo,
                mean_energy,
                std_energy,
                max_energy
            ])
            
            return temporal_features
            
        except Exception as e:
            logger.error(f"Temporal feature extraction failed: {str(e)}")
            raise FeatureExtractionError(f"Failed to extract temporal features: {str(e)}", feature_type="temporal")
    
    def preprocess_and_normalize(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess and normalize audio data.
        
        Args:
            audio_data: Raw audio time series
            
        Returns:
            Preprocessed and normalized audio data
        """
        try:
            # Remove DC offset
            audio_data = audio_data - np.mean(audio_data)
            
            # Normalize to [-1, 1] range
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {str(e)}")
            raise AudioProcessingError(f"Failed to preprocess audio: {str(e)}")
    
    def extract_features(self, audio_data: np.ndarray, sample_rate: int) -> AudioFeatures:
        """
        Extract comprehensive audio features for ML analysis.
        
        Args:
            audio_data: Audio time series
            sample_rate: Sample rate of audio
            
        Returns:
            AudioFeatures object containing all extracted features
        """
        try:
            # Preprocess audio
            processed_audio = self.preprocess_and_normalize(audio_data)
            
            # Extract different types of features
            mfcc_features = self.extract_mfcc_features(processed_audio, sample_rate)
            spectral_features = self.extract_spectral_features(processed_audio, sample_rate)
            temporal_features = self.extract_temporal_features(processed_audio, sample_rate)
            
            # Calculate duration
            duration = len(audio_data) / sample_rate
            
            # Convert numpy arrays to lists for Pydantic model
            return AudioFeatures(
                mfcc=mfcc_features.tolist(),
                spectral=spectral_features.tolist(),
                temporal=temporal_features.tolist(),
                duration=duration,
                sample_rate=sample_rate
            )
            
        except Exception as e:
            if isinstance(e, AudioProcessingError):
                raise
            logger.error(f"Feature extraction failed: {str(e)}")
            raise AudioProcessingError(f"Failed to extract audio features: {str(e)}")
    
    def process_base64_audio(self, base64_data: str, audio_format: str) -> AudioFeatures:
        """
        Complete audio processing pipeline from base64 input to features.
        
        Args:
            base64_data: Base64-encoded audio string
            audio_format: Expected audio format (must be "mp3")
            
        Returns:
            AudioFeatures object containing extracted features
            
        Raises:
            InvalidAudioFormatError: If audio format is not supported
            Base64DecodingError: If base64 decoding fails
            MP3ValidationError: If MP3 validation fails
            AudioProcessingError: If any other processing step fails
        """
        try:
            # Validate audio format
            if audio_format.lower() != "mp3":
                raise InvalidAudioFormatError(audio_format)
            
            # Decode base64 data
            audio_bytes = self.decode_base64_audio(base64_data)
            
            # Validate MP3 format
            self.validate_mp3_format(audio_bytes)
            
            # Load audio data
            audio_data, sample_rate = self.load_audio_from_bytes(audio_bytes)
            
            # Extract features
            features = self.extract_features(audio_data, sample_rate)
            
            return features
            
        except (InvalidAudioFormatError, Base64DecodingError, MP3ValidationError, FeatureExtractionError):
            raise
        except Exception as e:
            logger.error(f"Audio processing pipeline failed: {str(e)}")
            raise AudioProcessingError(f"Audio processing failed: {str(e)}")