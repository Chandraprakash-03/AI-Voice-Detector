"""
Unit tests for audio processing functionality.

Tests specific examples, edge cases, and error conditions for the AudioProcessor class.
Requirements: 3.1, 3.2, 3.3, 3.4
"""

import pytest
import base64
import io
import numpy as np
from unittest.mock import patch, MagicMock

from src.audio.processor import AudioProcessor, AudioFeatures, AudioProcessingError


class TestAudioProcessor:
    """Unit tests for AudioProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = AudioProcessor()
    
    def test_init_default_sample_rate(self):
        """Test AudioProcessor initialization with default sample rate."""
        processor = AudioProcessor()
        assert processor.target_sample_rate == 22050
    
    def test_init_custom_sample_rate(self):
        """Test AudioProcessor initialization with custom sample rate."""
        processor = AudioProcessor(target_sample_rate=44100)
        assert processor.target_sample_rate == 44100


class TestBase64Decoding:
    """Unit tests for base64 decoding functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = AudioProcessor()
    
    def test_valid_base64_decoding(self):
        """Test successful base64 decoding with valid data."""
        # Test data
        original_data = b"test audio data"
        base64_data = base64.b64encode(original_data).decode('ascii')
        
        # Decode
        result = self.processor.decode_base64_audio(base64_data)
        
        # Verify
        assert isinstance(result, bytes)
        assert result == original_data
    
    def test_empty_base64_string(self):
        """Test base64 decoding with empty string."""
        with pytest.raises(AudioProcessingError) as exc_info:
            self.processor.decode_base64_audio("")
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_whitespace_only_base64(self):
        """Test base64 decoding with whitespace-only string."""
        with pytest.raises(AudioProcessingError) as exc_info:
            self.processor.decode_base64_audio("   \t\n  ")
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_invalid_base64_characters(self):
        """Test base64 decoding with invalid characters."""
        with pytest.raises(AudioProcessingError) as exc_info:
            self.processor.decode_base64_audio("invalid!@#$%")
        
        error_message = str(exc_info.value).lower()
        assert "invalid" in error_message or "base64" in error_message
    
    def test_malformed_base64_padding(self):
        """Test base64 decoding with malformed padding."""
        with pytest.raises(AudioProcessingError) as exc_info:
            self.processor.decode_base64_audio("abc")  # Invalid padding
        
        error_message = str(exc_info.value).lower()
        assert "invalid" in error_message or "base64" in error_message
    
    def test_base64_with_valid_padding(self):
        """Test base64 decoding with proper padding."""
        # Create properly padded base64
        original_data = b"test"
        base64_data = base64.b64encode(original_data).decode('ascii')
        
        result = self.processor.decode_base64_audio(base64_data)
        assert result == original_data


class TestMP3FormatValidation:
    """Unit tests for MP3 format validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = AudioProcessor()
    
    def test_valid_mp3_id3_header(self):
        """Test MP3 validation with ID3 header."""
        # MP3 with ID3v2 header
        mp3_data = b"ID3" + b"some audio data here"
        
        result = self.processor.validate_mp3_format(mp3_data)
        assert result is True
    
    def test_valid_mp3_frame_header(self):
        """Test MP3 validation with frame header."""
        # MP3 with frame sync header
        mp3_data = bytes([0xFF, 0xFB]) + b"audio frame data"
        
        result = self.processor.validate_mp3_format(mp3_data)
        assert result is True
    
    def test_valid_mp3_frame_header_variants(self):
        """Test MP3 validation with different valid frame headers."""
        # Test different valid sync patterns
        valid_headers = [
            bytes([0xFF, 0xFB]),  # MPEG-1 Layer 3
            bytes([0xFF, 0xFA]),  # MPEG-1 Layer 2
            bytes([0xFF, 0xE0]),  # Minimum valid sync
            bytes([0xFF, 0xFF]),  # Maximum valid sync
        ]
        
        for header in valid_headers:
            mp3_data = header + b"audio data"
            result = self.processor.validate_mp3_format(mp3_data)
            assert result is True, f"Header {header.hex()} should be valid"
    
    def test_too_short_data(self):
        """Test MP3 validation with data too short."""
        with pytest.raises(AudioProcessingError) as exc_info:
            self.processor.validate_mp3_format(b"ab")  # Only 2 bytes
        
        assert "too short" in str(exc_info.value).lower()
    
    def test_empty_data(self):
        """Test MP3 validation with empty data."""
        with pytest.raises(AudioProcessingError) as exc_info:
            self.processor.validate_mp3_format(b"")
        
        assert "too short" in str(exc_info.value).lower()
    
    def test_invalid_mp3_header(self):
        """Test MP3 validation with invalid header."""
        # Data that doesn't start with ID3 or valid sync pattern
        invalid_data = b"WAV" + b"not mp3 data"
        
        with pytest.raises(AudioProcessingError) as exc_info:
            self.processor.validate_mp3_format(invalid_data)
        
        error_message = str(exc_info.value).lower()
        assert "invalid" in error_message and "mp3" in error_message
    
    def test_invalid_sync_pattern(self):
        """Test MP3 validation with invalid sync pattern."""
        # First byte is 0xFF but second byte doesn't match sync pattern
        invalid_data = bytes([0xFF, 0x00]) + b"not valid sync"
        
        with pytest.raises(AudioProcessingError) as exc_info:
            self.processor.validate_mp3_format(invalid_data)
        
        error_message = str(exc_info.value).lower()
        assert "invalid" in error_message and "mp3" in error_message


class TestAudioLoading:
    """Unit tests for audio loading functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = AudioProcessor()
    
    @patch('librosa.load')
    def test_successful_audio_loading(self, mock_librosa_load):
        """Test successful audio loading with librosa."""
        # Mock librosa.load return values
        mock_audio_data = np.array([0.1, 0.2, 0.3, 0.4])
        mock_sample_rate = 22050
        mock_librosa_load.return_value = (mock_audio_data, mock_sample_rate)
        
        # Test data
        audio_bytes = b"fake mp3 data"
        
        # Load audio
        audio_data, sample_rate = self.processor.load_audio_from_bytes(audio_bytes)
        
        # Verify
        assert np.array_equal(audio_data, mock_audio_data)
        assert sample_rate == mock_sample_rate
        
        # Verify librosa was called correctly
        mock_librosa_load.assert_called_once()
        call_args = mock_librosa_load.call_args
        assert call_args[1]['sr'] == 22050  # target sample rate
        assert call_args[1]['mono'] is True
    
    @patch('librosa.load')
    def test_audio_loading_with_custom_sample_rate(self, mock_librosa_load):
        """Test audio loading with custom sample rate."""
        # Custom processor with different sample rate
        processor = AudioProcessor(target_sample_rate=44100)
        
        # Mock librosa.load
        mock_librosa_load.return_value = (np.array([0.1, 0.2]), 44100)
        
        # Load audio
        audio_bytes = b"fake mp3 data"
        audio_data, sample_rate = processor.load_audio_from_bytes(audio_bytes)
        
        # Verify sample rate parameter
        call_args = mock_librosa_load.call_args
        assert call_args[1]['sr'] == 44100
    
    @patch('librosa.load')
    def test_empty_audio_data_error(self, mock_librosa_load):
        """Test error handling when librosa returns empty audio data."""
        # Mock librosa to return empty array
        mock_librosa_load.return_value = (np.array([]), 22050)
        
        audio_bytes = b"fake mp3 data"
        
        with pytest.raises(AudioProcessingError) as exc_info:
            self.processor.load_audio_from_bytes(audio_bytes)
        
        assert "empty" in str(exc_info.value).lower()
    
    @patch('librosa.load')
    def test_librosa_loading_error(self, mock_librosa_load):
        """Test error handling when librosa fails to load audio."""
        # Mock librosa to raise an exception
        mock_librosa_load.side_effect = Exception("Librosa loading failed")
        
        audio_bytes = b"invalid audio data"
        
        with pytest.raises(AudioProcessingError) as exc_info:
            self.processor.load_audio_from_bytes(audio_bytes)
        
        error_message = str(exc_info.value).lower()
        assert "failed to load" in error_message


class TestFeatureExtraction:
    """Unit tests for audio feature extraction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = AudioProcessor()
        # Create sample audio data
        self.sample_audio = np.array([0.1, 0.2, -0.1, 0.3, -0.2, 0.4] * 1000)  # Longer for realistic features
        self.sample_rate = 22050
    
    @patch('librosa.feature.mfcc')
    def test_mfcc_feature_extraction(self, mock_mfcc):
        """Test MFCC feature extraction."""
        # Mock MFCC return value (13 x N matrix)
        mock_mfcc_features = np.random.rand(13, 100)
        mock_mfcc.return_value = mock_mfcc_features
        
        # Extract MFCC features
        result = self.processor.extract_mfcc_features(self.sample_audio, self.sample_rate)
        
        # Verify
        assert np.array_equal(result, mock_mfcc_features)
        
        # Verify librosa.feature.mfcc was called with correct parameters
        mock_mfcc.assert_called_once()
        call_args = mock_mfcc.call_args
        assert np.array_equal(call_args[1]['y'], self.sample_audio)
        assert call_args[1]['sr'] == self.sample_rate
        assert call_args[1]['n_mfcc'] == 13
        assert call_args[1]['n_fft'] == 2048
        assert call_args[1]['hop_length'] == 512
    
    @patch('librosa.feature.mfcc')
    def test_mfcc_extraction_error(self, mock_mfcc):
        """Test error handling in MFCC extraction."""
        # Mock MFCC to raise an exception
        mock_mfcc.side_effect = Exception("MFCC extraction failed")
        
        with pytest.raises(AudioProcessingError) as exc_info:
            self.processor.extract_mfcc_features(self.sample_audio, self.sample_rate)
        
        error_message = str(exc_info.value).lower()
        assert "failed to extract mfcc" in error_message
    
    @patch('librosa.feature.spectral_centroid')
    @patch('librosa.feature.spectral_rolloff')
    @patch('librosa.feature.spectral_bandwidth')
    @patch('librosa.feature.zero_crossing_rate')
    def test_spectral_feature_extraction(self, mock_zcr, mock_bandwidth, mock_rolloff, mock_centroid):
        """Test spectral feature extraction."""
        # Mock return values
        mock_centroid.return_value = np.array([[1.0, 2.0, 3.0]])
        mock_rolloff.return_value = np.array([[4.0, 5.0, 6.0]])
        mock_bandwidth.return_value = np.array([[7.0, 8.0, 9.0]])
        mock_zcr.return_value = np.array([[0.1, 0.2, 0.3]])
        
        # Extract spectral features
        result = self.processor.extract_spectral_features(self.sample_audio, self.sample_rate)
        
        # Verify shape and content
        assert result.shape == (4, 3)  # 4 features x 3 time frames
        
        # Verify all functions were called
        mock_centroid.assert_called_once()
        mock_rolloff.assert_called_once()
        mock_bandwidth.assert_called_once()
        mock_zcr.assert_called_once()
    
    @patch('librosa.feature.rms')
    @patch('librosa.beat.beat_track')
    def test_temporal_feature_extraction(self, mock_beat_track, mock_rms):
        """Test temporal feature extraction."""
        # Mock return values
        mock_rms.return_value = np.array([[0.1, 0.2, 0.15, 0.3]])
        mock_beat_track.return_value = (120.0, np.array([0, 100, 200]))  # tempo, beats
        
        # Extract temporal features
        result = self.processor.extract_temporal_features(self.sample_audio, self.sample_rate)
        
        # Verify
        assert len(result) == 4  # tempo, mean_energy, std_energy, max_energy
        assert result[0] == 120.0  # tempo
        assert result[1] == np.mean([0.1, 0.2, 0.15, 0.3])  # mean energy
        assert result[2] == np.std([0.1, 0.2, 0.15, 0.3])  # std energy
        assert result[3] == 0.3  # max energy
    
    def test_audio_preprocessing_and_normalization(self):
        """Test audio preprocessing and normalization."""
        # Create test audio with DC offset and varying amplitude
        test_audio = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) + 0.5  # Add DC offset
        
        # Preprocess
        result = self.processor.preprocess_and_normalize(test_audio)
        
        # Verify DC removal (mean should be close to 0)
        assert abs(np.mean(result)) < 1e-10
        
        # Verify normalization (max absolute value should be 1.0)
        assert abs(np.max(np.abs(result)) - 1.0) < 1e-10
    
    def test_preprocessing_zero_audio(self):
        """Test preprocessing with all-zero audio."""
        zero_audio = np.zeros(100)
        
        result = self.processor.preprocess_and_normalize(zero_audio)
        
        # Should remain all zeros
        assert np.allclose(result, zero_audio)
    
    @patch.object(AudioProcessor, 'extract_mfcc_features')
    @patch.object(AudioProcessor, 'extract_spectral_features')
    @patch.object(AudioProcessor, 'extract_temporal_features')
    @patch.object(AudioProcessor, 'preprocess_and_normalize')
    def test_complete_feature_extraction(self, mock_preprocess, mock_temporal, mock_spectral, mock_mfcc):
        """Test complete feature extraction pipeline."""
        # Mock preprocessing
        mock_preprocess.return_value = self.sample_audio
        
        # Mock feature extraction methods
        mock_mfcc.return_value = np.random.rand(13, 100)
        mock_spectral.return_value = np.random.rand(4, 100)
        mock_temporal.return_value = np.array([120.0, 0.2, 0.1, 0.3])
        
        # Extract features
        result = self.processor.extract_features(self.sample_audio, self.sample_rate)
        
        # Verify result type and structure
        assert isinstance(result, AudioFeatures)
        assert isinstance(result.mfcc, list)
        assert isinstance(result.spectral, list)
        assert isinstance(result.temporal, list)
        assert isinstance(result.duration, float)
        assert isinstance(result.sample_rate, int)
        
        # Verify duration calculation
        expected_duration = len(self.sample_audio) / self.sample_rate
        assert abs(result.duration - expected_duration) < 1e-6
        
        # Verify all methods were called
        mock_preprocess.assert_called_once()
        mock_mfcc.assert_called_once()
        mock_spectral.assert_called_once()
        mock_temporal.assert_called_once()


class TestCompleteProcessingPipeline:
    """Unit tests for the complete audio processing pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = AudioProcessor()
    
    def test_unsupported_audio_format(self):
        """Test processing with unsupported audio format."""
        base64_data = base64.b64encode(b"test data").decode('ascii')
        
        with pytest.raises(AudioProcessingError) as exc_info:
            self.processor.process_base64_audio(base64_data, "wav")
        
        error_message = str(exc_info.value).lower()
        assert "unsupported" in error_message and "format" in error_message
    
    def test_case_sensitive_format_validation(self):
        """Test that audio format validation is case-sensitive."""
        base64_data = base64.b64encode(b"test data").decode('ascii')
        
        # Test various case combinations
        invalid_formats = ["MP3", "Mp3", "mP3"]
        
        for format_variant in invalid_formats:
            with pytest.raises(AudioProcessingError) as exc_info:
                self.processor.process_base64_audio(base64_data, format_variant)
            
            error_message = str(exc_info.value).lower()
            # The error could be either "unsupported" (format check) or "invalid mp3 format" (validation check)
            assert "unsupported" in error_message or "invalid mp3 format" in error_message
    
    @patch.object(AudioProcessor, 'decode_base64_audio')
    @patch.object(AudioProcessor, 'validate_mp3_format')
    @patch.object(AudioProcessor, 'load_audio_from_bytes')
    @patch.object(AudioProcessor, 'extract_features')
    def test_successful_complete_pipeline(self, mock_extract, mock_load, mock_validate, mock_decode):
        """Test successful execution of complete processing pipeline."""
        # Mock all pipeline stages
        mock_decode.return_value = b"fake mp3 data"
        mock_validate.return_value = True
        mock_load.return_value = (np.array([0.1, 0.2, 0.3]), 22050)
        mock_extract.return_value = AudioFeatures(
            mfcc=[[1.0, 2.0], [3.0, 4.0]],
            spectral=[[0.1, 0.2], [0.3, 0.4]],
            temporal=[120.0, 0.2, 0.1, 0.3],
            duration=1.5,
            sample_rate=22050
        )
        
        # Process audio
        base64_data = base64.b64encode(b"test mp3 data").decode('ascii')
        result = self.processor.process_base64_audio(base64_data, "mp3")
        
        # Verify result
        assert isinstance(result, AudioFeatures)
        
        # Verify all pipeline stages were called in order
        mock_decode.assert_called_once_with(base64_data)
        mock_validate.assert_called_once_with(b"fake mp3 data")
        mock_load.assert_called_once_with(b"fake mp3 data")
        mock_extract.assert_called_once()
    
    @patch.object(AudioProcessor, 'decode_base64_audio')
    def test_pipeline_failure_at_base64_stage(self, mock_decode):
        """Test pipeline failure at base64 decoding stage."""
        # Mock base64 decoding to fail
        mock_decode.side_effect = AudioProcessingError("Base64 decoding failed")
        
        base64_data = "invalid_base64"
        
        with pytest.raises(AudioProcessingError) as exc_info:
            self.processor.process_base64_audio(base64_data, "mp3")
        
        assert "Base64 decoding failed" in str(exc_info.value)
    
    @patch.object(AudioProcessor, 'decode_base64_audio')
    @patch.object(AudioProcessor, 'validate_mp3_format')
    def test_pipeline_failure_at_validation_stage(self, mock_validate, mock_decode):
        """Test pipeline failure at MP3 validation stage."""
        # Mock successful decoding but failed validation
        mock_decode.return_value = b"not mp3 data"
        mock_validate.side_effect = AudioProcessingError("Invalid MP3 format")
        
        base64_data = base64.b64encode(b"not mp3 data").decode('ascii')
        
        with pytest.raises(AudioProcessingError) as exc_info:
            self.processor.process_base64_audio(base64_data, "mp3")
        
        assert "Invalid MP3 format" in str(exc_info.value)


class TestAudioFeatures:
    """Unit tests for AudioFeatures data model."""
    
    def test_audio_features_creation(self):
        """Test AudioFeatures model creation with valid data."""
        features = AudioFeatures(
            mfcc=[[1.0, 2.0], [3.0, 4.0]],
            spectral=[[0.1, 0.2], [0.3, 0.4]],
            temporal=[120.0, 0.2, 0.1, 0.3],
            duration=1.5,
            sample_rate=22050
        )
        
        assert features.mfcc == [[1.0, 2.0], [3.0, 4.0]]
        assert features.spectral == [[0.1, 0.2], [0.3, 0.4]]
        assert features.temporal == [120.0, 0.2, 0.1, 0.3]
        assert features.duration == 1.5
        assert features.sample_rate == 22050
    
    def test_audio_features_serialization(self):
        """Test AudioFeatures model serialization to dict."""
        features = AudioFeatures(
            mfcc=[[1.0, 2.0]],
            spectral=[[0.1, 0.2]],
            temporal=[120.0],
            duration=1.0,
            sample_rate=22050
        )
        
        # Test dict conversion
        features_dict = features.model_dump()
        
        assert "mfcc" in features_dict
        assert "spectral" in features_dict
        assert "temporal" in features_dict
        assert "duration" in features_dict
        assert "sample_rate" in features_dict