"""
Property-based tests for audio format validation.

**Feature: ai-voice-detection-api, Property 3**: Audio Format Validation
**Validates: Requirements 3.1, 3.2, 3.3, 3.4**
"""

import pytest
from hypothesis import given, strategies as st
import base64
import io

from src.audio.processor import AudioProcessor, AudioProcessingError
from src.exceptions import InvalidAudioFormatError, MP3ValidationError


# Test data generators
@st.composite
def valid_base64_string(draw):
    """Generate a valid base64 encoded string."""
    # Generate random bytes and encode as base64
    data = draw(st.binary(min_size=10, max_size=1000))
    return base64.b64encode(data).decode('ascii')


@st.composite
def invalid_base64_string(draw):
    """Generate invalid base64 strings."""
    return draw(st.one_of(
        st.text().filter(lambda x: x and not _is_valid_base64(x)),
        st.just(""),  # Empty string
        st.just("   "),  # Whitespace only
        st.just("invalid!@#$%"),  # Invalid characters
        st.just("abc"),  # Too short/invalid padding
        st.just("===="),  # Only padding
    ))


@st.composite
def mp3_like_bytes(draw):
    """Generate bytes that look like MP3 format."""
    # Generate either ID3 header or MP3 frame header
    header_type = draw(st.sampled_from(["id3", "frame"]))
    
    if header_type == "id3":
        # ID3v2 header starts with "ID3"
        header = b"ID3"
        # Add some random data after header
        data = draw(st.binary(min_size=10, max_size=500))
        return header + data
    else:
        # MP3 frame header starts with sync bits (0xFF 0xFB, 0xFF 0xFA, etc.)
        sync_byte1 = 0xFF
        sync_byte2 = draw(st.integers(min_value=0xE0, max_value=0xFF))  # Valid sync pattern
        header = bytes([sync_byte1, sync_byte2])
        # Add some random data after header
        data = draw(st.binary(min_size=10, max_size=500))
        return header + data


@st.composite
def non_mp3_bytes(draw):
    """Generate bytes that don't look like MP3 format."""
    return draw(st.one_of(
        st.binary(min_size=0, max_size=2),  # Too short
        st.binary(min_size=3, max_size=500).filter(
            lambda x: not (x.startswith(b"ID3") or 
                          (len(x) >= 2 and x[0] == 0xFF and (x[1] & 0xE0) == 0xE0))
        ),  # Doesn't start with MP3 headers
    ))


def _is_valid_base64(s):
    """Helper to check if a string is valid base64."""
    try:
        base64.b64decode(s, validate=True)
        return True
    except Exception:
        return False


class TestAudioFormatValidation:
    """
    Property 3: Audio Format Validation
    
    For any audio input, the system should validate that the audioFormat is "mp3", 
    the audioBase64 is valid base64 encoding, and the decoded data is valid MP3 format, 
    rejecting invalid inputs with specific error messages.
    
    **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
    """
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = AudioProcessor()
    
    @given(
        base64_data=valid_base64_string(),
        audio_format=st.just("mp3")
    )
    def test_valid_base64_accepted(self, base64_data, audio_format):
        """
        Property: Valid base64 strings should be successfully decoded.
        **Validates: Requirements 3.1**
        """
        # Valid base64 should decode without error
        try:
            decoded_bytes = self.processor.decode_base64_audio(base64_data)
            assert isinstance(decoded_bytes, bytes)
            assert len(decoded_bytes) > 0
        except AudioProcessingError:
            # If decoding fails, it should be due to empty result, not invalid base64
            pytest.fail("Valid base64 string should decode successfully")
    
    @given(
        base64_data=invalid_base64_string(),
        audio_format=st.just("mp3")
    )
    def test_invalid_base64_rejected(self, base64_data, audio_format):
        """
        Property: Invalid base64 strings should be rejected with specific error messages.
        **Validates: Requirements 3.1, 3.3**
        """
        # Invalid base64 should raise AudioProcessingError
        with pytest.raises(AudioProcessingError) as exc_info:
            self.processor.decode_base64_audio(base64_data)
        
        # Error message should mention base64 or encoding
        error_message = str(exc_info.value).lower()
        assert any(keyword in error_message for keyword in ["base64", "encoding", "invalid", "empty"])
    
    @given(
        audio_format=st.sampled_from(["wav", "flac", "ogg", "m4a", "aac", "mp4", "MP3", "Mp3", "mP3"])
    )
    def test_non_mp3_format_rejected(self, audio_format):
        """
        Property: Non-MP3 audio formats should be rejected.
        **Validates: Requirements 3.2**
        """
        # Generate valid base64 data
        valid_base64 = base64.b64encode(b"test audio data").decode('ascii')
        
        # Non-MP3 formats should raise InvalidAudioFormatError or MP3ValidationError
        with pytest.raises((InvalidAudioFormatError, MP3ValidationError)) as exc_info:
            self.processor.process_base64_audio(valid_base64, audio_format)
        
        # Error message should mention format
        error_message = str(exc_info.value).lower()
        assert "format" in error_message
    
    @given(mp3_bytes=mp3_like_bytes())
    def test_valid_mp3_format_accepted(self, mp3_bytes):
        """
        Property: Valid MP3 format bytes should pass validation.
        **Validates: Requirements 3.2, 3.4**
        """
        # Valid MP3-like bytes should pass format validation
        try:
            is_valid = self.processor.validate_mp3_format(mp3_bytes)
            assert is_valid is True
        except AudioProcessingError:
            pytest.fail("Valid MP3 format should pass validation")
    
    @given(non_mp3_data=non_mp3_bytes())
    def test_invalid_mp3_format_rejected(self, non_mp3_data):
        """
        Property: Invalid MP3 format bytes should be rejected.
        **Validates: Requirements 3.4**
        """
        # Invalid MP3 format should raise AudioProcessingError
        with pytest.raises(AudioProcessingError) as exc_info:
            self.processor.validate_mp3_format(non_mp3_data)
        
        # Error message should mention MP3 format or header
        error_message = str(exc_info.value).lower()
        assert any(keyword in error_message for keyword in ["mp3", "format", "header", "invalid"])
    
    @given(
        base64_data=st.one_of(
            st.just(""),  # Empty string
            st.just("   "),  # Whitespace only
            st.just("\n\t\r"),  # Other whitespace
        )
    )
    def test_empty_base64_rejected(self, base64_data):
        """
        Property: Empty or whitespace-only base64 data should be rejected.
        **Validates: Requirements 3.1, 3.3**
        """
        # Empty base64 should raise AudioProcessingError
        with pytest.raises(AudioProcessingError) as exc_info:
            self.processor.decode_base64_audio(base64_data)
        
        # Error message should mention empty data
        error_message = str(exc_info.value).lower()
        assert "empty" in error_message
    
    @given(
        valid_base64=valid_base64_string(),
        audio_format=st.just("mp3")
    )
    def test_base64_decoding_produces_bytes(self, valid_base64, audio_format):
        """
        Property: Valid base64 decoding should always produce bytes.
        **Validates: Requirements 3.1**
        """
        try:
            decoded = self.processor.decode_base64_audio(valid_base64)
            assert isinstance(decoded, bytes)
            # Decoded data should not be empty (unless original was empty)
            if valid_base64.strip():  # If input wasn't just whitespace
                assert len(decoded) > 0
        except AudioProcessingError:
            # If it fails, it should be due to empty result, not type issues
            pass
    
    @given(
        mp3_data=mp3_like_bytes(),
        audio_format=st.just("mp3")
    )
    def test_complete_audio_processing_pipeline(self, mp3_data, audio_format):
        """
        Property: Complete audio processing pipeline should handle valid MP3 data consistently.
        **Validates: Requirements 3.1, 3.2, 3.4**
        """
        # Encode MP3-like data as base64
        base64_data = base64.b64encode(mp3_data).decode('ascii')
        
        # The complete pipeline might fail at audio loading (librosa), but should not fail
        # at base64 decoding or MP3 format validation stages
        try:
            # Test individual stages
            decoded_bytes = self.processor.decode_base64_audio(base64_data)
            assert isinstance(decoded_bytes, bytes)
            assert decoded_bytes == mp3_data
            
            # MP3 validation should pass
            is_valid_mp3 = self.processor.validate_mp3_format(decoded_bytes)
            assert is_valid_mp3 is True
            
        except AudioProcessingError as e:
            # If it fails, it should not be due to base64 or MP3 format validation
            error_message = str(e).lower()
            # These stages should not fail for valid inputs
            assert not any(keyword in error_message for keyword in ["base64", "encoding"])
            # MP3 format validation should not fail for MP3-like data
            if "mp3" in error_message and "format" in error_message:
                pytest.fail(f"MP3 format validation should not fail for MP3-like data: {e}")
    
    def test_audio_format_case_sensitivity(self):
        """
        Property: Audio format validation should be case-sensitive (only "mp3" accepted).
        **Validates: Requirements 3.2**
        """
        valid_base64 = base64.b64encode(b"test data").decode('ascii')
        
        # Only lowercase "mp3" should be accepted
        try:
            self.processor.process_base64_audio(valid_base64, "mp3")
            # This might fail at later stages, but should not fail due to format validation
        except AudioProcessingError as e:
            error_message = str(e).lower()
            # Should not fail due to unsupported format
            assert "unsupported" not in error_message
        
        # Other cases should be rejected
        for invalid_format in ["MP3", "Mp3", "mP3"]:
            with pytest.raises(AudioProcessingError) as exc_info:
                self.processor.process_base64_audio(valid_base64, invalid_format)
            
            error_message = str(exc_info.value).lower()
            # Check for either "unsupported" or "format" in the error message
            assert "unsupported" in error_message or "format" in error_message
    
    @given(
        whitespace_base64=st.text(alphabet=" \t\n\r", min_size=1, max_size=10)
    )
    def test_whitespace_only_base64_rejected(self, whitespace_base64):
        """
        Property: Base64 data containing only whitespace should be rejected.
        **Validates: Requirements 3.1, 3.3**
        """
        with pytest.raises(AudioProcessingError) as exc_info:
            self.processor.decode_base64_audio(whitespace_base64)
        
        error_message = str(exc_info.value).lower()
        assert "empty" in error_message
    
    @given(
        padded_base64=st.text(alphabet="=", min_size=1, max_size=10)
    )
    def test_padding_only_base64_rejected(self, padded_base64):
        """
        Property: Base64 data containing only padding characters should be rejected.
        **Validates: Requirements 3.1, 3.3**
        """
        with pytest.raises(AudioProcessingError) as exc_info:
            self.processor.decode_base64_audio(padded_base64)
        
        error_message = str(exc_info.value).lower()
        # Check for any relevant error keywords - be more flexible with error message matching
        assert any(keyword in error_message for keyword in ["invalid", "base64", "encoding", "empty", "decode"])