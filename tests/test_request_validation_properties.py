"""
Property-based tests for request validation completeness.

This module contains property-based tests that validate the request validation
behavior of the AI-Generated Voice Detection API models.
"""

import pytest
from hypothesis import given, strategies as st
from pydantic import ValidationError
import base64
import json

from src.models.schemas import VoiceDetectionRequest


# Test data generators
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
UNSUPPORTED_LANGUAGES = ["French", "German", "Spanish", "Chinese", "Japanese", "Korean", "Arabic"]
VALID_AUDIO_FORMATS = ["mp3"]
INVALID_AUDIO_FORMATS = ["wav", "flac", "ogg", "m4a", "aac", "mp4"]

# Generate valid base64 strings
@st.composite
def valid_base64_string(draw):
    """Generate a valid base64 encoded string."""
    # Generate random bytes and encode as base64
    data = draw(st.binary(min_size=10, max_size=1000))
    return base64.b64encode(data).decode('ascii')

# Generate invalid base64 strings
@st.composite  
def invalid_base64_string(draw):
    """Generate invalid base64 strings."""
    return draw(st.one_of(
        st.text().filter(lambda x: x and not _is_valid_base64(x)),
        st.just(""),  # Empty string
        st.just("   "),  # Whitespace only
        st.just("invalid!@#$%"),  # Invalid characters
        st.just("abc"),  # Too short/invalid padding
    ))

def _is_valid_base64(s):
    """Helper to check if a string is valid base64."""
    try:
        base64.b64decode(s, validate=True)
        return True
    except Exception:
        return False


class TestRequestValidationCompleteness:
    """
    Property 1: Request Validation Completeness
    
    For any HTTP request to the voice detection endpoint, the system should 
    validate all required fields (language, audioFormat, audioBase64) and 
    return descriptive error messages for any missing or invalid fields.
    
    **Validates: Requirements 1.2, 1.4, 8.2, 8.3**
    """
    
    @given(
        language=st.sampled_from(SUPPORTED_LANGUAGES),
        audio_format=st.sampled_from(VALID_AUDIO_FORMATS),
        audio_base64=valid_base64_string()
    )
    def test_valid_requests_are_accepted(self, language, audio_format, audio_base64):
        """
        Property: Valid requests with all required fields should be accepted.
        **Validates: Requirements 1.2, 1.4**
        """
        request_data = {
            "language": language,
            "audioFormat": audio_format,
            "audioBase64": audio_base64
        }
        
        # Valid requests should not raise ValidationError
        request = VoiceDetectionRequest(**request_data)
        
        # Verify the fields are set correctly
        assert request.language == language
        assert request.audioFormat == audio_format
        assert request.audioBase64 == audio_base64
    
    @given(
        language=st.one_of(st.none(), st.sampled_from(UNSUPPORTED_LANGUAGES)),
        audio_format=st.sampled_from(VALID_AUDIO_FORMATS),
        audio_base64=valid_base64_string()
    )
    def test_invalid_language_rejected(self, language, audio_format, audio_base64):
        """
        Property: Requests with invalid or missing language should be rejected.
        **Validates: Requirements 1.2, 8.2, 8.3**
        """
        request_data = {
            "audioFormat": audio_format,
            "audioBase64": audio_base64
        }
        
        if language is not None:
            request_data["language"] = language
        
        # Invalid language should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            VoiceDetectionRequest(**request_data)
        
        # Verify error contains descriptive message
        error_details = exc_info.value.errors()
        assert len(error_details) > 0
        
        # Check that the error is related to language field
        language_errors = [e for e in error_details if e['loc'] == ('language',)]
        assert len(language_errors) > 0
    
    @given(
        language=st.sampled_from(SUPPORTED_LANGUAGES),
        audio_format=st.one_of(st.none(), st.sampled_from(INVALID_AUDIO_FORMATS)),
        audio_base64=valid_base64_string()
    )
    def test_invalid_audio_format_rejected(self, language, audio_format, audio_base64):
        """
        Property: Requests with invalid or missing audioFormat should be rejected.
        **Validates: Requirements 1.2, 8.2, 8.3**
        """
        request_data = {
            "language": language,
            "audioBase64": audio_base64
        }
        
        if audio_format is not None:
            request_data["audioFormat"] = audio_format
        
        # Invalid audio format should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            VoiceDetectionRequest(**request_data)
        
        # Verify error contains descriptive message
        error_details = exc_info.value.errors()
        assert len(error_details) > 0
        
        # Check that the error is related to audioFormat field
        format_errors = [e for e in error_details if e['loc'] == ('audioFormat',)]
        assert len(format_errors) > 0
    
    @given(
        language=st.sampled_from(SUPPORTED_LANGUAGES),
        audio_format=st.sampled_from(VALID_AUDIO_FORMATS),
        audio_base64=st.one_of(st.none(), invalid_base64_string())
    )
    def test_invalid_audio_base64_rejected(self, language, audio_format, audio_base64):
        """
        Property: Requests with invalid or missing audioBase64 should be rejected.
        **Validates: Requirements 1.4, 8.2, 8.3**
        """
        request_data = {
            "language": language,
            "audioFormat": audio_format
        }
        
        if audio_base64 is not None:
            request_data["audioBase64"] = audio_base64
        
        # Invalid base64 should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            VoiceDetectionRequest(**request_data)
        
        # Verify error contains descriptive message
        error_details = exc_info.value.errors()
        assert len(error_details) > 0
        
        # Check that the error is related to audioBase64 field
        base64_errors = [e for e in error_details if e['loc'] == ('audioBase64',)]
        assert len(base64_errors) > 0
    
    @given(
        missing_fields=st.lists(
            st.sampled_from(["language", "audioFormat", "audioBase64"]),
            min_size=1,
            max_size=3,
            unique=True
        )
    )
    def test_missing_required_fields_rejected(self, missing_fields):
        """
        Property: Requests missing any required fields should be rejected with descriptive errors.
        **Validates: Requirements 1.2, 8.2, 8.3**
        """
        # Start with a complete valid request
        complete_request = {
            "language": "English",
            "audioFormat": "mp3",
            "audioBase64": base64.b64encode(b"test audio data").decode('ascii')
        }
        
        # Remove the specified fields
        incomplete_request = {k: v for k, v in complete_request.items() if k not in missing_fields}
        
        # Missing required fields should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            VoiceDetectionRequest(**incomplete_request)
        
        # Verify error contains information about missing fields
        error_details = exc_info.value.errors()
        assert len(error_details) >= len(missing_fields)
        
        # Check that errors are reported for the missing fields
        error_fields = {tuple(e['loc']) for e in error_details}
        for field in missing_fields:
            assert (field,) in error_fields
    
    @given(
        extra_fields=st.dictionaries(
            st.text().filter(lambda x: x not in ["language", "audioFormat", "audioBase64"]),
            st.text(),
            min_size=1,
            max_size=3
        )
    )
    def test_extra_fields_ignored_or_rejected(self, extra_fields):
        """
        Property: Requests with extra fields should be handled consistently.
        **Validates: Requirements 1.2**
        """
        # Create a valid base request
        base_request = {
            "language": "English",
            "audioFormat": "mp3",
            "audioBase64": base64.b64encode(b"test audio data").decode('ascii')
        }
        
        # Add extra fields
        request_with_extras = {**base_request, **extra_fields}
        
        # Pydantic should either ignore extra fields or handle them consistently
        # In this case, we expect it to ignore extra fields (default behavior)
        request = VoiceDetectionRequest(**request_with_extras)
        
        # Verify the core fields are still correct
        assert request.language == "English"
        assert request.audioFormat == "mp3"
        assert request.audioBase64 == base_request["audioBase64"]


class TestLanguageSupportValidation:
    """
    Property 4: Language Support Validation
    
    For any language field value, the system should accept only the five 
    supported languages (Tamil, English, Hindi, Malayalam, Telugu) and 
    reject all other values with error responses.
    
    **Validates: Requirements 4.1, 4.2**
    """
    
    @given(language=st.sampled_from(SUPPORTED_LANGUAGES))
    def test_supported_languages_accepted(self, language):
        """
        Property: All supported languages should be accepted.
        **Validates: Requirements 4.1**
        """
        request_data = {
            "language": language,
            "audioFormat": "mp3",
            "audioBase64": base64.b64encode(b"test audio data").decode('ascii')
        }
        
        # Supported languages should not raise ValidationError
        request = VoiceDetectionRequest(**request_data)
        assert request.language == language
    
    @given(language=st.sampled_from(UNSUPPORTED_LANGUAGES))
    def test_unsupported_languages_rejected(self, language):
        """
        Property: All unsupported languages should be rejected.
        **Validates: Requirements 4.2**
        """
        request_data = {
            "language": language,
            "audioFormat": "mp3",
            "audioBase64": base64.b64encode(b"test audio data").decode('ascii')
        }
        
        # Unsupported languages should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            VoiceDetectionRequest(**request_data)
        
        # Verify error is related to language field
        error_details = exc_info.value.errors()
        language_errors = [e for e in error_details if e['loc'] == ('language',)]
        assert len(language_errors) > 0
        
        # Verify error message mentions the invalid language or contains validation info
        error_messages = [e['msg'] for e in language_errors]
        # Just verify that there's an error message - don't be too strict about content
        assert all(msg for msg in error_messages)  # Ensure messages are not empty
    
    @given(
        language=st.text().filter(
            lambda x: x not in SUPPORTED_LANGUAGES and x not in UNSUPPORTED_LANGUAGES and x.strip()
        )
    )
    def test_arbitrary_languages_rejected(self, language):
        """
        Property: Any arbitrary language string not in supported list should be rejected.
        **Validates: Requirements 4.1, 4.2**
        """
        request_data = {
            "language": language,
            "audioFormat": "mp3",
            "audioBase64": base64.b64encode(b"test audio data").decode('ascii')
        }
        
        # Arbitrary languages should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            VoiceDetectionRequest(**request_data)
        
        # Verify error is related to language field
        error_details = exc_info.value.errors()
        language_errors = [e for e in error_details if e['loc'] == ('language',)]
        assert len(language_errors) > 0
    
    @given(
        language=st.one_of(
            st.integers(),
            st.floats(),
            st.booleans(),
            st.lists(st.text()),
            st.dictionaries(st.text(), st.text())
        )
    )
    def test_non_string_languages_rejected(self, language):
        """
        Property: Non-string language values should be rejected.
        **Validates: Requirements 4.1, 4.2**
        """
        request_data = {
            "language": language,
            "audioFormat": "mp3",
            "audioBase64": base64.b64encode(b"test audio data").decode('ascii')
        }
        
        # Non-string languages should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            VoiceDetectionRequest(**request_data)
        
        # Verify error is related to language field
        error_details = exc_info.value.errors()
        language_errors = [e for e in error_details if e['loc'] == ('language',)]
        assert len(language_errors) > 0
    
    def test_exactly_five_languages_supported(self):
        """
        Property: The system should support exactly five languages.
        **Validates: Requirements 4.1**
        """
        # This is a deterministic test to ensure we maintain exactly 5 languages
        assert len(SUPPORTED_LANGUAGES) == 5
        assert set(SUPPORTED_LANGUAGES) == {"Tamil", "English", "Hindi", "Malayalam", "Telugu"}
        
        # Verify each language works individually
        for language in SUPPORTED_LANGUAGES:
            request_data = {
                "language": language,
                "audioFormat": "mp3",
                "audioBase64": base64.b64encode(b"test audio data").decode('ascii')
            }
            request = VoiceDetectionRequest(**request_data)
            assert request.language == language
    
    @given(
        language_case_variant=st.sampled_from([
            "tamil", "TAMIL", "TaMiL",
            "english", "ENGLISH", "EnGlIsH", 
            "hindi", "HINDI", "HiNdI",
            "malayalam", "MALAYALAM", "MaLaYaLaM",
            "telugu", "TELUGU", "TeLuGu"
        ])
    )
    def test_case_sensitive_language_validation(self, language_case_variant):
        """
        Property: Language validation should be case-sensitive.
        **Validates: Requirements 4.1, 4.2**
        """
        request_data = {
            "language": language_case_variant,
            "audioFormat": "mp3",
            "audioBase64": base64.b64encode(b"test audio data").decode('ascii')
        }
        
        # Case variants should be rejected (only exact case matches accepted)
        with pytest.raises(ValidationError) as exc_info:
            VoiceDetectionRequest(**request_data)
        
        # Verify error is related to language field
        error_details = exc_info.value.errors()
        language_errors = [e for e in error_details if e['loc'] == ('language',)]
        assert len(language_errors) > 0