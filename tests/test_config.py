"""
Tests for configuration module.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import (
    Config, Platform, CaptionStyle, WhisperModel,
    PlatformPreset, TranscriptionConfig, HighlightConfig,
    ClippingConfig, CaptionConfig
)


class TestConfig:
    """Test cases for Config class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Config()

        assert config.platform == Platform.YOUTUBE_SHORTS
        assert config.transcription.model == WhisperModel.BASE
        assert config.highlight.target_clips == 5
        assert config.clipping.smart_framing is True
        assert config.caption.style == CaptionStyle.VIRAL

    def test_platform_preset(self):
        """Test platform presets."""
        youtube = PlatformPreset.youtube_shorts()
        assert youtube.width == 1080
        assert youtube.height == 1920
        assert youtube.max_duration == 60

        tiktok = PlatformPreset.tiktok()
        assert tiktok.max_duration == 180

    def test_update_for_platform(self):
        """Test updating config for platform."""
        config = Config()
        config.highlight.max_clip_duration = 120

        config.update_for_platform(Platform.YOUTUBE_SHORTS)

        # Should be clamped to platform max
        assert config.highlight.max_clip_duration == 60

    def test_validate(self):
        """Test configuration validation."""
        config = Config()
        issues = config.validate()
        assert len(issues) == 0

        # Invalid config
        config.highlight.min_clip_duration = 100
        config.highlight.max_clip_duration = 50
        issues = config.validate()
        assert len(issues) > 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = Config()
        d = config.to_dict()

        assert isinstance(d, dict)
        assert 'platform' in d
        assert 'transcription' in d
        assert d['platform'] == 'youtube_shorts'


class TestTranscriptionConfig:
    """Test cases for TranscriptionConfig."""

    def test_default_values(self):
        """Test default transcription config."""
        config = TranscriptionConfig()

        assert config.model == WhisperModel.BASE
        assert config.language is None
        assert config.word_timestamps is True
        assert config.device == 'auto'


class TestHighlightConfig:
    """Test cases for HighlightConfig."""

    def test_default_filler_words(self):
        """Test default filler words list."""
        config = HighlightConfig()

        assert 'um' in config.filler_words
        assert 'uh' in config.filler_words
        assert 'like' in config.filler_words

    def test_weights_sum(self):
        """Test that scoring weights sum to 1.0."""
        config = HighlightConfig()

        total = (
            config.emotional_weight +
            config.insight_weight +
            config.engagement_weight +
            config.pacing_weight
        )

        assert abs(total - 1.0) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
