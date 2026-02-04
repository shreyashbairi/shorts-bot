"""
Configuration management for Shorts Bot.
Handles all pipeline parameters with sensible defaults.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import yaml


def detect_cuda_available() -> bool:
    """
    Detect if CUDA is available on this system.

    Returns:
        True if CUDA is available, False otherwise
    """
    # Check via PyTorch (most reliable)
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except ImportError:
        pass

    # Check via ctranslate2
    try:
        import ctranslate2
        supported = ctranslate2.get_supported_compute_types("cuda")
        if supported:
            return True
    except (ImportError, RuntimeError, Exception):
        pass

    return False


def get_default_gpu_enabled() -> bool:
    """
    Get the default value for gpu_enabled based on system capabilities.

    Returns:
        True if CUDA GPU is available, False otherwise
    """
    return detect_cuda_available()


class Platform(Enum):
    """Supported output platforms."""
    YOUTUBE_SHORTS = "youtube_shorts"
    INSTAGRAM_REELS = "instagram_reels"
    TIKTOK = "tiktok"
    CUSTOM = "custom"


class CaptionStyle(Enum):
    """Caption style presets."""
    MINIMAL = "minimal"
    BOLD = "bold"
    VIRAL = "viral"
    SUBTITLE = "subtitle"
    KARAOKE = "karaoke"


class WhisperModel(Enum):
    """Available Whisper model sizes."""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"


@dataclass
class PlatformPreset:
    """Platform-specific video settings."""
    name: str
    width: int
    height: int
    max_duration: int  # seconds
    min_duration: int  # seconds
    fps: int
    bitrate: str
    audio_bitrate: str

    @classmethod
    def youtube_shorts(cls) -> "PlatformPreset":
        return cls(
            name="YouTube Shorts",
            width=1080,
            height=1920,
            max_duration=60,
            min_duration=15,
            fps=30,
            bitrate="8M",
            audio_bitrate="192k"
        )

    @classmethod
    def instagram_reels(cls) -> "PlatformPreset":
        return cls(
            name="Instagram Reels",
            width=1080,
            height=1920,
            max_duration=90,
            min_duration=15,
            fps=30,
            bitrate="8M",
            audio_bitrate="192k"
        )

    @classmethod
    def tiktok(cls) -> "PlatformPreset":
        return cls(
            name="TikTok",
            width=1080,
            height=1920,
            max_duration=180,
            min_duration=15,
            fps=30,
            bitrate="6M",
            audio_bitrate="192k"
        )

    @classmethod
    def from_platform(cls, platform: Platform) -> "PlatformPreset":
        presets = {
            Platform.YOUTUBE_SHORTS: cls.youtube_shorts,
            Platform.INSTAGRAM_REELS: cls.instagram_reels,
            Platform.TIKTOK: cls.tiktok,
        }
        return presets.get(platform, cls.youtube_shorts)()


@dataclass
class CaptionConfig:
    """Caption styling configuration."""
    style: CaptionStyle = CaptionStyle.VIRAL
    font_name: str = "Arial-Bold"
    font_size: int = 60
    font_color: str = "#FFFFFF"
    stroke_color: str = "#000000"
    stroke_width: int = 3
    highlight_color: str = "#FFFF00"
    highlight_current_word: bool = True
    position: str = "center"  # top, center, bottom
    margin_bottom: int = 200
    max_words_per_line: int = 4
    animation: str = "pop"  # none, pop, fade, bounce
    uppercase: bool = True
    add_emoji: bool = False
    shadow: bool = True
    shadow_color: str = "#000000"
    shadow_offset: int = 4


@dataclass
class TranscriptionConfig:
    """Transcription settings."""
    model: WhisperModel = WhisperModel.BASE
    language: Optional[str] = None  # Auto-detect if None
    task: str = "transcribe"  # or "translate"
    word_timestamps: bool = True
    vad_filter: bool = True  # Voice Activity Detection
    device: str = "auto"  # cpu, cuda, auto
    compute_type: str = "auto"  # float16, int8, auto
    batch_size: int = 16


@dataclass
class HighlightConfig:
    """Highlight detection settings."""
    min_clip_duration: int = 15  # seconds
    max_clip_duration: int = 60  # seconds
    target_clips: int = 5  # Number of clips to generate

    # Scoring weights
    emotional_weight: float = 0.3
    insight_weight: float = 0.3
    engagement_weight: float = 0.2
    pacing_weight: float = 0.2

    # Filler word handling
    remove_fillers: bool = True
    filler_aggressiveness: float = 0.5  # 0-1, higher = more aggressive
    filler_words: List[str] = field(default_factory=lambda: [
        "um", "uh", "er", "ah", "like", "you know", "basically",
        "actually", "literally", "sort of", "kind of", "i mean",
        "right", "so", "well", "anyway", "honestly"
    ])

    # Content analysis with LLM
    use_llm: bool = False  # Use local LLM for better analysis
    llm_model: str = "llama3.1:8b-instruct-q4_K_M"  # Model name/path
    llm_backend: str = "llama_cpp"  # llama_cpp, ollama
    llm_model_path: Optional[str] = None  # Path to GGUF file (for llama_cpp)
    llm_n_ctx: int = 2048  # Context window size
    llm_n_threads: int = 6  # Number of CPU threads
    llm_n_gpu_layers: int = -1  # -1 = all layers on GPU (Metal on M2)
    llm_temperature: float = 0.1  # Lower = more deterministic
    llm_max_tokens: int = 256  # Max response tokens

    # Silence detection
    min_silence_duration: float = 0.5  # seconds
    silence_threshold: float = -40  # dB


@dataclass
class ClippingConfig:
    """Video clipping settings."""
    output_format: str = "mp4"
    codec: str = "libx264"
    audio_codec: str = "aac"
    preset: str = "medium"  # ultrafast, fast, medium, slow
    crf: int = 23  # Quality (0-51, lower = better)

    # Smart framing
    smart_framing: bool = True
    face_detection: bool = True
    face_detection_confidence: float = 0.5
    tracking_smoothness: float = 0.3
    fallback_crop: str = "center"  # center, top, bottom

    # Padding for context
    clip_padding_start: float = 0.5  # seconds before
    clip_padding_end: float = 0.5  # seconds after


@dataclass
class Config:
    """Master configuration for the entire pipeline."""
    # Paths
    input_dir: Path = field(default_factory=lambda: Path("input"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    temp_dir: Path = field(default_factory=lambda: Path("temp"))
    models_dir: Path = field(default_factory=lambda: Path("models"))

    # Platform
    platform: Platform = Platform.YOUTUBE_SHORTS
    platform_preset: Optional[PlatformPreset] = None

    # Module configs
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    highlight: HighlightConfig = field(default_factory=HighlightConfig)
    clipping: ClippingConfig = field(default_factory=ClippingConfig)
    caption: CaptionConfig = field(default_factory=CaptionConfig)

    # Pipeline settings
    batch_mode: bool = False
    keep_temp_files: bool = False
    overwrite_existing: bool = False
    export_json: bool = True  # Export metadata for automation

    # Performance
    num_workers: int = 4
    gpu_enabled: bool = field(default_factory=get_default_gpu_enabled)

    def __post_init__(self):
        """Initialize platform preset if not provided."""
        if self.platform_preset is None:
            self.platform_preset = PlatformPreset.from_platform(self.platform)

        # Ensure paths are Path objects
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        self.temp_dir = Path(self.temp_dir)
        self.models_dir = Path(self.models_dir)

    def update_for_platform(self, platform: Platform):
        """Update configuration for a specific platform."""
        self.platform = platform
        self.platform_preset = PlatformPreset.from_platform(platform)

        # Adjust highlight config based on platform limits
        if self.highlight.max_clip_duration > self.platform_preset.max_duration:
            self.highlight.max_clip_duration = self.platform_preset.max_duration

    @classmethod
    def from_file(cls, path: Path) -> "Config":
        """Load configuration from YAML or JSON file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r') as f:
            if path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        # Handle nested configs
        if 'transcription' in data and isinstance(data['transcription'], dict):
            if 'model' in data['transcription']:
                data['transcription']['model'] = WhisperModel(data['transcription']['model'])
            data['transcription'] = TranscriptionConfig(**data['transcription'])

        if 'highlight' in data and isinstance(data['highlight'], dict):
            data['highlight'] = HighlightConfig(**data['highlight'])

        if 'clipping' in data and isinstance(data['clipping'], dict):
            data['clipping'] = ClippingConfig(**data['clipping'])

        if 'caption' in data and isinstance(data['caption'], dict):
            if 'style' in data['caption']:
                data['caption']['style'] = CaptionStyle(data['caption']['style'])
            data['caption'] = CaptionConfig(**data['caption'])

        if 'platform' in data:
            data['platform'] = Platform(data['platform'])

        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        def convert_value(v):
            if isinstance(v, Enum):
                return v.value
            if isinstance(v, Path):
                return str(v)
            if hasattr(v, '__dataclass_fields__'):
                return {k: convert_value(getattr(v, k)) for k in v.__dataclass_fields__}
            if isinstance(v, list):
                return [convert_value(i) for i in v]
            return v

        return {k: convert_value(v) for k, v in self.__dict__.items()}

    def save(self, path: Path):
        """Save configuration to file."""
        path = Path(path)
        data = self.to_dict()

        with open(path, 'w') as f:
            if path.suffix in ['.yaml', '.yml']:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            else:
                json.dump(data, f, indent=2)

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        if self.highlight.min_clip_duration >= self.highlight.max_clip_duration:
            issues.append("min_clip_duration must be less than max_clip_duration")

        if self.highlight.target_clips < 1:
            issues.append("target_clips must be at least 1")

        if self.clipping.crf < 0 or self.clipping.crf > 51:
            issues.append("crf must be between 0 and 51")

        weights_sum = (
            self.highlight.emotional_weight +
            self.highlight.insight_weight +
            self.highlight.engagement_weight +
            self.highlight.pacing_weight
        )
        if abs(weights_sum - 1.0) > 0.01:
            issues.append(f"Highlight weights should sum to 1.0, got {weights_sum}")

        return issues


def get_preset_config(preset_name: str) -> "Config":
    """
    Get a preset configuration with proper GPU auto-detection.

    Args:
        preset_name: One of 'fast', 'balanced', 'quality', 'max_quality'

    Returns:
        Config object with appropriate settings
    """
    gpu_available = get_default_gpu_enabled()

    presets = {
        "fast": Config(
            transcription=TranscriptionConfig(model=WhisperModel.TINY),
            clipping=ClippingConfig(preset="ultrafast", smart_framing=False),
            gpu_enabled=gpu_available,
        ),
        "balanced": Config(
            transcription=TranscriptionConfig(model=WhisperModel.BASE),
            clipping=ClippingConfig(preset="medium"),
            gpu_enabled=gpu_available,
        ),
        "quality": Config(
            transcription=TranscriptionConfig(model=WhisperModel.MEDIUM),
            clipping=ClippingConfig(preset="slow", crf=18),
            gpu_enabled=gpu_available,
        ),
        "max_quality": Config(
            transcription=TranscriptionConfig(model=WhisperModel.LARGE_V3),
            clipping=ClippingConfig(preset="slow", crf=15),
            caption=CaptionConfig(style=CaptionStyle.VIRAL),
            gpu_enabled=gpu_available,
        ),
    }

    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")

    return presets[preset_name]


# Default presets for quick setup (use get_preset_config() for proper GPU detection)
# These are kept for backward compatibility but may not auto-detect GPU correctly
PRESET_CONFIGS = {
    "fast": Config(
        transcription=TranscriptionConfig(model=WhisperModel.TINY),
        clipping=ClippingConfig(preset="ultrafast", smart_framing=False),
    ),
    "balanced": Config(
        transcription=TranscriptionConfig(model=WhisperModel.BASE),
        clipping=ClippingConfig(preset="medium"),
    ),
    "quality": Config(
        transcription=TranscriptionConfig(model=WhisperModel.MEDIUM),
        clipping=ClippingConfig(preset="slow", crf=18),
    ),
    "max_quality": Config(
        transcription=TranscriptionConfig(model=WhisperModel.LARGE_V3),
        clipping=ClippingConfig(preset="slow", crf=15),
        caption=CaptionConfig(style=CaptionStyle.VIRAL),
    ),
}
