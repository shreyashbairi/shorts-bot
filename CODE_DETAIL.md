# Shorts Bot - Developer Code Documentation

This document provides detailed explanations of every component in the Shorts Bot codebase. Use this as a reference for understanding, modifying, and debugging the system.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Data Flow](#data-flow)
3. [Core Module: Configuration](#core-module-configuration)
4. [Core Module: Logger](#core-module-logger)
5. [Core Module: Pipeline](#core-module-pipeline)
6. [Module: Input Handler](#module-input-handler)
7. [Module: Transcriber](#module-transcriber)
8. [Module: Highlight Detector](#module-highlight-detector)
9. [Module: Video Clipper](#module-video-clipper)
10. [Module: Caption Generator](#module-caption-generator)
11. [Utilities: File Utils](#utilities-file-utils)
12. [Utilities: Video Utils](#utilities-video-utils)
13. [Utilities: Text Utils](#utilities-text-utils)
14. [CLI Interface](#cli-interface)
15. [Streamlit Web UI](#streamlit-web-ui)
16. [Debugging Guide](#debugging-guide)
17. [Common Issues & Solutions](#common-issues--solutions)

---

## System Overview

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│                    ┌─────────────┐    ┌─────────────┐                       │
│                    │   CLI.py    │    │ Streamlit   │                       │
│                    │  (argparse) │    │   Web UI    │                       │
│                    └──────┬──────┘    └──────┬──────┘                       │
│                           │                  │                              │
│                           └────────┬─────────┘                              │
│                                    ▼                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                              CORE LAYER                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  config.py      │  │  pipeline.py    │  │  logger.py      │             │
│  │  - Config       │  │  - Pipeline     │  │  - get_logger   │             │
│  │  - Presets      │  │  - Orchestrator │  │  - ColorFormat  │             │
│  └─────────────────┘  └────────┬────────┘  └─────────────────┘             │
│                                │                                            │
├────────────────────────────────┼────────────────────────────────────────────┤
│                         MODULES LAYER                                        │
│                                ▼                                            │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐ │
│  │input_handler │──▶│ transcriber  │──▶│  highlight   │──▶│video_clipper │ │
│  │              │   │              │   │  _detector   │   │              │ │
│  │ - InputFile  │   │ - Transcript │   │ - Highlight  │   │ - ClipResult │ │
│  │ - BatchInput │   │ - Segment    │   │ - Detection  │   │ - CropRegion │ │
│  └──────────────┘   └──────────────┘   └──────────────┘   └──────┬───────┘ │
│                                                                   │         │
│                                                                   ▼         │
│                                                           ┌──────────────┐  │
│                                                           │caption_gen   │  │
│                                                           │              │  │
│                                                           │- CaptionSeq  │  │
│                                                           │- ASS/SRT     │  │
│                                                           └──────────────┘  │
├─────────────────────────────────────────────────────────────────────────────┤
│                           UTILITIES LAYER                                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  file_utils.py  │  │  video_utils.py │  │  text_utils.py  │             │
│  │  - find_files   │  │  - get_info     │  │  - filler_det   │             │
│  │  - ensure_dir   │  │  - extract_aud  │  │  - formatting   │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### File Structure

```
shorts-bot/
├── cli.py                          # Command-line interface entry point
├── requirements.txt                # Python dependencies
├── setup.py                        # Package installation
├── src/
│   ├── __init__.py
│   ├── core/                       # Core system components
│   │   ├── __init__.py
│   │   ├── config.py              # Configuration management (450 lines)
│   │   ├── logger.py              # Logging utilities (120 lines)
│   │   └── pipeline.py            # Main orchestrator (480 lines)
│   ├── modules/                    # Processing modules
│   │   ├── __init__.py
│   │   ├── input_handler.py       # File input processing (280 lines)
│   │   ├── transcriber.py         # Speech-to-text (380 lines)
│   │   ├── highlight_detector.py  # Highlight detection (520 lines)
│   │   ├── video_clipper.py       # Video clipping (420 lines)
│   │   └── caption_generator.py   # Caption generation (480 lines)
│   ├── utils/                      # Shared utilities
│   │   ├── __init__.py
│   │   ├── file_utils.py          # File operations (180 lines)
│   │   ├── video_utils.py         # FFmpeg wrappers (200 lines)
│   │   └── text_utils.py          # Text processing (280 lines)
│   └── ui/
│       ├── __init__.py
│       └── streamlit_app.py       # Web interface (400 lines)
└── config/
    └── default.yaml               # Default configuration
```

---

## Data Flow

### Complete Processing Pipeline

```
1. INPUT STAGE
   ┌─────────────┐
   │ User Input  │  video.mp4 / directory / URL
   └──────┬──────┘
          ▼
   ┌─────────────────────────────────────────┐
   │         InputHandler.process_file()     │
   │  - Validate file exists                 │
   │  - Determine media type (video/audio)   │
   │  - Extract video metadata via FFprobe   │
   │  - Extract audio to WAV (16kHz mono)    │
   └──────┬──────────────────────────────────┘
          │
          ▼ Returns: InputFile object
          │   - path: Path to original file
          │   - media_type: VIDEO or AUDIO
          │   - video_info: VideoInfo (duration, resolution, fps)
          │   - audio_path: Path to extracted WAV

2. TRANSCRIPTION STAGE
   ┌─────────────────────────────────────────┐
   │        Transcriber.transcribe()         │
   │  - Load Whisper model (lazy loading)    │
   │  - Process audio file                   │
   │  - Extract word-level timestamps        │
   │  - Build Segment objects                │
   └──────┬──────────────────────────────────┘
          │
          ▼ Returns: Transcript object
          │   - segments: List[Segment]
          │   - text: Full transcript string
          │   - language: Detected language
          │   - duration: Total duration
          │   - word_count: Total words

3. HIGHLIGHT DETECTION STAGE
   ┌─────────────────────────────────────────┐
   │   HighlightDetector.detect_highlights() │
   │  - Score each segment for:              │
   │    * Emotional content (0-1)            │
   │    * Key insights (0-1)                 │
   │    * Engagement potential (0-1)         │
   │    * Pacing/energy (0-1)                │
   │  - Detect filler segments               │
   │  - Group high-scoring segments          │
   │  - Create clip candidates               │
   └──────┬──────────────────────────────────┘
          │
          ▼ Returns: HighlightResult object
          │   - highlights: List[Highlight]
          │   - total_duration: float
          │   - coverage: float (% of video)
          │   - filler_segments: List[Tuple]

4. VIDEO CLIPPING STAGE
   ┌─────────────────────────────────────────┐
   │       VideoClipper.create_clips()        │
   │  - For each highlight:                  │
   │    * Detect faces in time range         │
   │    * Calculate crop region              │
   │    * Build FFmpeg filter graph          │
   │    * Execute clip creation              │
   └──────┬──────────────────────────────────┘
          │
          ▼ Returns: List[ClipResult]
          │   - path: Output video path
          │   - highlight: Original highlight
          │   - duration: Clip duration
          │   - width/height: Dimensions

5. CAPTION STAGE
   ┌─────────────────────────────────────────┐
   │   CaptionGenerator.generate_captions()   │
   │  - Extract relevant transcript segments │
   │  - Format into caption lines            │
   │  - Generate ASS subtitle file           │
   │  - Burn captions into video via FFmpeg  │
   └──────┬──────────────────────────────────┘
          │
          ▼ Returns: Path to captioned video

6. OUTPUT STAGE
   ┌─────────────────────────────────────────┐
   │  Final outputs in output/video_name/:   │
   │  - *_captioned.mp4  (final clips)       │
   │  - *_transcript.json                    │
   │  - *_highlights.json                    │
   │  - *.srt, *.edl (export formats)        │
   └─────────────────────────────────────────┘
```

---

## Core Module: Configuration

**File:** `src/core/config.py`

### Purpose

Centralizes all configuration parameters with type safety, validation, and serialization support.

### Key Classes

#### `Platform` (Enum)

```python
class Platform(Enum):
    YOUTUBE_SHORTS = "youtube_shorts"
    INSTAGRAM_REELS = "instagram_reels"
    TIKTOK = "tiktok"
    CUSTOM = "custom"
```

**Usage:** Determines output video specifications.

#### `CaptionStyle` (Enum)

```python
class CaptionStyle(Enum):
    MINIMAL = "minimal"    # Simple white text
    BOLD = "bold"          # Large, impactful
    VIRAL = "viral"        # Word-by-word highlight
    SUBTITLE = "subtitle"  # Traditional subtitles
    KARAOKE = "karaoke"    # Color-changing highlight
```

**Usage:** Controls caption rendering style.

#### `WhisperModel` (Enum)

```python
class WhisperModel(Enum):
    TINY = "tiny"          # ~39M params, fastest
    BASE = "base"          # ~74M params, good balance
    SMALL = "small"        # ~244M params
    MEDIUM = "medium"      # ~769M params
    LARGE = "large"        # ~1550M params
    LARGE_V2 = "large-v2"  # Improved large
    LARGE_V3 = "large-v3"  # Latest, best accuracy
```

**Usage:** Determines transcription model. Larger = more accurate but slower.

#### `PlatformPreset` (Dataclass)

```python
@dataclass
class PlatformPreset:
    name: str           # "YouTube Shorts"
    width: int          # 1080
    height: int         # 1920
    max_duration: int   # 60 seconds
    min_duration: int   # 15 seconds
    fps: int            # 30
    bitrate: str        # "8M"
    audio_bitrate: str  # "192k"
```

**Purpose:** Defines platform-specific video requirements.

**Factory Methods:**

- `youtube_shorts()` → 1080x1920, 60s max
- `instagram_reels()` → 1080x1920, 90s max
- `tiktok()` → 1080x1920, 180s max

#### `TranscriptionConfig` (Dataclass)

```python
@dataclass
class TranscriptionConfig:
    model: WhisperModel = WhisperModel.BASE
    language: Optional[str] = None      # Auto-detect if None
    task: str = "transcribe"            # or "translate"
    word_timestamps: bool = True        # Required for captions
    vad_filter: bool = True             # Voice Activity Detection
    device: str = "auto"                # cpu, cuda, auto
    compute_type: str = "auto"          # float16, int8, auto
    batch_size: int = 16
```

**Key Parameters Explained:**

- `word_timestamps`: MUST be True for word-by-word captions
- `vad_filter`: Filters out non-speech segments, improves accuracy
- `device`: "auto" selects CUDA if available, else CPU
- `compute_type`: "float16" for GPU, "int8" for CPU optimization

#### `HighlightConfig` (Dataclass)

```python
@dataclass
class HighlightConfig:
    min_clip_duration: int = 15         # Minimum clip length
    max_clip_duration: int = 60         # Maximum clip length
    target_clips: int = 5               # How many clips to generate

    # Scoring weights (must sum to 1.0)
    emotional_weight: float = 0.3       # Weight for emotional content
    insight_weight: float = 0.3         # Weight for key insights
    engagement_weight: float = 0.2      # Weight for questions/CTA
    pacing_weight: float = 0.2          # Weight for speech pacing

    # Filler handling
    remove_fillers: bool = True
    filler_aggressiveness: float = 0.5  # 0-1, higher = stricter
    filler_words: List[str] = [...]     # Words to detect

    # Optional LLM refinement
    use_llm: bool = False               # Use Ollama for analysis
    llm_model: str = "mistral"          # Ollama model name
```

**Debugging Weights:**
If highlights aren't detecting emotional content well, increase `emotional_weight`.
If too many fillers are included, increase `filler_aggressiveness`.

#### `ClippingConfig` (Dataclass)

```python
@dataclass
class ClippingConfig:
    output_format: str = "mp4"
    codec: str = "libx264"              # H.264 encoder
    audio_codec: str = "aac"
    preset: str = "medium"              # Encoding speed vs compression
    crf: int = 23                       # Quality (0-51, lower=better)

    # Smart framing
    smart_framing: bool = True          # Enable face tracking
    face_detection: bool = True         # Use OpenCV face detection
    face_detection_confidence: float = 0.5
    tracking_smoothness: float = 0.3    # Camera movement smoothing
    fallback_crop: str = "center"       # If no face: center, top, bottom

    # Timing
    clip_padding_start: float = 0.5     # Seconds before highlight
    clip_padding_end: float = 0.5       # Seconds after highlight
```

**FFmpeg Preset Values:**

- `ultrafast`: Fastest, largest file
- `fast`: Good speed, reasonable size
- `medium`: Balanced (recommended)
- `slow`: Better compression, slower
- `veryslow`: Best compression, slowest

**CRF (Constant Rate Factor):**

- 0: Lossless (huge files)
- 18: Visually lossless
- 23: Default, good quality
- 28: Smaller files, visible quality loss
- 51: Worst quality

#### `CaptionConfig` (Dataclass)

```python
@dataclass
class CaptionConfig:
    style: CaptionStyle = CaptionStyle.VIRAL
    font_name: str = "Arial-Bold"
    font_size: int = 60
    font_color: str = "#FFFFFF"         # White
    stroke_color: str = "#000000"       # Black outline
    stroke_width: int = 3
    highlight_color: str = "#FFFF00"    # Yellow highlight
    highlight_current_word: bool = True # Word-by-word highlight
    position: str = "center"            # top, center, bottom
    margin_bottom: int = 200            # Pixels from bottom
    max_words_per_line: int = 4         # Words before line break
    animation: str = "pop"              # none, pop, fade, bounce
    uppercase: bool = True              # CAPS for viral style
    add_emoji: bool = False             # Auto-add emojis
    shadow: bool = True                 # Drop shadow
    shadow_color: str = "#000000"
    shadow_offset: int = 4
```

#### `Config` (Main Configuration Class)

```python
@dataclass
class Config:
    # Paths
    input_dir: Path = Path("input")
    output_dir: Path = Path("output")
    temp_dir: Path = Path("temp")
    models_dir: Path = Path("models")

    # Platform
    platform: Platform = Platform.YOUTUBE_SHORTS
    platform_preset: Optional[PlatformPreset] = None

    # Sub-configs
    transcription: TranscriptionConfig
    highlight: HighlightConfig
    clipping: ClippingConfig
    caption: CaptionConfig

    # Pipeline settings
    batch_mode: bool = False
    keep_temp_files: bool = False       # For debugging
    overwrite_existing: bool = False
    export_json: bool = True

    # Performance
    num_workers: int = 4
    gpu_enabled: bool = True
```

**Key Methods:**

```python
def update_for_platform(self, platform: Platform):
    """Switch all settings to match a platform."""
    # Updates platform_preset and adjusts highlight config
    # to respect platform duration limits

def from_file(cls, path: Path) -> "Config":
    """Load config from YAML or JSON file."""

def to_dict(self) -> Dict[str, Any]:
    """Serialize config for export."""

def validate(self) -> List[str]:
    """Return list of configuration issues."""
```

### Preset Configurations

```python
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
    ),
}
```

### Debugging Configuration Issues

**Problem:** Config validation fails

```python
config = Config()
issues = config.validate()
for issue in issues:
    print(f"Config issue: {issue}")
```

**Problem:** Platform settings not applied

```python
# Wrong: Creating preset after config
config = Config()
config.platform = Platform.TIKTOK  # preset not updated!

# Correct: Use update method
config = Config()
config.update_for_platform(Platform.TIKTOK)  # Updates preset too
```

---

## Core Module: Logger

**File:** `src/core/logger.py`

### Purpose

Provides consistent, colored logging across all modules with optional file output.

### Classes

#### `ColorFormatter`

```python
class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
```

**Purpose:** Adds ANSI color codes to log level names for terminal output.

### Functions

#### `get_logger(name, level, log_file, use_colors)`

```python
def get_logger(
    name: str,                          # Usually __name__
    level: int = logging.INFO,
    log_file: Optional[Path] = None,    # Optional file output
    use_colors: bool = True
) -> logging.Logger:
```

**Usage in modules:**

```python
from src.core.logger import get_logger
logger = get_logger(__name__)

logger.debug("Detailed info for debugging")
logger.info("Normal operation info")
logger.warning("Something unexpected but not fatal")
logger.error("Operation failed")
```

#### `setup_global_logging(level, log_dir)`

```python
def setup_global_logging(
    level: int = logging.INFO,
    log_dir: Optional[Path] = None
) -> Path:
    """Configure root logger for entire application."""
```

**Call this once at startup (in cli.py or streamlit_app.py).**

#### `ProgressLogger`

```python
class ProgressLogger:
    def __init__(self, logger, total, description):
        """Track progress of long operations."""

    def update(self, n=1, message=""):
        """Increment progress by n steps."""

    def complete(self, message=""):
        """Mark operation as complete."""
```

**Usage:**

```python
progress = ProgressLogger(logger, total=100, description="Processing")
for i in range(100):
    # do work
    progress.update(1, f"Item {i}")
progress.complete("All done!")
```

### Debugging with Logs

**Enable verbose logging:**

```bash
python cli.py process video.mp4 --verbose
```

**Check log files:**

```python
# Logs are written to: logs/shorts_bot_YYYYMMDD_HHMMSS.log
setup_global_logging(level=logging.DEBUG, log_dir=Path("logs"))
```

**Log format:**

```
HH:MM:SS | LEVEL | module_name | message
14:23:45 | INFO | transcriber | Transcription complete in 45.2s
```

---

## Core Module: Pipeline

**File:** `src/core/pipeline.py`

### Purpose

Orchestrates the entire video processing workflow, coordinating all modules.

### Classes

#### `PipelineStage` (Enum)

```python
class PipelineStage(Enum):
    INPUT = "input"
    TRANSCRIPTION = "transcription"
    HIGHLIGHT_DETECTION = "highlight_detection"
    CLIPPING = "clipping"
    CAPTIONING = "captioning"
    EXPORT = "export"
    COMPLETE = "complete"
```

**Used for:** Progress tracking and callbacks.

#### `ProcessingResult` (Dataclass)

```python
@dataclass
class ProcessingResult:
    input_file: InputFile               # Original input
    transcript: Optional[Transcript]    # Transcription result
    highlights: Optional[HighlightResult]
    clips: List[ClipResult]             # Generated clips
    output_dir: Optional[Path]          # Output location
    success: bool = True
    error: Optional[str] = None         # Error message if failed
    processing_time: float = 0.0        # Total time in seconds
    metadata: Dict[str, Any]            # Additional info
```

**Methods:**

- `to_dict()`: Serialize for JSON export
- `save(path)`: Write result to JSON file

#### `BatchResult` (Dataclass)

```python
@dataclass
class BatchResult:
    results: List[ProcessingResult]
    total_time: float
    successful: int
    failed: int
```

#### `Pipeline` (Main Orchestrator)

```python
class Pipeline:
    def __init__(
        self,
        config: Config,
        progress_callback: Optional[Callable] = None
    ):
        """
        Initialize pipeline with configuration.

        Args:
            config: Pipeline configuration
            progress_callback: Function(stage, progress, message)
                              Called during processing for UI updates
        """
        self.config = config
        self.progress_callback = progress_callback

        # Initialize modules (some use lazy loading)
        self.input_handler = InputHandler(config)
        self.transcriber = Transcriber(config)
        self.highlight_detector = HighlightDetector(config)
        self.video_clipper = VideoClipper(config)
        self.caption_generator = CaptionGenerator(config)
```

### Key Methods

#### `process_file(file_path, output_name)`

```python
def process_file(
    self,
    file_path: Path,
    output_name: Optional[str] = None
) -> ProcessingResult:
    """
    Process a single video through entire pipeline.

    Steps:
    1. Input validation and audio extraction
    2. Transcription with Whisper
    3. Highlight detection and ranking
    4. Video clipping with smart framing
    5. Caption generation and burning
    6. Export metadata and cleanup

    Returns:
        ProcessingResult with all outputs
    """
```

**Internal Flow:**

```python
# Stage 1: Input
input_file = self.input_handler.process_file(file_path)
# Validates file, extracts metadata, extracts audio to WAV

# Stage 2: Transcription
transcript = self.transcriber.transcribe_with_cache(input_file.audio_path)
# Returns Transcript with word-level timing

# Stage 3: Highlight Detection
highlights = self.highlight_detector.detect_highlights(transcript, audio_path)
# Returns HighlightResult with ranked clips

# Stage 4: Clipping
clips = self.video_clipper.create_clips(source_video, highlights.highlights)
# Returns List[ClipResult] with vertical videos

# Stage 5: Captioning
for clip in clips:
    captions = self.caption_generator.generate_captions(
        transcript, clip.highlight.start, clip.highlight.end
    )
    self.caption_generator.burn_captions(clip.path, captions)
```

#### `process_batch(input_paths, parallel)`

```python
def process_batch(
    self,
    input_paths: List[Path],
    parallel: bool = False  # Not yet implemented
) -> BatchResult:
    """Process multiple videos sequentially."""
```

#### `process_directory(directory, recursive)`

```python
def process_directory(
    self,
    directory: Path,
    recursive: bool = False
) -> BatchResult:
    """Find and process all media files in directory."""
```

#### `preview_highlights(file_path, quick_mode)`

```python
def preview_highlights(
    self,
    file_path: Path,
    quick_mode: bool = True
) -> List[Highlight]:
    """
    Quick analysis without full processing.
    Uses tiny Whisper model for speed.
    Returns highlight candidates for preview.
    """
```

#### `create_single_clip(file_path, start_time, end_time, ...)`

```python
def create_single_clip(
    self,
    file_path: Path,
    start_time: float,
    end_time: float,
    output_path: Optional[Path] = None,
    add_captions: bool = True
) -> ClipResult:
    """Create a manual clip with specified times."""
```

#### `cleanup()`

```python
def cleanup(self):
    """Remove all temporary files from all modules."""
    self.input_handler.cleanup()
    self.transcriber.cleanup()
    self.video_clipper.cleanup()
    self.caption_generator.cleanup()
```

### Progress Callback

```python
def my_progress_callback(stage: PipelineStage, progress: float, message: str):
    """
    Called during processing.

    Args:
        stage: Current pipeline stage
        progress: 0.0 to 1.0 within current stage
        message: Human-readable status
    """
    print(f"[{stage.value}] {progress*100:.0f}% - {message}")

pipeline = Pipeline(config, progress_callback=my_progress_callback)
```

### Debugging Pipeline Issues

**Problem:** Pipeline fails silently

```python
result = pipeline.process_file(video_path)
if not result.success:
    print(f"Error: {result.error}")
    # Check result.metadata for additional context
```

**Problem:** Specific stage fails

```python
# Check the result object for partial data
if result.transcript is None:
    print("Transcription failed")
elif result.highlights is None:
    print("Highlight detection failed")
elif not result.clips:
    print("Clipping failed")
```

**Problem:** Need to debug intermediate outputs

```python
config.keep_temp_files = True  # Don't delete temp files
# Check temp/ directory for intermediate files
```

---

## Module: Input Handler

**File:** `src/modules/input_handler.py`

### Purpose

Validates input files, extracts metadata, and prepares audio for transcription.

### Classes

#### `MediaType` (Enum)

```python
class MediaType(Enum):
    VIDEO = "video"
    AUDIO = "audio"
    UNKNOWN = "unknown"
```

#### `InputFile` (Dataclass)

```python
@dataclass
class InputFile:
    path: Path                          # Original file path
    media_type: MediaType               # VIDEO or AUDIO
    video_info: Optional[VideoInfo]     # Metadata from FFprobe
    audio_path: Optional[Path] = None   # Extracted audio WAV
    metadata: Dict[str, Any]            # Additional info
    file_hash: Optional[str] = None     # For deduplication

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def duration(self) -> float:
        return self.video_info.duration if self.video_info else 0.0

    @property
    def is_valid(self) -> bool:
        return self.media_type != MediaType.UNKNOWN and self.path.exists()
```

#### `BatchInput` (Dataclass)

```python
@dataclass
class BatchInput:
    files: List[InputFile]
    total_duration: float = 0.0         # Sum of all durations
    processed_count: int = 0
    failed_count: int = 0

    def __iter__(self):
        """Iterate over files."""
        for f in self.files:
            yield f
```

#### `InputHandler`

```python
class InputHandler:
    def __init__(self, config: Config):
        self.config = config
        self.temp_dir = config.temp_dir / "input"
        self._processed_hashes = set()  # For deduplication
```

### Key Methods

#### `process_file(file_path, extract_audio_immediately)`

```python
def process_file(
    self,
    file_path: Path,
    extract_audio_immediately: bool = True
) -> InputFile:
    """
    Process a single input file.

    Steps:
    1. Check file exists
    2. Determine media type (video/audio)
    3. Extract video info via FFprobe
    4. Calculate file hash (if batch mode)
    5. Extract audio to WAV (16kHz mono)

    Returns:
        InputFile ready for transcription

    Raises:
        FileNotFoundError: File doesn't exist
        ValueError: Unsupported format
    """
```

**Audio Extraction Details:**

```python
# Audio is extracted to:
# - Sample rate: 16000 Hz (Whisper optimal)
# - Channels: Mono
# - Format: WAV (uncompressed for quality)
# - Location: temp/input/{filename}_audio.wav
```

#### `process_directory(directory, recursive)`

```python
def process_directory(
    self,
    directory: Path,
    recursive: bool = False
) -> BatchInput:
    """
    Find all media files in directory.

    Supported extensions:
    - Video: .mp4, .mkv, .avi, .mov, .webm, .flv, .wmv
    - Audio: .mp3, .wav, .aac, .flac, .ogg, .m4a
    """
```

#### `process_urls(urls)`

```python
def process_urls(self, urls: List[str]) -> BatchInput:
    """
    Download and process videos from URLs.
    Uses yt-dlp for downloading.

    Requires: yt-dlp installed (pip install yt-dlp)
    """
```

#### `validate_input(input_file)`

```python
def validate_input(self, input_file: InputFile) -> List[str]:
    """
    Validate input meets requirements.

    Checks:
    - File exists
    - Duration >= min_clip_duration
    - Resolution >= 480p (warning)
    - Long video warning (>1 hour)

    Returns:
        List of issues (empty if valid)
    """
```

### Debugging Input Issues

**Problem:** "Unsupported file format"

```python
# Check file extension is supported
from src.utils.file_utils import is_video_file, is_audio_file
print(f"Is video: {is_video_file(path)}")
print(f"Is audio: {is_audio_file(path)}")
```

**Problem:** Audio extraction fails

```python
# Check FFmpeg is installed
import subprocess
result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
print(result.returncode)  # Should be 0

# Check audio extraction manually
from src.utils.video_utils import extract_audio
extract_audio(video_path, output_path, sample_rate=16000, mono=True)
```

**Problem:** Video info extraction fails

```python
from src.utils.video_utils import get_video_info
info = get_video_info(video_path)
print(f"Duration: {info.duration}")
print(f"Resolution: {info.width}x{info.height}")
print(f"FPS: {info.fps}")
```

---

## Module: Transcriber

**File:** `src/modules/transcriber.py`

### Purpose

Converts speech to text using OpenAI Whisper with word-level timestamps.

### Classes

#### `Word` (Dataclass)

```python
@dataclass
class Word:
    text: str                   # The word itself
    start: float                # Start time in seconds
    end: float                  # End time in seconds
    confidence: float = 1.0     # Model confidence (0-1)

    @property
    def duration(self) -> float:
        return self.end - self.start
```

#### `Segment` (Dataclass)

```python
@dataclass
class Segment:
    id: int                     # Segment index
    text: str                   # Full segment text
    start: float                # Start time
    end: float                  # End time
    words: List[Word]           # Word-level timing
    confidence: float = 1.0
    speaker: Optional[str]      # Future: speaker diarization

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def word_count(self) -> int:
        return len(self.text.split())
```

#### `Transcript` (Dataclass)

```python
@dataclass
class Transcript:
    segments: List[Segment]     # All transcribed segments
    text: str                   # Full text concatenated
    language: str               # Detected language code
    duration: float             # Total audio duration
    word_count: int             # Total word count
    metadata: Dict[str, Any]    # Backend info, model used, etc.
```

**Key Methods:**

```python
def get_segment_at_time(self, time_seconds: float) -> Optional[Segment]:
    """Find segment containing this timestamp."""

def get_segments_in_range(self, start: float, end: float) -> List[Segment]:
    """Get all segments overlapping time range."""

def get_text_in_range(self, start: float, end: float) -> str:
    """Get concatenated text in time range."""

def to_dict(self) -> Dict[str, Any]:
    """Serialize for JSON export."""

def save(self, path: Path):
    """Save to JSON file."""

@classmethod
def load(cls, path: Path) -> 'Transcript':
    """Load from JSON file."""
```

#### `Transcriber`

```python
class Transcriber:
    def __init__(self, config: Config):
        self.config = config
        self.trans_config = config.transcription
        self.model = None           # Lazy loaded
        self._backend = None        # "faster-whisper" or "openai-whisper"
```

### Key Methods

#### `_load_model()`

```python
def _load_model(self):
    """
    Lazy load Whisper model.

    Tries backends in order:
    1. faster-whisper (recommended, uses CTranslate2)
    2. openai-whisper (original implementation)

    Raises:
        ImportError: No backend available
    """
```

**Model Loading Logic:**

```python
# faster-whisper (preferred)
from faster_whisper import WhisperModel
model = WhisperModel(
    model_size,                 # "base", "medium", etc.
    device=device,              # "cuda" or "cpu"
    compute_type=compute_type   # "float16" or "int8"
)

# openai-whisper (fallback)
import whisper
model = whisper.load_model(model_size, device=device)
```

#### `transcribe(audio_path, language)`

```python
def transcribe(
    self,
    audio_path: Path,
    language: Optional[str] = None
) -> Transcript:
    """
    Transcribe audio file.

    Args:
        audio_path: Path to audio file (WAV recommended)
        language: Language code (None for auto-detect)

    Returns:
        Transcript with segments and word timings
    """
```

**Processing Flow:**

```python
# 1. Load model (if not loaded)
self._load_model()

# 2. Run transcription
if self._backend == "faster-whisper":
    segments_gen, info = self.model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,
        beam_size=5
    )
    # Iterate generator to get segments

elif self._backend == "openai-whisper":
    result = self.model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True
    )

# 3. Build Transcript object from results
```

#### `transcribe_with_cache(audio_path, cache_dir, language)`

```python
def transcribe_with_cache(
    self,
    audio_path: Path,
    cache_dir: Optional[Path] = None,
    language: Optional[str] = None
) -> Transcript:
    """
    Transcribe with caching to avoid re-processing.

    Cache key: {file_hash}_{model_name}.json
    """
```

#### `get_word_timings(transcript)`

```python
def get_word_timings(self, transcript: Transcript) -> List[Dict[str, Any]]:
    """
    Extract flat list of all word timings.

    Returns:
        [{'text': 'hello', 'start': 0.0, 'end': 0.5, ...}, ...]

    If word timings not available, estimates from segment timing.
    """
```

#### `export_srt(transcript, output_path)` / `export_vtt(transcript, output_path)`

```python
def export_srt(self, transcript: Transcript, output_path: Path) -> Path:
    """Export as SRT subtitle file."""

def export_vtt(self, transcript: Transcript, output_path: Path) -> Path:
    """Export as WebVTT subtitle file."""
```

### Debugging Transcription

**Problem:** "No Whisper backend available"

```bash
# Install faster-whisper (recommended)
pip install faster-whisper

# Or install openai-whisper
pip install openai-whisper
```

**Problem:** CUDA out of memory

```python
# Use smaller model
config.transcription.model = WhisperModel.TINY

# Or force CPU
config.transcription.device = "cpu"
```

**Problem:** Wrong language detected

```python
# Force specific language
config.transcription.language = "en"  # English
config.transcription.language = "es"  # Spanish
```

**Problem:** Poor transcription quality

```python
# Use larger model
config.transcription.model = WhisperModel.MEDIUM  # or LARGE_V3

# Check audio quality
# - Ensure 16kHz sample rate
# - Ensure mono channel
# - Check for background noise
```

**Problem:** Missing word timestamps

```python
# Ensure word_timestamps is enabled
config.transcription.word_timestamps = True

# Check segments have words
for segment in transcript.segments:
    print(f"Segment {segment.id}: {len(segment.words)} words")
    if not segment.words:
        print("  WARNING: No word timings!")
```

---

## Module: Highlight Detector

**File:** `src/modules/highlight_detector.py`

### Purpose

Analyzes transcripts to identify the most engaging segments for short-form content.

### Classes

#### `HighlightReason` (Enum)

```python
class HighlightReason(Enum):
    EMOTIONAL_PEAK = "emotional_peak"       # Strong emotions
    KEY_INSIGHT = "key_insight"             # Important information
    STRONG_OPINION = "strong_opinion"       # Personal views
    ENGAGING_QUESTION = "engaging_question" # Questions to audience
    DRAMATIC_PAUSE = "dramatic_pause"       # Impactful pacing
    HIGH_ENERGY = "high_energy"             # Fast, energetic speech
    STORY_MOMENT = "story_moment"           # Narrative elements
    QUOTABLE = "quotable"                   # Short, punchy statements
    TOPIC_INTRO = "topic_intro"             # New topic introduction
    CONCLUSION = "conclusion"               # Wrap-up moments
```

#### `Highlight` (Dataclass)

```python
@dataclass
class Highlight:
    start: float                    # Start time in seconds
    end: float                      # End time in seconds
    text: str                       # Transcript text for this range
    score: float                    # Overall score (0-1)
    reasons: List[HighlightReason]  # Why this was selected
    metadata: Dict[str, Any]        # Additional info

    @property
    def duration(self) -> float:
        return self.end - self.start
```

#### `HighlightResult` (Dataclass)

```python
@dataclass
class HighlightResult:
    highlights: List[Highlight]             # Ranked highlights
    total_duration: float                   # Video duration
    coverage: float                         # % of video in highlights
    filler_segments: List[Tuple[float, float]]  # Filler-heavy segments
    silence_segments: List[Tuple[float, float]] # Silent segments

    def get_top_highlights(self, n: int) -> List[Highlight]:
        """Get top N highlights by score."""
```

#### `HighlightDetector`

```python
class HighlightDetector:
    def __init__(self, config: Config):
        self.config = config
        self.hl_config = config.highlight

        # Word lists for detection
        self.emotion_words = {
            'positive': {'amazing', 'incredible', ...},
            'negative': {'terrible', 'awful', ...},
            'intense': {'absolutely', 'completely', ...},
            'opinion': {'think', 'believe', ...},
        }

        # Regex patterns for insights
        self.insight_patterns = [
            r'\b(the key (is|thing)|what matters)\b',
            r'\b(here\'?s the (thing|secret))\b',
            ...
        ]

        # Story-telling patterns
        self.story_patterns = [
            r'\b(one (time|day)|remember when)\b',
            r'\b(and then|suddenly)\b',
            ...
        ]
```

### Key Methods

#### `detect_highlights(transcript, audio_path)`

```python
def detect_highlights(
    self,
    transcript: Transcript,
    audio_path: Optional[Path] = None
) -> HighlightResult:
    """
    Main detection method.

    Steps:
    1. Score each segment
    2. Detect filler segments
    3. Detect silence (if audio provided)
    4. Group high-scoring segments into clips
    5. Optionally refine with LLM

    Returns:
        HighlightResult with ranked highlights
    """
```

**Processing Flow:**

```python
# 1. Score each segment
segment_scores = []
for segment in transcript.segments:
    score, reasons = self._score_segment(segment, transcript)
    segment_scores.append({
        'segment': segment,
        'score': score,
        'reasons': reasons
    })

# 2. Detect filler segments (>40% filler words)
filler_segments = self._detect_filler_segments(transcript)

# 3. Detect silence segments
silence_segments = self._detect_silence_segments(audio_path)

# 4. Create clips from high-scoring segments
highlights = self._create_clips(
    segment_scores,
    transcript.duration,
    filler_segments,
    silence_segments
)

# 5. Optional LLM refinement
if self.hl_config.use_llm:
    highlights = self._refine_with_llm(highlights, transcript)
```

#### `_score_segment(segment, transcript)`

```python
def _score_segment(
    self,
    segment: Segment,
    transcript: Transcript
) -> Tuple[float, List[HighlightReason]]:
    """
    Score a single segment (0-1).

    Scoring factors:
    1. Emotional content (emotion words, punctuation)
    2. Insight potential (patterns, statistics)
    3. Engagement (questions, direct address)
    4. Pacing (words per minute)
    5. Story elements
    6. Strong opinions
    7. Quotability
    8. Filler word penalty
    """
```

**Scoring Breakdown:**

```python
# Emotional scoring (0-1)
emotion_score = self._score_emotional_content(segment.text)
# - Counts emotion words from word lists
# - Adds weight for exclamation marks
# - Adds weight for ALL CAPS words

# Insight scoring (0-1)
insight_score = self._score_insights(segment.text)
# - Matches against insight_patterns regex
# - Adds weight for numbers/statistics
# - Adds weight for comparison language

# Engagement scoring (0-1)
engagement_score = self._score_engagement(segment.text)
# - Counts question marks
# - Detects direct address ("you", "your")
# - Matches call-to-action patterns

# Pacing scoring (0-1)
pacing_score = self._score_pacing(segment)
# Optimal: 140-180 WPM → 0.8
# Good: 120-140 or 180-200 WPM → 0.6
# Acceptable: 100-120 or 200-220 WPM → 0.4
# Poor: < 100 or > 220 WPM → 0.2

# Filler penalty (0-0.5)
filler_penalty = self._calculate_filler_penalty(segment.text)
# filler_density = filler_count / word_count
# penalty = min(filler_density * 2, 0.5)

# Final score
score = (
    emotion_score * emotional_weight +
    insight_score * insight_weight +
    engagement_score * engagement_weight +
    pacing_score * pacing_weight
    - filler_penalty
)
score = max(0.0, min(1.0, score))  # Clamp to 0-1
```

#### `_create_clips(segment_scores, total_duration, filler_segments, silence_segments)`

```python
def _create_clips(...) -> List[Highlight]:
    """
    Group scored segments into highlight clips.

    Algorithm:
    1. Sort segments by score (descending)
    2. For each high-scoring segment:
       a. Start with segment boundaries
       b. Expand to include adjacent high-scoring segments
       c. Check duration constraints (min/max)
       d. Check overlap with existing highlights
       e. Check overlap with fillers/silence
       f. Create Highlight if valid
    3. Return top N highlights
    """
```

### Debugging Highlight Detection

**Problem:** No highlights found

```python
result = detector.detect_highlights(transcript)
print(f"Found {len(result.highlights)} highlights")

# Check segment scores
for segment in transcript.segments[:10]:
    score, reasons = detector._score_segment(segment, transcript)
    print(f"Segment {segment.id}: score={score:.2f}, reasons={reasons}")
```

**Problem:** Low scores across all segments

```python
# Check scoring weights
print(f"Emotional weight: {config.highlight.emotional_weight}")
print(f"Insight weight: {config.highlight.insight_weight}")
# Weights should sum to 1.0

# Check filler word settings
print(f"Filler aggressiveness: {config.highlight.filler_aggressiveness}")
# Try lowering this if too many segments are penalized
```

**Problem:** Wrong segments selected

```python
# Export all segment scores for analysis
import json
scores = []
for segment in transcript.segments:
    score, reasons = detector._score_segment(segment, transcript)
    scores.append({
        'id': segment.id,
        'text': segment.text[:50],
        'start': segment.start,
        'end': segment.end,
        'score': score,
        'reasons': [r.value for r in reasons]
    })
with open('segment_scores.json', 'w') as f:
    json.dump(scores, f, indent=2)
```

**Problem:** Clips too short/long

```python
# Adjust duration settings
config.highlight.min_clip_duration = 20  # Minimum 20s
config.highlight.max_clip_duration = 45  # Maximum 45s
```

---

## Module: Video Clipper

**File:** `src/modules/video_clipper.py`

### Purpose

Creates vertical video clips from horizontal source with smart framing.

### Classes

#### `FacePosition` (Dataclass)

```python
@dataclass
class FacePosition:
    x: int              # Center X coordinate
    y: int              # Center Y coordinate
    width: int          # Face bounding box width
    height: int         # Face bounding box height
    confidence: float   # Detection confidence
    timestamp: float    # Time in video
```

#### `CropRegion` (Dataclass)

```python
@dataclass
class CropRegion:
    x: int              # Left edge of crop
    y: int              # Top edge of crop
    width: int          # Crop width
    height: int         # Crop height

    def to_ffmpeg_filter(self) -> str:
        """Generate FFmpeg crop filter string."""
        return f"crop={self.width}:{self.height}:{self.x}:{self.y}"
```

#### `ClipResult` (Dataclass)

```python
@dataclass
class ClipResult:
    path: Path              # Output video path
    highlight: Highlight    # Source highlight
    duration: float         # Actual clip duration
    width: int              # Output width (1080)
    height: int             # Output height (1920)
    file_size: int          # File size in bytes
    metadata: Dict[str, Any]
```

#### `VideoClipper`

```python
class VideoClipper:
    def __init__(self, config: Config):
        self.config = config
        self.clip_config = config.clipping
        self.platform = config.platform_preset
        self._face_detector = None
```

### Key Methods

#### `create_clips(source_path, highlights, output_prefix)`

```python
def create_clips(
    self,
    source_path: Path,
    highlights: List[Highlight],
    output_prefix: str = "clip"
) -> List[ClipResult]:
    """
    Create video clips from highlights.

    Steps for each highlight:
    1. Detect faces in time range (if enabled)
    2. Calculate crop region for vertical aspect
    3. Build FFmpeg command
    4. Execute clip creation
    5. Verify output
    """
```

#### `_calculate_crop_region(video_info, face_data, start_time, end_time)`

```python
def _calculate_crop_region(
    self,
    video_info: VideoInfo,
    face_data: Optional[Dict],
    start_time: float,
    end_time: float
) -> CropRegion:
    """
    Calculate crop for 9:16 vertical aspect ratio.

    Algorithm:
    1. Calculate crop dimensions for 9:16 aspect
       crop_height = source_height
       crop_width = source_height * (9/16)

    2. If crop_width > source_width:
       # Source is narrower, crop vertically
       crop_width = source_width
       crop_height = source_width / (9/16)

    3. Determine horizontal position:
       - If face detected: center on face
       - Else: use fallback_crop setting (center/left/right)

    4. Clamp crop to stay within frame
    """
```

**Visual Example:**

```
Source (1920x1080 landscape):
┌─────────────────────────────────────┐
│                                     │
│        ┌─────────┐                  │
│        │  FACE   │                  │
│        └─────────┘                  │
│                                     │
└─────────────────────────────────────┘

Crop region for 9:16 (607x1080):
         ┌───────┐
         │       │
         │ FACE  │
         │       │
         └───────┘

Final output (1080x1920):
    ┌─────────┐
    │         │
    │  FACE   │
    │         │
    │         │
    └─────────┘
```

#### `_build_ffmpeg_command(...)`

```python
def _build_ffmpeg_command(
    self,
    source_path: Path,
    output_path: Path,
    start_time: float,
    duration: float,
    crop_region: CropRegion,
    video_info: VideoInfo
) -> List[str]:
    """
    Build FFmpeg command for clip creation.

    Filter graph:
    1. crop={w}:{h}:{x}:{y}     - Crop to 9:16 aspect
    2. scale={target_w}:{target_h} - Scale to 1080x1920
    3. fps={target_fps}         - Adjust frame rate
    """
```

**Generated Command:**

```bash
ffmpeg -y \
    -ss {start_time} \
    -i {source_path} \
    -t {duration} \
    -vf "crop=607:1080:656:0,scale=1080:1920,fps=30" \
    -c:v libx264 \
    -preset medium \
    -crf 23 \
    -c:a aac \
    -b:a 192k \
    -movflags +faststart \
    {output_path}
```

#### `_detect_faces_opencv(video_path, time_ranges, video_info)`

```python
def _detect_faces_opencv(
    self,
    video_path: Path,
    time_ranges: List[Tuple[float, float]],
    video_info: VideoInfo
) -> List[Dict]:
    """
    Detect faces using OpenCV Haar Cascades.

    Samples one frame per second in relevant time ranges.
    Returns list of face positions with timestamps.
    """
```

**Face Detection Code:**

```python
import cv2

# Load Haar Cascade
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Open video
cap = cv2.VideoCapture(str(video_path))

# For each time in range
frame_num = int(current_time * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
ret, frame = cap.read()

# Convert to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces
detected = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,    # Image pyramid scale
    minNeighbors=5,     # Required detections
    minSize=(50, 50)    # Minimum face size
)
```

### Debugging Video Clipping

**Problem:** Clips not created

```python
# Check FFmpeg is available
import subprocess
result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
if result.returncode != 0:
    print("FFmpeg not found!")

# Check source video
from src.utils.video_utils import get_video_info
info = get_video_info(source_path)
print(f"Source: {info.width}x{info.height}, {info.duration}s")
```

**Problem:** Wrong crop region

```python
# Debug crop calculation
clipper = VideoClipper(config)
video_info = get_video_info(video_path)

crop = clipper._calculate_crop_region(
    video_info, None, 0, 10
)
print(f"Crop: x={crop.x}, y={crop.y}, w={crop.width}, h={crop.height}")
print(f"Aspect ratio: {crop.width/crop.height:.4f}")  # Should be ~0.5625 (9/16)
```

**Problem:** Face detection not working

```python
# Check OpenCV is installed
try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
except ImportError:
    print("OpenCV not installed: pip install opencv-python")

# Test face detection directly
import cv2
cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
# If cascade.empty() is True, cascade file not found
```

**Problem:** Output quality is poor

```python
# Lower CRF for better quality
config.clipping.crf = 18  # Lower = better (18-23 recommended)

# Use slower preset for better compression
config.clipping.preset = "slow"
```

---

## Module: Caption Generator

**File:** `src/modules/caption_generator.py`

### Purpose

Generates and burns styled captions into video clips.

### Classes

#### `CaptionLine` (Dataclass)

```python
@dataclass
class CaptionLine:
    text: str                           # Caption text
    start: float                        # Start time
    end: float                          # End time
    words: List[Dict[str, Any]]         # Word-level timing for highlights
```

#### `CaptionSequence` (Dataclass)

```python
@dataclass
class CaptionSequence:
    lines: List[CaptionLine]
    style: CaptionStyle
    duration: float

    def to_ass_dialogue(self, style_name="Default") -> List[str]:
        """Convert to ASS subtitle format."""
```

#### `CaptionGenerator`

```python
class CaptionGenerator:
    # Style presets define default settings for each style
    STYLE_PRESETS = {
        CaptionStyle.VIRAL: {
            'font_size': 72,
            'font_color': '#FFFFFF',
            'stroke_width': 4,
            'position': 'center',
            'highlight_current_word': True,
            'highlight_color': '#FFFF00',
            'uppercase': True,
            'animation': 'pop',
            'max_words_per_line': 3,
        },
        # ... other styles
    }
```

### Key Methods

#### `generate_captions(transcript, clip_start, clip_end)`

```python
def generate_captions(
    self,
    transcript: Transcript,
    clip_start: float,
    clip_end: float
) -> CaptionSequence:
    """
    Generate captions for a clip time range.

    If highlight_current_word enabled:
        Creates word-by-word captions with timing
    Else:
        Creates sentence-based captions
    """
```

#### `_create_word_highlight_captions(segments, clip_start, clip_end)`

```python
def _create_word_highlight_captions(...) -> List[CaptionLine]:
    """
    Create viral-style word-by-word captions.

    Groups words into lines (max_words_per_line)
    Each line shows for duration of its words.
    """
```

**Example Output:**

```
Time 0.0-0.8:  "THIS IS"     (2 words)
Time 0.8-1.5:  "REALLY"      (1 word, highlighted)
Time 1.5-2.2:  "IMPORTANT"   (1 word)
```

#### `burn_captions(video_path, captions, output_path)`

```python
def burn_captions(
    self,
    video_path: Path,
    captions: CaptionSequence,
    output_path: Optional[Path] = None
) -> Path:
    """
    Burn captions into video using FFmpeg ASS filter.

    Steps:
    1. Generate ASS subtitle file
    2. Run FFmpeg with ass filter
    3. Clean up ASS file
    """
```

#### `_generate_ass_file(captions, video_path)`

```python
def _generate_ass_file(
    self,
    captions: CaptionSequence,
    video_path: Path
) -> Path:
    """
    Generate ASS (Advanced SubStation Alpha) subtitle file.
    """
```

**ASS File Structure:**

```
[Script Info]
Title: Shorts Bot Captions
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, ...
Style: Default,Arial-Bold,72,&H00FFFFFF,&H00FFFFFF,&H00000000,...

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:01.50,Default,,0,0,0,,THIS IS REALLY
```

#### `_generate_word_highlight_dialogue(captions)`

```python
def _generate_word_highlight_dialogue(captions: CaptionSequence) -> str:
    """
    Generate ASS dialogue with word-by-word highlighting.

    For each word timing:
    1. Build text with current word highlighted
    2. Use ASS override tags for color/scale
    """
```

**ASS Override Tags Used:**

```
{\c&H00FFFF00&}  - Change color to yellow (BGR format)
{\c&HFFFFFF&}    - Reset to white
{\fscx110}       - Scale X to 110%
{\fscy110}       - Scale Y to 110%
```

**Example Dialogue:**

```
Dialogue: 0,0:00:00.00,0:00:00.50,Default,,0,0,0,,{\fscx110\fscy110\c&H00FFFF&}THIS{\fscx100\fscy100\c&HFFFFFF&} IS
Dialogue: 0,0:00:00.50,0:00:01.00,Default,,0,0,0,,THIS {\fscx110\fscy110\c&H00FFFF&}IS{\fscx100\fscy100\c&HFFFFFF&}
```

### Debugging Captions

**Problem:** Captions not appearing

```python
# Check ASS file was generated
ass_path = caption_generator.temp_dir / "captions.ass"
if ass_path.exists():
    with open(ass_path) as f:
        print(f.read()[:1000])  # Print first 1000 chars

# Test FFmpeg ASS filter
cmd = f"ffmpeg -i video.mp4 -vf \"ass='captions.ass'\" -c:a copy output.mp4"
```

**Problem:** Captions out of sync

```python
# Check timing relative to clip
for line in captions.lines:
    print(f"{line.start:.2f}-{line.end:.2f}: {line.text}")

# Ensure times are relative to clip, not source
# clip_start should be subtracted from segment times
```

**Problem:** Wrong font

```python
# Check available fonts
import subprocess
result = subprocess.run(['fc-list'], capture_output=True, text=True)
print(result.stdout)  # Lists installed fonts

# Use a known font
config.caption.font_name = "DejaVu Sans"
```

**Problem:** Word highlighting not working

```python
# Check word timings exist
for line in captions.lines:
    print(f"Line: {line.text}")
    print(f"  Words: {len(line.words)}")
    for word in line.words:
        print(f"    {word['text']}: {word['start']:.2f}-{word['end']:.2f}")
```

---

## Utilities: File Utils

**File:** `src/utils/file_utils.py`

### Functions

#### `ensure_dir(path)`

```python
def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist. Returns path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
```

#### `find_media_files(directory, include_video, include_audio, recursive)`

```python
def find_media_files(
    directory: Path,
    include_video: bool = True,
    include_audio: bool = True,
    recursive: bool = False
) -> List[Path]:
    """
    Find all media files in directory.

    Video extensions: .mp4, .mkv, .avi, .mov, .webm, .flv, .wmv
    Audio extensions: .mp3, .wav, .aac, .flac, .ogg, .m4a
    """
```

#### `is_video_file(path)` / `is_audio_file(path)`

```python
def is_video_file(path: Path) -> bool:
    """Check if file has video extension."""

def is_audio_file(path: Path) -> bool:
    """Check if file has audio extension."""
```

#### `get_file_hash(path, algorithm)`

```python
def get_file_hash(path: Path, algorithm: str = "md5") -> str:
    """
    Calculate file hash for deduplication.
    Reads file in chunks for memory efficiency.
    """
```

#### `get_safe_filename(name)`

```python
def get_safe_filename(name: str) -> str:
    """
    Convert string to safe filename.
    Removes special characters, replaces spaces with underscores.
    """
```

---

## Utilities: Video Utils

**File:** `src/utils/video_utils.py`

### Classes

#### `VideoInfo` (Dataclass)

```python
@dataclass
class VideoInfo:
    duration: float         # Duration in seconds
    width: int              # Frame width
    height: int             # Frame height
    fps: float              # Frames per second
    codec: str              # Video codec name
    bitrate: Optional[int]  # Bitrate in bits/s
    audio_codec: Optional[str]
    audio_sample_rate: Optional[int]

    @property
    def duration_formatted(self) -> str:
        """Format as HH:MM:SS."""

    @property
    def aspect_ratio(self) -> float:
        """Width / Height."""
```

### Functions

#### `get_video_info(video_path)`

```python
def get_video_info(video_path: Path) -> VideoInfo:
    """
    Extract video metadata using FFprobe.

    FFprobe command:
    ffprobe -v quiet -print_format json -show_format -show_streams {path}
    """
```

#### `extract_audio(video_path, output_path, sample_rate, mono)`

```python
def extract_audio(
    video_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
    mono: bool = True
) -> Path:
    """
    Extract audio from video using FFmpeg.

    FFmpeg command:
    ffmpeg -i {input} -vn -acodec pcm_s16le -ar {sample_rate} -ac {1|2} {output}
    """
```

#### `detect_silence(audio_path, threshold_db, min_duration)`

```python
def detect_silence(
    audio_path: Path,
    threshold_db: float = -40,
    min_duration: float = 0.5
) -> List[Tuple[float, float]]:
    """
    Detect silent segments in audio.

    FFmpeg command:
    ffmpeg -i {input} -af silencedetect=n={threshold}dB:d={duration} -f null -

    Returns:
        List of (start, end) tuples for silent segments
    """
```

---

## Utilities: Text Utils

**File:** `src/utils/text_utils.py`

### Functions

#### `detect_filler_words(text, filler_words, filler_phrases, aggressiveness)`

```python
def detect_filler_words(
    text: str,
    filler_words: Optional[Set[str]] = None,
    filler_phrases: Optional[List[str]] = None,
    aggressiveness: float = 0.5
) -> List[Tuple[int, int, str]]:
    """
    Detect filler words and phrases.

    Returns:
        List of (start_index, end_index, filler_text) tuples

    Aggressiveness:
        0.0-0.6: Only obvious fillers (um, uh)
        0.7-1.0: Include common words (like, just, so)
    """
```

#### `remove_filler_words(text, ...)`

```python
def remove_filler_words(text: str, ...) -> str:
    """Remove detected fillers from text."""
```

#### `calculate_speech_rate(text, duration_seconds)`

```python
def calculate_speech_rate(text: str, duration_seconds: float) -> float:
    """
    Calculate words per minute.

    Normal speech: 120-150 WPM
    Fast speech: 150-180 WPM
    Very fast: > 180 WPM
    """
```

#### `format_for_captions(text, max_words_per_line, max_chars_per_line, uppercase)`

```python
def format_for_captions(
    text: str,
    max_words_per_line: int = 4,
    max_chars_per_line: int = 30,
    uppercase: bool = True
) -> List[str]:
    """Split text into caption-friendly lines."""
```

---

## CLI Interface

**File:** `cli.py`

### Commands

#### `process`

```bash
python cli.py process <input> [options]
```

**Flow:**

1. Parse arguments with argparse
2. Build Config from arguments
3. Create Pipeline with config
4. Call `pipeline.process_file()` or `pipeline.process_directory()`
5. Print results

#### `preview`

```bash
python cli.py preview <video> [--quick]
```

**Flow:**

1. Create Config with fast settings
2. Call `pipeline.preview_highlights()`
3. Print highlight candidates

#### `clip`

```bash
python cli.py clip <video> --start <s> --end <s> [--no-captions]
```

**Flow:**

1. Create manual Highlight object
2. Call `pipeline.create_single_clip()`

#### `transcribe`

```bash
python cli.py transcribe <video> [--format json|srt|vtt|txt]
```

**Flow:**

1. Process input file
2. Run transcription
3. Export in requested format

### Debugging CLI

**Problem:** Command not recognized

```bash
python cli.py --help
```

**Problem:** Arguments not parsed

```bash
# Use verbose mode
python cli.py process video.mp4 --verbose
```

---

## Streamlit Web UI

**File:** `src/ui/streamlit_app.py`

### Structure

```python
def main():
    init_session_state()    # Initialize st.session_state
    render_sidebar()        # Configuration options

    # Tab-based navigation
    tab1, tab2, tab3, tab4 = st.tabs([...])

    with tab1:
        render_main_content()    # File upload + process
    with tab2:
        render_preview_tab()     # Quick preview
    with tab3:
        render_manual_clip_tab() # Manual clipping
    with tab4:
        # About page
```

### Session State

```python
st.session_state = {
    'processing': False,        # Is processing running
    'current_stage': None,      # Current pipeline stage
    'progress': 0.0,            # Progress percentage
    'status_message': '',       # Status text
    'result': None,             # ProcessingResult
    'highlights': None,         # Preview highlights
    'config': None,             # Current config
}
```

### Debugging Streamlit

**Problem:** File upload fails

```python
# Check file size limit (default 200MB)
# In ~/.streamlit/config.toml:
# [server]
# maxUploadSize = 500
```

**Problem:** Session state not persisting

```python
# Always check if key exists
if 'result' not in st.session_state:
    st.session_state.result = None
```

---

## Debugging Guide

### General Debugging Steps

1. **Enable verbose logging**

   ```python
   from src.core.logger import setup_global_logging
   import logging
   setup_global_logging(level=logging.DEBUG, log_dir=Path("logs"))
   ```

2. **Keep temporary files**

   ```python
   config.keep_temp_files = True
   # Check temp/ directory for intermediate files
   ```

3. **Test modules individually**

   ```python
   # Test just transcription
   transcriber = Transcriber(config)
   transcript = transcriber.transcribe(audio_path)

   # Test just highlight detection
   detector = HighlightDetector(config)
   highlights = detector.detect_highlights(transcript)
   ```

4. **Check intermediate outputs**
   ```python
   # Save and inspect
   transcript.save(Path("debug_transcript.json"))
   highlights.save(Path("debug_highlights.json"))
   ```

### Performance Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run your code
pipeline.process_file(video_path)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 time consumers
```

### Memory Monitoring

```python
import tracemalloc

tracemalloc.start()

# Run your code
pipeline.process_file(video_path)

current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024 / 1024:.1f} MB")
print(f"Peak: {peak / 1024 / 1024:.1f} MB")
tracemalloc.stop()
```

---

## Common Issues & Solutions

### Issue: "No module named 'src'"

**Cause:** Python can't find the src package.

**Solution:**

```bash
# Run from project root
cd /path/to/shorts-bot
python cli.py process video.mp4

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/shorts-bot"
```

### Issue: FFmpeg errors

**Cause:** FFmpeg not installed or wrong version.

**Solution:**

```bash
# Check FFmpeg
ffmpeg -version

# Install on Ubuntu
sudo apt update && sudo apt install ffmpeg

# Install on macOS
brew install ffmpeg
```

### Issue: CUDA out of memory

**Cause:** GPU doesn't have enough VRAM for model.

**Solution:**

```python
# Use smaller model
config.transcription.model = WhisperModel.TINY

# Or force CPU
config.transcription.device = "cpu"
config.gpu_enabled = False
```

### Issue: Captions not synced

**Cause:** Timing offset between transcript and clip.

**Solution:**

```python
# Verify transcript timing
for segment in transcript.segments[:5]:
    print(f"{segment.start:.2f}s: {segment.text[:50]}")

# Ensure caption times are relative to clip start
# In generate_captions(), times should be: segment.start - clip_start
```

### Issue: Face detection finds no faces

**Cause:** Face too small, profile view, or poor lighting.

**Solution:**

```python
# Lower detection threshold
config.clipping.face_detection_confidence = 0.3

# Or disable face detection
config.clipping.face_detection = False
config.clipping.fallback_crop = "center"
```

### Issue: Output video quality is poor

**Cause:** High CRF value or fast preset.

**Solution:**

```python
# Better quality
config.clipping.crf = 18      # Lower = better
config.clipping.preset = "slow"  # Better compression
```

### Issue: Processing is very slow

**Cause:** Large Whisper model, slow encoding preset, or CPU-only.

**Solution:**

```python
# Faster transcription
config.transcription.model = WhisperModel.BASE  # or TINY

# Faster encoding
config.clipping.preset = "fast"  # or "ultrafast"

# Enable GPU
config.gpu_enabled = True
config.transcription.device = "cuda"
```

---

## Appendix: FFmpeg Filter Reference

### Crop Filter

```
crop=w:h:x:y
    w = output width
    h = output height
    x = left offset
    y = top offset
```

### Scale Filter

```
scale=w:h
    w = target width (or -1 for auto)
    h = target height (or -1 for auto)
```

### ASS Subtitle Filter

```
ass=filename.ass
    Burns ASS subtitles into video
```

### Common Filter Chain

```
-vf "crop=607:1080:656:0,scale=1080:1920,fps=30"
```

---

## Appendix: ASS Format Reference

### Colors

ASS uses BGR (Blue-Green-Red) format with alpha:

```
&HAABBGGRR
    AA = Alpha (00=opaque, FF=transparent)
    BB = Blue
    GG = Green
    RR = Red

Example: &H00FFFF00 = Yellow (BGR: 00FFFF)
```

### Override Tags

```
{\c&H00FFFF00&}  - Primary color
{\3c&H000000&}   - Outline color
{\fscx110}       - Scale X to 110%
{\fscy110}       - Scale Y to 110%
{\b1}            - Bold on
{\b0}            - Bold off
{\an5}           - Alignment: middle-center
```

### Alignment Values

```
7 8 9    (top)
4 5 6    (middle)
1 2 3    (bottom)
(left, center, right)
```

---

_Document Version: 1.0_
_Last Updated: 2024_
