# Shorts Bot

**Convert long-form videos into viral short-form content - 100% free and local**

Shorts Bot is a production-ready, fully open-source system for automatically converting long-form videos (podcasts, lectures, rants, discussions) into short-form vertical videos optimized for YouTube Shorts, Instagram Reels, and TikTok.

## Features

- **Automatic Highlight Detection** - AI-powered identification of the most engaging moments
- **High-Accuracy Transcription** - Local speech-to-text using OpenAI Whisper
- **Viral-Style Captions** - Word-by-word highlighting with customizable styles
- **Smart Framing** - Face detection for optimal vertical cropping
- **Platform Presets** - Optimized settings for YouTube Shorts, Instagram Reels, and TikTok
- **Batch Processing** - Process multiple videos at once
- **100% Free & Local** - No API costs, no cloud dependencies

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/shorts-bot.git
cd shorts-bot

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (if not already installed)
# Ubuntu/Debian: sudo apt install ffmpeg
# macOS: brew install ffmpeg
# Windows: Download from https://ffmpeg.org/download.html
```

### Basic Usage

#### Command Line
```bash
# Process a single video
python cli.py process video.mp4

# Process for TikTok with 10 clips
python cli.py process video.mp4 --platform tiktok --clips 10

# Quick preview of highlights
python cli.py preview video.mp4

# Create a manual clip
python cli.py clip video.mp4 --start 30 --end 60
```

#### Web Interface
```bash
# Start the Streamlit web app
streamlit run src/ui/streamlit_app.py
```

Then open `http://localhost:8501` in your browser.

## System Architecture

```
+------------------------------------------------------------------+
|                         SHORTS BOT                                |
+------------------------------------------------------------------+
|                                                                   |
|  +----------+   +--------------+   +-----------------+            |
|  |  INPUT   |-->| TRANSCRIPTION|-->| HIGHLIGHT       |            |
|  | HANDLER  |   |   (Whisper)  |   | DETECTION       |            |
|  +----------+   +--------------+   +--------+--------+            |
|       |                                     |                     |
|       |              +----------------------+                     |
|       |              v                                            |
|       |        +--------------+   +-----------------+             |
|       +------->|    VIDEO     |-->|    CAPTION      |             |
|                |   CLIPPER    |   |   GENERATOR     |             |
|                +--------------+   +--------+--------+             |
|                                            |                      |
|                                            v                      |
|                                   +-----------------+             |
|                                   |     OUTPUT      |             |
|                                   | (Vertical Clips)|             |
|                                   +-----------------+             |
|                                                                   |
+-------------------------------------------------------------------+
```

## Pipeline Stages

### 1. Input Handling
- Accepts MP4, MKV, AVI, MOV, WebM video files
- Supports audio-only files (MP3, WAV, etc.)
- Batch processing for multiple files
- URL download support via yt-dlp

### 2. Transcription
- Uses OpenAI Whisper for speech-to-text
- Supports multiple model sizes (tiny to large-v3)
- Word-level timestamps for precise captions
- 99+ language support with auto-detection

### 3. Highlight Detection
- Analyzes transcript for engaging content:
  - Emotional peaks (excitement, passion)
  - Key insights and quotable moments
  - Strong opinions and rants
  - Story-telling moments
- Identifies and filters out:
  - Filler words (um, uh, like)
  - Silent segments
  - Low-information content
- Outputs ranked clips with scores and reasons

### 4. Video Clipping
- Converts horizontal (16:9) to vertical (9:16)
- Smart framing with face detection
- Configurable clip duration (15-180 seconds)
- Platform-specific output settings

### 5. Caption Generation
- Multiple caption styles:
  - **Viral**: Word-by-word pop animation
  - **Bold**: Large, impactful text
  - **Minimal**: Clean, simple subtitles
  - **Karaoke**: Color-changing highlight
- Customizable fonts, colors, and positioning
- ASS subtitle format for precise rendering

## Configuration

### Quality Presets

| Preset | Whisper Model | Encoding | Use Case |
|--------|--------------|----------|----------|
| Fast | tiny | ultrafast | Quick testing |
| Balanced | base | medium | Daily use |
| Quality | medium | slow | Final production |
| Max Quality | large-v3 | slow | Best results |

### Platform Settings

| Platform | Resolution | Max Duration | FPS |
|----------|-----------|--------------|-----|
| YouTube Shorts | 1080x1920 | 60s | 30 |
| Instagram Reels | 1080x1920 | 90s | 30 |
| TikTok | 1080x1920 | 180s | 30 |

### Configuration File

Create a `config.yaml` file for custom settings:

```yaml
platform: youtube_shorts
transcription:
  model: base
  language: null  # Auto-detect
  word_timestamps: true
highlight:
  target_clips: 5
  min_clip_duration: 15
  max_clip_duration: 60
  emotional_weight: 0.3
  insight_weight: 0.3
caption:
  style: viral
  font_size: 72
  highlight_current_word: true
  uppercase: true
clipping:
  smart_framing: true
  face_detection: true
```

## CLI Reference

### Process Command
```bash
python cli.py process <input> [options]

Options:
  --batch              Batch process directory
  -r, --recursive      Scan subdirectories
  -o, --output         Output directory
  -p, --platform       Target platform (youtube_shorts|instagram_reels|tiktok)
  -n, --clips          Number of clips to generate
  --min-duration       Minimum clip duration (seconds)
  --max-duration       Maximum clip duration (seconds)
  --preset             Quality preset (fast|balanced|quality|max_quality)
  -m, --model          Whisper model size
  -l, --language       Language code
  --caption-style      Caption style (viral|bold|minimal|subtitle|karaoke)
  --no-captions        Disable captions
  --no-smart-framing   Disable face tracking
  -c, --config         Path to config file
```

### Preview Command
```bash
python cli.py preview <video> [--quick]
```

### Clip Command
```bash
python cli.py clip <video> --start <seconds> --end <seconds> [--output <path>] [--no-captions]
```

### Transcribe Command
```bash
python cli.py transcribe <video> [--output <path>] [--format json|srt|vtt|txt] [--model <model>]
```

## Output Structure

```
output/
  video_name/
    video_name_transcript.json    # Full transcript with timings
    video_name.srt                # Subtitle file
    video_name_highlights.json    # Detected highlights
    video_name_clips.json         # Clip metadata
    video_name.edl                # Edit Decision List
    video_name_01_captioned.mp4   # Generated clip 1
    video_name_01_captioned.jpg   # Thumbnail 1
    video_name_02_captioned.mp4   # Generated clip 2
    ...
```

## Hardware Requirements

### Minimum
- CPU: 4 cores
- RAM: 8 GB
- Storage: 10 GB free
- GPU: Not required (CPU-only mode available)

### Recommended
- CPU: 8+ cores
- RAM: 16 GB
- Storage: SSD with 50 GB free
- GPU: NVIDIA with 6+ GB VRAM (for faster processing)

## Dependencies

### Core
- Python 3.9+
- FFmpeg 4.0+
- faster-whisper or openai-whisper

### Python Packages
- faster-whisper (recommended) or openai-whisper
- opencv-python (for face detection)
- streamlit (for web UI)
- pyyaml (for configuration)

## Troubleshooting

### Common Issues

**"No Whisper backend available"**
```bash
pip install faster-whisper
# or
pip install openai-whisper
```

**"FFmpeg not found"**
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows - Download from ffmpeg.org and add to PATH
```

**"CUDA out of memory"**
- Use a smaller Whisper model: `--model tiny` or `--model base`
- Process shorter videos
- Set `device: cpu` in config

**"Face detection not working"**
```bash
pip install opencv-python-headless
```

## Project Structure

```
shorts-bot/
  cli.py                    # Command-line interface
  requirements.txt          # Python dependencies
  setup.py                  # Package setup
  config/                   # Configuration files
    default.yaml
  src/
    __init__.py
    core/
      __init__.py
      config.py             # Configuration management
      logger.py             # Logging utilities
      pipeline.py           # Main orchestrator
    modules/
      __init__.py
      input_handler.py      # File input processing
      transcriber.py        # Whisper transcription
      highlight_detector.py # Highlight detection
      video_clipper.py      # Video clipping
      caption_generator.py  # Caption generation
    utils/
      __init__.py
      file_utils.py         # File operations
      video_utils.py        # Video utilities
      text_utils.py         # Text processing
    ui/
      __init__.py
      streamlit_app.py      # Web interface
  tests/                    # Unit tests
  input/                    # Input files
  output/                   # Generated clips
  temp/                     # Temporary files
```

## Future Improvements

- [ ] Speaker diarization (identify different speakers)
- [ ] Multi-language caption support
- [ ] Custom caption animations
- [ ] Direct upload to platforms
- [ ] Real-time processing
- [ ] GPU acceleration for video encoding
- [ ] A/B testing for thumbnails
- [ ] Analytics dashboard

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Optimized Whisper
- [FFmpeg](https://ffmpeg.org/) - Video processing
- [OpenCV](https://opencv.org/) - Face detection
- [Streamlit](https://streamlit.io/) - Web interface

---

**Made with love for content creators**
