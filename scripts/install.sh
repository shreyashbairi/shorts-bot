#!/bin/bash
# Shorts Bot - Installation Script
# Run with: bash scripts/install.sh

set -e

echo "========================================"
echo "  Shorts Bot - Installation Script"
echo "========================================"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1-2)
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)"
    exit 1
fi
echo "[OK] Python $PYTHON_VERSION detected"

# Check FFmpeg
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -n1)
    echo "[OK] FFmpeg detected: $FFMPEG_VERSION"
else
    echo "[WARNING] FFmpeg not found. Please install FFmpeg:"
    echo "  Ubuntu/Debian: sudo apt install ffmpeg"
    echo "  macOS: brew install ffmpeg"
    echo "  Windows: Download from https://ffmpeg.org/download.html"
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "[OK] Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create directories
echo ""
echo "Creating directories..."
mkdir -p input output temp models config

# Check GPU availability
echo ""
echo "Checking GPU availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'[OK] CUDA GPU detected: {torch.cuda.get_device_name(0)}')
    print(f'     CUDA version: {torch.version.cuda}')
else:
    print('[INFO] No CUDA GPU detected - will use CPU')
    print('       GPU processing is optional but speeds up transcription')
" 2>/dev/null || echo "[INFO] PyTorch not installed - GPU check skipped"

# Test Whisper
echo ""
echo "Testing Whisper installation..."
python3 -c "
try:
    from faster_whisper import WhisperModel
    print('[OK] faster-whisper installed')
except ImportError:
    try:
        import whisper
        print('[OK] openai-whisper installed')
    except ImportError:
        print('[ERROR] No Whisper backend found')
        print('        Run: pip install faster-whisper')
"

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "
import sys
sys.path.insert(0, '.')
from src.core.config import Config
from src.core.pipeline import Pipeline
print('[OK] Core modules loaded successfully')
"

echo ""
echo "========================================"
echo "  Installation Complete!"
echo "========================================"
echo ""
echo "Quick start:"
echo "  1. Activate venv: source venv/bin/activate"
echo "  2. Process video: python cli.py process video.mp4"
echo "  3. Or use web UI: streamlit run src/ui/streamlit_app.py"
echo ""
echo "For help: python cli.py --help"
