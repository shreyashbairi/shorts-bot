#!/usr/bin/env python3
"""
Shorts Bot - Dependency Checker
Run with: python scripts/check_dependencies.py
"""

import sys
import subprocess
import shutil


def check_python():
    """Check Python version."""
    version = sys.version_info
    if version >= (3, 9):
        print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"[ERROR] Python 3.9+ required (found {version.major}.{version.minor})")
        return False


def check_ffmpeg():
    """Check FFmpeg installation."""
    if shutil.which('ffmpeg'):
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True
            )
            version = result.stdout.split('\n')[0]
            print(f"[OK] FFmpeg: {version}")
            return True
        except Exception as e:
            print(f"[ERROR] FFmpeg found but error: {e}")
            return False
    else:
        print("[ERROR] FFmpeg not found")
        print("  Install: sudo apt install ffmpeg (Linux)")
        print("           brew install ffmpeg (macOS)")
        return False


def check_whisper():
    """Check Whisper installation."""
    try:
        from faster_whisper import WhisperModel
        print("[OK] faster-whisper installed (recommended)")
        return True
    except ImportError:
        pass

    try:
        import whisper
        print("[OK] openai-whisper installed")
        return True
    except ImportError:
        pass

    print("[ERROR] No Whisper backend found")
    print("  Install: pip install faster-whisper")
    return False


def check_opencv():
    """Check OpenCV installation."""
    try:
        import cv2
        print(f"[OK] OpenCV {cv2.__version__}")
        return True
    except ImportError:
        print("[WARNING] OpenCV not installed (face detection disabled)")
        print("  Install: pip install opencv-python")
        return False


def check_streamlit():
    """Check Streamlit installation."""
    try:
        import streamlit
        print(f"[OK] Streamlit {streamlit.__version__}")
        return True
    except ImportError:
        print("[WARNING] Streamlit not installed (web UI disabled)")
        print("  Install: pip install streamlit")
        return False


def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print(f"[OK] CUDA GPU: {device_name} (CUDA {cuda_version})")
            return True
        else:
            print("[INFO] No CUDA GPU detected (CPU mode)")
            return False
    except ImportError:
        print("[INFO] PyTorch not installed (GPU detection unavailable)")
        return False


def check_core_modules():
    """Check core module imports."""
    try:
        sys.path.insert(0, '.')
        from src.core.config import Config
        from src.core.pipeline import Pipeline
        from src.modules.input_handler import InputHandler
        from src.modules.transcriber import Transcriber
        from src.modules.highlight_detector import HighlightDetector
        from src.modules.video_clipper import VideoClipper
        from src.modules.caption_generator import CaptionGenerator
        print("[OK] All core modules importable")
        return True
    except ImportError as e:
        print(f"[ERROR] Module import failed: {e}")
        return False


def main():
    print("=" * 50)
    print("  Shorts Bot - Dependency Check")
    print("=" * 50)
    print()

    checks = [
        ("Python", check_python),
        ("FFmpeg", check_ffmpeg),
        ("Whisper", check_whisper),
        ("OpenCV", check_opencv),
        ("Streamlit", check_streamlit),
        ("GPU", check_gpu),
        ("Core Modules", check_core_modules),
    ]

    results = {}
    for name, check_func in checks:
        print(f"\nChecking {name}...")
        results[name] = check_func()

    print("\n" + "=" * 50)
    print("  Summary")
    print("=" * 50)

    required = ["Python", "FFmpeg", "Whisper", "Core Modules"]
    optional = ["OpenCV", "Streamlit", "GPU"]

    all_required = all(results.get(r, False) for r in required)

    print("\nRequired:")
    for r in required:
        status = "OK" if results.get(r, False) else "MISSING"
        print(f"  [{status}] {r}")

    print("\nOptional:")
    for o in optional:
        status = "OK" if results.get(o, False) else "NOT FOUND"
        print(f"  [{status}] {o}")

    print()
    if all_required:
        print("Ready to use! Run: python cli.py --help")
        return 0
    else:
        print("Please install missing required dependencies.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
