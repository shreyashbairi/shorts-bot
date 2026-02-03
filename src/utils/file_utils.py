"""
File system utilities for Shorts Bot.
Handles temporary files, directory management, and cleanup.
"""

import os
import shutil
import tempfile
import hashlib
from pathlib import Path
from typing import Optional, List, Generator
from contextlib import contextmanager
import uuid

from ..core.logger import get_logger

logger = get_logger(__name__)

# Supported video formats
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.m4v'}
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a', '.wma'}


def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists

    Returns:
        The path (for chaining)
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_temp_path(
    base_dir: Path,
    prefix: str = "",
    suffix: str = "",
    unique: bool = True
) -> Path:
    """
    Generate a temporary file path.

    Args:
        base_dir: Base directory for temp files
        prefix: Filename prefix
        suffix: Filename suffix (e.g., '.mp4')
        unique: Whether to add unique identifier

    Returns:
        Path object for temporary file
    """
    base_dir = ensure_dir(Path(base_dir))

    if unique:
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{prefix}_{unique_id}{suffix}" if prefix else f"{unique_id}{suffix}"
    else:
        filename = f"{prefix}{suffix}"

    return base_dir / filename


def cleanup_temp(temp_dir: Path, keep_recent: int = 0) -> int:
    """
    Clean up temporary files.

    Args:
        temp_dir: Directory to clean
        keep_recent: Number of recent files to keep (0 = delete all)

    Returns:
        Number of files deleted
    """
    temp_dir = Path(temp_dir)

    if not temp_dir.exists():
        return 0

    deleted = 0

    if keep_recent == 0:
        # Delete everything
        try:
            shutil.rmtree(temp_dir)
            ensure_dir(temp_dir)
            logger.info(f"Cleaned up temp directory: {temp_dir}")
        except Exception as e:
            logger.error(f"Failed to clean temp directory: {e}")
    else:
        # Keep N most recent files
        files = sorted(temp_dir.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True)
        for f in files[keep_recent:]:
            try:
                if f.is_file():
                    f.unlink()
                else:
                    shutil.rmtree(f)
                deleted += 1
            except Exception as e:
                logger.warning(f"Failed to delete {f}: {e}")

    return deleted


@contextmanager
def temp_directory(base_dir: Optional[Path] = None, cleanup: bool = True):
    """
    Context manager for temporary directory operations.

    Args:
        base_dir: Base directory (uses system temp if None)
        cleanup: Whether to delete directory on exit

    Yields:
        Path to temporary directory
    """
    if base_dir:
        base_dir = ensure_dir(Path(base_dir))
        temp_path = base_dir / f"tmp_{uuid.uuid4().hex[:8]}"
        temp_path.mkdir()
    else:
        temp_path = Path(tempfile.mkdtemp())

    try:
        yield temp_path
    finally:
        if cleanup and temp_path.exists():
            shutil.rmtree(temp_path)


def find_media_files(
    directory: Path,
    include_video: bool = True,
    include_audio: bool = True,
    recursive: bool = False
) -> List[Path]:
    """
    Find all media files in a directory.

    Args:
        directory: Directory to search
        include_video: Include video files
        include_audio: Include audio-only files
        recursive: Search subdirectories

    Returns:
        List of media file paths, sorted by name
    """
    directory = Path(directory)

    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []

    extensions = set()
    if include_video:
        extensions.update(VIDEO_EXTENSIONS)
    if include_audio:
        extensions.update(AUDIO_EXTENSIONS)

    pattern = "**/*" if recursive else "*"
    files = []

    for path in directory.glob(pattern):
        if path.is_file() and path.suffix.lower() in extensions:
            files.append(path)

    return sorted(files, key=lambda p: p.name.lower())


def is_video_file(path: Path) -> bool:
    """Check if a file is a supported video format."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def is_audio_file(path: Path) -> bool:
    """Check if a file is a supported audio format."""
    return Path(path).suffix.lower() in AUDIO_EXTENSIONS


def get_file_hash(path: Path, algorithm: str = "md5") -> str:
    """
    Calculate hash of a file for caching/deduplication.

    Args:
        path: File path
        algorithm: Hash algorithm (md5, sha256, etc.)

    Returns:
        Hex digest of file hash
    """
    hash_func = hashlib.new(algorithm)

    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def get_safe_filename(name: str, max_length: int = 200) -> str:
    """
    Convert a string to a safe filename.

    Args:
        name: Original name
        max_length: Maximum filename length

    Returns:
        Safe filename string
    """
    # Remove or replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    safe_name = name

    for char in unsafe_chars:
        safe_name = safe_name.replace(char, '_')

    # Remove leading/trailing whitespace and dots
    safe_name = safe_name.strip(' .')

    # Truncate if too long
    if len(safe_name) > max_length:
        safe_name = safe_name[:max_length]

    return safe_name or "unnamed"


def copy_with_progress(
    src: Path,
    dst: Path,
    chunk_size: int = 1024 * 1024,
    callback=None
) -> Path:
    """
    Copy a file with progress callback.

    Args:
        src: Source file path
        dst: Destination file path
        chunk_size: Size of chunks to copy
        callback: Optional callback(bytes_copied, total_bytes)

    Returns:
        Destination path
    """
    src = Path(src)
    dst = Path(dst)

    ensure_dir(dst.parent)

    total_size = src.stat().st_size
    copied = 0

    with open(src, 'rb') as fsrc:
        with open(dst, 'wb') as fdst:
            while True:
                chunk = fsrc.read(chunk_size)
                if not chunk:
                    break
                fdst.write(chunk)
                copied += len(chunk)
                if callback:
                    callback(copied, total_size)

    return dst


def get_unique_output_path(base_path: Path) -> Path:
    """
    Get a unique output path by adding a counter if file exists.

    Args:
        base_path: Desired output path

    Returns:
        Unique path (may be original if doesn't exist)
    """
    base_path = Path(base_path)

    if not base_path.exists():
        return base_path

    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent

    counter = 1
    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1
