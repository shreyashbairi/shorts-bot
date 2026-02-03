"""
Video processing utilities using FFmpeg.
Provides video information extraction, audio extraction, and format conversion.
"""

import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from ..core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VideoInfo:
    """Container for video metadata."""
    path: Path
    duration: float  # seconds
    width: int
    height: int
    fps: float
    codec: str
    bitrate: int  # bits per second
    audio_codec: Optional[str]
    audio_sample_rate: Optional[int]
    audio_channels: Optional[int]
    format_name: str
    size_bytes: int

    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio (width/height)."""
        return self.width / self.height if self.height > 0 else 0

    @property
    def is_vertical(self) -> bool:
        """Check if video is in vertical format."""
        return self.height > self.width

    @property
    def is_horizontal(self) -> bool:
        """Check if video is in horizontal format."""
        return self.width > self.height

    @property
    def duration_formatted(self) -> str:
        """Get duration as HH:MM:SS string."""
        hours = int(self.duration // 3600)
        minutes = int((self.duration % 3600) // 60)
        seconds = int(self.duration % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def check_ffmpeg() -> bool:
    """Check if FFmpeg is installed and accessible."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_video_info(path: Path) -> VideoInfo:
    """
    Extract video metadata using FFprobe.

    Args:
        path: Path to video file

    Returns:
        VideoInfo object with metadata

    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If FFprobe fails
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        str(path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFprobe failed: {e.stderr}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse FFprobe output: {e}")

    # Find video and audio streams
    video_stream = None
    audio_stream = None

    for stream in data.get('streams', []):
        if stream.get('codec_type') == 'video' and not video_stream:
            video_stream = stream
        elif stream.get('codec_type') == 'audio' and not audio_stream:
            audio_stream = stream

    if not video_stream:
        # Might be audio-only file
        format_info = data.get('format', {})
        return VideoInfo(
            path=path,
            duration=float(format_info.get('duration', 0)),
            width=0,
            height=0,
            fps=0,
            codec='audio_only',
            bitrate=int(format_info.get('bit_rate', 0)),
            audio_codec=audio_stream.get('codec_name') if audio_stream else None,
            audio_sample_rate=int(audio_stream.get('sample_rate', 0)) if audio_stream else None,
            audio_channels=int(audio_stream.get('channels', 0)) if audio_stream else None,
            format_name=format_info.get('format_name', 'unknown'),
            size_bytes=int(format_info.get('size', 0))
        )

    # Parse frame rate
    fps_str = video_stream.get('r_frame_rate', '30/1')
    try:
        num, den = map(int, fps_str.split('/'))
        fps = num / den if den > 0 else 30.0
    except (ValueError, ZeroDivisionError):
        fps = 30.0

    format_info = data.get('format', {})

    return VideoInfo(
        path=path,
        duration=float(format_info.get('duration', 0)),
        width=int(video_stream.get('width', 0)),
        height=int(video_stream.get('height', 0)),
        fps=fps,
        codec=video_stream.get('codec_name', 'unknown'),
        bitrate=int(format_info.get('bit_rate', 0)),
        audio_codec=audio_stream.get('codec_name') if audio_stream else None,
        audio_sample_rate=int(audio_stream.get('sample_rate', 0)) if audio_stream else None,
        audio_channels=int(audio_stream.get('channels', 0)) if audio_stream else None,
        format_name=format_info.get('format_name', 'unknown'),
        size_bytes=int(format_info.get('size', 0))
    )


def extract_audio(
    video_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
    mono: bool = True,
    format: str = "wav"
) -> Path:
    """
    Extract audio from video file.

    Args:
        video_path: Path to video file
        output_path: Path for output audio file
        sample_rate: Audio sample rate (16000 for Whisper)
        mono: Convert to mono
        format: Output format (wav, mp3, etc.)

    Returns:
        Path to extracted audio file
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-vn',  # No video
        '-acodec', 'pcm_s16le' if format == 'wav' else 'libmp3lame',
        '-ar', str(sample_rate),
    ]

    if mono:
        cmd.extend(['-ac', '1'])

    cmd.extend(['-y', str(output_path)])

    logger.debug(f"Extracting audio: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Audio extraction failed: {e.stderr.decode()}")

    return output_path


def get_frame_at_time(
    video_path: Path,
    time_seconds: float,
    output_path: Path
) -> Path:
    """
    Extract a single frame from video at specified time.

    Args:
        video_path: Path to video file
        time_seconds: Time in seconds
        output_path: Path for output image

    Returns:
        Path to extracted frame
    """
    cmd = [
        'ffmpeg',
        '-ss', str(time_seconds),
        '-i', str(video_path),
        '-vframes', '1',
        '-y', str(output_path)
    ]

    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def cut_video_segment(
    input_path: Path,
    output_path: Path,
    start_time: float,
    end_time: float,
    codec: str = "copy",
    audio_codec: str = "copy"
) -> Path:
    """
    Cut a segment from video file.

    Args:
        input_path: Source video path
        output_path: Output video path
        start_time: Start time in seconds
        end_time: End time in seconds
        codec: Video codec (copy for fast cut, libx264 for re-encode)
        audio_codec: Audio codec

    Returns:
        Path to output video
    """
    duration = end_time - start_time

    cmd = [
        'ffmpeg',
        '-ss', str(start_time),
        '-i', str(input_path),
        '-t', str(duration),
        '-c:v', codec,
        '-c:a', audio_codec,
        '-y', str(output_path)
    ]

    logger.debug(f"Cutting video: {start_time:.2f}s to {end_time:.2f}s")

    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Video cut failed: {e.stderr.decode()}")

    return output_path


def resize_video(
    input_path: Path,
    output_path: Path,
    width: int,
    height: int,
    maintain_aspect: bool = True
) -> Path:
    """
    Resize video to specified dimensions.

    Args:
        input_path: Source video path
        output_path: Output video path
        width: Target width
        height: Target height
        maintain_aspect: Add padding to maintain aspect ratio

    Returns:
        Path to output video
    """
    if maintain_aspect:
        # Scale to fit within dimensions, then pad
        filter_complex = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black"
        )
    else:
        filter_complex = f"scale={width}:{height}"

    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-vf', filter_complex,
        '-c:a', 'copy',
        '-y', str(output_path)
    ]

    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def detect_silence(
    audio_path: Path,
    threshold_db: float = -40,
    min_duration: float = 0.5
) -> list[Tuple[float, float]]:
    """
    Detect silent segments in audio.

    Args:
        audio_path: Path to audio file
        threshold_db: Silence threshold in dB
        min_duration: Minimum silence duration in seconds

    Returns:
        List of (start, end) tuples for silent segments
    """
    cmd = [
        'ffmpeg',
        '-i', str(audio_path),
        '-af', f'silencedetect=noise={threshold_db}dB:d={min_duration}',
        '-f', 'null', '-'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stderr

    silences = []
    current_start = None

    for line in output.split('\n'):
        if 'silence_start:' in line:
            try:
                current_start = float(line.split('silence_start:')[1].strip().split()[0])
            except (IndexError, ValueError):
                continue
        elif 'silence_end:' in line and current_start is not None:
            try:
                end = float(line.split('silence_end:')[1].strip().split()[0])
                silences.append((current_start, end))
                current_start = None
            except (IndexError, ValueError):
                continue

    return silences


def get_volume_levels(
    audio_path: Path,
    interval: float = 0.5
) -> list[Tuple[float, float]]:
    """
    Get volume levels throughout audio file.

    Args:
        audio_path: Path to audio file
        interval: Measurement interval in seconds

    Returns:
        List of (timestamp, volume_db) tuples
    """
    cmd = [
        'ffmpeg',
        '-i', str(audio_path),
        '-af', f'astats=metadata=1:reset={int(1/interval)}',
        '-f', 'null', '-'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    # Parse volume data from stderr
    # This is a simplified version - full implementation would parse lavfi metadata

    return []  # Placeholder - would need more complex parsing


def concatenate_videos(
    input_paths: list[Path],
    output_path: Path,
    transition: Optional[str] = None
) -> Path:
    """
    Concatenate multiple videos into one.

    Args:
        input_paths: List of video paths to concatenate
        output_path: Output video path
        transition: Optional transition effect

    Returns:
        Path to output video
    """
    # Create concat file
    concat_file = output_path.parent / "concat_list.txt"

    with open(concat_file, 'w') as f:
        for path in input_paths:
            f.write(f"file '{path}'\n")

    cmd = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(concat_file),
        '-c', 'copy',
        '-y', str(output_path)
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
    finally:
        concat_file.unlink()

    return output_path
