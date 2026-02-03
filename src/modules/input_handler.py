"""
Input handler module for processing video and audio files.
Validates inputs, extracts metadata, and prepares files for processing.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Generator
from dataclasses import dataclass, field
from enum import Enum
import json

from ..core.config import Config
from ..core.logger import get_logger
from ..utils.file_utils import (
    find_media_files,
    is_video_file,
    is_audio_file,
    ensure_dir,
    get_file_hash,
    get_safe_filename
)
from ..utils.video_utils import get_video_info, VideoInfo, extract_audio

logger = get_logger(__name__)


class MediaType(Enum):
    """Type of input media."""
    VIDEO = "video"
    AUDIO = "audio"
    UNKNOWN = "unknown"


@dataclass
class InputFile:
    """Represents a validated input file ready for processing."""
    path: Path
    media_type: MediaType
    video_info: Optional[VideoInfo]
    audio_path: Optional[Path] = None  # Extracted audio for processing
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_hash: Optional[str] = None

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def duration(self) -> float:
        if self.video_info:
            return self.video_info.duration
        return 0.0

    @property
    def is_valid(self) -> bool:
        return self.media_type != MediaType.UNKNOWN and self.path.exists()


@dataclass
class BatchInput:
    """Container for batch processing multiple files."""
    files: List[InputFile]
    total_duration: float = 0.0
    processed_count: int = 0
    failed_count: int = 0

    def __post_init__(self):
        self.total_duration = sum(f.duration for f in self.files)

    def __iter__(self) -> Generator[InputFile, None, None]:
        for f in self.files:
            yield f

    def __len__(self) -> int:
        return len(self.files)


class InputHandler:
    """
    Handles input file validation, metadata extraction, and preparation.

    This module is the entry point for all media files into the pipeline.
    It validates files, extracts necessary metadata, and prepares audio
    for transcription.
    """

    def __init__(self, config: Config):
        """
        Initialize the input handler.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.temp_dir = ensure_dir(config.temp_dir / "input")
        self._processed_hashes: set = set()

    def process_file(
        self,
        file_path: Path,
        extract_audio_immediately: bool = True
    ) -> InputFile:
        """
        Process a single input file.

        Args:
            file_path: Path to media file
            extract_audio_immediately: Whether to extract audio now

        Returns:
            InputFile object with metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        logger.info(f"Processing input file: {file_path.name}")

        # Determine media type
        if is_video_file(file_path):
            media_type = MediaType.VIDEO
        elif is_audio_file(file_path):
            media_type = MediaType.AUDIO
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        # Get video info (works for audio too)
        try:
            video_info = get_video_info(file_path)
            logger.debug(f"Duration: {video_info.duration_formatted}, "
                        f"Resolution: {video_info.width}x{video_info.height}")
        except Exception as e:
            logger.error(f"Failed to extract video info: {e}")
            video_info = None

        # Calculate file hash for deduplication
        file_hash = None
        if self.config.batch_mode:
            file_hash = get_file_hash(file_path)
            if file_hash in self._processed_hashes:
                logger.warning(f"Duplicate file detected (skipping): {file_path.name}")

        # Create InputFile object
        input_file = InputFile(
            path=file_path,
            media_type=media_type,
            video_info=video_info,
            file_hash=file_hash,
            metadata={
                "original_filename": file_path.name,
                "file_size": file_path.stat().st_size,
            }
        )

        # Extract audio if requested (needed for transcription)
        if extract_audio_immediately and media_type == MediaType.VIDEO:
            input_file.audio_path = self._extract_audio(input_file)
        elif media_type == MediaType.AUDIO:
            # Audio file can be used directly, but may need resampling
            input_file.audio_path = self._prepare_audio(file_path)

        return input_file

    def process_directory(
        self,
        directory: Path,
        recursive: bool = False
    ) -> BatchInput:
        """
        Process all media files in a directory.

        Args:
            directory: Directory to scan
            recursive: Whether to scan subdirectories

        Returns:
            BatchInput containing all valid files
        """
        directory = Path(directory)
        logger.info(f"Scanning directory: {directory}")

        media_files = find_media_files(
            directory,
            include_video=True,
            include_audio=True,
            recursive=recursive
        )

        logger.info(f"Found {len(media_files)} media files")

        input_files = []
        for file_path in media_files:
            try:
                input_file = self.process_file(file_path, extract_audio_immediately=False)
                input_files.append(input_file)
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")

        batch = BatchInput(files=input_files)
        logger.info(f"Batch prepared: {len(batch)} files, "
                   f"total duration: {batch.total_duration / 60:.1f} minutes")

        return batch

    def process_urls(self, urls: List[str]) -> BatchInput:
        """
        Download and process videos from URLs.
        Uses yt-dlp for downloading (must be installed).

        Args:
            urls: List of video URLs

        Returns:
            BatchInput containing downloaded files
        """
        import subprocess

        input_files = []
        download_dir = ensure_dir(self.temp_dir / "downloads")

        for url in urls:
            try:
                logger.info(f"Downloading: {url}")

                # Use yt-dlp to download
                output_template = str(download_dir / "%(title)s.%(ext)s")
                cmd = [
                    'yt-dlp',
                    '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                    '-o', output_template,
                    '--no-playlist',
                    url
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    logger.error(f"Download failed: {result.stderr}")
                    continue

                # Find the downloaded file
                for line in result.stdout.split('\n'):
                    if '[download] Destination:' in line:
                        file_path = Path(line.split('Destination:')[1].strip())
                        break
                    elif '[download]' in line and 'has already been downloaded' in line:
                        # Extract path from "already downloaded" message
                        file_path = Path(line.split('[download]')[1].split('has already')[0].strip())
                        break
                else:
                    logger.warning(f"Could not find downloaded file for {url}")
                    continue

                input_file = self.process_file(file_path)
                input_file.metadata['source_url'] = url
                input_files.append(input_file)

            except Exception as e:
                logger.error(f"Failed to download {url}: {e}")

        return BatchInput(files=input_files)

    def _extract_audio(self, input_file: InputFile) -> Path:
        """
        Extract audio from video file for transcription.

        Args:
            input_file: Input file to extract audio from

        Returns:
            Path to extracted audio file
        """
        safe_name = get_safe_filename(input_file.name)
        audio_path = self.temp_dir / f"{safe_name}_audio.wav"

        if audio_path.exists() and not self.config.overwrite_existing:
            logger.debug(f"Using cached audio: {audio_path}")
            return audio_path

        logger.info(f"Extracting audio from: {input_file.name}")

        return extract_audio(
            video_path=input_file.path,
            output_path=audio_path,
            sample_rate=16000,  # Whisper optimal sample rate
            mono=True
        )

    def _prepare_audio(self, audio_path: Path) -> Path:
        """
        Prepare audio file for transcription (resample if needed).

        Args:
            audio_path: Path to audio file

        Returns:
            Path to prepared audio file
        """
        safe_name = get_safe_filename(audio_path.stem)
        output_path = self.temp_dir / f"{safe_name}_prepared.wav"

        if output_path.exists() and not self.config.overwrite_existing:
            logger.debug(f"Using cached audio: {output_path}")
            return output_path

        logger.info(f"Preparing audio: {audio_path.name}")

        return extract_audio(
            video_path=audio_path,  # Works for audio files too
            output_path=output_path,
            sample_rate=16000,
            mono=True
        )

    def validate_input(self, input_file: InputFile) -> List[str]:
        """
        Validate an input file for processing requirements.

        Args:
            input_file: Input file to validate

        Returns:
            List of validation warnings/errors (empty if valid)
        """
        issues = []

        if not input_file.path.exists():
            issues.append(f"File does not exist: {input_file.path}")
            return issues

        if input_file.video_info:
            # Check duration
            min_duration = self.config.highlight.min_clip_duration
            if input_file.duration < min_duration:
                issues.append(f"Video too short ({input_file.duration:.1f}s < {min_duration}s)")

            # Check if video is excessively long (warning only)
            if input_file.duration > 3600:  # 1 hour
                issues.append(f"Warning: Long video ({input_file.duration / 60:.0f} minutes) may take significant time")

            # Check resolution
            if input_file.media_type == MediaType.VIDEO:
                if input_file.video_info.width < 480 or input_file.video_info.height < 480:
                    issues.append(f"Low resolution video ({input_file.video_info.width}x{input_file.video_info.height})")

        return issues

    def get_processing_estimate(self, input_file: InputFile) -> Dict[str, Any]:
        """
        Estimate processing requirements for a file.

        Args:
            input_file: Input file to estimate

        Returns:
            Dictionary with estimates
        """
        duration = input_file.duration

        # Rough estimates based on typical processing times
        estimates = {
            "transcription_factor": 0.3 if self.config.transcription.model.value in ['tiny', 'base'] else 1.0,
            "estimated_clips": min(
                self.config.highlight.target_clips,
                int(duration / self.config.highlight.min_clip_duration)
            ),
            "input_duration_seconds": duration,
            "is_large_file": duration > 1800,  # 30 minutes
        }

        # Processing time depends on model and hardware
        if self.config.gpu_enabled:
            estimates["transcription_factor"] *= 0.3

        return estimates

    def cleanup(self):
        """Clean up temporary files created by input handler."""
        if not self.config.keep_temp_files:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.debug("Cleaned up input handler temp files")

    def export_metadata(self, input_file: InputFile, output_path: Path):
        """
        Export input file metadata to JSON.

        Args:
            input_file: Input file
            output_path: Path for JSON output
        """
        metadata = {
            "source_file": str(input_file.path),
            "media_type": input_file.media_type.value,
            "duration": input_file.duration,
            "file_hash": input_file.file_hash,
            **input_file.metadata
        }

        if input_file.video_info:
            metadata.update({
                "width": input_file.video_info.width,
                "height": input_file.video_info.height,
                "fps": input_file.video_info.fps,
                "codec": input_file.video_info.codec,
            })

        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
