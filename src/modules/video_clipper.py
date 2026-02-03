"""
Video clipping module with smart framing.
Handles conversion from horizontal to vertical format with face tracking.
"""

import json
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import math

from ..core.config import Config, ClippingConfig, PlatformPreset
from ..core.logger import get_logger
from ..utils.file_utils import ensure_dir, get_safe_filename
from ..utils.video_utils import get_video_info, VideoInfo
from .highlight_detector import Highlight

logger = get_logger(__name__)


@dataclass
class FacePosition:
    """Face detection result for a frame."""
    x: int  # Center X
    y: int  # Center Y
    width: int
    height: int
    confidence: float
    timestamp: float


@dataclass
class CropRegion:
    """Crop region for vertical video conversion."""
    x: int  # Left edge
    y: int  # Top edge
    width: int
    height: int

    def to_ffmpeg_filter(self) -> str:
        return f"crop={self.width}:{self.height}:{self.x}:{self.y}"


@dataclass
class ClipResult:
    """Result of video clip generation."""
    path: Path
    highlight: Highlight
    duration: float
    width: int
    height: int
    file_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'path': str(self.path),
            'start': self.highlight.start,
            'end': self.highlight.end,
            'duration': self.duration,
            'width': self.width,
            'height': self.height,
            'file_size': self.file_size,
            'reasons': [r.value for r in self.highlight.reasons],
            'score': self.highlight.score,
            'metadata': self.metadata
        }


class VideoClipper:
    """
    Creates short-form vertical video clips from horizontal source.

    Features:
    - Smart framing with face detection
    - Smooth camera tracking
    - Platform-specific output presets
    - Quality optimization
    """

    def __init__(self, config: Config):
        """
        Initialize the video clipper.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.clip_config = config.clipping
        self.platform = config.platform_preset
        self.temp_dir = ensure_dir(config.temp_dir / "clips")
        self.output_dir = ensure_dir(config.output_dir)

        self._face_detector = None

    def create_clips(
        self,
        source_path: Path,
        highlights: List[Highlight],
        output_prefix: str = "clip"
    ) -> List[ClipResult]:
        """
        Create video clips from highlights.

        Args:
            source_path: Source video path
            highlights: List of highlights to clip
            output_prefix: Prefix for output filenames

        Returns:
            List of ClipResult objects
        """
        source_path = Path(source_path)
        video_info = get_video_info(source_path)

        logger.info(f"Creating {len(highlights)} clips from {source_path.name}")

        # Detect faces throughout video if smart framing enabled
        face_data = None
        if self.clip_config.smart_framing and self.clip_config.face_detection:
            face_data = self._detect_faces_for_highlights(source_path, highlights, video_info)

        results = []
        for i, highlight in enumerate(highlights):
            try:
                clip_path = self._create_single_clip(
                    source_path=source_path,
                    highlight=highlight,
                    video_info=video_info,
                    face_data=face_data,
                    output_name=f"{output_prefix}_{i + 1:02d}",
                    clip_index=i
                )

                if clip_path and clip_path.exists():
                    clip_info = get_video_info(clip_path)
                    result = ClipResult(
                        path=clip_path,
                        highlight=highlight,
                        duration=clip_info.duration,
                        width=clip_info.width,
                        height=clip_info.height,
                        file_size=clip_path.stat().st_size,
                        metadata={'index': i, 'source': str(source_path)}
                    )
                    results.append(result)
                    logger.info(f"Created clip {i + 1}: {clip_path.name} "
                               f"({result.duration:.1f}s, {result.file_size // 1024}KB)")

            except Exception as e:
                logger.error(f"Failed to create clip {i + 1}: {e}")

        return results

    def _create_single_clip(
        self,
        source_path: Path,
        highlight: Highlight,
        video_info: VideoInfo,
        face_data: Optional[Dict],
        output_name: str,
        clip_index: int
    ) -> Path:
        """Create a single video clip."""
        output_path = self.output_dir / f"{output_name}.{self.clip_config.output_format}"

        # Calculate clip timing with padding
        start_time = max(0, highlight.start - self.clip_config.clip_padding_start)
        end_time = min(video_info.duration, highlight.end + self.clip_config.clip_padding_end)
        duration = end_time - start_time

        # Determine crop region
        crop_region = self._calculate_crop_region(
            video_info=video_info,
            face_data=face_data,
            start_time=start_time,
            end_time=end_time
        )

        # Build FFmpeg command
        cmd = self._build_ffmpeg_command(
            source_path=source_path,
            output_path=output_path,
            start_time=start_time,
            duration=duration,
            crop_region=crop_region,
            video_info=video_info
        )

        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            raise RuntimeError(f"Failed to create clip: {e.stderr}")

        return output_path

    def _calculate_crop_region(
        self,
        video_info: VideoInfo,
        face_data: Optional[Dict],
        start_time: float,
        end_time: float
    ) -> CropRegion:
        """
        Calculate the crop region for vertical video.

        Args:
            video_info: Source video information
            face_data: Face detection data
            start_time: Clip start time
            end_time: Clip end time

        Returns:
            CropRegion for FFmpeg
        """
        target_width = self.platform.width
        target_height = self.platform.height
        target_aspect = target_width / target_height  # 9:16 = 0.5625

        source_width = video_info.width
        source_height = video_info.height

        # Calculate crop dimensions to achieve target aspect ratio
        # We crop the source to match 9:16, then scale if needed
        crop_height = source_height
        crop_width = int(source_height * target_aspect)

        if crop_width > source_width:
            # Source is narrower than target, crop vertically instead
            crop_width = source_width
            crop_height = int(source_width / target_aspect)

        # Determine crop position
        if face_data and self.clip_config.face_detection:
            # Use face tracking to center crop
            center_x = self._get_average_face_position(face_data, start_time, end_time)
            if center_x is None:
                center_x = source_width // 2
        else:
            # Fallback to configured crop position
            if self.clip_config.fallback_crop == "center":
                center_x = source_width // 2
            elif self.clip_config.fallback_crop == "left":
                center_x = crop_width // 2
            else:  # right
                center_x = source_width - crop_width // 2

        # Calculate crop X position (centered on face/target)
        crop_x = center_x - crop_width // 2
        crop_x = max(0, min(crop_x, source_width - crop_width))

        # Y position (usually centered or slightly top)
        crop_y = max(0, (source_height - crop_height) // 2)

        return CropRegion(
            x=crop_x,
            y=crop_y,
            width=crop_width,
            height=crop_height
        )

    def _get_average_face_position(
        self,
        face_data: Dict,
        start_time: float,
        end_time: float
    ) -> Optional[int]:
        """Get average face X position in time range."""
        if not face_data or 'faces' not in face_data:
            return None

        relevant_faces = [
            f for f in face_data['faces']
            if start_time <= f['timestamp'] <= end_time
        ]

        if not relevant_faces:
            return None

        avg_x = sum(f['x'] for f in relevant_faces) / len(relevant_faces)
        return int(avg_x)

    def _build_ffmpeg_command(
        self,
        source_path: Path,
        output_path: Path,
        start_time: float,
        duration: float,
        crop_region: CropRegion,
        video_info: VideoInfo
    ) -> List[str]:
        """Build FFmpeg command for clip creation."""
        target_width = self.platform.width
        target_height = self.platform.height

        # Build filter graph
        filters = []

        # 1. Crop to vertical aspect ratio
        filters.append(crop_region.to_ffmpeg_filter())

        # 2. Scale to target resolution
        filters.append(f"scale={target_width}:{target_height}")

        # 3. Set frame rate
        if video_info.fps != self.platform.fps:
            filters.append(f"fps={self.platform.fps}")

        filter_string = ','.join(filters)

        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-ss', str(start_time),
            '-i', str(source_path),
            '-t', str(duration),
            '-vf', filter_string,
            '-c:v', self.clip_config.codec,
            '-preset', self.clip_config.preset,
            '-crf', str(self.clip_config.crf),
            '-c:a', self.clip_config.audio_codec,
            '-b:a', self.platform.audio_bitrate,
            '-movflags', '+faststart',  # Web optimization
            str(output_path)
        ]

        return cmd

    def _detect_faces_for_highlights(
        self,
        video_path: Path,
        highlights: List[Highlight],
        video_info: VideoInfo
    ) -> Dict:
        """
        Detect faces in relevant portions of video.

        Args:
            video_path: Source video path
            highlights: Highlights to analyze
            video_info: Video metadata

        Returns:
            Dictionary with face detection data
        """
        logger.info("Detecting faces for smart framing")

        # Determine which time ranges need face detection
        time_ranges = []
        for h in highlights:
            start = max(0, h.start - self.clip_config.clip_padding_start)
            end = min(video_info.duration, h.end + self.clip_config.clip_padding_end)
            time_ranges.append((start, end))

        # Merge overlapping ranges
        time_ranges = self._merge_time_ranges(time_ranges)

        faces = []

        try:
            # Try using OpenCV for face detection
            faces = self._detect_faces_opencv(video_path, time_ranges, video_info)
        except ImportError:
            logger.warning("OpenCV not available, using center crop fallback")
        except Exception as e:
            logger.warning(f"Face detection failed: {e}, using center crop")

        return {'faces': faces}

    def _detect_faces_opencv(
        self,
        video_path: Path,
        time_ranges: List[Tuple[float, float]],
        video_info: VideoInfo
    ) -> List[Dict]:
        """Detect faces using OpenCV."""
        import cv2

        # Load face detector
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)

        cap = cv2.VideoCapture(str(video_path))
        faces = []

        fps = video_info.fps
        sample_interval = 1.0  # Sample every second

        for start_time, end_time in time_ranges:
            current_time = start_time

            while current_time < end_time:
                frame_num = int(current_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to grayscale for detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces
                detected = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(50, 50)
                )

                for (x, y, w, h) in detected:
                    faces.append({
                        'x': x + w // 2,  # Center X
                        'y': y + h // 2,  # Center Y
                        'width': w,
                        'height': h,
                        'timestamp': current_time,
                        'confidence': 1.0
                    })

                current_time += sample_interval

        cap.release()

        logger.debug(f"Detected {len(faces)} face instances")
        return faces

    def _merge_time_ranges(
        self,
        ranges: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Merge overlapping time ranges."""
        if not ranges:
            return []

        sorted_ranges = sorted(ranges, key=lambda x: x[0])
        merged = [sorted_ranges[0]]

        for start, end in sorted_ranges[1:]:
            last_start, last_end = merged[-1]

            if start <= last_end:
                # Overlapping, merge
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))

        return merged

    def create_preview(
        self,
        source_path: Path,
        highlight: Highlight,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Create a quick preview (lower quality) of a clip.

        Args:
            source_path: Source video path
            highlight: Highlight to preview
            output_path: Optional output path

        Returns:
            Path to preview video
        """
        if output_path is None:
            output_path = self.temp_dir / f"preview_{highlight.start:.0f}.mp4"

        video_info = get_video_info(source_path)
        crop_region = self._calculate_crop_region(
            video_info=video_info,
            face_data=None,
            start_time=highlight.start,
            end_time=highlight.end
        )

        # Quick preview settings
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(highlight.start),
            '-i', str(source_path),
            '-t', str(highlight.duration),
            '-vf', f"{crop_region.to_ffmpeg_filter()},scale=540:960",
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '28',
            '-c:a', 'aac',
            '-b:a', '128k',
            str(output_path)
        ]

        subprocess.run(cmd, capture_output=True, check=True)
        return output_path

    def create_thumbnail(
        self,
        video_path: Path,
        time_offset: float = 0.5,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Create a thumbnail image from video.

        Args:
            video_path: Video file path
            time_offset: Time offset (0-1, percentage into video)
            output_path: Optional output path

        Returns:
            Path to thumbnail image
        """
        video_info = get_video_info(video_path)
        timestamp = video_info.duration * time_offset

        if output_path is None:
            output_path = video_path.with_suffix('.jpg')

        cmd = [
            'ffmpeg', '-y',
            '-ss', str(timestamp),
            '-i', str(video_path),
            '-vframes', '1',
            '-q:v', '2',
            str(output_path)
        ]

        subprocess.run(cmd, capture_output=True, check=True)
        return output_path

    def export_clips_metadata(
        self,
        clips: List[ClipResult],
        output_path: Path
    ) -> Path:
        """
        Export clip metadata for automation workflows.

        Args:
            clips: List of clip results
            output_path: Output JSON path

        Returns:
            Path to metadata file
        """
        metadata = {
            'clips': [clip.to_dict() for clip in clips],
            'total_clips': len(clips),
            'platform': self.platform.name,
            'resolution': f"{self.platform.width}x{self.platform.height}",
        }

        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return output_path

    def concatenate_clips(
        self,
        clip_paths: List[Path],
        output_prefix: str = "combined"
    ) -> Path:
        """
        Concatenate multiple video clips into a single video.

        Args:
            clip_paths: List of video file paths to concatenate
            output_prefix: Prefix for output filename

        Returns:
            Path to concatenated video
        """
        if not clip_paths:
            raise ValueError("At least one clip path is required")

        if len(clip_paths) == 1:
            return clip_paths[0]

        output_path = self.output_dir / f"{output_prefix}.{self.clip_config.output_format}"

        # Create a temporary file list for FFmpeg concat demuxer
        concat_file = self.temp_dir / "concat_list.txt"

        with open(concat_file, 'w') as f:
            for clip_path in clip_paths:
                # Escape single quotes and write file path
                escaped_path = str(clip_path).replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")

        logger.info(f"Concatenating {len(clip_paths)} clips into {output_path.name}")

        # Use FFmpeg concat demuxer
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file),
            '-c', 'copy',  # Copy streams without re-encoding
            str(output_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            # If copy fails (different codecs), try re-encoding
            logger.warning("Direct copy failed, re-encoding clips")
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c:v', self.clip_config.codec,
                '-preset', self.clip_config.preset,
                '-crf', str(self.clip_config.crf),
                '-c:a', self.clip_config.audio_codec,
                '-b:a', self.platform.audio_bitrate,
                str(output_path)
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Clean up concat file
        try:
            concat_file.unlink()
        except:
            pass

        logger.info(f"Created concatenated video: {output_path.name}")
        return output_path

    def cleanup(self):
        """Clean up temporary files."""
        if not self.config.keep_temp_files:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
