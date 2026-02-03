"""
Pipeline orchestrator for the Shorts Bot video processing system.
Coordinates all modules for end-to-end video processing.
"""

import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .config import Config, Platform
from .logger import get_logger, ProgressLogger
from ..utils.file_utils import ensure_dir, get_safe_filename
from ..modules.input_handler import InputHandler, InputFile, MediaType
from ..modules.transcriber import Transcriber, Transcript
from ..modules.highlight_detector import HighlightDetector, HighlightResult, Highlight
from ..modules.video_clipper import VideoClipper, ClipResult
from ..modules.caption_generator import CaptionGenerator, CaptionSequence

logger = get_logger(__name__)


class PipelineStage(Enum):
    """Pipeline processing stages."""
    INPUT = "input"
    TRANSCRIPTION = "transcription"
    HIGHLIGHT_DETECTION = "highlight_detection"
    CLIPPING = "clipping"
    CAPTIONING = "captioning"
    EXPORT = "export"
    COMPLETE = "complete"


@dataclass
class ProcessingResult:
    """Result of processing a single video."""
    input_file: InputFile
    transcript: Optional[Transcript] = None
    highlights: Optional[HighlightResult] = None
    clips: List[ClipResult] = field(default_factory=list)
    output_dir: Optional[Path] = None
    success: bool = True
    error: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON export."""
        return {
            'source': str(self.input_file.path),
            'success': self.success,
            'error': self.error,
            'processing_time': self.processing_time,
            'transcript': {
                'word_count': self.transcript.word_count if self.transcript else 0,
                'duration': self.transcript.duration if self.transcript else 0,
                'language': self.transcript.language if self.transcript else None,
            } if self.transcript else None,
            'highlights': {
                'count': len(self.highlights.highlights) if self.highlights else 0,
                'coverage': self.highlights.coverage if self.highlights else 0,
            } if self.highlights else None,
            'clips': [clip.to_dict() for clip in self.clips],
            'output_dir': str(self.output_dir) if self.output_dir else None,
            'metadata': self.metadata
        }

    def save(self, path: Path):
        """Save result to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class BatchResult:
    """Result of batch processing multiple videos."""
    results: List[ProcessingResult]
    total_time: float
    successful: int
    failed: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_videos': len(self.results),
            'successful': self.successful,
            'failed': self.failed,
            'total_time': self.total_time,
            'results': [r.to_dict() for r in self.results]
        }


class Pipeline:
    """
    Main orchestrator for the video processing pipeline.

    Coordinates the flow from input to output:
    1. Input handling and validation
    2. Audio extraction and transcription
    3. Highlight detection
    4. Video clipping with smart framing
    5. Caption generation and burning
    6. Export and cleanup
    """

    def __init__(
        self,
        config: Config,
        progress_callback: Optional[Callable[[PipelineStage, float, str], None]] = None
    ):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration
            progress_callback: Optional callback for progress updates
                              (stage, progress 0-1, message)
        """
        self.config = config
        self.progress_callback = progress_callback

        # Ensure directories exist
        ensure_dir(config.input_dir)
        ensure_dir(config.output_dir)
        ensure_dir(config.temp_dir)

        # Initialize modules (lazy loading where possible)
        self.input_handler = InputHandler(config)
        self.transcriber = Transcriber(config)
        self.highlight_detector = HighlightDetector(config)
        self.video_clipper = VideoClipper(config)
        self.caption_generator = CaptionGenerator(config)

        self._current_stage = PipelineStage.INPUT

    def _update_progress(self, stage: PipelineStage, progress: float, message: str):
        """Update progress via callback if available."""
        self._current_stage = stage
        if self.progress_callback:
            self.progress_callback(stage, progress, message)
        logger.info(f"[{stage.value}] {message}")

    def process_file(
        self,
        file_path: Path,
        output_name: Optional[str] = None
    ) -> ProcessingResult:
        """
        Process a single video file through the entire pipeline.

        Args:
            file_path: Path to input video/audio file
            output_name: Optional custom name for outputs

        Returns:
            ProcessingResult with all outputs and metadata
        """
        file_path = Path(file_path)
        start_time = time.time()

        if output_name is None:
            output_name = get_safe_filename(file_path.stem)

        # Create output directory for this video
        video_output_dir = ensure_dir(self.config.output_dir / output_name)

        result = ProcessingResult(
            input_file=InputFile(
                path=file_path,
                media_type=MediaType.UNKNOWN,
                video_info=None
            ),
            output_dir=video_output_dir
        )

        try:
            # Stage 1: Input handling
            self._update_progress(PipelineStage.INPUT, 0.0, f"Processing input: {file_path.name}")
            input_file = self.input_handler.process_file(file_path)
            result.input_file = input_file

            # Validate input
            issues = self.input_handler.validate_input(input_file)
            if any("File does not exist" in i or "too short" in i for i in issues):
                raise ValueError(f"Input validation failed: {'; '.join(issues)}")

            for issue in issues:
                logger.warning(f"Input warning: {issue}")

            self._update_progress(PipelineStage.INPUT, 1.0, "Input processed successfully")

            # Stage 2: Transcription
            self._update_progress(PipelineStage.TRANSCRIPTION, 0.0, "Starting transcription")

            audio_path = input_file.audio_path
            if audio_path is None:
                raise RuntimeError("Audio extraction failed")

            transcript = self.transcriber.transcribe_with_cache(audio_path)
            result.transcript = transcript

            # Save transcript
            transcript.save(video_output_dir / f"{output_name}_transcript.json")
            self.transcriber.export_srt(transcript, video_output_dir / f"{output_name}.srt")

            self._update_progress(
                PipelineStage.TRANSCRIPTION, 1.0,
                f"Transcription complete: {transcript.word_count} words"
            )

            # Stage 3: Highlight Detection
            self._update_progress(PipelineStage.HIGHLIGHT_DETECTION, 0.0, "Detecting highlights")

            highlights = self.highlight_detector.detect_highlights(
                transcript,
                audio_path=audio_path
            )
            result.highlights = highlights

            # Save highlights
            highlights.save(video_output_dir / f"{output_name}_highlights.json")

            self._update_progress(
                PipelineStage.HIGHLIGHT_DETECTION, 1.0,
                f"Found {len(highlights.highlights)} potential clips"
            )

            # Stage 4: Video Clipping
            self._update_progress(PipelineStage.CLIPPING, 0.0, "Creating video clips")

            source_video = input_file.path
            if input_file.media_type == MediaType.AUDIO:
                # For audio-only, skip clipping or create visualized audio
                logger.warning("Audio-only input - skipping video clipping")
                clips = []
            else:
                # Get top highlights for clipping
                top_highlights = highlights.get_top_highlights(self.config.highlight.target_clips)

                clips = self.video_clipper.create_clips(
                    source_path=source_video,
                    highlights=top_highlights,
                    output_prefix=output_name
                )

            result.clips = clips
            self._update_progress(
                PipelineStage.CLIPPING, 1.0,
                f"Created {len(clips)} video clips"
            )

            # Stage 5: Captioning
            self._update_progress(PipelineStage.CAPTIONING, 0.0, "Adding captions")

            captioned_clips = []
            for i, clip in enumerate(clips):
                self._update_progress(
                    PipelineStage.CAPTIONING,
                    i / len(clips),
                    f"Captioning clip {i + 1}/{len(clips)}"
                )

                # Generate captions for this clip's time range
                clip_captions = self.caption_generator.generate_captions(
                    transcript=transcript,
                    clip_start=clip.highlight.start,
                    clip_end=clip.highlight.end
                )

                # Add emoji emphasis if enabled
                if self.config.caption.add_emoji:
                    clip_captions = self.caption_generator.add_emoji_emphasis(clip_captions)

                # Burn captions into video
                captioned_path = self.caption_generator.burn_captions(
                    video_path=clip.path,
                    captions=clip_captions,
                    output_path=video_output_dir / f"{clip.path.stem}_captioned.mp4"
                )

                # Update clip with captioned path
                clip.path = captioned_path
                clip.metadata['has_captions'] = True
                captioned_clips.append(clip)

                # Export caption data
                self.caption_generator.export_json(
                    clip_captions,
                    video_output_dir / f"{clip.path.stem}_captions.json"
                )

            result.clips = captioned_clips
            self._update_progress(PipelineStage.CAPTIONING, 1.0, "Captions added")

            # Stage 6: Export
            self._update_progress(PipelineStage.EXPORT, 0.0, "Exporting metadata")

            # Export clips metadata
            self.video_clipper.export_clips_metadata(
                clips=captioned_clips,
                output_path=video_output_dir / f"{output_name}_clips.json"
            )

            # Create thumbnails
            for clip in captioned_clips:
                self.video_clipper.create_thumbnail(
                    clip.path,
                    output_path=clip.path.with_suffix('.jpg')
                )

            # Export EDL for video editors
            if highlights.highlights:
                self.highlight_detector.export_edl(
                    highlights.highlights,
                    video_output_dir / f"{output_name}.edl"
                )

            self._update_progress(PipelineStage.EXPORT, 1.0, "Export complete")

            # Mark as complete
            result.success = True
            result.processing_time = time.time() - start_time
            result.metadata = {
                'processed_at': datetime.now().isoformat(),
                'platform': self.config.platform.value,
                'model': self.config.transcription.model.value,
            }

            self._update_progress(
                PipelineStage.COMPLETE, 1.0,
                f"Processing complete in {result.processing_time:.1f}s"
            )

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            result.success = False
            result.error = str(e)
            result.processing_time = time.time() - start_time

        # Save processing result
        result.save(video_output_dir / f"{output_name}_result.json")

        return result

    def process_batch(
        self,
        input_paths: List[Path],
        parallel: bool = False
    ) -> BatchResult:
        """
        Process multiple videos.

        Args:
            input_paths: List of input file paths
            parallel: Whether to process in parallel (not yet implemented)

        Returns:
            BatchResult with all processing results
        """
        start_time = time.time()
        results = []
        successful = 0
        failed = 0

        logger.info(f"Starting batch processing of {len(input_paths)} files")

        for i, path in enumerate(input_paths):
            logger.info(f"\nProcessing {i + 1}/{len(input_paths)}: {path.name}")

            try:
                result = self.process_file(path)
                results.append(result)

                if result.success:
                    successful += 1
                else:
                    failed += 1

            except Exception as e:
                logger.error(f"Failed to process {path.name}: {e}")
                failed += 1
                results.append(ProcessingResult(
                    input_file=InputFile(
                        path=path,
                        media_type=MediaType.UNKNOWN,
                        video_info=None
                    ),
                    success=False,
                    error=str(e)
                ))

        total_time = time.time() - start_time

        batch_result = BatchResult(
            results=results,
            total_time=total_time,
            successful=successful,
            failed=failed
        )

        logger.info(f"\nBatch complete: {successful} successful, {failed} failed, "
                   f"total time: {total_time:.1f}s")

        return batch_result

    def process_directory(
        self,
        directory: Path,
        recursive: bool = False
    ) -> BatchResult:
        """
        Process all media files in a directory.

        Args:
            directory: Directory to process
            recursive: Whether to scan subdirectories

        Returns:
            BatchResult with all processing results
        """
        batch_input = self.input_handler.process_directory(directory, recursive)
        input_paths = [f.path for f in batch_input.files]
        return self.process_batch(input_paths)

    def preview_highlights(
        self,
        file_path: Path,
        quick_mode: bool = True
    ) -> List[Highlight]:
        """
        Generate a quick preview of highlights without full processing.

        Args:
            file_path: Path to video file
            quick_mode: Use faster settings for preview

        Returns:
            List of detected highlights
        """
        # Use faster settings for preview
        original_model = self.config.transcription.model
        if quick_mode:
            from .config import WhisperModel
            self.config.transcription.model = WhisperModel.TINY

        try:
            input_file = self.input_handler.process_file(file_path)

            if input_file.audio_path is None:
                raise RuntimeError("Audio extraction failed")

            transcript = self.transcriber.transcribe(input_file.audio_path)
            highlights = self.highlight_detector.detect_highlights(transcript)

            return highlights.highlights

        finally:
            # Restore original model
            self.config.transcription.model = original_model

    def create_single_clip(
        self,
        file_path: Path,
        start_time: float,
        end_time: float,
        output_path: Optional[Path] = None,
        add_captions: bool = True
    ) -> ClipResult:
        """
        Create a single clip with manual time selection.

        Args:
            file_path: Source video path
            start_time: Clip start time in seconds
            end_time: Clip end time in seconds
            output_path: Optional output path
            add_captions: Whether to add captions

        Returns:
            ClipResult
        """
        input_file = self.input_handler.process_file(file_path)

        # Create manual highlight
        highlight = Highlight(
            start=start_time,
            end=end_time,
            text="",
            score=1.0,
            reasons=[]
        )

        # Create clip
        clips = self.video_clipper.create_clips(
            source_path=file_path,
            highlights=[highlight],
            output_prefix="manual_clip"
        )

        if not clips:
            raise RuntimeError("Failed to create clip")

        clip = clips[0]

        # Add captions if requested
        if add_captions and input_file.audio_path:
            transcript = self.transcriber.transcribe(input_file.audio_path)

            captions = self.caption_generator.generate_captions(
                transcript=transcript,
                clip_start=start_time,
                clip_end=end_time
            )

            captioned_path = self.caption_generator.burn_captions(
                video_path=clip.path,
                captions=captions,
                output_path=output_path
            )
            clip.path = captioned_path

        return clip

    def create_multi_segment_clip(
        self,
        file_path: Path,
        segments: List[Dict[str, float]],
        output_path: Optional[Path] = None,
        add_captions: bool = True
    ) -> ClipResult:
        """
        Create a single clip from multiple segments concatenated together.

        Args:
            file_path: Source video path
            segments: List of dicts with 'start' and 'end' times in seconds
            output_path: Optional output path
            add_captions: Whether to add captions

        Returns:
            ClipResult with concatenated video
        """
        if not segments:
            raise ValueError("At least one segment is required")

        input_file = self.input_handler.process_file(file_path)

        # Get transcript if we need captions
        transcript = None
        if add_captions and input_file.audio_path:
            transcript = self.transcriber.transcribe(input_file.audio_path)

        # Create individual clips for each segment
        temp_clips = []
        for i, seg in enumerate(segments):
            highlight = Highlight(
                start=seg['start'],
                end=seg['end'],
                text="",
                score=1.0,
                reasons=[]
            )

            clips = self.video_clipper.create_clips(
                source_path=file_path,
                highlights=[highlight],
                output_prefix=f"segment_{i}"
            )

            if clips:
                temp_clips.append(clips[0])

        if not temp_clips:
            raise RuntimeError("Failed to create any clips")

        # If only one segment, just return that clip (optionally with captions)
        if len(temp_clips) == 1:
            clip = temp_clips[0]
            if add_captions and transcript:
                captions = self.caption_generator.generate_captions(
                    transcript=transcript,
                    clip_start=segments[0]['start'],
                    clip_end=segments[0]['end']
                )
                captioned_path = self.caption_generator.burn_captions(
                    video_path=clip.path,
                    captions=captions,
                    output_path=output_path
                )
                clip.path = captioned_path
            return clip

        # Concatenate multiple clips
        concatenated_path = self.video_clipper.concatenate_clips(
            clip_paths=[c.path for c in temp_clips],
            output_prefix="combined_clip"
        )

        # Calculate total duration
        total_duration = sum(seg['end'] - seg['start'] for seg in segments)

        # Get video info for dimensions
        from ..utils.video_utils import get_video_info
        concat_info = get_video_info(concatenated_path)

        # Create result for concatenated clip
        combined_highlight = Highlight(
            start=0,
            end=total_duration,
            text="Combined from multiple segments",
            score=1.0,
            reasons=[],
            metadata={'segments': segments}
        )

        final_clip = ClipResult(
            path=concatenated_path,
            highlight=combined_highlight,
            duration=total_duration,
            width=concat_info.width,
            height=concat_info.height,
            file_size=concatenated_path.stat().st_size if concatenated_path.exists() else 0,
            metadata={'segment_count': len(segments)}
        )

        # Add captions if requested
        if add_captions and transcript:
            # Generate captions for all segments combined
            all_captions = None
            time_offset = 0
            for seg in segments:
                seg_captions = self.caption_generator.generate_captions(
                    transcript=transcript,
                    clip_start=seg['start'],
                    clip_end=seg['end']
                )
                if all_captions is None:
                    all_captions = seg_captions
                else:
                    # Merge captions with time offset
                    all_captions = self.caption_generator.merge_captions(
                        all_captions, seg_captions, time_offset
                    )
                time_offset += seg['end'] - seg['start']

            if all_captions:
                captioned_path = self.caption_generator.burn_captions(
                    video_path=final_clip.path,
                    captions=all_captions,
                    output_path=output_path
                )
                final_clip.path = captioned_path

        # Clean up temporary segment clips
        for clip in temp_clips:
            try:
                if clip.path.exists():
                    clip.path.unlink()
            except Exception:
                pass

        return final_clip

    def cleanup(self):
        """Clean up all temporary files."""
        logger.info("Cleaning up temporary files")
        self.input_handler.cleanup()
        self.transcriber.cleanup()
        self.video_clipper.cleanup()
        self.caption_generator.cleanup()

    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'current_stage': self._current_stage.value,
            'config': {
                'platform': self.config.platform.value,
                'model': self.config.transcription.model.value,
                'target_clips': self.config.highlight.target_clips,
            }
        }
