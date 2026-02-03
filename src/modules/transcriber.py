"""
Transcription module using OpenAI Whisper.
Provides high-accuracy speech-to-text with word-level timestamps.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
import time

from ..core.config import Config, TranscriptionConfig, WhisperModel
from ..core.logger import get_logger
from ..utils.file_utils import ensure_dir

logger = get_logger(__name__)


@dataclass
class Word:
    """A single transcribed word with timing."""
    text: str
    start: float
    end: float
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Segment:
    """A transcribed segment (typically a sentence or phrase)."""
    id: int
    text: str
    start: float
    end: float
    words: List[Word] = field(default_factory=list)
    confidence: float = 1.0
    speaker: Optional[str] = None  # For future speaker diarization

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class Transcript:
    """Complete transcript with all segments and metadata."""
    segments: List[Segment]
    text: str  # Full text
    language: str
    duration: float
    word_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_segment_at_time(self, time_seconds: float) -> Optional[Segment]:
        """Get the segment containing the specified time."""
        for segment in self.segments:
            if segment.start <= time_seconds <= segment.end:
                return segment
        return None

    def get_segments_in_range(self, start: float, end: float) -> List[Segment]:
        """Get all segments within a time range."""
        return [
            s for s in self.segments
            if s.start < end and s.end > start
        ]

    def get_text_in_range(self, start: float, end: float) -> str:
        """Get transcript text within a time range."""
        segments = self.get_segments_in_range(start, end)
        return ' '.join(s.text for s in segments)

    def to_dict(self) -> Dict[str, Any]:
        """Convert transcript to dictionary."""
        return {
            'text': self.text,
            'language': self.language,
            'duration': self.duration,
            'word_count': self.word_count,
            'segments': [
                {
                    'id': s.id,
                    'text': s.text,
                    'start': s.start,
                    'end': s.end,
                    'words': [
                        {'text': w.text, 'start': w.start, 'end': w.end, 'confidence': w.confidence}
                        for w in s.words
                    ]
                }
                for s in self.segments
            ],
            'metadata': self.metadata
        }

    def save(self, path: Path):
        """Save transcript to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> 'Transcript':
        """Load transcript from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        segments = []
        for seg_data in data['segments']:
            words = [
                Word(
                    text=w['text'],
                    start=w['start'],
                    end=w['end'],
                    confidence=w.get('confidence', 1.0)
                )
                for w in seg_data.get('words', [])
            ]
            segments.append(Segment(
                id=seg_data['id'],
                text=seg_data['text'],
                start=seg_data['start'],
                end=seg_data['end'],
                words=words
            ))

        return cls(
            segments=segments,
            text=data['text'],
            language=data['language'],
            duration=data['duration'],
            word_count=data['word_count'],
            metadata=data.get('metadata', {})
        )


class Transcriber:
    """
    Transcription engine using OpenAI Whisper.

    Supports multiple Whisper backends:
    - faster-whisper (recommended): Fast inference with CTranslate2
    - openai-whisper: Original implementation
    - whisper.cpp: Lightweight C++ implementation
    """

    def __init__(self, config: Config):
        """
        Initialize the transcriber.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.trans_config = config.transcription
        self.temp_dir = ensure_dir(config.temp_dir / "transcription")
        self.model = None
        self._backend = None

    def _load_model(self):
        """Load the Whisper model (lazy loading)."""
        if self.model is not None:
            return

        model_size = self.trans_config.model.value
        logger.info(f"Loading Whisper model: {model_size}")

        # Try faster-whisper first (recommended)
        try:
            from faster_whisper import WhisperModel

            # Determine compute type
            compute_type = self.trans_config.compute_type
            if compute_type == "auto":
                compute_type = "float16" if self.config.gpu_enabled else "int8"

            # Determine device
            device = self.trans_config.device
            if device == "auto":
                device = "cuda" if self.config.gpu_enabled else "cpu"

            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type
            )
            self._backend = "faster-whisper"
            logger.info(f"Loaded faster-whisper model on {device}")
            return

        except ImportError:
            logger.debug("faster-whisper not available, trying alternatives")

        # Try original whisper
        try:
            import whisper

            device = "cuda" if self.config.gpu_enabled else "cpu"
            self.model = whisper.load_model(model_size, device=device)
            self._backend = "openai-whisper"
            logger.info(f"Loaded openai-whisper model on {device}")
            return

        except ImportError:
            logger.debug("openai-whisper not available")

        raise ImportError(
            "No Whisper backend available. Please install one of:\n"
            "  pip install faster-whisper  (recommended)\n"
            "  pip install openai-whisper"
        )

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None
    ) -> Transcript:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to audio file
            language: Language code (None for auto-detection)

        Returns:
            Transcript object with segments and word timings
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._load_model()

        logger.info(f"Transcribing: {audio_path.name}")
        start_time = time.time()

        if self._backend == "faster-whisper":
            transcript = self._transcribe_faster_whisper(audio_path, language)
        else:
            transcript = self._transcribe_openai_whisper(audio_path, language)

        elapsed = time.time() - start_time
        logger.info(f"Transcription complete in {elapsed:.1f}s "
                   f"({transcript.duration / elapsed:.1f}x realtime)")

        return transcript

    def _transcribe_faster_whisper(
        self,
        audio_path: Path,
        language: Optional[str]
    ) -> Transcript:
        """Transcribe using faster-whisper backend."""
        segments_gen, info = self.model.transcribe(
            str(audio_path),
            language=language or self.trans_config.language,
            task=self.trans_config.task,
            word_timestamps=self.trans_config.word_timestamps,
            vad_filter=self.trans_config.vad_filter,
            beam_size=5
        )

        segments = []
        full_text_parts = []

        for i, seg in enumerate(segments_gen):
            words = []
            if hasattr(seg, 'words') and seg.words:
                for word_info in seg.words:
                    words.append(Word(
                        text=word_info.word.strip(),
                        start=word_info.start,
                        end=word_info.end,
                        confidence=word_info.probability if hasattr(word_info, 'probability') else 1.0
                    ))

            segment = Segment(
                id=i,
                text=seg.text.strip(),
                start=seg.start,
                end=seg.end,
                words=words
            )
            segments.append(segment)
            full_text_parts.append(seg.text.strip())

        full_text = ' '.join(full_text_parts)

        return Transcript(
            segments=segments,
            text=full_text,
            language=info.language,
            duration=info.duration,
            word_count=len(full_text.split()),
            metadata={
                'backend': 'faster-whisper',
                'model': self.trans_config.model.value,
                'language_probability': info.language_probability
            }
        )

    def _transcribe_openai_whisper(
        self,
        audio_path: Path,
        language: Optional[str]
    ) -> Transcript:
        """Transcribe using openai-whisper backend."""
        result = self.model.transcribe(
            str(audio_path),
            language=language or self.trans_config.language,
            task=self.trans_config.task,
            word_timestamps=self.trans_config.word_timestamps,
            verbose=False
        )

        segments = []
        for i, seg in enumerate(result['segments']):
            words = []
            if 'words' in seg:
                for word_info in seg['words']:
                    words.append(Word(
                        text=word_info['word'].strip(),
                        start=word_info['start'],
                        end=word_info['end'],
                        confidence=word_info.get('probability', 1.0)
                    ))

            segment = Segment(
                id=i,
                text=seg['text'].strip(),
                start=seg['start'],
                end=seg['end'],
                words=words
            )
            segments.append(segment)

        # Calculate total duration from last segment
        duration = segments[-1].end if segments else 0.0

        return Transcript(
            segments=segments,
            text=result['text'].strip(),
            language=result['language'],
            duration=duration,
            word_count=len(result['text'].split()),
            metadata={
                'backend': 'openai-whisper',
                'model': self.trans_config.model.value
            }
        )

    def transcribe_with_cache(
        self,
        audio_path: Path,
        cache_dir: Optional[Path] = None,
        language: Optional[str] = None
    ) -> Transcript:
        """
        Transcribe with caching to avoid re-processing.

        Args:
            audio_path: Path to audio file
            cache_dir: Directory for cache files
            language: Language code

        Returns:
            Transcript (from cache if available)
        """
        if cache_dir is None:
            cache_dir = self.temp_dir / "cache"
        ensure_dir(cache_dir)

        # Create cache key from file hash and config
        from ..utils.file_utils import get_file_hash
        file_hash = get_file_hash(audio_path)
        cache_key = f"{file_hash}_{self.trans_config.model.value}"
        cache_path = cache_dir / f"{cache_key}.json"

        if cache_path.exists():
            logger.info(f"Loading cached transcript: {cache_path.name}")
            return Transcript.load(cache_path)

        # Transcribe and cache
        transcript = self.transcribe(audio_path, language)
        transcript.save(cache_path)

        return transcript

    def get_word_timings(self, transcript: Transcript) -> List[Dict[str, Any]]:
        """
        Extract flat list of all word timings from transcript.

        Args:
            transcript: Transcript object

        Returns:
            List of word timing dictionaries
        """
        word_timings = []

        for segment in transcript.segments:
            if segment.words:
                for word in segment.words:
                    word_timings.append({
                        'text': word.text,
                        'start': word.start,
                        'end': word.end,
                        'confidence': word.confidence,
                        'segment_id': segment.id
                    })
            else:
                # Estimate word timings if not available
                words = segment.text.split()
                if words:
                    duration_per_word = segment.duration / len(words)
                    for i, word_text in enumerate(words):
                        word_timings.append({
                            'text': word_text,
                            'start': segment.start + i * duration_per_word,
                            'end': segment.start + (i + 1) * duration_per_word,
                            'confidence': segment.confidence,
                            'segment_id': segment.id
                        })

        return word_timings

    def export_srt(self, transcript: Transcript, output_path: Path) -> Path:
        """
        Export transcript as SRT subtitle file.

        Args:
            transcript: Transcript object
            output_path: Output file path

        Returns:
            Path to SRT file
        """
        def format_timestamp(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

        lines = []
        for i, segment in enumerate(transcript.segments, 1):
            lines.append(str(i))
            lines.append(f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}")
            lines.append(segment.text)
            lines.append("")

        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        return output_path

    def export_vtt(self, transcript: Transcript, output_path: Path) -> Path:
        """
        Export transcript as WebVTT subtitle file.

        Args:
            transcript: Transcript object
            output_path: Output file path

        Returns:
            Path to VTT file
        """
        def format_timestamp(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

        lines = ["WEBVTT", ""]

        for segment in transcript.segments:
            lines.append(f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}")
            lines.append(segment.text)
            lines.append("")

        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        return output_path

    def cleanup(self):
        """Clean up temporary files and unload model."""
        self.model = None
        if not self.config.keep_temp_files:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
