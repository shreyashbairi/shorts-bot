"""
Caption generation module for burned-in video captions.
Supports multiple styles including viral-style word-by-word highlighting.
"""

import json
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import math
import re

from ..core.config import Config, CaptionConfig, CaptionStyle
from ..core.logger import get_logger
from ..utils.file_utils import ensure_dir
from .transcriber import Transcript, Segment, Word

logger = get_logger(__name__)


@dataclass
class CaptionLine:
    """A single line of caption with timing."""
    text: str
    start: float
    end: float
    words: List[Dict[str, Any]] = field(default_factory=list)  # Word-level timing

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class CaptionSequence:
    """A sequence of captions for a video clip."""
    lines: List[CaptionLine]
    style: CaptionStyle
    duration: float

    def to_ass_dialogue(self, style_name: str = "Default") -> List[str]:
        """Convert to ASS subtitle format dialogue lines."""
        def format_time(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}:{minutes:02d}:{secs:05.2f}"

        dialogues = []
        for line in self.lines:
            start_time = format_time(line.start)
            end_time = format_time(line.end)
            text = line.text.replace('\n', '\\N')
            dialogues.append(
                f"Dialogue: 0,{start_time},{end_time},{style_name},,0,0,0,,{text}"
            )
        return dialogues


class CaptionGenerator:
    """
    Generates styled captions for video clips.

    Features:
    - Multiple caption styles (minimal, bold, viral, karaoke)
    - Word-by-word highlighting
    - Customizable fonts, colors, and animations
    - FFmpeg ASS subtitle burning
    """

    # Style presets
    STYLE_PRESETS = {
        CaptionStyle.MINIMAL: {
            'font_size': 48,
            'font_color': '#FFFFFF',
            'stroke_width': 2,
            'position': 'bottom',
            'highlight_current_word': False,
            'uppercase': False,
            'animation': 'none',
        },
        CaptionStyle.BOLD: {
            'font_size': 64,
            'font_color': '#FFFFFF',
            'stroke_width': 4,
            'position': 'center',
            'highlight_current_word': False,
            'uppercase': True,
            'animation': 'none',
        },
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
        CaptionStyle.SUBTITLE: {
            'font_size': 42,
            'font_color': '#FFFFFF',
            'stroke_width': 2,
            'position': 'bottom',
            'highlight_current_word': False,
            'uppercase': False,
            'animation': 'none',
            'background': True,
        },
        CaptionStyle.KARAOKE: {
            'font_size': 60,
            'font_color': '#FFFFFF',
            'stroke_width': 3,
            'position': 'center',
            'highlight_current_word': True,
            'highlight_color': '#00FF00',
            'uppercase': True,
            'animation': 'fade',
        },
    }

    def __init__(self, config: Config):
        """
        Initialize the caption generator.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.caption_config = config.caption
        self.temp_dir = ensure_dir(config.temp_dir / "captions")

        # Merge preset with user config
        self._apply_style_preset()

    def _apply_style_preset(self):
        """Apply style preset defaults to caption config."""
        preset = self.STYLE_PRESETS.get(self.caption_config.style, {})

        # Apply preset values only if not explicitly set
        # (We can't easily detect "not set" with dataclasses, so we just use preset)
        for key, value in preset.items():
            if hasattr(self.caption_config, key):
                # Preserve user-specified values by not overwriting
                pass

    def generate_captions(
        self,
        transcript: Transcript,
        clip_start: float,
        clip_end: float
    ) -> CaptionSequence:
        """
        Generate captions for a video clip from transcript.

        Args:
            transcript: Full video transcript
            clip_start: Clip start time (in source video)
            clip_end: Clip end time (in source video)

        Returns:
            CaptionSequence ready for rendering
        """
        # Extract relevant segments
        segments = transcript.get_segments_in_range(clip_start, clip_end)

        if not segments:
            logger.warning(f"No transcript segments found for {clip_start:.1f}-{clip_end:.1f}")
            return CaptionSequence(lines=[], style=self.caption_config.style, duration=0)

        # Create caption lines based on style
        if self.caption_config.highlight_current_word:
            lines = self._create_word_highlight_captions(segments, clip_start, clip_end)
        else:
            lines = self._create_sentence_captions(segments, clip_start, clip_end)

        return CaptionSequence(
            lines=lines,
            style=self.caption_config.style,
            duration=clip_end - clip_start
        )

    def _create_sentence_captions(
        self,
        segments: List[Segment],
        clip_start: float,
        clip_end: float
    ) -> List[CaptionLine]:
        """Create traditional sentence-based captions."""
        lines = []

        for segment in segments:
            # Adjust timing relative to clip start
            start = max(0, segment.start - clip_start)
            end = min(clip_end - clip_start, segment.end - clip_start)

            if end <= start:
                continue

            text = segment.text.strip()
            if self.caption_config.uppercase:
                text = text.upper()

            # Split long sentences
            formatted_lines = self._format_caption_text(text)

            for formatted_text in formatted_lines:
                lines.append(CaptionLine(
                    text=formatted_text,
                    start=start,
                    end=end,
                    words=[]
                ))

        return lines

    def _create_word_highlight_captions(
        self,
        segments: List[Segment],
        clip_start: float,
        clip_end: float
    ) -> List[CaptionLine]:
        """Create word-by-word highlight captions (viral style)."""
        lines = []
        max_words = self.caption_config.max_words_per_line

        # Collect all words with timing
        all_words = []
        for segment in segments:
            if segment.words:
                for word in segment.words:
                    if clip_start <= word.start <= clip_end:
                        all_words.append({
                            'text': word.text,
                            'start': word.start - clip_start,
                            'end': word.end - clip_start
                        })
            else:
                # Estimate word timing
                words = segment.text.split()
                if words:
                    duration_per_word = segment.duration / len(words)
                    for i, word_text in enumerate(words):
                        word_start = segment.start + i * duration_per_word
                        word_end = word_start + duration_per_word
                        if clip_start <= word_start <= clip_end:
                            all_words.append({
                                'text': word_text,
                                'start': word_start - clip_start,
                                'end': word_end - clip_start
                            })

        # Group words into caption lines
        word_groups = []
        current_group = []

        for word in all_words:
            current_group.append(word)

            if len(current_group) >= max_words:
                word_groups.append(current_group)
                current_group = []

        if current_group:
            word_groups.append(current_group)

        # Create caption lines with word highlighting
        for group in word_groups:
            if not group:
                continue

            start_time = group[0]['start']
            end_time = group[-1]['end']

            # Build text with potential highlighting
            words_text = [w['text'] for w in group]
            if self.caption_config.uppercase:
                words_text = [w.upper() for w in words_text]

            text = ' '.join(words_text)

            lines.append(CaptionLine(
                text=text,
                start=start_time,
                end=end_time,
                words=group
            ))

        return lines

    def _format_caption_text(self, text: str) -> List[str]:
        """Format text for display, splitting into multiple lines if needed."""
        max_chars = 35  # Characters per line
        max_words = self.caption_config.max_words_per_line

        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            word_len = len(word) + (1 if current_line else 0)

            if len(current_line) >= max_words or current_length + word_len > max_chars:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += word_len

        if current_line:
            lines.append(' '.join(current_line))

        return lines

    def burn_captions(
        self,
        video_path: Path,
        captions: CaptionSequence,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Burn captions into video using FFmpeg.

        Args:
            video_path: Input video path
            captions: Caption sequence to burn
            output_path: Optional output path

        Returns:
            Path to output video with captions
        """
        video_path = Path(video_path)

        if output_path is None:
            output_path = video_path.with_stem(f"{video_path.stem}_captioned")

        logger.info(f"Burning captions into {video_path.name}")

        # Generate ASS subtitle file
        ass_path = self._generate_ass_file(captions, video_path)

        # Burn subtitles with FFmpeg
        cmd = [
            'ffmpeg', '-y',
            '-i', str(video_path),
            '-vf', f"ass='{ass_path}'",
            '-c:v', self.config.clipping.codec,
            '-preset', self.config.clipping.preset,
            '-crf', str(self.config.clipping.crf),
            '-c:a', 'copy',
            str(output_path)
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            raise RuntimeError(f"Failed to burn captions: {e.stderr}")

        # Clean up ASS file
        if not self.config.keep_temp_files:
            ass_path.unlink()

        return output_path

    def _generate_ass_file(
        self,
        captions: CaptionSequence,
        video_path: Path
    ) -> Path:
        """Generate ASS subtitle file for FFmpeg."""
        from ..utils.video_utils import get_video_info

        video_info = get_video_info(video_path)
        ass_path = self.temp_dir / f"{video_path.stem}_captions.ass"

        # Calculate position
        if self.caption_config.position == 'top':
            alignment = 8  # Top center
            margin_v = 50
        elif self.caption_config.position == 'center':
            alignment = 5  # Middle center
            margin_v = 0
        else:  # bottom
            alignment = 2  # Bottom center
            margin_v = self.caption_config.margin_bottom

        # Convert hex colors to ASS format (BGR)
        def hex_to_ass(hex_color: str) -> str:
            hex_color = hex_color.lstrip('#')
            r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
            return f"&H00{b:02X}{g:02X}{r:02X}"

        primary_color = hex_to_ass(self.caption_config.font_color)
        outline_color = hex_to_ass(self.caption_config.stroke_color)
        highlight_color = hex_to_ass(self.caption_config.highlight_color)

        # ASS header
        ass_content = f"""[Script Info]
Title: Shorts Bot Captions
ScriptType: v4.00+
PlayResX: {video_info.width}
PlayResY: {video_info.height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{self.caption_config.font_name},{self.caption_config.font_size},{primary_color},{primary_color},{outline_color},&H80000000,1,0,0,0,100,100,0,0,1,{self.caption_config.stroke_width},{self.caption_config.shadow_offset if self.caption_config.shadow else 0},{alignment},10,10,{margin_v},1
Style: Highlight,{self.caption_config.font_name},{self.caption_config.font_size},{highlight_color},{highlight_color},{outline_color},&H80000000,1,0,0,0,100,100,0,0,1,{self.caption_config.stroke_width},{self.caption_config.shadow_offset if self.caption_config.shadow else 0},{alignment},10,10,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        # Generate dialogue lines
        if self.caption_config.highlight_current_word and captions.lines:
            # Word-by-word highlighting
            ass_content += self._generate_word_highlight_dialogue(captions)
        else:
            # Standard captions
            dialogues = captions.to_ass_dialogue("Default")
            ass_content += '\n'.join(dialogues)

        with open(ass_path, 'w', encoding='utf-8') as f:
            f.write(ass_content)

        return ass_path

    def _generate_word_highlight_dialogue(self, captions: CaptionSequence) -> str:
        """Generate ASS dialogue with word-by-word highlighting."""
        def format_time(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}:{minutes:02d}:{secs:05.2f}"

        dialogues = []

        for line in captions.lines:
            if not line.words:
                # No word timing, show full line
                start = format_time(line.start)
                end = format_time(line.end)
                text = line.text
                if self.caption_config.uppercase:
                    text = text.upper()
                dialogues.append(
                    f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}"
                )
                continue

            # Generate separate dialogue for each word highlight state
            for i, word_data in enumerate(line.words):
                word_start = format_time(word_data['start'])
                word_end = format_time(word_data['end'])

                # Build text with current word highlighted
                text_parts = []
                for j, w in enumerate(line.words):
                    word_text = w['text']
                    if self.caption_config.uppercase:
                        word_text = word_text.upper()

                    if j == i:
                        # Current word - highlighted
                        if self.caption_config.animation == 'pop':
                            # Scale animation
                            text_parts.append(f"{{\\fscx110\\fscy110\\c{self._get_highlight_ass_color()}}}{word_text}{{\\fscx100\\fscy100\\c&HFFFFFF&}}")
                        else:
                            text_parts.append(f"{{\\c{self._get_highlight_ass_color()}}}{word_text}{{\\c&HFFFFFF&}}")
                    else:
                        text_parts.append(word_text)

                text = ' '.join(text_parts)
                dialogues.append(
                    f"Dialogue: 0,{word_start},{word_end},Default,,0,0,0,,{text}"
                )

        return '\n'.join(dialogues)

    def _get_highlight_ass_color(self) -> str:
        """Get highlight color in ASS format."""
        hex_color = self.caption_config.highlight_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f"&H00{b:02X}{g:02X}{r:02X}&"

    def export_srt(
        self,
        captions: CaptionSequence,
        output_path: Path
    ) -> Path:
        """Export captions as SRT file."""
        def format_timestamp(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

        lines = []
        for i, caption in enumerate(captions.lines, 1):
            lines.append(str(i))
            lines.append(f"{format_timestamp(caption.start)} --> {format_timestamp(caption.end)}")
            lines.append(caption.text)
            lines.append("")

        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        return output_path

    def export_json(
        self,
        captions: CaptionSequence,
        output_path: Path
    ) -> Path:
        """Export captions as JSON for automation."""
        data = {
            'style': captions.style.value,
            'duration': captions.duration,
            'lines': [
                {
                    'text': line.text,
                    'start': line.start,
                    'end': line.end,
                    'words': line.words
                }
                for line in captions.lines
            ]
        }

        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        return output_path

    def generate_capcut_export(
        self,
        captions: CaptionSequence,
        output_path: Path
    ) -> Path:
        """
        Generate CapCut-compatible caption data.

        CapCut uses a JSON format for text templates.
        """
        capcut_data = {
            "type": "text",
            "tracks": [{
                "type": "text",
                "segments": []
            }]
        }

        for line in captions.lines:
            segment = {
                "start_time": int(line.start * 1000000),  # Microseconds
                "end_time": int(line.end * 1000000),
                "text": line.text,
                "style": {
                    "font_size": self.caption_config.font_size,
                    "font_color": self.caption_config.font_color,
                    "bold": True,
                    "alignment": "center"
                }
            }
            capcut_data["tracks"][0]["segments"].append(segment)

        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(capcut_data, f, indent=2)

        return output_path

    def add_emoji_emphasis(
        self,
        captions: CaptionSequence
    ) -> CaptionSequence:
        """Add emoji emphasis to captions based on content."""
        if not self.caption_config.add_emoji:
            return captions

        emoji_map = {
            # Emotions
            r'\b(love|amazing|incredible|awesome)\b': ' ❤️',
            r'\b(laugh|funny|hilarious|lol)\b': ' 😂',
            r'\b(wow|woah|whoa|omg)\b': ' 😮',
            r'\b(angry|mad|furious)\b': ' 😠',
            r'\b(sad|crying|tears)\b': ' 😢',

            # Actions
            r'\b(money|cash|rich|dollar)\b': ' 💰',
            r'\b(fire|hot|lit|🔥)\b': ' 🔥',
            r'\b(think|thought|brain)\b': ' 🧠',
            r'\b(point|key|important)\b': ' 👉',

            # Emphasis
            r'!{2,}': ' ‼️',
            r'\?{2,}': ' ❓',
        }

        new_lines = []
        for line in captions.lines:
            text = line.text
            for pattern, emoji in emoji_map.items():
                if re.search(pattern, text, re.IGNORECASE):
                    text = re.sub(pattern, lambda m: m.group() + emoji, text, flags=re.IGNORECASE)
                    break  # Only add one emoji per line

            new_lines.append(CaptionLine(
                text=text,
                start=line.start,
                end=line.end,
                words=line.words
            ))

        return CaptionSequence(
            lines=new_lines,
            style=captions.style,
            duration=captions.duration
        )

    def cleanup(self):
        """Clean up temporary files."""
        if not self.config.keep_temp_files:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
