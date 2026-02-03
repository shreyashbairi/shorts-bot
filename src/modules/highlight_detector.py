"""
Highlight detection module for identifying interesting video segments.
Uses transcript analysis, audio features, and optional LLM for smart selection.
"""

import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import subprocess

from ..core.config import Config, HighlightConfig
from ..core.logger import get_logger
from ..utils.file_utils import ensure_dir
from ..utils.text_utils import (
    detect_filler_words,
    remove_filler_words,
    split_into_sentences,
    calculate_speech_rate,
    detect_emotional_peaks
)
from .transcriber import Transcript, Segment

logger = get_logger(__name__)


class HighlightReason(Enum):
    """Reasons why a segment was selected as a highlight."""
    EMOTIONAL_PEAK = "emotional_peak"
    KEY_INSIGHT = "key_insight"
    STRONG_OPINION = "strong_opinion"
    ENGAGING_QUESTION = "engaging_question"
    DRAMATIC_PAUSE = "dramatic_pause"
    HIGH_ENERGY = "high_energy"
    STORY_MOMENT = "story_moment"
    QUOTABLE = "quotable"
    TOPIC_INTRO = "topic_intro"
    CONCLUSION = "conclusion"


@dataclass
class Highlight:
    """A detected highlight segment."""
    start: float
    end: float
    text: str
    score: float  # 0-1, higher = better
    reasons: List[HighlightReason]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        return {
            'start': self.start,
            'end': self.end,
            'duration': self.duration,
            'text': self.text,
            'score': self.score,
            'reasons': [r.value for r in self.reasons],
            'metadata': self.metadata
        }


@dataclass
class HighlightResult:
    """Result of highlight detection."""
    highlights: List[Highlight]
    total_duration: float
    coverage: float  # Percentage of video covered by highlights
    filler_segments: List[Tuple[float, float]] = field(default_factory=list)
    silence_segments: List[Tuple[float, float]] = field(default_factory=list)

    def get_top_highlights(self, n: int) -> List[Highlight]:
        """Get top N highlights by score."""
        return sorted(self.highlights, key=lambda h: h.score, reverse=True)[:n]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'highlights': [h.to_dict() for h in self.highlights],
            'total_duration': self.total_duration,
            'coverage': self.coverage,
            'highlight_count': len(self.highlights)
        }

    def save(self, path: Path):
        """Save highlights to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)


class HighlightDetector:
    """
    Detects highlight-worthy segments from transcripts.

    Uses multiple signals to identify interesting content:
    - Emotional language and punctuation
    - Speech patterns (rate, pauses)
    - Content analysis (insights, opinions)
    - Optional LLM-based analysis
    """

    def __init__(self, config: Config):
        """
        Initialize the highlight detector.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.hl_config = config.highlight
        self.temp_dir = ensure_dir(config.temp_dir / "highlights")

        # Emotional and engagement indicators
        self.emotion_words = {
            'positive': {'amazing', 'incredible', 'wonderful', 'love', 'excited', 'happy', 'great', 'best', 'awesome', 'fantastic'},
            'negative': {'terrible', 'awful', 'hate', 'angry', 'frustrated', 'worst', 'horrible', 'disgusting', 'annoying'},
            'intense': {'absolutely', 'definitely', 'completely', 'totally', 'extremely', 'insane', 'crazy', 'unbelievable', 'mind-blowing'},
            'opinion': {'think', 'believe', 'feel', 'opinion', 'honestly', 'frankly', 'truth'},
        }

        self.insight_patterns = [
            r'\b(the key (is|thing|point)|what matters|important to understand)\b',
            r'\b(here\'?s the (thing|deal|secret)|the truth is|let me tell you)\b',
            r'\b(most people (don\'?t|think|believe)|contrary to|actually)\b',
            r'\b(this is (why|how|what)|the reason)\b',
            r'\b(never|always|every time|without exception)\b',
        ]

        self.story_patterns = [
            r'\b(one (time|day)|remember when|this happened|true story)\b',
            r'\b(so there (I|we) (was|were)|picture this|imagine)\b',
            r'\b(and then|suddenly|out of nowhere|next thing)\b',
        ]

    def detect_highlights(
        self,
        transcript: Transcript,
        audio_path: Optional[Path] = None
    ) -> HighlightResult:
        """
        Detect highlight segments from transcript.

        Args:
            transcript: Transcription result
            audio_path: Optional audio file for additional analysis

        Returns:
            HighlightResult with ranked highlights
        """
        logger.info("Detecting highlights from transcript")

        # Score each segment
        segment_scores = []
        for segment in transcript.segments:
            score, reasons = self._score_segment(segment, transcript)
            segment_scores.append({
                'segment': segment,
                'score': score,
                'reasons': reasons
            })

        # Detect silence and filler segments
        filler_segments = self._detect_filler_segments(transcript)
        silence_segments = []
        if audio_path:
            silence_segments = self._detect_silence_segments(audio_path)

        # Group high-scoring segments into clips
        highlights = self._create_clips(
            segment_scores,
            transcript.duration,
            filler_segments,
            silence_segments
        )

        # Use LLM for additional refinement if enabled
        if self.hl_config.use_llm:
            highlights = self._refine_with_llm(highlights, transcript)

        # Calculate coverage
        total_highlight_duration = sum(h.duration for h in highlights)
        coverage = total_highlight_duration / transcript.duration if transcript.duration > 0 else 0

        result = HighlightResult(
            highlights=highlights,
            total_duration=transcript.duration,
            coverage=coverage,
            filler_segments=filler_segments,
            silence_segments=silence_segments
        )

        logger.info(f"Found {len(highlights)} potential highlights "
                   f"({coverage * 100:.1f}% coverage)")

        return result

    def _score_segment(
        self,
        segment: Segment,
        transcript: Transcript
    ) -> Tuple[float, List[HighlightReason]]:
        """
        Score a single segment for highlight potential.

        Args:
            segment: Segment to score
            transcript: Full transcript for context

        Returns:
            Tuple of (score, reasons)
        """
        # Start with a base score so all segments have some value
        score = 0.1
        reasons = []
        text_lower = segment.text.lower()

        # Give a small bonus for segments with more content
        word_count = len(segment.text.split())
        if word_count >= 5:
            score += 0.05
        if word_count >= 10:
            score += 0.05

        # Emotional content scoring (use lower threshold)
        emotion_score = self._score_emotional_content(segment.text)
        if emotion_score > 0.1:
            score += emotion_score * self.hl_config.emotional_weight
            if emotion_score > 0.2:
                reasons.append(HighlightReason.EMOTIONAL_PEAK)

        # Insight/key point scoring (use lower threshold)
        insight_score = self._score_insights(segment.text)
        if insight_score > 0.1:
            score += insight_score * self.hl_config.insight_weight
            if insight_score > 0.2:
                reasons.append(HighlightReason.KEY_INSIGHT)

        # Engagement scoring (questions, direct address) (use lower threshold)
        engagement_score = self._score_engagement(segment.text)
        if engagement_score > 0.1:
            score += engagement_score * self.hl_config.engagement_weight
            if '?' in segment.text:
                reasons.append(HighlightReason.ENGAGING_QUESTION)

        # Pacing/energy scoring (always add some contribution)
        pacing_score = self._score_pacing(segment)
        score += pacing_score * self.hl_config.pacing_weight * 0.5
        if pacing_score > 0.5:
            reasons.append(HighlightReason.HIGH_ENERGY)

        # Story detection
        if self._is_story_moment(segment.text):
            score += 0.2
            reasons.append(HighlightReason.STORY_MOMENT)

        # Strong opinion detection
        if self._is_strong_opinion(segment.text):
            score += 0.15
            reasons.append(HighlightReason.STRONG_OPINION)

        # Quotable content (short, punchy statements)
        if self._is_quotable(segment):
            score += 0.1
            reasons.append(HighlightReason.QUOTABLE)

        # Penalize segments with too many filler words (but less severely)
        filler_penalty = self._calculate_filler_penalty(segment.text) * 0.5
        score -= filler_penalty

        # Normalize score to 0-1 range (ensure minimum of 0.05)
        score = max(0.05, min(1.0, score))

        return score, reasons

    def _score_emotional_content(self, text: str) -> float:
        """Score text for emotional content."""
        text_lower = text.lower()
        words = set(text_lower.split())

        score = 0.0

        # Check emotion word categories
        for category, emotion_words in self.emotion_words.items():
            matches = len(words & emotion_words)
            if category == 'intense':
                score += matches * 0.15
            else:
                score += matches * 0.1

        # Check punctuation (exclamation marks indicate emotion)
        exclamation_count = text.count('!')
        score += min(exclamation_count * 0.1, 0.3)

        # Check for ALL CAPS words (emphasis)
        caps_words = len(re.findall(r'\b[A-Z]{3,}\b', text))
        score += min(caps_words * 0.1, 0.2)

        return min(score, 1.0)

    def _score_insights(self, text: str) -> float:
        """Score text for insight/key point content."""
        text_lower = text.lower()
        score = 0.0

        for pattern in self.insight_patterns:
            if re.search(pattern, text_lower):
                score += 0.25

        # Check for numbers/statistics (often indicate insights)
        number_matches = len(re.findall(r'\b\d+(?:\.\d+)?%?\b', text))
        score += min(number_matches * 0.1, 0.3)

        # Check for comparison language
        comparison_words = {'more', 'less', 'better', 'worse', 'faster', 'slower', 'bigger', 'smaller'}
        if any(word in text_lower for word in comparison_words):
            score += 0.1

        return min(score, 1.0)

    def _score_engagement(self, text: str) -> float:
        """Score text for engagement potential."""
        score = 0.0

        # Questions engage viewers
        question_count = text.count('?')
        score += min(question_count * 0.2, 0.4)

        # Direct address ("you", "your")
        text_lower = text.lower()
        if re.search(r'\byou(r)?\b', text_lower):
            score += 0.15

        # Call to action language
        cta_patterns = [
            r'\b(think about|imagine|consider|look at)\b',
            r'\b(here\'?s (what|how|why))\b',
            r'\b(let me (show|tell|explain))\b',
        ]
        for pattern in cta_patterns:
            if re.search(pattern, text_lower):
                score += 0.15

        return min(score, 1.0)

    def _score_pacing(self, segment: Segment) -> float:
        """Score segment based on speech pacing."""
        if segment.duration <= 0:
            return 0.5

        wpm = calculate_speech_rate(segment.text, segment.duration)

        # Optimal range for engaging content: 140-180 WPM
        # Too slow can be boring, too fast can be hard to follow
        if 140 <= wpm <= 180:
            return 0.8
        elif 120 <= wpm < 140 or 180 < wpm <= 200:
            return 0.6
        elif 100 <= wpm < 120 or 200 < wpm <= 220:
            return 0.4
        else:
            return 0.2

    def _is_story_moment(self, text: str) -> bool:
        """Check if text contains story-telling elements."""
        text_lower = text.lower()
        for pattern in self.story_patterns:
            if re.search(pattern, text_lower):
                return True
        return False

    def _is_strong_opinion(self, text: str) -> bool:
        """Check if text expresses a strong opinion."""
        opinion_patterns = [
            r'\bi (think|believe|feel) (that )?',
            r'\bin my (opinion|view|experience)',
            r'\b(honestly|frankly|truthfully)',
            r'\bthe (problem|issue|thing) (is|with)',
        ]
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in opinion_patterns)

    def _is_quotable(self, segment: Segment) -> bool:
        """Check if segment is quotable (short, punchy, complete thought)."""
        word_count = len(segment.text.split())

        # Quotable segments are typically 5-20 words
        if not (5 <= word_count <= 20):
            return False

        # Should be a complete thought (ends with punctuation)
        if not segment.text.rstrip()[-1] in '.!?':
            return False

        # Contains impactful language
        impact_words = {'never', 'always', 'only', 'best', 'worst', 'must', 'need'}
        text_lower = segment.text.lower()
        if any(word in text_lower for word in impact_words):
            return True

        return False

    def _calculate_filler_penalty(self, text: str) -> float:
        """Calculate penalty based on filler word density."""
        fillers = detect_filler_words(
            text,
            filler_words=set(self.hl_config.filler_words),
            aggressiveness=self.hl_config.filler_aggressiveness
        )

        word_count = len(text.split())
        if word_count == 0:
            return 0

        filler_density = len(fillers) / word_count
        return min(filler_density * 2, 0.5)  # Max 0.5 penalty

    def _detect_filler_segments(
        self,
        transcript: Transcript
    ) -> List[Tuple[float, float]]:
        """
        Detect segments dominated by filler words.

        Returns:
            List of (start, end) tuples for filler-heavy segments
        """
        filler_segments = []

        for segment in transcript.segments:
            fillers = detect_filler_words(
                segment.text,
                filler_words=set(self.hl_config.filler_words),
                aggressiveness=self.hl_config.filler_aggressiveness
            )

            word_count = len(segment.text.split())
            if word_count > 0:
                filler_ratio = len(fillers) / word_count
                if filler_ratio > 0.4:  # More than 40% fillers
                    filler_segments.append((segment.start, segment.end))

        return filler_segments

    def _detect_silence_segments(self, audio_path: Path) -> List[Tuple[float, float]]:
        """Detect silent segments in audio."""
        from ..utils.video_utils import detect_silence

        return detect_silence(
            audio_path,
            threshold_db=self.hl_config.silence_threshold,
            min_duration=self.hl_config.min_silence_duration
        )

    def _create_clips(
        self,
        segment_scores: List[Dict],
        total_duration: float,
        filler_segments: List[Tuple[float, float]],
        silence_segments: List[Tuple[float, float]]
    ) -> List[Highlight]:
        """
        Group scored segments into highlight clips.

        Args:
            segment_scores: List of segment score data
            total_duration: Total video duration
            filler_segments: Segments to avoid
            silence_segments: Silent segments to avoid

        Returns:
            List of Highlight objects
        """
        min_duration = self.hl_config.min_clip_duration
        max_duration = self.hl_config.max_clip_duration

        # Sort by score to start with best segments
        sorted_scores = sorted(
            segment_scores,
            key=lambda x: x['score'],
            reverse=True
        )

        highlights = []
        used_ranges = []

        def overlaps_used(start: float, end: float) -> bool:
            for used_start, used_end in used_ranges:
                if start < used_end and end > used_start:
                    return True
            return False

        def overlaps_filler(start: float, end: float) -> float:
            """Calculate percentage overlap with filler segments."""
            overlap = 0
            for f_start, f_end in filler_segments + silence_segments:
                overlap_start = max(start, f_start)
                overlap_end = min(end, f_end)
                if overlap_start < overlap_end:
                    overlap += overlap_end - overlap_start
            return overlap / (end - start) if end > start else 0

        # Determine score threshold dynamically
        # Use a lower threshold to ensure we find highlights
        if sorted_scores:
            max_score = sorted_scores[0]['score']
            # Use 10% of max score as minimum threshold, with floor of 0.05
            score_threshold = max(0.05, max_score * 0.1)
        else:
            score_threshold = 0.05

        logger.debug(f"Using score threshold: {score_threshold:.3f}")

        for scored in sorted_scores:
            if len(highlights) >= self.hl_config.target_clips * 2:  # Get extra, we'll trim later
                break

            segment = scored['segment']

            # Skip very low-scoring segments (use dynamic threshold)
            if scored['score'] < score_threshold:
                continue

            # Try to expand this segment into a full clip
            clip_start = segment.start
            clip_end = segment.end

            # Find adjacent segments to include (more lenient scoring)
            for other_scored in segment_scores:
                other = other_scored['segment']
                if other.id == segment.id:
                    continue

                # Check if adjacent (within 2 seconds) and has any positive score
                is_adjacent = (
                    abs(other.end - clip_start) < 2.0 or
                    abs(other.start - clip_end) < 2.0
                )

                if is_adjacent and other_scored['score'] > 0.0:
                    new_start = min(clip_start, other.start)
                    new_end = max(clip_end, other.end)

                    # Check duration constraints
                    if new_end - new_start <= max_duration:
                        clip_start = new_start
                        clip_end = new_end

            # Apply padding
            clip_start = max(0, clip_start - 0.5)
            clip_end = min(total_duration, clip_end + 0.5)

            # Check duration
            duration = clip_end - clip_start
            if duration < min_duration:
                # Try to extend more aggressively
                needed = min_duration - duration
                clip_start = max(0, clip_start - needed)
                clip_end = min(total_duration, clip_end + needed)
                duration = clip_end - clip_start

            # If still too short but video is short, allow shorter clips
            effective_min = min(min_duration, total_duration * 0.5)
            if duration < effective_min or duration > max_duration:
                continue

            # Skip if overlaps with already selected highlights
            if overlaps_used(clip_start, clip_end):
                continue

            # Be more lenient with filler overlap (50% instead of 30%)
            if overlaps_filler(clip_start, clip_end) > 0.5:
                continue

            # Get text for this clip range
            clip_segments = [
                s for s in segment_scores
                if s['segment'].start >= clip_start and s['segment'].end <= clip_end
            ]
            clip_text = ' '.join(s['segment'].text for s in clip_segments)

            # Combine reasons from all included segments
            all_reasons = set()
            for s in clip_segments:
                all_reasons.update(s['reasons'])

            # If no reasons, add a generic one
            if not all_reasons:
                all_reasons.add(HighlightReason.KEY_INSIGHT)

            highlight = Highlight(
                start=clip_start,
                end=clip_end,
                text=clip_text,
                score=scored['score'],
                reasons=list(all_reasons),
                metadata={
                    'segment_count': len(clip_segments),
                    'primary_segment_id': segment.id
                }
            )

            highlights.append(highlight)
            used_ranges.append((clip_start, clip_end))

        # Fallback: if no highlights found, create clips from the highest scoring segments
        if not highlights and sorted_scores:
            logger.info("No highlights met criteria, creating fallback clips from best segments")
            highlights = self._create_fallback_clips(sorted_scores, total_duration, used_ranges)

        # Sort by time and return top N
        highlights = sorted(highlights, key=lambda h: h.score, reverse=True)
        highlights = highlights[:self.hl_config.target_clips]
        highlights = sorted(highlights, key=lambda h: h.start)

        return highlights

    def _create_fallback_clips(
        self,
        sorted_scores: List[Dict],
        total_duration: float,
        used_ranges: List[Tuple[float, float]]
    ) -> List[Highlight]:
        """
        Create fallback clips when normal detection fails.

        This ensures we always return some highlights by being very lenient
        with scoring and duration requirements.
        """
        highlights = []
        min_duration = min(self.hl_config.min_clip_duration, max(5, total_duration * 0.3))
        max_duration = self.hl_config.max_clip_duration

        def overlaps_used(start: float, end: float) -> bool:
            for used_start, used_end in used_ranges:
                if start < used_end and end > used_start:
                    return True
            return False

        for scored in sorted_scores[:self.hl_config.target_clips * 3]:
            if len(highlights) >= self.hl_config.target_clips:
                break

            segment = scored['segment']
            clip_start = max(0, segment.start - 1.0)
            clip_end = min(total_duration, segment.end + 1.0)

            # Extend to meet minimum duration
            duration = clip_end - clip_start
            if duration < min_duration:
                needed = min_duration - duration
                clip_start = max(0, clip_start - needed / 2)
                clip_end = min(total_duration, clip_end + needed / 2)
                duration = clip_end - clip_start

            # Skip if still too short or too long
            if duration < 3 or duration > max_duration:
                continue

            # Skip if overlaps
            if overlaps_used(clip_start, clip_end):
                continue

            # Get text for this range
            clip_text = segment.text
            reasons = scored.get('reasons', [])
            if not reasons:
                reasons = [HighlightReason.KEY_INSIGHT]

            highlight = Highlight(
                start=clip_start,
                end=clip_end,
                text=clip_text,
                score=max(scored['score'], 0.1),  # Ensure minimum score
                reasons=reasons,
                metadata={
                    'segment_count': 1,
                    'primary_segment_id': segment.id,
                    'fallback': True
                }
            )

            highlights.append(highlight)
            used_ranges.append((clip_start, clip_end))

        return highlights

    def _refine_with_llm(
        self,
        highlights: List[Highlight],
        transcript: Transcript
    ) -> List[Highlight]:
        """
        Use local LLM to refine highlight selection.

        Args:
            highlights: Initial highlights
            transcript: Full transcript

        Returns:
            Refined highlights
        """
        try:
            import subprocess

            # Check if ollama is available
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                logger.warning("Ollama not available, skipping LLM refinement")
                return highlights

            # Create prompt for LLM
            prompt = self._create_llm_prompt(highlights, transcript)

            # Call ollama
            result = subprocess.run(
                ['ollama', 'run', self.hl_config.llm_model, prompt],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                # Parse LLM response to adjust scores
                response = result.stdout
                highlights = self._parse_llm_response(highlights, response)

        except subprocess.TimeoutExpired:
            logger.warning("LLM request timed out")
        except Exception as e:
            logger.warning(f"LLM refinement failed: {e}")

        return highlights

    def _create_llm_prompt(
        self,
        highlights: List[Highlight],
        transcript: Transcript
    ) -> str:
        """Create prompt for LLM analysis."""
        prompt = """Analyze these video clip candidates and rate them for viral potential on a scale of 1-10.
Consider: emotional impact, shareability, clarity of message, hook potential.

Clips:
"""
        for i, h in enumerate(highlights, 1):
            prompt += f"\n{i}. [{h.duration:.1f}s] \"{h.text[:200]}...\"\n"

        prompt += "\nRespond with just the clip numbers and scores, e.g.: 1:8, 2:6, 3:9"

        return prompt

    def _parse_llm_response(
        self,
        highlights: List[Highlight],
        response: str
    ) -> List[Highlight]:
        """Parse LLM response to adjust highlight scores."""
        # Extract scores from response
        score_pattern = r'(\d+)\s*:\s*(\d+)'
        matches = re.findall(score_pattern, response)

        for clip_num, score in matches:
            try:
                idx = int(clip_num) - 1
                llm_score = int(score) / 10.0

                if 0 <= idx < len(highlights):
                    # Blend original score with LLM score
                    original = highlights[idx].score
                    highlights[idx].score = (original + llm_score) / 2
                    highlights[idx].metadata['llm_score'] = llm_score

            except (ValueError, IndexError):
                continue

        return highlights

    def export_edl(
        self,
        highlights: List[Highlight],
        output_path: Path,
        fps: float = 30.0
    ) -> Path:
        """
        Export highlights as EDL (Edit Decision List) for video editors.

        Args:
            highlights: List of highlights
            output_path: Output file path
            fps: Frame rate for timecode calculation

        Returns:
            Path to EDL file
        """
        def seconds_to_timecode(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            frames = int((seconds % 1) * fps)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"

        lines = ["TITLE: Shorts Bot Highlights", "FCM: NON-DROP FRAME", ""]

        for i, h in enumerate(highlights, 1):
            lines.append(f"{i:03d}  AX       V     C        "
                        f"{seconds_to_timecode(h.start)} {seconds_to_timecode(h.end)} "
                        f"{seconds_to_timecode(0)} {seconds_to_timecode(h.duration)}")
            lines.append(f"* REASON: {', '.join(r.value for r in h.reasons)}")
            lines.append("")

        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        return output_path
