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

    When LLM is enabled, uses the model as the PRIMARY method for:
    - Analyzing the full transcript for context
    - Identifying engaging moments with proper sentence boundaries
    - Scoring and ranking clips by viral potential
    - Creating cohesive narratives from multiple segments

    Falls back to rule-based detection when LLM is disabled.
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
        self._llm = None  # Lazy initialization

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

        # Detect silence and filler segments
        filler_segments = self._detect_filler_segments(transcript)
        silence_segments = []
        if audio_path:
            silence_segments = self._detect_silence_segments(audio_path)

        # Use LLM as PRIMARY highlight detector when enabled
        if self.hl_config.use_llm and getattr(self.hl_config, 'llm_full_analysis', True):
            logger.info("Using LLM as primary highlight detector")
            highlights = self._detect_highlights_with_llm(transcript)

            if highlights:
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

                logger.info(f"LLM found {len(highlights)} highlights ({coverage * 100:.1f}% coverage)")
                return result
            else:
                logger.warning("LLM detection failed, falling back to rule-based")

        # Fallback: Rule-based detection
        highlights = self._detect_highlights_rule_based(transcript, filler_segments, silence_segments)

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

        logger.info(f"Found {len(highlights)} potential highlights ({coverage * 100:.1f}% coverage)")
        return result

    def _detect_highlights_with_llm(self, transcript: Transcript) -> List[Highlight]:
        """
        Use LLM as the primary method for highlight detection.

        Sends the full transcript to the LLM and asks it to identify
        the most engaging segments with proper sentence boundaries.
        """
        llm = self._init_llm()
        if llm is None:
            return []

        try:
            # Build transcript with timestamps for LLM
            transcript_text = self._format_transcript_for_llm(transcript)

            # Create comprehensive prompt for highlight detection
            prompt = f"""Analyze this video transcript and identify the {self.hl_config.target_clips} most engaging clips for short-form content (YouTube Shorts, TikTok, Reels).

TRANSCRIPT WITH TIMESTAMPS:
{transcript_text}

REQUIREMENTS:
1. Each clip should be {self.hl_config.min_clip_duration}-{self.hl_config.max_clip_duration} seconds long
2. Clips MUST start and end at complete sentences - never cut mid-sentence
3. Look for: emotional moments, surprising insights, strong opinions, engaging questions, story peaks, quotable statements
4. Consider viral potential: hook strength, shareability, clarity of message
5. Clips should be self-contained and make sense without additional context

OUTPUT FORMAT (JSON array):
[
  {{"start": 12.5, "end": 45.2, "score": 9, "reason": "Strong hook with surprising insight about X"}},
  {{"start": 78.0, "end": 120.5, "score": 8, "reason": "Emotional story moment with clear takeaway"}}
]

Return ONLY the JSON array, no other text."""

            logger.debug("Sending full transcript to LLM for analysis")

            response = llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert video editor specializing in viral short-form content. You analyze transcripts to find the most engaging moments that will captivate viewers. Always ensure clips start and end at natural sentence boundaries for cohesive viewing."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.hl_config.llm_max_tokens,
                temperature=self.hl_config.llm_temperature,
            )

            response_text = response['choices'][0]['message']['content'].strip()
            logger.debug(f"LLM response: {response_text}")

            # Parse LLM response
            highlights = self._parse_llm_highlights(response_text, transcript)
            return highlights

        except Exception as e:
            logger.warning(f"LLM highlight detection failed: {e}")
            return []

    def _format_transcript_for_llm(self, transcript: Transcript) -> str:
        """Format transcript with timestamps for LLM analysis."""
        lines = []
        for segment in transcript.segments:
            # Format: [start-end] text
            lines.append(f"[{segment.start:.1f}-{segment.end:.1f}] {segment.text}")
        return "\n".join(lines)

    def _parse_llm_highlights(self, response: str, transcript: Transcript) -> List[Highlight]:
        """Parse LLM response into Highlight objects."""
        highlights = []

        try:
            # Try to extract JSON from response
            # Handle cases where LLM might include extra text
            json_match = re.search(r'\[[\s\S]*\]', response)
            if not json_match:
                logger.warning("No JSON array found in LLM response")
                return []

            json_str = json_match.group()
            clips_data = json.loads(json_str)

            for clip in clips_data:
                start = float(clip.get('start', 0))
                end = float(clip.get('end', 0))
                score = float(clip.get('score', 5)) / 10.0  # Normalize to 0-1
                reason = clip.get('reason', 'LLM selected')

                if end <= start:
                    continue

                # Validate against transcript duration
                if start < 0:
                    start = 0
                if end > transcript.duration:
                    end = transcript.duration

                # Get text for this time range
                text = self._get_text_for_range(transcript, start, end)

                # Adjust boundaries to sentence edges
                start, end, text = self._adjust_to_sentence_boundaries(transcript, start, end)

                if end - start < 3:  # Skip very short clips
                    continue

                highlight = Highlight(
                    start=start,
                    end=end,
                    text=text,
                    score=score,
                    reasons=[HighlightReason.KEY_INSIGHT],  # LLM-detected
                    metadata={
                        'llm_reason': reason,
                        'llm_detected': True,
                        'llm_score': score
                    }
                )
                highlights.append(highlight)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}")
        except Exception as e:
            logger.warning(f"Error parsing LLM highlights: {e}")

        # Sort by score and limit to target
        highlights = sorted(highlights, key=lambda h: h.score, reverse=True)
        highlights = highlights[:self.hl_config.target_clips]
        highlights = sorted(highlights, key=lambda h: h.start)

        return highlights

    def _get_text_for_range(self, transcript: Transcript, start: float, end: float) -> str:
        """Get transcript text for a time range."""
        texts = []
        for segment in transcript.segments:
            if segment.end > start and segment.start < end:
                texts.append(segment.text)
        return ' '.join(texts)

    def _adjust_to_sentence_boundaries(
        self,
        transcript: Transcript,
        start: float,
        end: float
    ) -> Tuple[float, float, str]:
        """
        Adjust clip boundaries to align with sentence edges.

        Returns:
            Tuple of (adjusted_start, adjusted_end, text)
        """
        # Find the segment that contains the start time
        start_segment = None
        end_segment = None

        for segment in transcript.segments:
            if segment.start <= start < segment.end:
                start_segment = segment
            if segment.start < end <= segment.end:
                end_segment = segment

        # Adjust start to beginning of sentence
        adjusted_start = start
        if start_segment:
            # Check if we're mid-sentence (not at segment start)
            if start > start_segment.start + 0.5:
                # Move to start of this segment
                adjusted_start = start_segment.start

        # Adjust end to end of sentence
        adjusted_end = end
        if end_segment:
            # Check if we're mid-sentence (not at segment end)
            if end < end_segment.end - 0.5:
                # Move to end of this segment
                adjusted_end = end_segment.end

        # Get the full text for adjusted range
        text = self._get_text_for_range(transcript, adjusted_start, adjusted_end)

        return adjusted_start, adjusted_end, text

    def create_cohesive_clip(
        self,
        transcript: Transcript,
        target_duration: float = 60.0
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Use LLM to create a cohesive clip by combining multiple transcript segments.

        This analyzes the full transcript and selects the best parts to combine
        into a single coherent narrative.

        Args:
            transcript: Full transcript
            target_duration: Target duration for the combined clip

        Returns:
            List of segment dicts with 'start' and 'end' times to combine
        """
        llm = self._init_llm()
        if llm is None:
            logger.warning("LLM not available for cohesive clip creation")
            return None

        try:
            transcript_text = self._format_transcript_for_llm(transcript)

            prompt = f"""Analyze this transcript and select segments to create ONE cohesive {target_duration:.0f}-second video clip.

TRANSCRIPT WITH TIMESTAMPS:
{transcript_text}

TASK:
Create a cohesive narrative by selecting multiple segments from different parts of the video.
The segments will be edited together to form one continuous, engaging clip.

REQUIREMENTS:
1. Total duration of all segments combined should be approximately {target_duration:.0f} seconds
2. Each segment MUST start and end at complete sentences
3. Segments should flow logically when combined (even if from different parts)
4. Prioritize: strong opening hook, valuable content, satisfying conclusion
5. The combined clip should tell a complete mini-story or deliver a clear message

OUTPUT FORMAT (JSON array of segments in playback order):
[
  {{"start": 5.0, "end": 18.5, "purpose": "Hook - attention-grabbing opening"}},
  {{"start": 45.2, "end": 62.0, "purpose": "Main content - key insight"}},
  {{"start": 120.0, "end": 135.5, "purpose": "Conclusion - memorable takeaway"}}
]

Return ONLY the JSON array, no other text."""

            logger.info("Using LLM to create cohesive clip from transcript")

            response = llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert video editor who creates compelling narratives by combining the best moments from longer videos. You understand pacing, story structure, and what makes content go viral."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.hl_config.llm_max_tokens,
                temperature=self.hl_config.llm_temperature,
            )

            response_text = response['choices'][0]['message']['content'].strip()
            logger.debug(f"LLM cohesive clip response: {response_text}")

            # Parse response
            json_match = re.search(r'\[[\s\S]*\]', response_text)
            if not json_match:
                return None

            segments = json.loads(json_match.group())

            # Validate and adjust segments
            validated_segments = []
            for seg in segments:
                start = float(seg.get('start', 0))
                end = float(seg.get('end', 0))

                if end <= start or start < 0 or end > transcript.duration:
                    continue

                # Adjust to sentence boundaries
                adj_start, adj_end, _ = self._adjust_to_sentence_boundaries(transcript, start, end)

                validated_segments.append({
                    'start': adj_start,
                    'end': adj_end,
                    'purpose': seg.get('purpose', '')
                })

            return validated_segments if validated_segments else None

        except Exception as e:
            logger.warning(f"Failed to create cohesive clip: {e}")
            return None

    def _detect_highlights_rule_based(
        self,
        transcript: Transcript,
        filler_segments: List[Tuple[float, float]],
        silence_segments: List[Tuple[float, float]]
    ) -> List[Highlight]:
        """Fallback rule-based highlight detection."""
        # Score each segment using simple heuristics
        segment_scores = []
        for segment in transcript.segments:
            score, reasons = self._score_segment_simple(segment)
            segment_scores.append({
                'segment': segment,
                'score': score,
                'reasons': reasons
            })

        # Group into clips
        highlights = self._create_clips(
            segment_scores,
            transcript.duration,
            filler_segments,
            silence_segments
        )

        return highlights

    def _score_segment_simple(self, segment: Segment) -> Tuple[float, List[HighlightReason]]:
        """Simple scoring for fallback mode."""
        score = 0.1
        reasons = []
        text = segment.text.lower()

        # Basic engagement signals
        if '?' in segment.text:
            score += 0.2
            reasons.append(HighlightReason.ENGAGING_QUESTION)

        if '!' in segment.text:
            score += 0.1

        # Length bonus (prefer substantial segments)
        word_count = len(segment.text.split())
        if word_count >= 10:
            score += 0.1
            reasons.append(HighlightReason.KEY_INSIGHT)

        return min(score, 1.0), reasons

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

    def _init_llm(self):
        """
        Initialize the LLM for highlight detection.

        Uses llama-cpp-python with Metal acceleration for Apple Silicon.
        Configured for full transcript analysis with larger context window.
        """
        if hasattr(self, '_llm') and self._llm is not None:
            return self._llm

        self._llm = None

        if self.hl_config.llm_backend == "llama_cpp":
            try:
                from llama_cpp import Llama

                model_path = self.hl_config.llm_model_path

                # Check if the configured path exists
                if model_path and Path(model_path).exists():
                    pass  # Use configured path
                else:
                    # Try to find model in models directory
                    models_dir = self.config.models_dir
                    possible_paths = [
                        models_dir / f"{model_name}.gguf",
                        models_dir / "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
                        models_dir / "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
                        Path.home() / ".cache" / "llama.cpp" / f"{model_name}.gguf",
                    ]
                    model_path = None
                    for p in possible_paths:
                        if p.exists():
                            model_path = str(p)
                            break

                if model_path is None or not Path(model_path).exists():
                    logger.warning(f"LLM model file not found. Please download and place in models/")
                    logger.info("Download Q5_K_M from: https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF")
                    logger.info("Place as: models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf")
                    return None

                logger.info(f"Loading LLM model from {model_path}")
                logger.info(f"Using Metal acceleration: n_gpu_layers={self.hl_config.llm_n_gpu_layers}")

                # Initialize with Apple M2 optimizations
                # Note: bf16 kernel warnings are normal on M2 - it falls back to f16
                # Note: n_ctx < n_ctx_train warning is expected - we don't need full context
                self._llm = Llama(
                    model_path=model_path,
                    n_ctx=self.hl_config.llm_n_ctx,
                    n_threads=self.hl_config.llm_n_threads,
                    n_gpu_layers=self.hl_config.llm_n_gpu_layers,  # -1 = all on Metal
                    verbose=False,  # Suppress Metal kernel warnings
                    use_mmap=True,  # Memory-mapped for efficiency
                    use_mlock=False,  # Don't lock memory on macOS
                    chat_format="llama-3",  # Use proper chat format to avoid duplicate tokens
                )

                logger.info("LLM initialized successfully with Metal acceleration")
                return self._llm

            except ImportError:
                logger.warning("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            except Exception as e:
                logger.warning(f"Failed to initialize llama-cpp: {e}")

        # Fallback to None (will use Ollama in _refine_with_llm)
        return None

    def _refine_with_llm(
        self,
        highlights: List[Highlight],
        transcript: Transcript
    ) -> List[Highlight]:
        """
        Use local LLM to refine highlight selection.

        Supports llama-cpp-python (with Metal on Apple Silicon) and Ollama.

        Args:
            highlights: Initial highlights
            transcript: Full transcript

        Returns:
            Refined highlights
        """
        if not highlights:
            return highlights

        # Try llama-cpp-python first (faster on Apple Silicon with Metal)
        if self.hl_config.llm_backend == "llama_cpp":
            llm = self._init_llm()
            if llm is not None:
                return self._refine_with_llama_cpp(highlights, transcript, llm)

        # Fallback to Ollama
        return self._refine_with_ollama(highlights, transcript)

    def _refine_with_llama_cpp(
        self,
        highlights: List[Highlight],
        transcript: Transcript,
        llm
    ) -> List[Highlight]:
        """Refine highlights using llama-cpp-python."""
        try:
            prompt = self._create_llm_prompt(highlights, transcript)

            logger.debug("Running LLM inference with Metal acceleration")

            # Use chat completion API - handles tokenization properly
            response = llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a video content analyst. Analyze video clips and rate their viral potential. Be concise and respond only with scores in the format: 1:8, 2:6, 3:9"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.hl_config.llm_max_tokens,
                temperature=self.hl_config.llm_temperature,
            )

            response_text = response['choices'][0]['message']['content'].strip()
            logger.debug(f"LLM response: {response_text}")

            highlights = self._parse_llm_response(highlights, response_text)

        except Exception as e:
            logger.warning(f"LLM refinement with llama-cpp failed: {e}")

        return highlights

    def _refine_with_ollama(
        self,
        highlights: List[Highlight],
        transcript: Transcript
    ) -> List[Highlight]:
        """Refine highlights using Ollama as fallback."""
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

            # Get model name (convert llama_cpp format to ollama format if needed)
            model_name = self.hl_config.llm_model
            if "q5_K_M" in model_name.lower():
                model_name = "llama3.1:8b"  # Ollama naming

            # Call ollama
            result = subprocess.run(
                ['ollama', 'run', model_name, prompt],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                response = result.stdout
                highlights = self._parse_llm_response(highlights, response)

        except subprocess.TimeoutExpired:
            logger.warning("Ollama request timed out")
        except Exception as e:
            logger.warning(f"Ollama refinement failed: {e}")

        return highlights

    def _create_llm_prompt(
        self,
        highlights: List[Highlight],
        transcript: Transcript
    ) -> str:
        """Create prompt for LLM analysis."""
        prompt = """Rate these video clip candidates for viral potential (1-10).
Consider: emotional impact, shareability, hook strength, clarity.

Clips:
"""
        for i, h in enumerate(highlights, 1):
            text_preview = h.text[:200] + "..." if len(h.text) > 200 else h.text
            prompt += f"\n{i}. [{h.duration:.1f}s] \"{text_preview}\"\n"

        prompt += "\nRespond with ONLY clip numbers and scores like: 1:8, 2:6, 3:9"

        return prompt

    def _parse_llm_response(
        self,
        highlights: List[Highlight],
        response: str
    ) -> List[Highlight]:
        """Parse LLM response to adjust highlight scores."""
        # Extract scores from response
        score_pattern = r'(\d+)\s*[:\-]\s*(\d+)'
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
                    logger.debug(f"Clip {clip_num}: original={original:.2f}, llm={llm_score:.2f}, final={highlights[idx].score:.2f}")

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
