"""
Text processing utilities for transcript analysis.
Handles filler word detection, text cleaning, and formatting.
"""

import re
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass

from ..core.logger import get_logger

logger = get_logger(__name__)


# Default filler words and phrases to detect
DEFAULT_FILLER_WORDS = {
    # Single words
    "um", "uh", "er", "ah", "eh", "hm", "hmm", "mhm",
    "like", "basically", "actually", "literally", "seriously",
    "honestly", "obviously", "definitely", "certainly", "absolutely",
    "really", "very", "just", "so", "well", "anyway", "anyways",
    "right", "okay", "ok", "yeah", "yep", "nope",

    # Multi-word fillers (will be handled separately)
}

DEFAULT_FILLER_PHRASES = [
    "you know", "i mean", "kind of", "sort of", "you see",
    "at the end of the day", "to be honest", "to be fair",
    "if you will", "if that makes sense", "does that make sense",
    "know what i mean", "the thing is", "the point is",
    "i think", "i guess", "i suppose", "i believe",
]


@dataclass
class TextSegment:
    """A segment of text with timing information."""
    text: str
    start: float
    end: float
    confidence: float = 1.0
    is_filler: bool = False
    word_timings: Optional[List[Dict]] = None


def clean_text(text: str, lowercase: bool = False) -> str:
    """
    Clean text by removing extra whitespace and normalizing.

    Args:
        text: Input text
        lowercase: Whether to convert to lowercase

    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    if lowercase:
        text = text.lower()

    return text


def detect_filler_words(
    text: str,
    filler_words: Optional[Set[str]] = None,
    filler_phrases: Optional[List[str]] = None,
    aggressiveness: float = 0.5
) -> List[Tuple[int, int, str]]:
    """
    Detect filler words and phrases in text.

    Args:
        text: Input text
        filler_words: Set of filler words to detect
        filler_phrases: List of filler phrases to detect
        aggressiveness: 0-1, higher = more fillers detected

    Returns:
        List of (start_idx, end_idx, filler) tuples
    """
    if filler_words is None:
        filler_words = DEFAULT_FILLER_WORDS

    if filler_phrases is None:
        filler_phrases = DEFAULT_FILLER_PHRASES

    text_lower = text.lower()
    fillers_found = []

    # Find filler phrases first (longer matches)
    for phrase in filler_phrases:
        pattern = r'\b' + re.escape(phrase) + r'\b'
        for match in re.finditer(pattern, text_lower):
            fillers_found.append((match.start(), match.end(), phrase))

    # Find single filler words
    words_pattern = r'\b(' + '|'.join(re.escape(w) for w in filler_words) + r')\b'
    for match in re.finditer(words_pattern, text_lower):
        word = match.group()

        # Check if this word is part of an already found phrase
        is_part_of_phrase = any(
            start <= match.start() < end
            for start, end, _ in fillers_found
        )

        if not is_part_of_phrase:
            # Apply aggressiveness filter
            # Common words like "so", "well", "just" are only flagged at higher aggressiveness
            common_fillers = {"so", "well", "just", "really", "very", "like"}
            if word in common_fillers and aggressiveness < 0.7:
                continue

            fillers_found.append((match.start(), match.end(), word))

    # Sort by position
    fillers_found.sort(key=lambda x: x[0])

    return fillers_found


def remove_filler_words(
    text: str,
    filler_words: Optional[Set[str]] = None,
    filler_phrases: Optional[List[str]] = None,
    aggressiveness: float = 0.5
) -> str:
    """
    Remove filler words and phrases from text.

    Args:
        text: Input text
        filler_words: Set of filler words
        filler_phrases: List of filler phrases
        aggressiveness: 0-1, higher = more aggressive removal

    Returns:
        Text with fillers removed
    """
    fillers = detect_filler_words(text, filler_words, filler_phrases, aggressiveness)

    if not fillers:
        return text

    # Remove fillers from end to start to preserve indices
    result = text
    for start, end, _ in reversed(fillers):
        # Remove the filler and any extra space
        before = result[:start].rstrip()
        after = result[end:].lstrip()

        # Maintain sentence structure
        if before and after:
            result = before + ' ' + after
        else:
            result = before + after

    return clean_text(result)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    # Handle common abbreviations
    text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr)\.\s', r'\1<DOT> ', text)
    text = re.sub(r'\b(etc|vs|viz|al|eg|ie|cf)\.\s', r'\1<DOT> ', text, flags=re.I)

    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Restore abbreviations
    sentences = [s.replace('<DOT>', '.') for s in sentences]

    return [s.strip() for s in sentences if s.strip()]


def format_for_captions(
    text: str,
    max_words_per_line: int = 4,
    max_chars_per_line: int = 30,
    uppercase: bool = True
) -> List[str]:
    """
    Format text for caption display.

    Args:
        text: Input text
        max_words_per_line: Maximum words per caption line
        max_chars_per_line: Maximum characters per line
        uppercase: Convert to uppercase

    Returns:
        List of caption lines
    """
    if uppercase:
        text = text.upper()

    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        word_length = len(word) + (1 if current_line else 0)  # +1 for space

        if (len(current_line) >= max_words_per_line or
            current_length + word_length > max_chars_per_line):
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += word_length

    if current_line:
        lines.append(' '.join(current_line))

    return lines


def calculate_speech_rate(
    text: str,
    duration_seconds: float
) -> float:
    """
    Calculate speech rate in words per minute.

    Args:
        text: Spoken text
        duration_seconds: Duration of speech

    Returns:
        Words per minute
    """
    words = len(text.split())
    if duration_seconds <= 0:
        return 0

    return (words / duration_seconds) * 60


def extract_key_phrases(text: str, max_phrases: int = 10) -> List[str]:
    """
    Extract key phrases from text using simple heuristics.

    Args:
        text: Input text
        max_phrases: Maximum number of phrases to extract

    Returns:
        List of key phrases
    """
    # Simple approach: find noun phrases and important words
    # For production, consider using spaCy or similar

    sentences = split_into_sentences(text)
    phrases = []

    # Look for phrases with emphasis patterns
    patterns = [
        r'\b(the (?:key|main|important|critical|essential) (?:point|thing|issue|problem))\b',
        r'\b(this is (?:why|how|what|when|where))\b',
        r'\b(the (?:truth|fact|reality) is)\b',
        r'\b((?:most|more) importantly)\b',
        r'\b((?:in|at) (?:the )?(?:end|beginning|middle))\b',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        phrases.extend(matches)

    return phrases[:max_phrases]


def detect_emotional_peaks(
    segments: List[TextSegment],
    threshold: float = 0.7
) -> List[int]:
    """
    Detect segments with high emotional content.
    Uses heuristics based on punctuation, word choice, etc.

    Args:
        segments: List of text segments with timing
        threshold: Emotion score threshold (0-1)

    Returns:
        Indices of high-emotion segments
    """
    emotional_indices = []

    # Words that indicate emotional content
    emotion_words = {
        'positive': {'amazing', 'incredible', 'wonderful', 'love', 'excited', 'happy', 'great', 'best'},
        'negative': {'terrible', 'awful', 'hate', 'angry', 'frustrated', 'worst', 'horrible'},
        'intense': {'absolutely', 'definitely', 'completely', 'totally', 'extremely', 'insane', 'crazy'},
    }

    all_emotion_words = set().union(*emotion_words.values())

    for i, segment in enumerate(segments):
        text_lower = segment.text.lower()
        score = 0.0

        # Check for emotion words
        words = set(text_lower.split())
        emotion_count = len(words & all_emotion_words)
        score += emotion_count * 0.2

        # Check for exclamation marks
        exclamation_count = segment.text.count('!')
        score += exclamation_count * 0.15

        # Check for question marks (engagement)
        question_count = segment.text.count('?')
        score += question_count * 0.1

        # Check for ALL CAPS words
        caps_words = len(re.findall(r'\b[A-Z]{2,}\b', segment.text))
        score += caps_words * 0.1

        # Check speech rate (faster = more emotional)
        duration = segment.end - segment.start
        if duration > 0:
            wpm = calculate_speech_rate(segment.text, duration)
            if wpm > 180:  # Fast speech
                score += 0.2
            elif wpm < 100:  # Slow, dramatic speech
                score += 0.1

        if score >= threshold:
            emotional_indices.append(i)

    return emotional_indices


def generate_hashtags(text: str, max_tags: int = 5) -> List[str]:
    """
    Generate relevant hashtags from text content.

    Args:
        text: Input text
        max_tags: Maximum number of hashtags

    Returns:
        List of hashtags
    """
    # Extract meaningful words
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

    # Count word frequency
    word_freq = {}
    stopwords = {'this', 'that', 'with', 'from', 'they', 'have', 'been', 'were', 'said', 'will'}

    for word in words:
        if word not in stopwords:
            word_freq[word] = word_freq.get(word, 0) + 1

    # Get top words
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_tags]

    return [f"#{word}" for word, _ in top_words]
