"""
2_preprocessor.py
Text normalization and word-ID encoding.

Implements the "word ID" scheme hinted at in the issue: every single word in a
phrase is assigned a stable integer ID, so phrases become sequences of IDs that
can be featurised (e.g. hashed / count-based n-grams over ID space) in addition
to the raw lexical TF-IDF signal.
"""

import re
import unicodedata
from typing import Iterable, List, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r'\s+')
_CONTROL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
# Characters commonly used to smuggle instructions past naive filters.
_INVISIBLE_RE = re.compile(r'[\u200b\u200c\u200d\u2060\ufeff\u00ad]')


def normalize_text(text: str) -> str:
    """
    Normalize a raw prompt for lexical analysis.

    Steps (order matters):
      1. NFKC unicode normalization  -> folds fancy quotes/dashes/ligatures.
      2. Strip zero-width / bidi / soft-hyphen control characters.
      3. Lowercase.
      4. Collapse all whitespace runs into a single space.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = unicodedata.normalize('NFKC', str(text))
    s = _INVISIBLE_RE.sub('', s)
    s = _CONTROL_RE.sub(' ', s)
    s = s.lower()
    s = _WS_RE.sub(' ', s).strip()
    return s


def normalize_series(texts: Iterable[str]) -> List[str]:
    """Vectorised convenience wrapper around :func:`normalize_text`."""
    return [normalize_text(t) for t in texts]


# ---------------------------------------------------------------------------
# Word-ID encoding
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str) -> List[str]:
    """Split a normalized prompt into word tokens."""
    return _TOKEN_RE.findall(text.lower())


class WordIdEncoder:
    """
    Assigns a stable integer ID to every distinct word (token) in the corpus.

    The ID space is built from a deterministic, sorted token list, so the same
    corpus always yields the same IDs regardless of row order or run order.
    Unknown tokens encountered at transform time map to ``-1``.

    Attributes:
        token_to_id_ : dict mapping token -> integer ID
        id_to_token_ : list mapping integer ID -> token
    """

    UNKNOWN_ID = -1

    def __init__(self) -> None:
        self.token_to_id_: dict[str, int] = {}
        self.id_to_token_: List[str] = []

    def fit(self, texts: Sequence[str]) -> "WordIdEncoder":
        """Build the vocabulary from an iterable of raw (or normalized) texts."""
        vocabulary = set()
        for text in texts:
            vocabulary.update(tokenize(normalize_text(text)))
        # Deterministic ordering -> deterministic IDs.
        ordered = sorted(vocabulary)
        self.token_to_id_ = {token: idx for idx, token in enumerate(ordered)}
        self.id_to_token_ = list(ordered)
        return self

    def transform(self, texts: Sequence[str]) -> List[List[int]]:
        """Encode each text as the list of its word IDs (unknown -> -1)."""
        encoded: List[List[int]] = []
        for text in texts:
            ids = []
            for token in tokenize(normalize_text(text)):
                ids.append(self.token_to_id_.get(token, self.UNKNOWN_ID))
            encoded.append(ids)
        return encoded

    def fit_transform(self, texts: Sequence[str]) -> List[List[int]]:
        """Fit the vocabulary and encode in one call."""
        return self.fit(texts).transform(texts)

    @property
    def vocabulary_size(self) -> int:
        """Number of distinct word IDs."""
        return len(self.token_to_id_)

    def id_to_token(self, word_id: int) -> str:
        """Reverse lookup: integer ID -> token."""
        if word_id < 0 or word_id >= len(self.id_to_token_):
            return "<unk>"
        return self.id_to_token_[word_id]

    def get_feature_names(self) -> List[str]:
        """Human readable feature names, one per word ID."""
        return [f"w{idx}:{token}" for idx, token in enumerate(self.id_to_token_)]


# ---------------------------------------------------------------------------
# Lightweight hand-crafted signal features
# ---------------------------------------------------------------------------

# Lexical markers that strongly co-occur with instruction-override attempts.
_JAILBREAK_MARKERS = (
    'ignore all previous', 'ignore previous', 'ignore the above', 'ignore any previous',
    'disregard all', 'disregard previous', 'forget everything', 'forget all previous',
    'no restrictions', 'without any restrictions', 'no ethical', 'without ethics',
    'no filter', 'no censorship', 'uncensored', 'unfiltered', 'bypass', 'override',
    'system prompt', 'developer mode', 'jailbreak', 'jail broken', 'do anything now',
    'you are now', 'act as', 'pretend to be', 'roleplay', 'role-play', 'simulate',
    'begin by saying', 'start by saying', 'must say', 'end your response with',
    'hypothetically', 'in this hypothetical', 'alternate universe', 'alternate reality',
    'fictional scenario', 'for educational purposes', 'purely academic',
)

# Markers that signal a refusal / safety-oriented framing.
_BENIGN_MARKERS = (
    'what is', 'what are', 'how do i', 'how to', 'explain', 'describe', 'summarize',
    'write a', 'list ', 'help me', 'can you tell me', 'difference between',
    'options:', 'which of the following', 'translate', 'define',
)


def marker_count(text: str, markers: Sequence[str]) -> int:
    """Count how many of the given marker phrases occur in the text."""
    s = normalize_text(text)
    return sum(1 for marker in markers if marker in s)


def build_style_features(texts: Sequence[str]) -> np.ndarray:
    """
    Build a small dense matrix of hand-crafted structural features.

    Columns:
      0: character length
      1: word count
      2: unique word count
      3: unique-word ratio (type/token ratio)
      4: number of sentences
      5: number of question marks
      6: number of exclamation marks
      7: number of newlines
      8: number of double-quote characters
      9: number of single-quote characters
      10: jailbreak marker count
      11: benign marker count
      12: jailbreak minus benign marker delta
      13: fraction of non-alphanumeric characters
      14: fraction of uppercase characters in the raw text
      15: number of backticks
      16: number of '{' characters (JSON / template payloads)
    """
    n = len(texts)
    out = np.zeros((n, 17), dtype=np.float64)
    for i, raw in enumerate(texts):
        raw = str(raw) if raw is not None else ""
        norm = normalize_text(raw)
        tokens = tokenize(norm)
        n_chars = len(raw)
        out[i, 0] = n_chars
        out[i, 1] = len(tokens)
        out[i, 2] = len(set(tokens))
        out[i, 3] = out[i, 2] / max(1, out[i, 1])
        out[i, 4] = norm.count('.') + norm.count('!') + norm.count('?')
        out[i, 5] = raw.count('?')
        out[i, 6] = raw.count('!')
        out[i, 7] = raw.count('\n')
        out[i, 8] = raw.count('"')
        out[i, 9] = raw.count("'")
        jail = marker_count(raw, _JAILBREAK_MARKERS)
        benign = marker_count(raw, _BENIGN_MARKERS)
        out[i, 10] = jail
        out[i, 11] = benign
        out[i, 12] = jail - benign
        out[i, 13] = (sum(1 for c in raw if not c.isalnum() and not c.isspace()) / max(1, n_chars))
        out[i, 14] = (sum(1 for c in raw if c.isupper()) / max(1, n_chars))
        out[i, 15] = raw.count('`')
        out[i, 16] = raw.count('{')
    return out


STYLE_FEATURE_NAMES = [
    'char_len', 'word_count', 'unique_words', 'type_token_ratio',
    'sentences', 'question_marks', 'exclamation_marks', 'newlines',
    'double_quotes', 'single_quotes', 'jailbreak_markers', 'benign_markers',
    'marker_delta', 'nonalnum_ratio', 'uppercase_ratio', 'backticks', 'braces',
]
