"""Feature extraction and explicit prompt-injection safeguards (step 4).

Two complementary representations are built from the *normalised* prompt:

1. ``build_tfidf_union`` - sparse word (1-3) + character (2-6) TF-IDF.
   Character n-grams catch obfuscation (``ign0re``, spaced-out words, unusual
   punctuation) that word n-grams miss.
2. ``handcrafted_features`` - dense counts/densities for families of known
   jailbreak safeguards (instruction override, password extraction, role-play,
   "unrestricted persona", harmful asks, known benchmark formats, ...).

The handcrafted block is appended to the sparse matrix for the boosting model
and is also reported as a per-class diagnostic table in the notebook.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

from step_01_config import (
    CHAR_NGRAM_RANGE,
    MIN_DF,
    SUBLINEAR_TF,
    WORD_NGRAM_RANGE,
)

# Each pattern is a "safeguard family" that catches a widely used jailbreak
# behaviour.  The list is intentionally broad but each entry is documented.
SAFEGUARD_PATTERNS: dict[str, str] = {
    "instruction_override": (
        r"\b(?:ignore|ignoring|disregard|forget|override|bypass)\b"
        r"[^.\n]{0,40}\b(?:previous|prior|above|earlier|all|any|your|the)\b"
        r"[^.\n]{0,30}\b(?:instruction|direction|rule|prompt|guideline|"
        r"restriction|filter|programming|command|order)"
    ),
    "instruction_words": (
        r"\b(?:ignore|disregard|forget)\b.{0,30}"
        r"\b(?:instruction|direction|rule|guideline|restriction|filter|prompt)s?\b"
    ),
    "system_prompt": (
        r"\b(?:system prompt|system message|initial prompt|original instruction|"
        r"your instruction|your programming|your guideline|your rule)"
    ),
    "secret_extraction": (
        r"\b(?:password|passcode|secret|hidden message|secret key|api key|credential)\b"
    ),
    "repeat_or_encode_secret": (
        r"\b(?:repeat|print|output|tell|reveal|spell|translate|reverse|summar\w*|"
        r"encode|decode|hex|base64)\b.{0,40}"
        r"\b(?:password|secret|instruction|prompt|word|character|letter|first|"
        r"last|previous)"
    ),
    "roleplay": (
        r"\b(?:pretend|roleplay|role-play|role play|act as|you are now|you will act|"
        r"assume the role|impersonate|simulate|emulate|respond as)\b"
    ),
    "fictional_framing": (
        r"\b(?:fictional|hypothetical|imaginary|story|scenario|simulation|"
        r"for research|educational purpose|thought experiment|game)\b"
    ),
    "unrestricted_persona": (
        r"\b(?:unrestricted|unfiltered|uncensored|without restriction|"
        r"no restriction|no rule|no guideline|no filter|free of|free from|"
        r"unbound|unleash|jailbr\w*|dan|developer mode|do anything now|evil|"
        r"amoral|immoral|unethical)\b"
    ),
    "new_directives": (
        r"\b(?:from now on|henceforth|for the rest of|starting now|new rule|"
        r"new instruction|new directive|updated instruction)\b"
    ),
    "harmful_request": (
        r"\b(?:weapon|bomb|explosive|poison|meth|cocaine|drug|kill|murder|attack|"
        r"hack|malware|virus|fraud|scam|steal|theft|credit card|social security|"
        r"suicide|self-harm|child|terror)\b"
    ),
    "compliance_pressure": (
        r"\b(?:do not refuse|must comply|you must|you will|i command|obey|comply|"
        r"answer me|no warning|without warning|do not mention|do not apologize|"
        r"never refuse)\b"
    ),
    "benign_benchmark_format": (
        r"\b(?:options?|premise|hypothesis|stream of consciousness|step-by-step|"
        r"let us think|lets think|is the hypothesis|can we conclude|entailed|"
        r"question:|answer:|student:|teacher:|passage|document id|snippet|"
        r"i have been asked the following|provided contextual information|"
        r"url:|title:)\b"
    ),
    "gandalf_style_probe": (
        r"\b(?:the password|first letter|last letter|second letter|third letter|"
        r"each character|character by character|letters of the|characters in)\b"
    ),
    "injection_formatting": (
        r"(?:\[{2,}|<\|.{0,20}\|>|###|---|\*\*\*|```|"
        r"^\s*system\s*:|^\s*assistant\s*:|^\s*user\s*:)"
    ),
}

_COMPILED = {
    name: re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    for name, pattern in SAFEGUARD_PATTERNS.items()
}


def build_tfidf_union(
    word_ngram_range: tuple[int, int] = WORD_NGRAM_RANGE,
    char_ngram_range: tuple[int, int] = CHAR_NGRAM_RANGE,
    min_df: int = MIN_DF,
) -> FeatureUnion:
    """Return an (unfitted) word + character TF-IDF feature union."""
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=word_ngram_range,
                    min_df=min_df,
                    sublinear_tf=SUBLINEAR_TF,
                    strip_accents=None,
                    lowercase=False,  # text is already normalised
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=char_ngram_range,
                    min_df=min_df,
                    sublinear_tf=SUBLINEAR_TF,
                    lowercase=False,
                ),
            ),
        ]
    )


def build_baseline_vectorizer() -> TfidfVectorizer:
    """Word-unigram TF-IDF used by the reference baseline."""
    from step_01_config import BASELINE_PARAMS

    return TfidfVectorizer(
        ngram_range=BASELINE_PARAMS["word_ngram_range"],
        min_df=BASELINE_PARAMS["min_df"],
        sublinear_tf=SUBLINEAR_TF,
        lowercase=False,
    )


def _prompt_features(text: str) -> list[float]:
    n_chars = max(len(text), 1)
    words = re.findall(r"[a-z0-9']+", text)
    n_words = max(len(words), 1)
    lines = text.split("\n")
    features = [
        np.log1p(n_chars),
        np.log1p(n_words),
        np.log1p(len(lines)),
        n_chars / n_words,
        sum(ch.isdigit() for ch in text) / n_chars,
        sum(ch.isupper() for ch in text) / n_chars,
        sum(ch in "!?" for ch in text) / n_chars,
        sum(ch in ".,;:" for ch in text) / n_chars,
        sum(ch == '"' for ch in text) / n_chars,
        sum(ch == "'" for ch in text) / n_chars,
        text.count("-") / n_chars,
        text.count("*") / n_chars,
        text.count("#") / n_chars,
        text.count("/") / n_chars,
        sum(ord(ch) > 127 for ch in text) / n_chars,
        len(set(words)) / n_words,
        (len(text) - len(text.rstrip())) / n_chars,
        len(set(re.findall(r"[aeiou]", text))) / n_chars,
    ]
    for pattern in _COMPILED.values():
        matches = pattern.findall(text)
        features.append(np.log1p(len(matches)))
        features.append(np.log1p(len(matches)) / (n_chars / 1000.0))
    return features


_BASE_FEATURE_NAMES = [
    "log_chars",
    "log_words",
    "log_lines",
    "avg_word_length",
    "digit_ratio",
    "upper_ratio",
    "question_exclamation_ratio",
    "punctuation_ratio",
    "double_quote_ratio",
    "single_quote_ratio",
    "dash_ratio",
    "star_ratio",
    "hash_ratio",
    "slash_ratio",
    "non_ascii_ratio",
    "unique_word_ratio",
    "trailing_whitespace",
    "vowel_diversity",
]
HANDCRAFTED_FEATURE_NAMES = tuple(
    _BASE_FEATURE_NAMES
    + [f"{name}_count" for name in SAFEGUARD_PATTERNS]
    + [f"{name}_density" for name in SAFEGUARD_PATTERNS]
)


def handcrafted_features(texts: Iterable[str]) -> np.ndarray:
    """Return the dense safeguard/format feature matrix (rows x features)."""
    rows = [_prompt_features(str(text)) for text in texts]
    return np.asarray(rows, dtype=np.float64)


def safeguard_signal_table(
    texts: Sequence[str], labels: Sequence[int]
) -> pd.DataFrame:
    """Per-safeguard mean count by class plus point-biserial correlation."""
    matrix = handcrafted_features(texts)
    labels = np.asarray(labels, dtype=np.float64)
    records = []
    for index, name in enumerate(HANDCRAFTED_FEATURE_NAMES):
        column = matrix[:, index]
        if column.std() == 0:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(column, labels)[0, 1])
        records.append(
            {
                "feature": name,
                "mean_benign": float(column[labels == 0].mean()),
                "mean_jailbreak": float(column[labels == 1].mean()),
                "correlation": correlation,
            }
        )
    return pd.DataFrame(records)


def top_safeguard_signals(
    texts: Sequence[str], labels: Sequence[int], n: int = 12
) -> pd.DataFrame:
    """Return the ``n`` safeguard features most correlated with the label."""
    table = safeguard_signal_table(texts, labels)
    return (
        table.reindex(table["correlation"].abs().sort_values(ascending=False).index)
        .head(n)
        .reset_index(drop=True)
    )
