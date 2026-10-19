"""Text normalisation, word IDs, duplicates, and dataset safeguards (step 3).

Hints from the task: "you can place an id on every single word in a phrase".
``build_word_ids`` assigns a stable integer id to every distinct token in the
training corpus, including stop words and single letters, and
``encode_words`` maps a prompt to that id sequence (unknown tokens -> -1).
"""

from __future__ import annotations

import re
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from step_01_config import NEAR_DUP_THRESHOLD

# Invisible characters that attackers insert to break keyword matching.
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)
_BIDI_CONTROLS = re.compile(r"[\u202a-\u202e\u2066-\u2069]")
_HORIZONTAL_WS = re.compile(r"[ \t\f\v]+")
_SPACE_AROUND_NEWLINE = re.compile(r" *\n *")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def normalize_text(text: object) -> str:
    """Normalise one prompt deterministically and idempotently.

    Steps: Unicode NFKC (folds full-width / compatibility characters), remove
    zero-width and bidi control characters, unify newlines, lowercase, collapse
    horizontal whitespace, trim.  Punctuation is preserved because it carries
    prompt-injection signal.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    value = value.translate(_ZERO_WIDTH)
    value = _BIDI_CONTROLS.sub("", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.lower()
    value = _HORIZONTAL_WS.sub(" ", value)
    value = _SPACE_AROUND_NEWLINE.sub("\n", value)
    value = _MANY_NEWLINES.sub("\n\n", value)
    return value.strip()


def normalize_texts(texts: Iterable[object]) -> list[str]:
    """Normalise a sequence of prompts."""
    return [normalize_text(text) for text in texts]


def tokenize(text: str) -> list[str]:
    """Split normalised text into word tokens (apostrophes kept)."""
    return _TOKEN_RE.findall(text)


def build_word_ids(texts: Sequence[str], max_words: int | None = None) -> dict[str, int]:
    """Assign an id to every distinct word, ordered by first appearance.

    ``max_words`` optionally caps the vocabulary (ids are then 0..max_words-1).
    """
    vocabulary: "OrderedDict[str, int]" = OrderedDict()
    for text in texts:
        for token in tokenize(normalize_text(text)):
            if token not in vocabulary:
                if max_words is not None and len(vocabulary) >= max_words:
                    continue
                vocabulary[token] = len(vocabulary)
    return dict(vocabulary)


def encode_words(text: str, word_ids: dict[str, int]) -> list[int]:
    """Map a prompt to word ids; unseen words map to ``-1``."""
    return [word_ids.get(token, -1) for token in tokenize(normalize_text(text))]


def word_id_examples(
    texts: Sequence[str], word_ids: dict[str, int], n_prompts: int = 2
) -> list[dict[str, object]]:
    """Return token/id pairs for the first ``n_prompts`` prompts."""
    examples = []
    for text in texts[:n_prompts]:
        tokens = tokenize(normalize_text(text))
        examples.append(
            {
                "prompt_excerpt": normalize_text(text)[:160],
                "tokens": tokens[:20],
                "ids": [word_ids.get(t, -1) for t in tokens[:20]],
                "token_count": len(tokens),
            }
        )
    return examples


def word_id_coverage(texts: Sequence[str], word_ids: dict[str, int]) -> float:
    """Fraction of tokens in ``texts`` that exist in the vocabulary."""
    total = known = 0
    for text in texts:
        for token in tokenize(normalize_text(text)):
            total += 1
            known += int(token in word_ids)
    return float(known / total) if total else 0.0


@dataclass(frozen=True)
class DuplicateReport:
    """Summary of duplicate / near-duplicate structure in a text corpus."""

    rows: int
    exact_duplicate_rows: int
    near_duplicate_groups: int
    rows_in_near_duplicate_groups: int
    largest_group: int
    group_ids: np.ndarray

    def to_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "exact_duplicate_rows": self.exact_duplicate_rows,
            "near_duplicate_groups": self.near_duplicate_groups,
            "rows_in_near_duplicate_groups": self.rows_in_near_duplicate_groups,
            "largest_group": self.largest_group,
        }


def near_duplicate_groups(
    texts: Sequence[str],
    threshold: float = NEAR_DUP_THRESHOLD,
    chunk_size: int = 512,
) -> np.ndarray:
    """Cluster near-duplicate prompts with character TF-IDF cosine similarity.

    Two prompts are linked when their cosine similarity is at least
    ``threshold``; connected components give the final group ids.  Similarity
    is computed in row chunks so memory stays bounded.
    """
    if len(texts) == 0:
        return np.zeros(0, dtype=np.int64)
    normalized = normalize_texts(texts)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(normalized)
    rows, cols = [], []
    for start in range(0, matrix.shape[0], chunk_size):
        block = cosine_similarity(matrix[start : start + chunk_size], matrix)
        local_rows = np.arange(block.shape[0]) + start
        for offset, row in enumerate(block):
            neighbours = np.flatnonzero(row >= threshold)
            neighbours = neighbours[neighbours > local_rows[offset]]
            rows.extend([local_rows[offset]] * len(neighbours))
            cols.extend(neighbours.tolist())
    if rows:
        graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(texts), len(texts)))
    else:
        graph = csr_matrix((len(texts), len(texts)))
    _, group_ids = connected_components(graph, directed=False)
    return group_ids.astype(np.int64)


def duplicate_report(
    texts: Sequence[str], threshold: float = NEAR_DUP_THRESHOLD
) -> DuplicateReport:
    """Detect exact duplicates and near-duplicate clusters."""
    normalized = normalize_texts(texts)
    exact_duplicates = int(len(normalized) - len(set(normalized)))
    groups = near_duplicate_groups(texts, threshold=threshold)
    if len(groups):
        sizes = np.bincount(groups)
        near_groups = int((sizes > 1).sum())
        rows_in_groups = int(sizes[sizes > 1].sum())
        largest = int(sizes.max())
    else:
        near_groups = rows_in_groups = largest = 0
    return DuplicateReport(
        rows=len(texts),
        exact_duplicate_rows=exact_duplicates,
        near_duplicate_groups=near_groups,
        rows_in_near_duplicate_groups=rows_in_groups,
        largest_group=largest,
        group_ids=groups,
    )


def label_conflicts(labels: Sequence[int], groups: np.ndarray) -> dict[str, object]:
    """Find near-duplicate groups whose members disagree on the label."""
    labels = np.asarray(labels, dtype=np.int64)
    conflicting_groups: list[int] = []
    for group in np.unique(groups):
        members = labels[groups == group]
        if members.size > 1 and len(np.unique(members)) > 1:
            conflicting_groups.append(int(group))
    return {
        "conflicting_groups": len(conflicting_groups),
        "conflicting_rows": int(np.isin(groups, conflicting_groups).sum())
        if conflicting_groups
        else 0,
        "group_ids": conflicting_groups[:20],
    }


def cross_corpus_near_duplicates(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    threshold: float = 0.90,
) -> dict[str, object]:
    """Check how many test prompts closely match a training prompt."""
    if not len(train_texts) or not len(test_texts):
        return {"pairs_above_threshold": 0, "threshold": threshold, "max_similarity": 0.0}
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(normalize_texts(list(train_texts) + list(test_texts)))
    train_matrix = matrix[: len(train_texts)]
    test_matrix = matrix[len(train_texts) :]
    max_similarity = 0.0
    pairs = 0
    for start in range(0, test_matrix.shape[0], 256):
        block = cosine_similarity(test_matrix[start : start + 256], train_matrix)
        block_max = float(block.max()) if block.size else 0.0
        max_similarity = max(max_similarity, block_max)
        pairs += int((block >= threshold).any(axis=1).sum())
    return {
        "pairs_above_threshold": pairs,
        "rows_with_match_above_threshold": pairs,
        "threshold": threshold,
        "max_similarity": max_similarity,
    }
