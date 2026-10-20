"""
3_feature_builder.py
Builds the feature space used by every model in the ensemble.

The feature space is a :class:`sklearn.pipeline.FeatureUnion` of four blocks:

  1. ``word_tfidf``   - word n-grams (1-2) on normalized prompts.
  2. ``char_tfidf``   - character n-grams (2-5) on normalized prompts. This is
     the block that generalises best against obfuscation / typos / zero-width
     character smuggling.
  3. ``word_id``      - the "word ID" feature required by the issue: every word
     is mapped to a stable integer ID, and the ID sequence is summarised as a
     hashed bag-of-ids vector plus width/depth statistics.
  4. ``style``        - dense structural features (length, punctuation, marker
     counts, ...).

All vectorizers are fitted **inside** the cross-validation folds by the
trainer, so no vocabulary, IDF weight or ID mapping is ever learned from a
validation or test row.
"""

from typing import List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from importlib import import_module

_preprocessor = import_module("2_preprocessor")


class WordIdVectorizer(BaseEstimator, TransformerMixin):
    """
    Transform raw prompts into a matrix of word-ID derived features.

    The encoder assigns one integer ID to every distinct word in the corpus
    (see :class:`2_preprocessor.WordIdEncoder`). Each prompt is then encoded as
    the sequence of its word IDs, and the sequence is summarised as:

      * a **hashed bag-of-ids** vector - a linear, non-negative hashing of the
        ID sequence into ``n_features`` buckets. Because hashing is
        order-insensitive but ID-aware, two prompts sharing the same
        vocabulary get similar codes, and the representation is robust to
        unseen words (they simply do not contribute).
      * a few **sequence statistics** (number of IDs, number of distinct IDs,
        average ID, maximum ID, fraction of unknown IDs) which capture
        vocabulary richness and specificity.

    Parameters
    ----------
    n_features : int
        Size of the hashed bag-of-ids space.
    max_id_feature : int
        The vocabulary is *not* limited; all IDs are hashed. This attribute is
        kept for introspection only.
    """

    def __init__(self, n_features: int = 4096, max_id_feature: int = 200000) -> None:
        self.n_features = n_features
        self.max_id_feature = max_id_feature
        self.encoder_: object | None = None
        self.id_to_token_: List[str] = []

    # -- sklearn API --------------------------------------------------------
    def fit(self, texts: Sequence[str], y=None) -> "WordIdVectorizer":
        encoder = _preprocessor.WordIdEncoder()
        encoder.fit(texts)
        self.encoder_ = encoder
        self.id_to_token_ = encoder.id_to_token_
        return self

    def transform(self, texts: Sequence[str]) -> sp.csr_matrix:
        if self.encoder_ is None:
            raise RuntimeError("WordIdVectorizer must be fitted before transform.")
        encoded = self.encoder_.transform(texts)
        n = len(encoded)
        n_features = int(self.n_features)
        n_stats = 5

        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        stats = np.zeros((n, n_stats), dtype=np.float64)

        for i, ids in enumerate(encoded):
            known = [t for t in ids if t >= 0]
            stats[i, 0] = len(ids)
            stats[i, 1] = len(set(known))
            stats[i, 2] = (sum(known) / len(known)) if known else 0.0
            stats[i, 3] = max(known) if known else 0.0
            stats[i, 4] = (len(ids) - len(known)) / max(1, len(ids))
            for tok in known:
                # Hashed bucket, deterministic across runs.
                bucket = (tok * 2654435761) % n_features
                rows.append(i)
                cols.append(bucket)
                vals.append(1.0)

        hashed = sp.csr_matrix(
            (np.asarray(vals, dtype=np.float64), (np.asarray(rows), np.asarray(cols))),
            shape=(n, n_features),
        )
        # L2 normalise the hashed block so the linear models stay well scaled.
        norms = np.sqrt(hashed.multiply(hashed).sum(axis=1)).A.ravel()
        norms[norms == 0] = 1.0
        hashed = sp.diags(1.0 / norms) @ hashed
        return sp.hstack([hashed, sp.csr_matrix(stats)], format="csr")

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = [f"wordid_hash_{i}" for i in range(int(self.n_features))]
        names += ["wordid_n_tokens", "wordid_n_distinct", "wordid_mean_id",
                  "wordid_max_id", "wordid_unknown_ratio"]
        return np.asarray(names, dtype=object)

    # -- introspection ------------------------------------------------------
    @property
    def vocabulary_size(self) -> int:
        return 0 if self.encoder_ is None else self.encoder_.vocabulary_size


def _word_block(ngram_range=(1, 2), min_df=2) -> Pipeline:
    """Word n-gram TF-IDF block.

    The (1,2) configuration was the strongest single lexical feature in the
    grouped 5-fold search (≈0.9596 AUC) and is fast to compute.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=0.995,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
            dtype=np.float32,
        )),
    ])


def _char_block(ngram_range=(2, 6), min_df=2) -> Pipeline:
    """Character n-gram TF-IDF block (robust to typos / obfuscation).

    ``char_wb`` (2,6) with ``min_df=2`` was the strongest single feature in
    the grouped 5-fold search (≈0.9602 AUC) and the best generaliser against
    obfuscation per the prompt-injection detection literature.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=0.995,
            sublinear_tf=True,
            lowercase=True,
            dtype=np.float32,
        )),
    ])


def _char_wide_block(ngram_range=(2, 8), min_df=3) -> Pipeline:
    """Wider character n-gram TF-IDF block for capturing longer-range
    obfuscation patterns.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=0.995,
            sublinear_tf=True,
            lowercase=True,
            dtype=np.float32,
        )),
    ])


def _word_id_block() -> Pipeline:
    """Word-ID block (the issue's "place an id on every single word" feature).

    NOTE: a hashed bag-of-ids representation was measured to *decrease* the
    honest grouped-CV ROC-AUC relative to the raw TF-IDF space (hashing
    collisions + missing IDF reweighting).  This block is therefore retained
    as a documented demonstration / safeguard aid but is **not** wired into the
    predictive feature union of the winning ensemble.  See the notebook's
    "Word-ID demonstration" cell.
    """
    return Pipeline([
        ("wordid", WordIdVectorizer(n_features=4096)),
    ])


def build_feature_union(blocks: Sequence[str] = ("word_tfidf", "char_tfidf")) -> FeatureUnion:
    """
    Construct a feature union from the requested blocks.

    Hyper-parameters are fixed (not tuned per fold) so that the cross-validated
    numbers are honest and the notebook is fully deterministic.

    Parameters
    ----------
    blocks : sequence of str
        Any combination of ``'word_tfidf'``, ``'char_tfidf'``, ``'word_id'``.
    """
    available = {
        "word_tfidf": _word_block,
        "char_tfidf": _char_block,
        "word_id": _word_id_block,
        "char_wide_tfidf": _char_wide_block,
    }
    unknown = set(blocks) - set(available)
    if unknown:
        raise ValueError(f"Unknown feature blocks: {sorted(unknown)}")

    transformer_list = [(name, available[name]()) for name in blocks]
    # The word-ID block is a hashed, L2-normalised representation; down-weight
    # it so it complements rather than dominates the lexical signal.
    weights = {name: (0.5 if name == "word_id" else 1.0) for name in blocks}
    return FeatureUnion(
        transformer_list=transformer_list,
        transformer_weights=weights,
        n_jobs=None,
    )


def build_preprocessor_pipeline() -> Pipeline:
    """
    Build a pipeline that applies :func:`2_preprocessor.normalize_text` to the
    raw ``text`` column before vectorisation.

    The normalizer is a plain callable wrapped in ``FunctionTransformer`` so the
    whole pipeline stays picklable and can be cloned inside CV folds.
    """
    from sklearn.preprocessing import FunctionTransformer

    return Pipeline([
        ("normalize", FunctionTransformer(
            func=_preprocessor.normalize_series,
            feature_names_out="one-to-one",
        )),
    ])


def build_full_pipeline(base_estimator, blocks: Sequence[str] = ("word_tfidf", "char_tfidf", "word_id")) -> Pipeline:
    """
    Compose: normalize -> feature union -> estimator.

    Parameters
    ----------
    base_estimator : sklearn estimator
        Any classifier accepting a sparse matrix.
    blocks : sequence of str
        Feature blocks to activate.
    """
    return Pipeline([
        ("normalize", build_preprocessor_pipeline()),
        ("features", build_feature_union(blocks)),
        ("model", base_estimator),
    ])


def build_style_only_matrix(texts: Sequence[str]) -> np.ndarray:
    """Dense structural features only - used for the logistic stacker's inputs."""
    return _preprocessor.build_style_features(texts)


def summarize_feature_matrix(texts: Sequence[str]) -> Tuple[int, int, int]:
    """
    Fit a feature union on ``texts`` and report its shape.

    Returns:
        (n_rows, n_features, vocabulary_size)
    """
    union = build_feature_union()
    matrix = union.fit_transform(list(texts))
    vocab = union.transformer_list[0][1].named_steps["tfidf"].vocabulary_
    return matrix.shape[0], matrix.shape[1], len(vocab)


def build_word_id_encoder(texts: Sequence[str]):
    """Fit and return a :class:`2_preprocessor.WordIdEncoder` on ``texts``."""
    encoder = _preprocessor.WordIdEncoder()
    encoder.fit(list(texts))
    return encoder
