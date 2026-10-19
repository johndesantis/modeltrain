# Jailbreak classification notebook

## Current result — important limitation

The only supplied dataset is `jailbreak.csv`: **666 rows, all labeled `true`**.
It has no benign examples, competition test prompts, or original prompt IDs/names.
After case/Unicode/whitespace normalization, 648 unique prompts remain for fitting.
The original input file is unchanged.

**It is not possible to train or validate a meaningful binary classifier from these
labels alone.** No outside or fabricated training data is used. The notebook fits
a single-class empirical-prior `DummyClassifier`. Consequently, every exported
`jailbreak_probability` is **1.0**. These are constant baseline outputs, **not
calibrated jailbreak risks**, perfect accuracy, or evidence of generalization.
ROC-AUC is undefined and stored as `null`, not assigned a fictitious score.

`submissions.csv` is an export of the existing training rows, **not a leaderboard-ready
submission**. No competition test set, sample submission schema, or leaderboard
result was available.

## Files

- `modeltrain.ipynb`: all data validation, training, evaluation, export, and self-tests;
  saved with outputs from a successful fresh-kernel execution.
- `submissions.csv`: all 666 input rows in their original order, including duplicates.
- `training_report.json`: input checksum, model mode, class counts, evaluation
  limitations, identifier provenance, and passed test names.
- `execute_notebook.py`: executes the notebook in a fresh kernel using the active
  Python interpreter, fails on cell errors, and saves successful execution outputs.
- `requirements.txt`: exact dependency versions used for the verified run (Python 3.11).

## Reproduce

From the repository root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python execute_notebook.py
```

This regenerates the notebook outputs, `submissions.csv`, and `training_report.json`.
After dependency installation, no internet, API token, GPU, or model download is
required. Alternatively, open the notebook in a Jupyter-compatible application with
these dependencies installed and run all cells in order.

A successful run prints `PASS: 40 checks`. Tests include both single-class cases,
the binary ensemble and grouped cross-validation path, deterministic refitting,
invalid inputs, conflicting duplicates, score bounds, metadata preservation, CSV
round trips, strict binary-label enforcement, and word-ID/TF-IDF consistency. Synthetic test fixtures exercise software behavior only; they are never
added to actual training data or reported as real validation performance.

## Strict binary mode

To prevent accidentally accepting the single-class baseline as a trained binary
classifier, run:

```bash
.venv/bin/python execute_notebook.py --require-binary
```

With the supplied positive-only CSV, this command **intentionally exits with an
error before fitting or writing artifacts**. It leaves existing notebook outputs,
CSV, and report unchanged. In an interactive notebook, set `REQUIRE_BINARY = True`
in the configuration cell for the same behavior. Normal mode explicitly permits
the documented baseline export.

## Word-ID demonstration

The notebook now fits an exploratory unigram TF-IDF vocabulary on the actual
supplied text: **8,709 word IDs**, a **648 × 8,709** sparse matrix, and **98,740**
nonzero weights. It displays token-to-ID pairs and the strongest TF-IDF features
for an example prompt. Repeated normalized words share an ID; single-letter words
are included; unknown tokens use `-1` in the demonstration encoder and are ignored
in vectorizer transforms.

This is unsupervised feature extraction, not successful binary classifier training.
The exploratory vectorizer is never reused inside classifier evaluation: each
training fold builds its own vocabulary to avoid leakage.

## Export schema

| Column | Meaning for the supplied file |
| --- | --- |
| `prompt_id` | Generated one-based source-row identifier, e.g. `row_000001`; not a competition ID. Stable for unchanged source-row order. |
| `prompt_name` | Generated whitespace-normalized excerpt of the prompt, at most 100 characters; not original metadata and not necessarily unique. |
| `jailbreak_probability` | Constant empirical-prior output `1.0` for this single-class dataset; not a validated risk estimate. |

Existing `prompt_id` and `prompt_name` columns are preserved if supplied in future
inputs; invalid blank or duplicate IDs are rejected. Full original prompts remain
in `jailbreak.csv`.

## Binary training path when real data becomes available

The notebook already implements an equal-weight soft-voting ensemble of word
unigram/bigram TF-IDF and character 3–5-gram TF-IDF logistic regressions. This path
runs only if the labeled input actually contains both benign (`0`/`false`/`benign`)
and jailbreak (`1`/`true`/`jailbreak`) examples.

It uses fixed hyperparameters and up to five stratified group folds. Vocabulary
and IDF fitting happen inside each fold. Normalized duplicates are deduplicated;
validation groups additionally collapse digit variants. Source/platform/community
metadata is excluded from features. Per-model and ensemble out-of-fold ROC-AUC are
reported only when class/group support permits valid folds. Semantic paraphrases
can still cross groups; an independently collected holdout is necessary.

For an actual competition, supply real two-class training data and the actual test
CSV, set the notebook's `TRAIN_CSV` and `TEST_CSV` explicitly, and verify the export
schema and IDs against the competition sample submission. Predicted probabilities
from the binary path are model estimates; independent calibration has not been
established. No leaderboard submission is uploaded automatically.
