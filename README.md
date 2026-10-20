# Jailbreak prompt classification

Binary classifier that separates **jailbreak** prompts from **benign** prompts and
exports a jailbreak likelihood in `[0, 1]` for every prompt in `test.csv`.

**Offline result: grouped 5-fold cross-validation ROC-AUC = `0.96421`**
(95% bootstrap CI `[0.95932, 0.96908]`), accuracy `0.8955` and F1 `0.8727` at the
`TARGET >= 0.5` operating point.

> The offline number is an estimate computed on the training split with
> near-duplicate-aware folds. The leaderboard score is computed on the hidden
> test labels and may differ.

## Export schema

| Column | Meaning |
| --- | --- |
| `Id` | the prompt id, copied verbatim from `test.csv` (1000 rows) |
| `TARGET` | jailbreak likelihood in `[0, 1]`; `TARGET >= 0.5` -> **jailbreak**, `TARGET < 0.5` -> **benign** |

The exported `submission.csv` is validated against `sample_submission.csv`
(columns, row count, id set, row order) before it is written.

## Files

| File | Role |
| --- | --- |
| `modeltrain.ipynb` | The project. 11 code cells, saved with the outputs of a successful fresh-kernel execution. |
| `1_data_loader.py` | Load `train.csv` / `test.csv`, validate the schema, assert there is no train/test leakage. |
| `2_preprocessor.py` | Unicode/whitespace normalisation, the **word-ID** encoder, style features. |
| `3_feature_builder.py` | TF-IDF word / character feature blocks and the full feature union. |
| `4_model_trainer.py` | Near-duplicate grouping, base models, grouped CV, stacking. |
| `5_evaluator.py` | ROC-AUC and supporting metrics, threshold sweep, bootstrap CI. |
| `6_submission_exporter.py` | Build / validate / write `submission.csv`. |
| `execute_notebook.py` | Execute the notebook in a fresh kernel; fails on any cell error. |
| `requirements.txt` | Dependency versions used for the verified run. |
| `submission.csv` | The exported predictions (1000 rows). |

The Python modules are named **in list order** and are imported directly by the
notebook cells (`importlib.import_module("1_data_loader")`, ...).

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python execute_notebook.py --save
```

`execute_notebook.py` runs the notebook in a fresh kernel using the active
interpreter, raises on the first failing cell, and only with `--save` writes the
executed notebook (with outputs) back to disk. No internet, API token, GPU or
model download is required.

## How the number was reached

### 1. Safeguards

The issue asked for more safeguards on the supplied CSVs, so the data is checked
*before* any model is fitted, and every model is fitted inside a leakage-aware
cross-validation loop.

* **Schema and label checks** - required columns, no missing text, ids unique and
  within range, labels restricted to `benign` / `jailbreak`.
* **No train/test leakage** - normalised (NFKC, lowercased, whitespace-collapsed)
  train and test prompts are intersected; the overlap must be empty. It is empty
  for the supplied data (0 shared prompts of 4000 / 1000).
* **No conflicting duplicates** - no prompt appears twice with two different
  labels.
* **Near-duplicate grouping** - prompts that are near-identical (character
  n-gram cosine similarity >= 0.80) are clustered with union-find and pinned to
  a single CV fold. Without this, a model can memorise a template from the
  training half and then "predict" its twin in the validation half, which
  inflates the offline score and hides the real generalisation gap. The
  supplied data yields 3959 groups over 4000 rows.
* **Fold-local fitting** - every vectorizer, IDF weight and word-ID map is fitted
  inside the fold. No vocabulary is ever learned from a validation or test row.
* **Out-of-fold stacking** - the meta-learner is fitted on out-of-fold
  predictions only, so it never sees a base model's prediction for a row that
  base model was trained on.

### 2. Features

Two complementary views of each prompt:

| Block | Configuration | Why |
| --- | --- | --- |
| `word_tfidf` | word 1-2 grams, `min_df=2`, sublinear TF, `strip_accents='unicode'` | captures the vocabulary a jailbreak uses |
| `char_tfidf` | `char_wb` 2-6 grams, `min_df=2`, sublinear TF | captures the *shape* of the text and survives typos, casing tricks and zero-width characters |

The character view is the stronger single model and is also the one that
generalises best against obfuscation, which matches the prompt-injection
detection literature (word-level TF-IDF + LR and character-level TF-IDF + SVM
are the standard strong baselines, and the character family is consistently
reported as the better generaliser).

### 3. Model

A four-member stack, each member cloned and refitted per fold:

| Member | Feature block | Estimator | Out-of-fold ROC-AUC |
| --- | --- | --- | --- |
| `word_lr` | `word_tfidf` | LogisticRegression, `C=64` | 0.95809 |
| `char_lr` | `char_tfidf` | LogisticRegression, `C=32` | 0.96024 |
| `char_svm` | `char_tfidf` | LinearSVC, `C=2` | 0.95877 |
| `word_svm` | `word_tfidf` | LinearSVC, `C=1` | 0.95822 |
| **stacked** | - | LogisticRegression meta-learner | **0.96421** |

The four members disagree on different prompts, which is where the stack
earns its lift over the best single model (0.96024 -> 0.96421).

### 4. Word IDs

The issue hints that *"you can place an id on every single word in a phrase"*.
`2_preprocessor.WordIdEncoder` implements exactly that: every distinct word in
the corpus is assigned a stable integer ID, so a prompt becomes a sequence of
IDs. The ID map is deterministic (tokens are sorted before IDs are handed out)
and is fitted on the training split only, so it is reproducible and leak-free.

The notebook demonstrates the scheme (vocabulary size, token -> ID table, a
worked example) and `2_preprocessor.build_style_features` adds dense structural
signals (length, punctuation, marker counts).

A hashed bag-of-ids block was also measured **as a predictive feature**. It
*lowered* the grouped-CV ROC-AUC (0.96003 -> 0.95881, a delta of `-0.00122`),
because hashing collides and the representation loses IDF reweighting. The
word-ID block is therefore deliberately **excluded from the predictive feature
union** - the numbers behind that decision are printed by Cell 5 of the
notebook, and the encoder is retained for inspection and for the near-duplicate
safeguard.

## What the notebook checks

`execute_notebook.py` reports `OK: 11 code cells executed`. The last cell runs
15 self-tests and fails the notebook if any of them breaks:

* `submission.csv` exists and has exactly the columns `['Id', 'TARGET']`;
* row count, id set and id order match `test.csv` **and** `sample_submission.csv`;
* no duplicate ids, no missing `TARGET`, every `TARGET` inside `[0, 1]`;
* the score column is not constant, and both classes are predicted;
* the exported file re-reads cleanly from disk;
* the grouped-CV ROC-AUC is at least `0.96`;
* every base model beats the 0.5 chance line;
* there is no normalised train/test text overlap;
* every prompt tokenises into at least one word ID.

Last verified run: **15/15 checks passed**.

## Caveats

* The dataset contains genuinely ambiguous rows - many `benign` prompts are
  role-play or fictional framings that superficially resemble jailbreaks
  (175 of 2401 `benign` training rows contain an explicit jailbreak-style
  keyword). This label noise, not model capacity, is the binding constraint on
  the achievable ROC-AUC.
* The offline score is measured on the training split. It is not a leaderboard
  result and no submission is uploaded automatically.
* `TARGET` is a model estimate. Independent probability calibration was not
  established, although the logistic stacker is calibrated by construction and
  the score distribution is reported in the notebook.
