# Jailbreak prompt classification

Binary classification of prompts as **jailbreak** (`1`) or **benign** (`0`).
`modeltrain.ipynb` trains the model, prints grouped cross-validation metrics,
runs its own test suite, and exports `submission.csv` with `Id,TARGET`, where
`TARGET` is the calibrated jailbreak likelihood in `[0, 1]`.

## Result

| Model | Grouped 5-fold CV ROC-AUC (2 seeds, 10 folds) |
| --- | --- |
| Baseline: word-unigram TF-IDF + logistic regression | **0.9418** ± 0.0075 |
| **Ensemble (6 models, rank fusion)** | **0.9724** ± 0.0034 |

The ensemble improves the internal cross-validated ROC-AUC by **+0.0306** over
the baseline. These numbers are *internal estimates*, not public leaderboard
scores; test labels are not part of this repository. The previous submission
referenced in issue #5 scored `0.94755` publicly, so this pipeline is a
substantial modelling change rather than a re-export of the same model.

## Data

| File | Rows | Columns |
| --- | --- | --- |
| `train.csv` | 4,000 | `Id`, `text`, `label` (`benign` 2,401 / `jailbreak` 1,599) |
| `test.csv` | 1,000 | `Id`, `text` |
| `sample_submission.csv` | 1,000 | `Id`, `TARGET` |

Every run performs these safeguards before training:

- schema, missing-value, duplicate-id, and train/test id-disjoint checks;
- strict label parsing (`benign`/`jailbreak`, `0`/`1`, `true`/`false`); unknown
  labels and single-class training files abort the run;
- Unicode NFKC normalisation, zero-width/bidi character removal, case and
  whitespace unification;
- exact-duplicate, near-duplicate, and conflicting-label detection (near
  duplicates are grouped so they cannot leak across CV folds);
- train/test near-duplicate audit (14 test prompts are ≥ 0.90 cosine-similar to
  a training prompt);
- output validation: exact `Id,TARGET` schema, matching id order, finite
  scores, non-constant scores, and scores inside `[0, 1]`.

## Method

1. **Text normalisation** — NFKC, invisible character removal, lowercasing,
   whitespace/newline cleanup. Punctuation is preserved because it carries
   injection signal.
2. **Word IDs** — every distinct token gets a stable integer id (21,017-word
   vocabulary; 97.0% of test tokens are covered). Repeated words share an id,
   unseen words map to `-1`.
3. **Features** — a `FeatureUnion` of word TF-IDF (1–3-grams, 80,802 terms) and
   character TF-IDF (2–6-grams, 113,222 terms), giving a 194,024-column sparse
   matrix. Character n-grams catch obfuscation such as `ign0re`, spaced letters,
   and punctuation tricks.
4. **Prompt-injection safeguards** — 46 dense features covering 14 attack
   families: instruction override, system-prompt probing, password/secret
   extraction, repeat/reverse/encode requests, role-play, fictional framing,
   unrestricted persona ("DAN", "no restrictions"), new directives, harmful
   asks, compliance pressure, known benign benchmark formats, Gandalf-style
   character probes, and injection formatting markers. These are appended to
   the TF-IDF matrix for the boosting model and reported as a diagnostic table.
5. **Base models**
   - logistic regression (`C=32`),
   - linear SVM (`C=1`),
   - complement Naive Bayes (`alpha=0.3`),
   - cosine k-NN with `k=10` and `k=20` (similarity-weighted),
   - LightGBM (400 trees) on TF-IDF plus safeguard features.
6. **Validation** — stratified **grouped** 5-fold CV, repeated with seeds 42 and
   7. Near-duplicate prompts (character TF-IDF cosine ≥ 0.80) always share a
   fold, so duplicate leakage cannot inflate the estimate.
7. **Ensemble** — equal-weight percentile-rank fusion of the six models. Rank
   fusion is scale-free, which matters because the SVM outputs margins while the
   other models output probabilities.
8. **Calibration** — Platt scaling fitted on the cross-validated ensemble ranks
   maps the final test ranks into `[0, 1]` without changing their order (AUC is
   rank-based).

## Notebook cells

`modeltrain.ipynb` contains **30 Python cells** (plus 11 markdown cells, 41 in
total). Every cell is small and focused: environment, configuration, data
loading/validation, normalisation, duplicates, word IDs, features, model
factories, folds, baseline CV, ensemble CV, results, full training, test
prediction, blending, calibration, export, validation, self-tests, report, and a
final summary. The count is also printed by the last cell and stored in
`training_report.json`.

## Files

| File | Role |
| --- | --- |
| `modeltrain.ipynb` | End-to-end notebook, saved with outputs from a clean execution |
| `step_01_config.py` | Paths, seeds, CV/folds, model hyper-parameters, blend weights |
| `step_02_data.py` | CSV loading, label parsing, schema/id/disjointness validation |
| `step_03_normalize.py` | Normalisation, tokenisation, word IDs, duplicate and conflict detection |
| `step_04_features.py` | TF-IDF union, 46 safeguard features, diagnostic tables |
| `step_05_models.py` | Model factories, matrix construction, score extraction |
| `step_06_validation.py` | Grouped folds, per-model and ensemble cross-validation |
| `step_07_ensemble.py` | Rank normalisation, weighted fusion, Platt calibration, full-data prediction |
| `step_08_submission.py` | Submission construction, validation, atomic CSV writing |
| `step_09_selfcheck.py` | 14 self-tests, including a synthetic end-to-end training run |
| `execute_notebook.py` | Runs the notebook in a fresh kernel and reports metrics |
| `submission.csv` | 1,000-row `Id,TARGET` export |
| `training_report.json` | Data audits, CV metrics, environment, checks, notebook cell counts |
| `requirements.txt` | Verified dependency versions |

## Reproduce

From the repository root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python execute_notebook.py
```

The run takes roughly 13 minutes on 2 CPU cores, regenerates the notebook
outputs, `submission.csv`, and `training_report.json`, and fails on any cell
error. The final lines report:

```
Cells executed: 30 Python, 11 markdown
Grouped CV ROC-AUC: baseline=0.94176, ensemble=0.97237
Self-checks: 14/14 passed
Notebook Python cells: 30
```

Alternatively, open `modeltrain.ipynb` in Jupyter and run all cells in order.

## Submission schema

```
Id,TARGET
1,0.451340
4,0.421665
...
```

- `Id` matches `test.csv` and `sample_submission.csv` order exactly.
- `TARGET` is a calibrated score in `[0, 1]`; higher means more jailbreak-like.
- 39.6% of test prompts score ≥ 0.5 and would be labelled `jailbreak` at that
  threshold (the training positive rate is 40.0%).

## What is verified

`step_09_selfcheck.py` runs 14 checks, all passing in the saved notebook:

1. text normalisation (NFKC, zero-width/bidi stripping, idempotence),
2. word-ID construction, reuse, unknown handling, and tokenisation,
3. strict label parsing with rejection of unknown labels,
4. training-frame schema / single-class / duplicate-id guards,
5. test-frame and sample-submission alignment guards,
6. 46 finite safeguard features with the expected direction on a planted
   instruction-override prompt,
7. TF-IDF union shape and term statistics,
8. rank normalisation and weighted fusion bounds/monotonicity,
9. Platt calibration bounds and monotonicity,
10. grouped-fold coverage and duplicate-group isolation,
11. exact/near duplicate and conflicting-label detection,
12. submission round-trip, wrong-order / NaN / out-of-range / constant guards,
13. model factory construction and a separable synthetic AUC of 1.000,
14. deterministic model fits and deterministic folds.

## Limitations

- Cross-validation cannot guarantee the public leaderboard score; only a
  submission can measure that.
- Near-duplicate grouping uses a fixed 0.80 cosine threshold.
- The model sees a single prompt, not conversation history; benign role-play and
  adversarial role-play can be textually nearly identical.
- Platt calibration is fitted on out-of-fold ranks; it preserves ordering (and
  therefore ROC-AUC) but is not a full independent probability audit.
- No external or fabricated training data is used, and the test prompts are never
  given labels.
