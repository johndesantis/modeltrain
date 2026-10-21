# Jailbreak prompt classifier — issue #5

`modeltrain.ipynb` is the complete, standalone implementation in **10 Python
code cells**. There are no companion `.py` modules. It uses the supplied
`train.csv` (4,000 labeled prompts), `test.csv` (1,000 prompts), and
`sample_submission.csv`. No external training data or model downloads are used.

## Verified results

The saved notebook was executed from beginning to end in a fresh Python 3.11
kernel with no cell errors. **26 regression checks passed**. The run regenerated
`submission.csv` and `training_report.json`.

| Development metric | Result |
| --- | ---: |
| Five-fold grouped OOF ROC-AUC | 0.95761 |
| Conditional group-bootstrap 95% interval | 0.94779–0.96605 |
| Accuracy at 0.5 | 0.88875 |
| F1 at 0.5 | 0.85958 |

**These are local development results, not public leaderboard results.** The
latest issue comment requests a public score of at least 0.97. That target has
not been demonstrated, and this result is not evidence of an improvement over
the previously reported public score of 0.95826. No leaderboard upload was made.
Passing software checks does not guarantee perfect classification or performance
on every possible input.

## Model and evaluation

Equal-weight probability blend of four members: word unigram/bigram TF-IDF and
character-boundary 2–6-gram TF-IDF, each with logistic regression and a sigmoid-
calibrated linear SVM. The SVM's three-fold calibration uses only outer-training
rows; its vocabulary is shared across inner calibration folds. The outer
validation fold is excluded from all predictive vocabulary, IDF, classifier, and
calibration fitting. Scores are model estimates, not independently calibrated
security guarantees.

An initial NB-weighted ensemble achieved 0.94881 grouped OOF ROC-AUC. NB members
underperformed and were replaced by SVMs. This comparison reused development
folds, so the final metric is not an untouched holdout estimate. No blend weights
were tuned on validation labels. Both experiments are documented in the notebook.

Safeguards:

- Validate CSV headers and row widths, required fields, unique/nonblank IDs,
  nonempty normalized prompts, and exactly the two expected training labels.
- Reject overlapping train/test IDs or normalized texts, conflicting labels on
  normalized duplicates, and sample/test ID order mismatches.
- Collapse consistent exact duplicates before fitting (none in the supplied data).
- Normalize Unicode NFKC, case, invisible format characters, and whitespace.
- Group character-hash cosine neighbors at similarity >= 0.80 using connected
  components; all members of each of the 3,855 groups stay in one CV fold.
  Hashing is stateless and only used for grouping, not prediction.
- Do not use IDs, test text for feature fitting, or sample TARGET values as features.
- Test invalid inputs, fold coverage, group isolation, held-out vocabulary
  isolation, repeated inference, score bounds, word IDs, and CSV round trips.
- Atomically replace `submission.csv` only after export checks succeed.

Lexical grouping cannot eliminate all semantic paraphrase or source leakage.
The bootstrap resamples complete groups but is conditional on fitted OOF
predictions; it does not account for training or model-selection uncertainty.

## Reproduce

From the repository root (Python 3.11):

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name modeltrain --display-name Modeltrain
.venv/bin/python - <<'PY'
import nbformat
from nbclient import NotebookClient
nb = nbformat.read('modeltrain.ipynb', as_version=4)
NotebookClient(nb, kernel_name='modeltrain', timeout=1800, allow_errors=False).execute()
nbformat.write(nb, 'modeltrain.ipynb')
PY
```

Alternatively, open the notebook with the installed environment and run all cells
in order. A CPU run took approximately one minute in the verified environment;
actual time and memory depend on hardware/data. Dependencies are pinned in
`requirements.txt`; exact core versions and input SHA-256 hashes are recorded in
`training_report.json`. No internet access is needed after installing dependencies.

## Outputs and word IDs

`submission.csv` has exactly **Id,TARGET**, with all 1,000 test IDs in their
original order and likelihoods in [0,1]. A likelihood **>= 0.5** is classified as
`jailbreak`, otherwise `benign`; labels are not added to the competition CSV.

The notebook demonstrates fitted word vocabulary IDs (52,166 unigram/bigram
entries). Repeated words share feature IDs, unknown words map to -1, and these
IDs are distinct from competition prompt IDs. The original CSVs are unchanged.

`training_report.json` records metrics, fold scores, checks, checksums, versions,
and limitations. The executed notebook includes all successful outputs.

## Research consulted before implementation

- [Issue #5 and latest score requests](https://github.com/johndesantis/modeltrain/issues/5)
- [scikit-learn text feature extraction/pipeline example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_text_feature_extraction.html)
