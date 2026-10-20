#!/usr/bin/env python3
"""
validate_pipeline.py
Strict 3-level nested cross-validation of the whole two-level pipeline.

The notebook reports the *2-level* estimate: the base models are
cross-validated with near-duplicate-aware folds, and the meta-learner is
cross-validated again (nested) on those out-of-fold predictions.  That is
already honest about the meta-learner, but the base predictions of the
meta-learner's training rows were produced by models that had seen the
meta-learner's validation rows - the classic residual stacking bias.

This script removes that last bit of bias with a genuine 3-level protocol:

  level 1  outer grouped folds (3).  The outer-validation block is never
           seen by any component.
  level 2  an inner cross-validation *inside the outer-training block only*
           produces the out-of-fold base predictions that the meta-learner
           is trained on.
  level 3  the base models are refitted on the whole outer-training block and
           the meta-learner (fitted on the inner out-of-fold predictions)
           scores the untouched outer-validation block.

Usage:
    python3 validate_pipeline.py

Runtime: roughly 8-10 minutes on a single core.
"""

import sys
import time

sys.path.insert(0, "/workspace/app")
import importlib

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

dl = importlib.import_module("1_data_loader")
pre = importlib.import_module("2_preprocessor")
ft = importlib.import_module("4_model_trainer")

BASE_DIR = "/workspace/app"
TRAIN_CSV = f"{BASE_DIR}/train.csv"
TEST_CSV = f"{BASE_DIR}/test.csv"

train_df, _ = dl.load_data(TRAIN_CSV, TEST_CSV)
texts = train_df["text"].astype(str).tolist()
y, _encoder = ft.encode_labels(train_df["label"])
groups = ft.build_near_duplicate_groups(texts)
style = pre.build_style_features(texts)
models = ft.build_base_models()

print(f"rows={len(texts)} style_features={style.shape[1]} base_models={len(models)}", flush=True)

outer = list(
    StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(
        np.zeros(len(y)), y, groups
    )
)

MODES = ("stack_rank", "meta_gb")
outer_scores = {m: [] for m in MODES}
outer_rows = []
joint_scores = []

for oi, (otr, ova) in enumerate(outer):
    t0 = time.time()
    otr_texts = [texts[i] for i in otr]
    ova_texts = [texts[i] for i in ova]
    inner_y = y[otr]
    inner_groups = groups[otr]

    # Level 2: inner CV on the outer-training block only.
    oof_inner, _ = ft.cross_validate(models, otr_texts, inner_y, inner_groups, n_splits=3)

    # Level 3: refit the base models on the whole outer-training block.
    base_fit = ft.fit_final_models(models, otr_texts, inner_y)
    base_ova = ft.predict_base_models(base_fit, ova_texts)

    for mode in MODES:
        fusion = ft.Fusion(mode).fit(oof_inner, inner_y, style[otr])
        outer_scores[mode].append(fusion.predict(base_ova, style[ova]))

    joint_idx = [i for i, (n, _) in enumerate(models) if n == "joint_lr"][0]
    joint_scores.append(base_ova[:, joint_idx])

    outer_rows.append(ova)
    print(f"outer fold {oi + 1}/{len(outer)} done in {time.time() - t0:.0f}s", flush=True)

rows = np.concatenate(outer_rows)
order = np.argsort(rows)
rows = rows[order]
y_eval = y[rows]

print()
print("=== TRUE 3-LEVEL nested cross-validation (outer blocks never seen) ===")
for mode in MODES:
    p = np.concatenate(outer_scores[mode])[order]
    print(f"  {mode:<12} ROC-AUC = {roc_auc_score(y_eval, p):.5f}   acc@0.5 = "
          f"{float(((p >= 0.5) == (y_eval == 1)).mean()):.4f}   "
          f"frac>=0.5 = {float((p >= 0.5).mean()):.3f}")
pj = np.concatenate(joint_scores)[order]
print(f"  {'joint_lr':<12} ROC-AUC = {roc_auc_score(y_eval, pj):.5f}   "
      f"(best single base model, same protocol)")
