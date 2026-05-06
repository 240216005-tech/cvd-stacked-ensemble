"""
DeLong's Test for Comparing Two AUCs
======================================
Non-parametric test for the difference between two correlated ROC-AUC
values (DeLong et al., 1988).  Used in the ablation study to compare
the stacked ensemble against XGBoost alone.

Usage
-----
    python src/delong.py --dataset kaggle

Reference
---------
    DeLong ER, DeLong DM, Clarke-Pearson DL.  Comparing the areas under
    two or more correlated receiver operating characteristic curves:
    a nonparametric approach.  Biometrics. 1988;44(3):837-845.
"""

import argparse
import os

import numpy as np
from scipy.stats import norm


# ------------------------------------------------------------------ #
#  DeLong AUC variance estimation                                    #
# ------------------------------------------------------------------ #
def _compute_midrank(x):
    """Compute midranks for the array ``x``."""
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    rank = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n - 1 and z[k + 1] == z[k]:
            k += 1
        for t in range(i, k + 1):
            rank[j[t]] = (i + k) / 2.0
        i = k + 1
    return rank


def _fast_delong(predictions_sorted_transposed, label_1_count):
    """Core DeLong computation.

    Parameters
    ----------
    predictions_sorted_transposed : ndarray, shape (2, n)
        Two sets of predictions sorted by ground truth label (positive last).
    label_1_count : int
        Number of positive samples.

    Returns
    -------
    aucs : ndarray, shape (2,)
    var  : ndarray, shape (2, 2) — covariance matrix
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m

    positive_examples = predictions_sorted_transposed[:, n:]
    negative_examples = predictions_sorted_transposed[:, :n]

    k = predictions_sorted_transposed.shape[0]
    aucs = np.empty(k)
    tx = np.empty([k, m])
    ty = np.empty([k, n])

    for i in range(k):
        # Structural components
        all_preds = np.concatenate([negative_examples[i], positive_examples[i]])
        ranks = _compute_midrank(all_preds)

        # Placement values
        tx[i] = ranks[n:]  # ranks of positive predictions
        ty[i] = ranks[:n]  # ranks of negative predictions

        aucs[i] = (np.sum(tx[i]) - m * (m + 1) / 2.0) / (m * n)

    # Covariance matrix
    sx = np.cov(tx)
    sy = np.cov(ty)

    if k == 1:
        sx = np.array([[np.var(tx[0])]])
        sy = np.array([[np.var(ty[0])]])

    var = sx / m + sy / n
    return aucs, var


def delong_test(y_true, pred_a, pred_b):
    """Perform DeLong's test comparing two ROC-AUC values.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Binary ground truth labels.
    pred_a : array-like, shape (n,)
        Predicted probabilities from model A.
    pred_b : array-like, shape (n,)
        Predicted probabilities from model B.

    Returns
    -------
    dict with keys:
        auc_a, auc_b       — the two AUC values
        z_statistic         — DeLong z statistic
        p_value             — two-sided p value
        auc_difference      — auc_a - auc_b
        se_difference       — standard error of the difference
    """
    y_true = np.asarray(y_true, dtype=int)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)

    # Sort by label: negatives first, then positives
    order = np.argsort(y_true)  # 0s first, then 1s
    y_sorted = y_true[order]
    label_1_count = int(np.sum(y_true))

    predictions = np.vstack([pred_a[order], pred_b[order]])

    aucs, cov = _fast_delong(predictions, label_1_count)

    diff = aucs[0] - aucs[1]
    se = np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])

    if se == 0:
        z = 0.0
        p = 1.0
    else:
        z = diff / se
        p = 2 * norm.sf(abs(z))  # two-sided

    return {
        "auc_a": aucs[0],
        "auc_b": aucs[1],
        "auc_difference": diff,
        "se_difference": se,
        "z_statistic": z,
        "p_value": p,
    }


# ------------------------------------------------------------------ #
#  CLI — compare stacked ensemble vs XGBoost from saved results       #
# ------------------------------------------------------------------ #
def main(dataset: str):
    d = os.path.join("results", dataset)
    y_test = np.load(os.path.join(d, "y_test.npy"))
    stack_probs = np.load(os.path.join(d, "final_probs.npy"))
    xgb_probs = np.load(os.path.join(d, "xgb_probs.npy"))
    cat_probs = np.load(os.path.join(d, "cat_probs.npy"))
    tab_probs = np.load(os.path.join(d, "tab_probs.npy"))

    print(f"\n{'='*60}")
    print(f"  DeLong's Test — Ablation Study ({dataset})")
    print(f"{'='*60}")

    comparisons = [
        ("Stacked Ensemble", stack_probs, "XGBoost", xgb_probs),
        ("Stacked Ensemble", stack_probs, "CatBoost", cat_probs),
        ("Stacked Ensemble", stack_probs, "TabNet", tab_probs),
    ]

    for name_a, pred_a, name_b, pred_b in comparisons:
        result = delong_test(y_test, pred_a, pred_b)
        print(f"\n  {name_a} vs {name_b}")
        print(f"    AUC ({name_a:>17s}): {result['auc_a']:.4f}")
        print(f"    AUC ({name_b:>17s}): {result['auc_b']:.4f}")
        print(f"    Difference:            {result['auc_difference']:+.4f}")
        print(f"    SE:                    {result['se_difference']:.4f}")
        print(f"    z-statistic:           {result['z_statistic']:.4f}")
        print(f"    p-value:               {result['p_value']:.6f}")

        if result["p_value"] < 0.05:
            print(f"    Conclusion: Significant difference (p < 0.05)")
        else:
            print(f"    Conclusion: No significant difference (p >= 0.05)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeLong's test for AUC comparison")
    parser.add_argument("--dataset", type=str, default="kaggle")
    args = parser.parse_args()
    main(args.dataset)
