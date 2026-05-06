"""
Evaluation & Visualisation
============================
Generate ROC curves (with bootstrap CI), calibration plots, confusion
matrices, SHAP summary plots, ablation study, and decision curve analysis.

Usage
-----
    python src/evaluate.py --dataset kaggle
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from dcurves import dca
from scipy.stats import ttest_rel
from sklearn.calibration import calibration_curve
from sklearn.metrics import (auc, confusion_matrix, ConfusionMatrixDisplay,
                             roc_auc_score, roc_curve)
from sklearn.utils import resample
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 12, "figure.dpi": 300})


def load_results(dataset: str):
    """Load saved .npy artefacts from results/<dataset>/."""
    d = os.path.join("results", dataset)
    return {
        "y_test": np.load(os.path.join(d, "y_test.npy")),
        "final_probs": np.load(os.path.join(d, "final_probs.npy")),
        "xgb_probs": np.load(os.path.join(d, "xgb_probs.npy")),
        "cat_probs": np.load(os.path.join(d, "cat_probs.npy")),
        "tab_probs": np.load(os.path.join(d, "tab_probs.npy")),
        "X_test": np.load(os.path.join(d, "X_test_pp.npy")),
        "X_train": np.load(os.path.join(d, "X_train_pp.npy")),
        "y_train": np.load(os.path.join(d, "y_train.npy")),
    }


# ------------------------------------------------------------------ #
#  1. ROC curve with 95 % bootstrap CI                                #
# ------------------------------------------------------------------ #
def plot_roc_bootstrap(y_test, probs, dataset, fig_dir, n_boot=1000):
    boot_aucs = []
    for _ in range(n_boot):
        idx = resample(range(len(y_test)), replace=True)
        fpr_b, tpr_b, _ = roc_curve(y_test[idx], probs[idx])
        boot_aucs.append(auc(fpr_b, tpr_b))
    ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])

    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2,
             label=f"Stacked Ensemble (AUC={roc_auc:.3f}, "
                   f"95% CI [{ci_lo:.3f}\u2013{ci_hi:.3f}])")
    plt.plot([0, 1], [0, 1], "k--")
    plt.fill_between(fpr, tpr, alpha=0.15)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve \u2013 {dataset}")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "roc_curve_bootstrap.png"))
    plt.close()
    print(f"  Saved roc_curve_bootstrap.png  (AUC={roc_auc:.3f})")


# ------------------------------------------------------------------ #
#  2. ROC comparison across models                                    #
# ------------------------------------------------------------------ #
def plot_roc_comparison(y_test, probs_dict, dataset, fig_dir):
    plt.figure(figsize=(8, 6))
    for name, probs in probs_dict.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Comparison \u2013 {dataset}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "roc_comparison.png"))
    plt.close()
    print("  Saved roc_comparison.png")


# ------------------------------------------------------------------ #
#  3. Calibration curve                                               #
# ------------------------------------------------------------------ #
def plot_calibration(y_test, probs, dataset, fig_dir):
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", lw=2, label="Stacked Ensemble")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration Curve \u2013 {dataset}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "calibration_curve.png"))
    plt.close()
    print("  Saved calibration_curve.png")


# ------------------------------------------------------------------ #
#  4. Confusion matrix                                                #
# ------------------------------------------------------------------ #
def plot_confusion_matrix(y_test, probs, dataset, fig_dir):
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["No CVD", "CVD"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix \u2013 {dataset}")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "confusion_matrix.png"))
    plt.close()

    tn, fp, fn, tp = cm.ravel()
    print(f"  Saved confusion_matrix.png  (TP={tp} FN={fn} FP={fp} TN={tn})")


# ------------------------------------------------------------------ #
#  5. SHAP summary                                                    #
# ------------------------------------------------------------------ #
def plot_shap(X_train, y_train, X_test, fig_dir, seed=42):
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                         eval_metric="logloss", random_state=seed)
    xgb.fit(X_train, y_train)
    explainer = shap.Explainer(xgb)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "shap_summary.png"), bbox_inches="tight")
    plt.close()
    print("  Saved shap_summary.png")


# ------------------------------------------------------------------ #
#  6. Ablation study                                                  #
# ------------------------------------------------------------------ #
def ablation_study(y_test, probs_dict, fig_dir):
    print("\n  === ABLATION STUDY ===")
    for name, probs in probs_dict.items():
        print(f"  {name:>20s}  AUC = {roc_auc_score(y_test, probs):.4f}")

    # Statistical test: stacked vs XGBoost
    if "XGBoost" in probs_dict and "Stacked Ensemble" in probs_dict:
        t_stat, p_val = ttest_rel(probs_dict["XGBoost"],
                                   probs_dict["Stacked Ensemble"])
        print(f"\n  Paired t-test (XGBoost vs Stacked): p = {p_val:.6f}")


# ------------------------------------------------------------------ #
#  7. Decision curve analysis                                         #
# ------------------------------------------------------------------ #
def plot_dca(y_test, probs, dataset, fig_dir):
    df = pd.DataFrame({"outcome": y_test, "stacked": probs})
    dca_result = dca(data=df, outcome="outcome", modelnames=["stacked"],
                     thresholds=np.arange(0, 0.5, 0.01))
    dca_result.plot()
    plt.title(f"Decision Curve Analysis \u2013 {dataset}")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "decision_curve.png"))
    plt.close()
    print("  Saved decision_curve.png")


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #
def main(dataset: str):
    fig_dir = os.path.join("figures", dataset)
    os.makedirs(fig_dir, exist_ok=True)

    print(f"\n>>> Loading results for '{dataset}'")
    r = load_results(dataset)

    probs_dict = {
        "XGBoost": r["xgb_probs"],
        "CatBoost": r["cat_probs"],
        "TabNet": r["tab_probs"],
        "Stacked Ensemble": r["final_probs"],
    }

    print("\n>>> Generating figures")
    plot_roc_bootstrap(r["y_test"], r["final_probs"], dataset, fig_dir)
    plot_roc_comparison(r["y_test"], probs_dict, dataset, fig_dir)
    plot_calibration(r["y_test"], r["final_probs"], dataset, fig_dir)
    plot_confusion_matrix(r["y_test"], r["final_probs"], dataset, fig_dir)
    plot_shap(r["X_train"], r["y_train"], r["X_test"], fig_dir)
    ablation_study(r["y_test"], probs_dict, fig_dir)
    plot_dca(r["y_test"], r["final_probs"], dataset, fig_dir)

    print(f"\n>>> All figures saved to {fig_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CVD Stacked Ensemble")
    parser.add_argument("--dataset", type=str, default="kaggle")
    args = parser.parse_args()
    main(args.dataset)
