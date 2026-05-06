"""
Stacked Ensemble for Cardiovascular Disease Prediction
=======================================================
Train XGBoost, CatBoost, and TabNet base learners with a Logistic Regression
meta-learner.  Saves trained models, test arrays, and stacked probabilities
to ``results/<dataset>/``.

Authors : Akhil Tripathi (https://orcid.org/0009-0009-1650-5787)
          Dr. Imran Khan
Affiliation: Dept. of Computer Science, HBTU Kanpur, India

Usage
-----
    python src/train.py --dataset kaggle --data_path data/kaggle_cvd.csv
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

warnings.filterwarnings("ignore", message="No early stopping will be performed")


# ------------------------------------------------------------------ #
#  Preprocessing                                                      #
# ------------------------------------------------------------------ #
def preprocess(X_train, X_test):
    """KNN imputation followed by standard scaling."""
    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_test_sc = scaler.transform(X_test_imp)
    return X_train_sc, X_test_sc


# ------------------------------------------------------------------ #
#  Base learners                                                      #
# ------------------------------------------------------------------ #
def train_base_models(X_train, y_train, X_test, seed=42):
    """Train XGBoost, CatBoost, and TabNet; return test probabilities."""
    xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        eval_metric="logloss", random_state=seed,
    )
    cat = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05,
        verbose=0, random_seed=seed,
    )
    tab = TabNetClassifier(
        n_d=16, n_a=16, n_steps=5, gamma=1.5,
        learning_rate=0.02, verbose=0, seed=seed,
    )

    xgb.fit(X_train, y_train)
    cat.fit(X_train, y_train)
    tab.fit(X_train, y_train.values if hasattr(y_train, "values") else y_train,
            max_epochs=50, patience=10)

    return xgb, cat, tab


# ------------------------------------------------------------------ #
#  Out-of-fold stacking                                               #
# ------------------------------------------------------------------ #
def generate_oof_predictions(X_train, y_train, n_splits=5, seed=42):
    """Generate leak-free out-of-fold probabilities for the meta-learner."""
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    n = len(X_train)
    xgb_oof = np.zeros(n)
    cat_oof = np.zeros(n)
    tab_oof = np.zeros(n)

    y_arr = y_train.values if hasattr(y_train, "values") else y_train

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_arr), 1):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]

        xgb_f = XGBClassifier(n_estimators=100, max_depth=3,
                               eval_metric="logloss", random_state=seed)
        cat_f = CatBoostClassifier(iterations=100, depth=4,
                                    verbose=0, random_seed=seed)
        tab_f = TabNetClassifier(verbose=0, seed=seed)

        xgb_f.fit(X_tr, y_tr)
        cat_f.fit(X_tr, y_tr)
        tab_f.fit(X_tr, y_tr, max_epochs=10)

        xgb_oof[val_idx] = xgb_f.predict_proba(X_val)[:, 1]
        cat_oof[val_idx] = cat_f.predict_proba(X_val)[:, 1]
        tab_oof[val_idx] = tab_f.predict_proba(X_val)[:, 1]

        print(f"  Fold {fold}/{n_splits} complete")

    return np.column_stack([xgb_oof, cat_oof, tab_oof])


# ------------------------------------------------------------------ #
#  Evaluation helpers                                                 #
# ------------------------------------------------------------------ #
def print_metrics(y_true, y_prob, dataset_name=""):
    """Print classification metrics and return them as a dict."""
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "Dataset": dataset_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_prob),
    }
    print(f"\n{'='*50}")
    print(f"  {dataset_name} — Stacked Ensemble Results")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k != "Dataset":
            print(f"  {k:>10s}: {v:.4f}")
    return metrics


# ------------------------------------------------------------------ #
#  Main pipeline                                                      #
# ------------------------------------------------------------------ #
def main(dataset_name: str, data_path: str, seed: int = 42):
    print(f"\n>>> Loading {dataset_name} from {data_path}")
    df = pd.read_csv(data_path)

    # Expect last column or a column named 'target' / 'TenYearCHD' / 'cardio'
    target_col = None
    for candidate in ["target", "TenYearCHD", "cardio", "HeartDisease"]:
        if candidate in df.columns:
            target_col = candidate
            break
    if target_col is None:
        target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]
    y = (y > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=seed,
    )

    print(">>> Preprocessing (KNN imputation + scaling)")
    X_train_pp, X_test_pp = preprocess(X_train.values, X_test.values)

    print(">>> Training base models (XGBoost, CatBoost, TabNet)")
    xgb, cat, tab = train_base_models(X_train_pp, y_train, X_test_pp, seed)

    print(">>> Generating out-of-fold predictions for meta-learner")
    oof_stack = generate_oof_predictions(X_train_pp, y_train, seed=seed)

    print(">>> Training meta-learner (Logistic Regression)")
    meta = LogisticRegression(random_state=seed)
    meta.fit(oof_stack, y_train)

    test_stack = np.column_stack([
        xgb.predict_proba(X_test_pp)[:, 1],
        cat.predict_proba(X_test_pp)[:, 1],
        tab.predict_proba(X_test_pp)[:, 1],
    ])
    final_probs = meta.predict_proba(test_stack)[:, 1]

    metrics = print_metrics(y_test, final_probs, dataset_name)

    # ---- Save artefacts ---- #
    out_dir = os.path.join("results", dataset_name.lower().replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_train_pp.npy"), X_train_pp)
    np.save(os.path.join(out_dir, "X_test_pp.npy"), X_test_pp)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train.values if hasattr(y_train, "values") else y_train)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test.values if hasattr(y_test, "values") else y_test)
    np.save(os.path.join(out_dir, "final_probs.npy"), final_probs)
    np.save(os.path.join(out_dir, "xgb_probs.npy"), xgb.predict_proba(X_test_pp)[:, 1])
    np.save(os.path.join(out_dir, "cat_probs.npy"), cat.predict_proba(X_test_pp)[:, 1])
    np.save(os.path.join(out_dir, "tab_probs.npy"), tab.predict_proba(X_test_pp)[:, 1])
    print(f"\n>>> Artefacts saved to {out_dir}/")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CVD Stacked Ensemble")
    parser.add_argument("--dataset", type=str, default="kaggle",
                        help="Dataset name (used for output folder)")
    parser.add_argument("--data_path", type=str, default="data/kaggle_cvd.csv",
                        help="Path to CSV file")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.dataset, args.data_path, args.seed)
