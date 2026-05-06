"""
External Validation on UCI Heart Disease Datasets
===================================================
Evaluate the stacked ensemble pipeline on Cleveland, Hungarian,
and Statlog cohorts with multiple random seeds for stability.

Usage
-----
    python src/external_validation.py
"""

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

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
#  Dataset loaders                                                    #
# ------------------------------------------------------------------ #
FEATURE_COLS = ["age", "sex", "trestbps", "chol", "thalach", "oldpeak"]
UCI_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
               "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]

DATASETS = {
    "Cleveland": "https://archive.ics.uci.edu/ml/machine-learning-databases/"
                 "heart-disease/processed.cleveland.data",
    "Hungarian": "https://archive.ics.uci.edu/ml/machine-learning-databases/"
                 "heart-disease/processed.hungarian.data",
    "Statlog":   "https://archive.ics.uci.edu/ml/machine-learning-databases/"
                 "statlog/heart/heart.dat",
}


def load_uci(url: str) -> pd.DataFrame:
    df = pd.read_csv(url, names=UCI_COLUMNS)
    df.replace("?", np.nan, inplace=True)
    df = df.apply(pd.to_numeric)
    df["target"] = (df["target"] > 0).astype(int)
    return df[FEATURE_COLS + ["target"]].dropna()


def load_statlog(url: str) -> pd.DataFrame:
    df = pd.read_csv(url, sep=r"\s+", names=UCI_COLUMNS)
    df["target"] = df["target"].map({1: 0, 2: 1})
    return df[FEATURE_COLS + ["target"]]


# ------------------------------------------------------------------ #
#  Single-seed evaluation                                             #
# ------------------------------------------------------------------ #
def evaluate_once(df: pd.DataFrame, seed: int = 42) -> dict:
    X = df[FEATURE_COLS].values
    y = df["target"].values

    imp = KNNImputer(n_neighbors=5)
    sc = StandardScaler()
    X = sc.fit_transform(imp.fit_transform(X))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=seed,
    )

    # Base learners
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                         eval_metric="logloss", random_state=seed)
    cat = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05,
                              verbose=0, random_seed=seed)
    tab = TabNetClassifier(verbose=0, seed=seed)

    xgb.fit(X_tr, y_tr)
    cat.fit(X_tr, y_tr)
    tab.fit(X_tr, y_tr, max_epochs=20)

    # Out-of-fold for meta-learner
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = {k: np.zeros(len(X_tr)) for k in ("xgb", "cat", "tab")}

    for tr_idx, val_idx in skf.split(X_tr, y_tr):
        Xf, Xv = X_tr[tr_idx], X_tr[val_idx]
        yf = y_tr[tr_idx]

        m_xgb = XGBClassifier(n_estimators=100, max_depth=3,
                               eval_metric="logloss", random_state=seed)
        m_cat = CatBoostClassifier(iterations=100, depth=4,
                                    verbose=0, random_seed=seed)
        m_tab = TabNetClassifier(verbose=0, seed=seed)

        m_xgb.fit(Xf, yf); m_cat.fit(Xf, yf); m_tab.fit(Xf, yf, max_epochs=10)

        oof["xgb"][val_idx] = m_xgb.predict_proba(Xv)[:, 1]
        oof["cat"][val_idx] = m_cat.predict_proba(Xv)[:, 1]
        oof["tab"][val_idx] = m_tab.predict_proba(Xv)[:, 1]

    meta = LogisticRegression(random_state=seed)
    meta.fit(np.column_stack(list(oof.values())), y_tr)

    test_stack = np.column_stack([
        xgb.predict_proba(X_te)[:, 1],
        cat.predict_proba(X_te)[:, 1],
        tab.predict_proba(X_te)[:, 1],
    ])
    probs = meta.predict_proba(test_stack)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return {
        "AUC": roc_auc_score(y_te, probs),
        "Accuracy": accuracy_score(y_te, preds),
        "Precision": precision_score(y_te, preds, zero_division=0),
        "Recall": recall_score(y_te, preds, zero_division=0),
        "F1": f1_score(y_te, preds, zero_division=0),
    }


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #
def main():
    seeds = [42, 123, 456, 789, 101112]
    all_results = []

    for name, url in DATASETS.items():
        print(f"\n{'='*55}")
        print(f"  {name} External Validation")
        print(f"{'='*55}")

        loader = load_statlog if name == "Statlog" else load_uci
        try:
            df = loader(url)
        except Exception as e:
            print(f"  ERROR loading {name}: {e}")
            continue

        print(f"  Samples: {len(df)}  |  Positive rate: {df['target'].mean():.2%}")

        seed_aucs = []
        for s in seeds:
            m = evaluate_once(df, seed=s)
            seed_aucs.append(m["AUC"])
            print(f"  Seed {s:>6d}: AUC = {m['AUC']:.4f}")

        mean_auc = np.mean(seed_aucs)
        std_auc = np.std(seed_aucs)
        print(f"\n  Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
        all_results.append({"Dataset": name, "Mean AUC": mean_auc, "Std": std_auc})

    print(f"\n{'='*55}")
    print("  SUMMARY")
    print(f"{'='*55}")
    summary = pd.DataFrame(all_results)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
