"""
Bayesian Hyperparameter Optimisation with Optuna
==================================================
Tunes XGBoost, CatBoost, and TabNet using Tree-structured Parzen
Estimators (TPE) with 30–50 trials per model.  Best parameters are
saved as JSON for use by ``train.py``.

Usage
-----
    python src/optimise.py --data_path data/kaggle_cvd.csv --n_trials 50
"""

import argparse
import json
import os
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_data(path: str):
    """Load CSV and return X, y arrays (preprocessed)."""
    df = pd.read_csv(path)
    target_col = None
    for c in ["target", "TenYearCHD", "cardio", "HeartDisease"]:
        if c in df.columns:
            target_col = c
            break
    if target_col is None:
        target_col = df.columns[-1]

    X = df.drop(columns=[target_col]).values
    y = (df[target_col] > 0).astype(int).values

    imp = KNNImputer(n_neighbors=5)
    sc = StandardScaler()
    X = sc.fit_transform(imp.fit_transform(X))
    return X, y


def cv_score(model_fn, X, y, n_splits=5, seed=42):
    """Return mean 5-fold CV AUC for a model constructor ``model_fn``."""
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr_idx, val_idx in skf.split(X, y):
        model = model_fn()
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        if isinstance(model, TabNetClassifier):
            model.fit(X_tr, y_tr, max_epochs=30, patience=5,
                      eval_set=[(X_val, y_val)], eval_metric=["auc"])
        else:
            model.fit(X_tr, y_tr)

        probs = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, probs))
    return np.mean(aucs)


# ------------------------------------------------------------------ #
#  Objective functions                                                #
# ------------------------------------------------------------------ #
def xgboost_objective(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
        "eval_metric": "logloss",
        "random_state": 42,
    }
    return cv_score(lambda: XGBClassifier(**params), X, y)


def catboost_objective(trial, X, y):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 500),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "random_strength": trial.suggest_float("random_strength", 1e-2, 10, log=True),
        "verbose": 0,
        "random_seed": 42,
    }
    return cv_score(lambda: CatBoostClassifier(**params), X, y)


def tabnet_objective(trial, X, y):
    params = {
        "n_d": trial.suggest_int("n_d", 8, 64),
        "n_a": trial.suggest_int("n_a", 8, 64),
        "n_steps": trial.suggest_int("n_steps", 3, 10),
        "gamma": trial.suggest_float("gamma", 1.0, 2.0),
        "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-2, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "verbose": 0,
        "seed": 42,
    }
    return cv_score(lambda: TabNetClassifier(**params), X, y)


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #
def main(data_path: str, n_trials: int, output_dir: str):
    print(f">>> Loading data from {data_path}")
    X, y = load_data(data_path)
    print(f"    Samples: {len(y)}  |  Features: {X.shape[1]}")

    os.makedirs(output_dir, exist_ok=True)
    best_params = {}

    # --- XGBoost ---
    print(f"\n>>> Optimising XGBoost ({n_trials} trials)")
    study_xgb = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=42))
    study_xgb.optimize(lambda t: xgboost_objective(t, X, y), n_trials=n_trials)
    best_params["xgboost"] = study_xgb.best_params
    print(f"    Best AUC: {study_xgb.best_value:.4f}")
    print(f"    Params:   {study_xgb.best_params}")

    # --- CatBoost ---
    print(f"\n>>> Optimising CatBoost ({n_trials} trials)")
    study_cat = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=42))
    study_cat.optimize(lambda t: catboost_objective(t, X, y), n_trials=n_trials)
    best_params["catboost"] = study_cat.best_params
    print(f"    Best AUC: {study_cat.best_value:.4f}")
    print(f"    Params:   {study_cat.best_params}")

    # --- TabNet ---
    print(f"\n>>> Optimising TabNet ({n_trials} trials)")
    study_tab = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=42))
    study_tab.optimize(lambda t: tabnet_objective(t, X, y), n_trials=n_trials)
    best_params["tabnet"] = study_tab.best_params
    print(f"    Best AUC: {study_tab.best_value:.4f}")
    print(f"    Params:   {study_tab.best_params}")

    # --- Save ---
    out_path = os.path.join(output_dir, "best_params.json")
    with open(out_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\n>>> Best parameters saved to {out_path}")

    # --- Summary ---
    print(f"\n{'='*55}")
    print("  OPTIMISATION SUMMARY")
    print(f"{'='*55}")
    print(f"  XGBoost  best CV AUC: {study_xgb.best_value:.4f}")
    print(f"  CatBoost best CV AUC: {study_cat.best_value:.4f}")
    print(f"  TabNet   best CV AUC: {study_tab.best_value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter optimisation with Optuna")
    parser.add_argument("--data_path", type=str, default="data/kaggle_cvd.csv")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of Optuna trials per model (30-50 recommended)")
    parser.add_argument("--output_dir", type=str, default="results/optuna")
    args = parser.parse_args()
    main(args.data_path, args.n_trials, args.output_dir)
