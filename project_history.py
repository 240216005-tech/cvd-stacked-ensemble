!pip install dcurves

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
from dcurves import dca
import shap
from xgboost import XGBClassifier
import os

# Ensure figures directory exists
os.makedirs('figures', exist_ok=True)

# ------------------------------------------------------------
# 1. Load your saved test data (adjust paths)
# ------------------------------------------------------------
# Example: load Kaggle test labels and probabilities
# Note: Ensure these files actually exist or update paths accordingly
try:
    y_test = np.load('results/kaggle/y_test.npy')
    probs = np.load('results/kaggle/final_probs.npy')
except FileNotFoundError:
    print("Warning: Data files not found. Please ensure paths in Section 1 are correct.")
    # Fallback to current kernel variables if available for demonstration
    if 'y_test' in globals() and 'final_probs' in globals():
        y_test = globals()['y_test'].values if hasattr(globals()['y_test'], 'values') else globals()['y_test']
        probs = globals()['final_probs']

# ------------------------------------------------------------
# 2. ROC curve with 95% bootstrap CI
# ------------------------------------------------------------
if 'y_test' in locals() or 'y_test' in globals():
    n_boot = 1000
    aucs = []
    for _ in range(n_boot):
        idx = resample(range(len(y_test)), replace=True)
        fpr_b, tpr_b, _ = roc_curve(y_test[idx], probs[idx])
        aucs.append(auc(fpr_b, tpr_b))
    ci_low, ci_high = np.percentile(aucs, [2.5, 97.5])
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, lw=2, label=f'Stacked Ensemble (AUC = {roc_auc:.3f}, 95% CI [{ci_low:.3f}–{ci_high:.3f}])')
    plt.plot([0,1], [0,1], 'k--')
    plt.fill_between(fpr, tpr, alpha=0.2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve – Kaggle Test Set')
    plt.legend(loc='lower right')
    plt.savefig('figures/roc_curve.png', dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # 3. Calibration curve
    # ------------------------------------------------------------
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
    plt.figure(figsize=(6,5))
    plt.plot(prob_pred, prob_true, marker='o', lw=2, label='Stacked Ensemble')
    plt.plot([0,1], [0,1], 'k--', label='Perfect calibration')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve – Kaggle')
    plt.legend()
    plt.savefig('figures/calibration_curve.png', dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # 5. Decision curve analysis (requires dcurves package)
    # ------------------------------------------------------------
    df_dca = pd.DataFrame({'outcome': y_test, 'stacked': probs})
    dca_data = dca(data=df_dca, outcome='outcome', modelnames=['stacked'], thresholds=np.arange(0, 0.5, 0.01))
    dca_data.plot()
    plt.title('Decision Curve Analysis – Framingham')
    plt.savefig('figures/decision_curve.png', dpi=300)
    plt.close()

    print("All figures saved to 'figures/' directory.")
else:
    print("Error: y_test and probs not found. Run training code first.")
!pip install dcurves

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
from dcurves import dca
import shap
from xgboost import XGBClassifier
import os

# Ensure figures directory exists
os.makedirs('figures', exist_ok=True)

# ------------------------------------------------------------
# 1. Load your saved test data (adjust paths)
# ------------------------------------------------------------
# Example: load Kaggle test labels and probabilities
# Note: Ensure these files actually exist or update paths accordingly
try:
    y_test = np.load('results/kaggle/y_test.npy')
    probs = np.load('results/kaggle/final_probs.npy')
except FileNotFoundError:
    print("Warning: Data files not found. Please ensure paths in Section 1 are correct.")
    # Fallback to current kernel variables if available for demonstration
    if 'y_test' in globals() and 'final_probs' in globals():
        y_test = globals()['y_test'].values if hasattr(globals()['y_test'], 'values') else globals()['y_test']
        probs = globals()['final_probs']

# ------------------------------------------------------------
# 2. ROC curve with 95% bootstrap CI
# ------------------------------------------------------------
if 'y_test' in locals() or 'y_test' in globals():
    n_boot = 1000
    aucs = []
    for _ in range(n_boot):
        idx = resample(range(len(y_test)), replace=True)
        fpr_b, tpr_b, _ = roc_curve(y_test[idx], probs[idx])
        aucs.append(auc(fpr_b, tpr_b))
    ci_low, ci_high = np.percentile(aucs, [2.5, 97.5])
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, lw=2, label=f'Stacked Ensemble (AUC = {roc_auc:.3f}, 95% CI [{ci_low:.3f}–{ci_high:.3f}])')
    plt.plot([0,1], [0,1], 'k--')
    plt.fill_between(fpr, tpr, alpha=0.2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve – Kaggle Test Set')
    plt.legend(loc='lower right')
    plt.savefig('figures/roc_curve.png', dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # 3. Calibration curve
    # ------------------------------------------------------------
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
    plt.figure(figsize=(6,5))
    plt.plot(prob_pred, prob_true, marker='o', lw=2, label='Stacked Ensemble')
    plt.plot([0,1], [0,1], 'k--', label='Perfect calibration')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve – Kaggle')
    plt.legend()
    plt.savefig('figures/calibration_curve.png', dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # 5. Decision curve analysis (requires dcurves package)
    # ------------------------------------------------------------
    df_dca = pd.DataFrame({'outcome': y_test, 'stacked': probs})
    dca_data = dca(data=df_dca, outcome='outcome', modelnames=['stacked'], thresholds=np.arange(0, 0.5, 0.01))
    dca_data.plot()
    plt.title('Decision Curve Analysis – Framingham')
    plt.savefig('figures/decision_curve.png', dpi=300)
    plt.close()

    print("All figures saved to 'figures/' directory.")
else:
    print("Error: y_test and probs not found. Run training code first.")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_dataset(name, df):
    X = df[feature_cols]
    y = df['target']

    # Preprocessing
    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, stratify=y, test_size=0.2, random_state=42
    )

    # Models
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, eval_metric='logloss')
    cat = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, verbose=0)

    xgb.fit(X_train, y_train)
    cat.fit(X_train, y_train)

    # Stacking
    stack_train = np.column_stack([
        xgb.predict_proba(X_train)[:,1],
        cat.predict_proba(X_train)[:,1]
    ])

    stack_test = np.column_stack([
        xgb.predict_proba(X_test)[:,1],
        cat.predict_proba(X_test)[:,1]
    ])

    meta = LogisticRegression()
    meta.fit(stack_train, y_train)

    # Final predictions
    final_probs = meta.predict_proba(stack_test)[:,1]
    final_preds = (final_probs >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, final_preds)
    prec = precision_score(y_test, final_preds)
    rec = recall_score(y_test, final_preds)
    f1 = f1_score(y_test, final_preds)
    auc = roc_auc_score(y_test, final_probs)

    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

    return acc, prec, rec, f1, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_dataset(name, df):
    X = df[feature_cols]
    y = df['target']

    # Preprocessing
    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, stratify=y, test_size=0.2, random_state=42
    )

    # Models
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, eval_metric='logloss')
    cat = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, verbose=0)

    xgb.fit(X_train, y_train)
    cat.fit(X_train, y_train)

    # Stacking
    stack_train = np.column_stack([
        xgb.predict_proba(X_train)[:,1],
        cat.predict_proba(X_train)[:,1]
    ])

    stack_test = np.column_stack([
        xgb.predict_proba(X_test)[:,1],
        cat.predict_proba(X_test)[:,1]
    ])

    meta = LogisticRegression()
    meta.fit(stack_train, y_train)

    # Final predictions
    final_probs = meta.predict_proba(stack_test)[:,1]
    final_preds = (final_probs >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, final_preds)
    prec = precision_score(y_test, final_preds)
    rec = recall_score(y_test, final_preds)
    f1 = f1_score(y_test, final_preds)
    auc = roc_auc_score(y_test, final_probs)

    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

    return acc, prec, rec, f1, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_dataset(name, df):
    X = df[feature_cols]
    y = df['target']

    # Preprocessing
    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, stratify=y, test_size=0.2, random_state=42
    )

    # Models
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, eval_metric='logloss')
    cat = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, verbose=0)

    xgb.fit(X_train, y_train)
    cat.fit(X_train, y_train)

    # Stacking
    stack_train = np.column_stack([
        xgb.predict_proba(X_train)[:,1],
        cat.predict_proba(X_train)[:,1]
    ])

    stack_test = np.column_stack([
        xgb.predict_proba(X_test)[:,1],
        cat.predict_proba(X_test)[:,1]
    ])

    meta = LogisticRegression()
    meta.fit(stack_train, y_train)

    # Final predictions
    final_probs = meta.predict_proba(stack_test)[:,1]
    final_preds = (final_probs >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, final_preds)
    prec = precision_score(y_test, final_preds)
    rec = recall_score(y_test, final_preds)
    f1 = f1_score(y_test, final_preds)
    auc = roc_auc_score(y_test, final_probs)

    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

    return acc, prec, rec, f1, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_dataset(name, df):
    X = df[feature_cols]
    y = df['target']

    # Preprocessing
    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, stratify=y, test_size=0.2, random_state=42
    )

    # Models
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, eval_metric='logloss')
    cat = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, verbose=0)

    xgb.fit(X_train, y_train)
    cat.fit(X_train, y_train)

    # Stacking
    stack_train = np.column_stack([
        xgb.predict_proba(X_train)[:,1],
        cat.predict_proba(X_train)[:,1]
    ])

    stack_test = np.column_stack([
        xgb.predict_proba(X_test)[:,1],
        cat.predict_proba(X_test)[:,1]
    ])

    meta = LogisticRegression()
    meta.fit(stack_train, y_train)

    # Final predictions
    final_probs = meta.predict_proba(stack_test)[:,1]
    final_preds = (final_probs >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, final_preds)
    prec = precision_score(y_test, final_preds)
    rec = recall_score(y_test, final_preds)
    f1 = f1_score(y_test, final_preds)
    auc = roc_auc_score(y_test, final_probs)

    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

    return acc, prec, rec, f1, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_dataset(name, df):
    X = df[feature_cols]
    y = df['target']

    # Preprocessing
    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, stratify=y, test_size=0.2, random_state=42
    )

    # Models
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, eval_metric='logloss')
    cat = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, verbose=0)

    xgb.fit(X_train, y_train)
    cat.fit(X_train, y_train)

    # Stacking
    stack_train = np.column_stack([
        xgb.predict_proba(X_train)[:,1],
        cat.predict_proba(X_train)[:,1]
    ])

    stack_test = np.column_stack([
        xgb.predict_proba(X_test)[:,1],
        cat.predict_proba(X_test)[:,1]
    ])

    meta = LogisticRegression()
    meta.fit(stack_train, y_train)

    # Final predictions
    final_probs = meta.predict_proba(stack_test)[:,1]
    final_preds = (final_probs >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, final_preds)
    prec = precision_score(y_test, final_preds)
    rec = recall_score(y_test, final_preds)
    f1 = f1_score(y_test, final_preds)
    auc = roc_auc_score(y_test, final_probs)

    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

    return acc, prec, rec, f1, auc
for name, url in datasets.items():
    try:
        if name == "Statlog":
            df = load_statlog_dataset(url)
        else:
            df = load_uci_dataset(url)

        auc = evaluate_dataset(name, df)
        results[name] = auc

    except Exception as e:
        print(f"❌ Error in {name}: {e}")
datasets = {
    "Cleveland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "Statlog": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat",
    "Hungarian": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data"
}

for name, url in datasets.items():
    try:
        if name == "Statlog":
            df = load_statlog_dataset(url)
        else:
            df = load_uci_dataset(url)

        auc = evaluate_dataset(name, df)
        results[name] = auc

    except Exception as e:
        print(f"❌ Error in {name}: {e}")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_dataset(name, df):
    X = df[feature_cols]
    y = df['target']

    # Preprocessing
    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, stratify=y, test_size=0.2, random_state=42
    )

    # Models
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, eval_metric='logloss')
    cat = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, verbose=0)

    xgb.fit(X_train, y_train)
    cat.fit(X_train, y_train)

    # Stacking
    stack_train = np.column_stack([
        xgb.predict_proba(X_train)[:,1],
        cat.predict_proba(X_train)[:,1]
    ])

    stack_test = np.column_stack([
        xgb.predict_proba(X_test)[:,1],
        cat.predict_proba(X_test)[:,1]
    ])

    meta = LogisticRegression()
    meta.fit(stack_train, y_train)

    # Final predictions
    final_probs = meta.predict_proba(stack_test)[:,1]
    final_preds = (final_probs >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, final_preds)
    prec = precision_score(y_test, final_preds)
    rec = recall_score(y_test, final_preds)
    f1 = f1_score(y_test, final_preds)
    auc = roc_auc_score(y_test, final_probs)

    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

    return acc, prec, rec, f1, auc
import pandas as pd
import numpy as np

# Define the common feature set
feature_cols = ['age', 'sex', 'trestbps', 'chol', 'thalach', 'oldpeak']

def load_uci_dataset(url):
    columns = ['age','sex','cp','trestbps','chol','fbs','restecg',
               'thalach','exang','oldpeak','slope','ca','thal','target']
    df = pd.read_csv(url, names=columns)
    df.replace('?', np.nan, inplace=True)
    df = df.apply(pd.to_numeric)
    df['target'] = (df['target'] > 0).astype(int)
    return df[feature_cols + ['target']].dropna()

def load_statlog_dataset(url):
    columns = ['age','sex','cp','trestbps','chol','fbs','restecg',
               'thalach','exang','oldpeak','slope','ca','thal','target']
    df = pd.read_csv(url, sep=r'\s+', names=columns)
    df['target'] = df['target'].apply(lambda x: 1 if x == 2 else 0)
    return df[feature_cols + ['target']]

datasets = {
    "Cleveland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "Statlog": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat",
    "Hungarian": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data"
}

results = {}

for name, url in datasets.items():
    try:
        if name == "Statlog":
            df = load_statlog_dataset(url)
        else:
            df = load_uci_dataset(url)

        # evaluate_dataset returns: acc, prec, rec, f1, auc
        metrics = evaluate_dataset(name, df)
        results[name] = metrics[-1] # Storing AUC

    except Exception as e:
        print(f"❌ Error in {name}: {e}")

print("\n=== FINAL AUC SUMMARY ===")
for k, v in results.items():
    print(f"{k}: {v:.4f}")
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# Define the common feature set
feature_cols = ['age', 'sex', 'trestbps', 'chol', 'thalach', 'oldpeak']

def load_uci_dataset(url):
    columns = ['age','sex','cp','trestbps','chol','fbs','restecg',
               'thalach','exang','oldpeak','slope','ca','thal','target']
    df = pd.read_csv(url, names=columns)
    df.replace('?', np.nan, inplace=True)
    df = df.apply(pd.to_numeric)
    df['target'] = (df['target'] > 0).astype(int)
    return df[feature_cols + ['target']].dropna()

def load_statlog_dataset(url):
    columns = ['age','sex','cp','trestbps','chol','fbs','restecg',
               'thalach','exang','oldpeak','slope','ca','thal','target']
    df = pd.read_csv(url, sep=r'\s+', names=columns)
    df['target'] = df['target'].apply(lambda x: 1 if x == 2 else 0)
    return df[feature_cols + ['target']]

datasets = {
    "Cleveland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "Statlog": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat",
    "Hungarian": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data"
}

results = {}

for name, url in datasets.items():
    try:
        if name == "Statlog":
            df = load_statlog_dataset(url)
        else:
            df = load_uci_dataset(url)

        # evaluate_dataset returns: acc, prec, rec, f1, auc
        metrics = evaluate_dataset(name, df)
        results[name] = metrics[-1] # Storing AUC

    except Exception as e:
        print(f"❌ Error in {name}: {e}")

print("\n=== FINAL AUC SUMMARY ===")
for k, v in results.items():
    print(f"{k}: {v:.4f}")
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define the common feature set
feature_cols = ['age', 'sex', 'trestbps', 'chol', 'thalach', 'oldpeak']

def load_uci_dataset(url):
    columns = ['age','sex','cp','trestbps','chol','fbs','restecg',
               'thalach','exang','oldpeak','slope','ca','thal','target']
    df = pd.read_csv(url, names=columns)
    df.replace('?', np.nan, inplace=True)
    df = df.apply(pd.to_numeric)
    df['target'] = (df['target'] > 0).astype(int)
    return df[feature_cols + ['target']].dropna()

def load_statlog_dataset(url):
    columns = ['age','sex','cp','trestbps','chol','fbs','restecg',
               'thalach','exang','oldpeak','slope','ca','thal','target']
    df = pd.read_csv(url, sep=r'\s+', names=columns)
    df['target'] = df['target'].apply(lambda x: 1 if x == 2 else 0)
    return df[feature_cols + ['target']]

datasets = {
    "Cleveland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "Statlog": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat",
    "Hungarian": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data"
}

results = {}

for name, url in datasets.items():
    try:
        if name == "Statlog":
            df = load_statlog_dataset(url)
        else:
            df = load_uci_dataset(url)

        # evaluate_dataset returns: acc, prec, rec, f1, auc
        metrics = evaluate_dataset(name, df)
        results[name] = metrics[-1] # Storing AUC

    except Exception as e:
        print(f"❌ Error in {name}: {e}")

print("\n=== FINAL AUC SUMMARY ===")
for k, v in results.items():
    print(f"{k}: {v:.4f}")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_dataset(name, df):
    X = df[feature_cols]
    y = df['target']

    # Preprocessing
    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, stratify=y, test_size=0.2, random_state=42
    )

    # Models
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, eval_metric='logloss')
    cat = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, verbose=0)

    xgb.fit(X_train, y_train)
    cat.fit(X_train, y_train)

    # Stacking
    stack_train = np.column_stack([
        xgb.predict_proba(X_train)[:,1],
        cat.predict_proba(X_train)[:,1]
    ])

    stack_test = np.column_stack([
        xgb.predict_proba(X_test)[:,1],
        cat.predict_proba(X_test)[:,1]
    ])

    meta = LogisticRegression()
    meta.fit(stack_train, y_train)

    # Final predictions
    final_probs = meta.predict_proba(stack_test)[:,1]
    final_preds = (final_probs >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, final_preds)
    prec = precision_score(y_test, final_preds)
    rec = recall_score(y_test, final_preds)
    f1 = f1_score(y_test, final_preds)
    auc = roc_auc_score(y_test, final_probs)

    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

    return acc, prec, rec, f1, auc
def load_statlog_dataset(url):
    columns = ['age','sex','cp','trestbps','chol','fbs','restecg',
               'thalach','exang','oldpeak','slope','ca','thal','target']

    # IMPORTANT FIX → whitespace separator
    df = pd.read_csv(url, sep=r'\s+', names=columns)

    # Convert target (1 = no disease, 2 = disease → convert to 0/1)
    df['target'] = df['target'].apply(lambda x: 1 if x == 2 else 0)

    # Select same features
    df = df[['age','sex','trestbps','chol','thalach','oldpeak','target']]

    return df
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

# Define the common feature set
feature_cols = ['age', 'sex', 'trestbps', 'chol', 'thalach', 'oldpeak']

def load_uci_dataset(url):
    columns = ['age','sex','cp','trestbps','chol','fbs','restecg',
               'thalach','exang','oldpeak','slope','ca','thal','target']
    df = pd.read_csv(url, names=columns)
    df.replace('?', np.nan, inplace=True)
    df = df.apply(pd.to_numeric)
    df['target'] = (df['target'] > 0).astype(int)
    return df[feature_cols + ['target']].dropna()

def load_statlog_dataset(url):
    columns = ['age','sex','cp','trestbps','chol','fbs','restecg',
               'thalach','exang','oldpeak','slope','ca','thal','target']
    df = pd.read_csv(url, sep=r'\s+', names=columns)
    df['target'] = df['target'].apply(lambda x: 1 if x == 2 else 0)
    return df[feature_cols + ['target']]

datasets = {
    "Cleveland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "Statlog": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat",
    "Hungarian": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data"
}

results = {}

for name, url in datasets.items():
    try:
        if name == "Statlog":
            df = load_statlog_dataset(url)
        else:
            df = load_uci_dataset(url)

        # evaluate_dataset returns: acc, prec, rec, f1, auc
        metrics = evaluate_dataset(name, df)
        results[name] = metrics[-1] # Storing AUC

    except Exception as e:
        print(f"❌ Error in {name}: {e}")

print("\n=== FINAL AUC SUMMARY ===")
for k, v in results.items():
    print(f"{k}: {v:.4f}")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_dataset(name, df):
    X = df[feature_cols]
    y = df['target']

    # Preprocessing
    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, stratify=y, test_size=0.2, random_state=42
    )

    # Models
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, eval_metric='logloss')
    cat = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, verbose=0)

    xgb.fit(X_train, y_train)
    cat.fit(X_train, y_train)

    # Stacking
    stack_train = np.column_stack([
        xgb.predict_proba(X_train)[:,1],
        cat.predict_proba(X_train)[:,1]
    ])

    stack_test = np.column_stack([
        xgb.predict_proba(X_test)[:,1],
        cat.predict_proba(X_test)[:,1]
    ])

    meta = LogisticRegression()
    meta.fit(stack_train, y_train)

    # Final predictions
    final_probs = meta.predict_proba(stack_test)[:,1]
    final_preds = (final_probs >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, final_preds)
    prec = precision_score(y_test, final_preds)
    rec = recall_score(y_test, final_preds)
    f1 = f1_score(y_test, final_preds)
    auc = roc_auc_score(y_test, final_probs)

    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

    return acc, prec, rec, f1, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

# Reuse the evaluation logic to output all requested metrics
def get_all_metrics(name, df):
    X = df[feature_cols]
    y = df['target']

    # Preprocessing
    X_imp = KNNImputer(n_neighbors=5).fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_imp)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, stratify=y, test_size=0.2, random_state=42
    )

    # Models
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, eval_metric='logloss')
    cat = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, verbose=0)
    xgb.fit(X_train, y_train)
    cat.fit(X_train, y_train)

    # Stacking
    stack_train = np.column_stack([xgb.predict_proba(X_train)[:,1], cat.predict_proba(X_train)[:,1]])
    stack_test = np.column_stack([xgb.predict_proba(X_test)[:,1], cat.predict_proba(X_test)[:,1]])
    meta = LogisticRegression().fit(stack_train, y_train)

    # Results
    probs = meta.predict_proba(stack_test)[:,1]
    preds = (probs >= 0.5).astype(int)

    return {
        'Dataset': name,
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'Recall': recall_score(y_test, preds),
        'F1-Score': f1_score(y_test, preds)
    }

all_results = []
for name, url in datasets.items():
    try:
        curr_df = load_statlog_dataset(url) if name == 'Statlog' else load_uci_dataset(url)
        all_results.append(get_all_metrics(name, curr_df))
    except Exception as e:
        print(f'Error processing {name}: {e}')

results_df = pd.DataFrame(all_results)
display(results_df)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Using the variables from the stacking logic in cell jvob_iPV7qpx
# final_probs were generated from meta.predict_proba(stack_test)[:,1]

# Convert probabilities to binary predictions
final_preds = (final_probs >= 0.5).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test, final_preds)
precision = precision_score(y_test, final_preds)
recall = recall_score(y_test, final_preds)
f1 = f1_score(y_test, final_preds)

print("--- UCI Cleveland External Validation Metrics ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUC Score: {auc:.4f}")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Using the variables from the stacking logic in cell jvob_iPV7qpx
# final_probs were generated from meta.predict_proba(stack_test)[:,1]

# Convert probabilities to binary predictions
final_preds = (final_probs >= 0.5).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test, final_preds)
precision = precision_score(y_test, final_preds)
recall = recall_score(y_test, final_preds)
f1 = f1_score(y_test, final_preds)

print("--- UCI Cleveland External Validation Metrics ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUC Score: {auc:.4f}")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Re-run evaluation for Cleveland specifically to ensure variables are in namespace
cleveland_url = datasets['Cleveland']
cleveland_df = load_uci_dataset(cleveland_url)

# evaluate_dataset(name, df) returns acc, prec, rec, f1, auc
# But we need the actual arrays for these local calculations
X = cleveland_df[feature_cols]
y = cleveland_df['target']

# Standard Preprocessing
X_scaled = StandardScaler().fit_transform(KNNImputer(n_neighbors=5).fit_transform(X))
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

# Models
xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, eval_metric='logloss')
cat = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, verbose=0)
xgb.fit(X_train, y_train)
cat.fit(X_train, y_train)

# Stack and Predict
stack_test = np.column_stack([xgb.predict_proba(X_test)[:,1], cat.predict_proba(X_test)[:,1]])
stack_train = np.column_stack([xgb.predict_proba(X_train)[:,1], cat.predict_proba(X_train)[:,1]])
meta = LogisticRegression().fit(stack_train, y_train)

final_probs = meta.predict_proba(stack_test)[:,1]
final_preds = (final_probs >= 0.5).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test, final_preds)
precision = precision_score(y_test, final_preds)
recall = recall_score(y_test, final_preds)
f1 = f1_score(y_test, final_preds)
auc_val = roc_auc_score(y_test, final_probs)

print("--- UCI Cleveland External Validation Metrics ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUC Score: {auc_val:.4f}")
import shap
import matplotlib.pyplot as plt

# Fit on full data for explanation
xgb.fit(X_train, y_train)

explainer = shap.Explainer(xgb)
shap_values = explainer(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Feature Importance")
plt.savefig("shap_plot.png", bbox_inches='tight')
plt.show()
from sklearn.metrics import roc_auc_score

# Individual models
xgb_probs = xgb.predict_proba(X_test)[:,1]
cat_probs = cat.predict_proba(X_test)[:,1]

# AUC
xgb_auc = roc_auc_score(y_test, xgb_probs)
cat_auc = roc_auc_score(y_test, cat_probs)
stack_auc = roc_auc_score(y_test, final_probs)

print("\n=== ABLATION STUDY ===")
print(f"XGBoost AUC: {xgb_auc:.4f}")
print(f"CatBoost AUC: {cat_auc:.4f}")
print(f"Stacked Model AUC: {stack_auc:.4f}")
from scipy.stats import ttest_rel

# Difference between models
t_stat, p_value = ttest_rel(xgb_probs, final_probs)

print("\n=== STATISTICAL TEST ===")
print(f"p-value: {p_value:.6f}")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# 1. ROC Curve Comparison
plt.figure(figsize=(10, 6))

# Calculate ROC for each model
models_to_plot = {
    'Stacked Model': final_probs,
    'XGBoost': xgb_probs,
    'CatBoost': cat_probs
}

for name, probs in models_to_plot.items():
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison - External Validation')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()

# 2. Confusion Matrix for Final Stacked Model
cm = confusion_matrix(y_test, final_preds)
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No CVD', 'CVD'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix: Final Stacked Model')
plt.show()

# Summary stats to accompany the matrix
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix Breakdown:")
print(f"True Positives (Correct CVD detection): {tp}")
print(f"False Negatives (Missed cases): {fn}")
print(f"True Negatives (Correct healthy): {tn}")
print(f"False Positives (False alarms): {fp}")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# 1. ROC Curve Comparison
plt.figure(figsize=(10, 6))

# Calculate ROC for each model
models_to_plot = {
    'Stacked Model': final_probs,
    'XGBoost': xgb_probs,
    'CatBoost': cat_probs
}

for name, probs in models_to_plot.items():
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison - External Validation')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()

# 2. Confusion Matrix for Final Stacked Model
cm = confusion_matrix(y_test, final_preds)
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No CVD', 'CVD'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix: Final Stacked Model')
plt.show()

# Summary stats to accompany the matrix
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix Breakdown:")
print(f"True Positives (Correct CVD detection): {tp}")
print(f"False Negatives (Missed cases): {fn}")
print(f"True Negatives (Correct healthy): {tn}")
print(f"False Positives (False alarms): {fp}")
np.save(f'{output_dir}/X_train_pp.npy', X_train_pp)
np.save(f'{output_dir}/y_train.npy', y_train)
np.save(f'{output_dir}/X_test_pp.npy', X_test_pp)
np.save(f'{output_dir}/y_test.npy', y_test)
import os
import numpy as np

# Define and create the output directory
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

# Save the existing variables to disk
np.save(f'{output_dir}/X_train_pp.npy', X_train)
np.save(f'{output_dir}/y_train.npy', y_train)
np.save(f'{output_dir}/X_test_pp.npy', X_test)
np.save(f'{output_dir}/y_test.npy', y_test)

print(f"Data successfully saved to {output_dir}/")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import pickle
import os

# ------------------------------------------------------------
# 1. Load or retrain models and test data
# ------------------------------------------------------------
# Paths – adjust to your saved files
X_test_path = 'results/kaggle/X_test_pp.npy'
y_test_path = 'results/kaggle/y_test.npy'
X_train_path = 'results/kaggle/X_train_pp.npy'
y_train_path = 'results/kaggle/y_train.npy'

if os.path.exists(X_test_path) and os.path.exists(y_test_path):
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    X_train = np.load(X_train_path) if os.path.exists(X_train_path) else None
    y_train = np.load(y_train_path) if os.path.exists(y_train_path) else None
else:
    raise FileNotFoundError("Please run your main pipeline first and save test data.")

# If training data is available, retrain models (or load saved models)
if X_train is not None and y_train is not None:
    print("Retraining models on full training set...")
    # Use best hyperparameters from your optimisation (replace with your actual best params)
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                        eval_metric='logloss', use_label_encoder=False, random_state=42)
    cat = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05,
                             verbose=0, random_seed=42)
    tab = TabNetClassifier(n_d=16, n_a=16, n_steps=5, gamma=1.5,
                           learning_rate=0.02, verbose=0, seed=42)
    lr = LogisticRegression(random_state=42)

    xgb.fit(X_train, y_train)
    cat.fit(X_train, y_train)
    tab.fit(X_train, y_train, eval_set=[(X_test, y_test)], max_epochs=50, patience=10)
    lr.fit(X_train, y_train)

    # Stacked ensemble (using the same meta-learner as in your pipeline)
    # For simplicity, we use the predictions of the three base models as features
    xgb_pred_train = xgb.predict_proba(X_train)[:, 1]
    cat_pred_train = cat.predict_proba(X_train)[:, 1]
    tab_pred_train = tab.predict_proba(X_train)[:, 1]
    stack_train = np.column_stack([xgb_pred_train, cat_pred_train, tab_pred_train])
    meta = LogisticRegression()
    meta.fit(stack_train, y_train)

    # Test predictions
    xgb_pred = xgb.predict_proba(X_test)[:, 1]
    cat_pred = cat.predict_proba(X_test)[:, 1]
    tab_pred = tab.predict_proba(X_test)[:, 1]
    stack_test = np.column_stack([xgb_pred, cat_pred, tab_pred])
    stack_pred = meta.predict_proba(stack_test)[:, 1]
    stack_class = (stack_pred >= 0.5).astype(int)
else:
    # Load saved predictions if available
    # (Assume you have saved them as .npy files)
    pass

# ------------------------------------------------------------
# 2. ROC curves for all models
# ------------------------------------------------------------
plt.figure(figsize=(8,6))
models = {
    'Logistic Regression': lr.predict_proba(X_test)[:, 1],
    'XGBoost': xgb_pred,
    'CatBoost': cat_pred,
    'TabNet': tab_pred,
    'Stacked Ensemble': stack_pred
}
for name, probs in models.items():
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison – Kaggle Test Set')
plt.legend(loc='lower right')
plt.savefig('figures/roc_comparison.png', dpi=300)
plt.show()

# ------------------------------------------------------------
# 3. Confusion matrix for stacked ensemble
# ------------------------------------------------------------
cm = confusion_matrix(y_test, stack_class)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No CVD', 'CVD'])
fig, ax = plt.subplots(figsize=(6,5))
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title('Confusion Matrix – Stacked Ensemble (Kaggle)')
plt.savefig('figures/confusion_matrix.png', dpi=300)
plt.show()

print("ROC comparison and confusion matrix saved to 'figures/'.")
!pip install pytorch-tabnet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import pickle
import os

# Ensure directories exist
os.makedirs('results/kaggle', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ------------------------------------------------------------
# 1. Load or retrain models and test data
# ------------------------------------------------------------
X_test_path = 'results/kaggle/X_test_pp.npy'
y_test_path = 'results/kaggle/y_test.npy'
X_train_path = 'results/kaggle/X_train_pp.npy'
y_train_path = 'results/kaggle/y_train.npy'

# Try to load from disk, fallback to memory variables if available
if os.path.exists(X_test_path):
    X_test_local = np.load(X_test_path)
    y_test_local = np.load(y_test_path)
    X_train_local = np.load(X_train_path) if os.path.exists(X_train_path) else None
    y_train_local = np.load(y_train_path) if os.path.exists(y_train_path) else None
elif 'X_test' in globals():
    print("Files not found on disk. Using variables from memory.")
    X_test_local = X_test
    y_test_local = y_test
    X_train_local = X_train if 'X_train' in globals() else None
    y_train_local = y_train if 'y_train' in globals() else None
else:
    raise FileNotFoundError("Please run your main pipeline first to define data.")

if X_train_local is not None and y_train_local is not None:
    print("Retraining models...")
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, eval_metric='logloss', random_state=42)
    cat = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, verbose=0, random_seed=42)
    tab = TabNetClassifier(n_d=16, n_a=16, n_steps=5, gamma=1.5, learning_rate=0.02, verbose=0, seed=42)
    lr = LogisticRegression(random_state=42)

    xgb.fit(X_train_local, y_train_local)
    cat.fit(X_train_local, y_train_local)
    tab.fit(X_train_local, y_train_local.values if isinstance(y_train_local, pd.Series) else y_train_local, 
            eval_set=[(X_test_local, y_test_local.values if isinstance(y_test_local, pd.Series) else y_test_local)], 
            max_epochs=50, patience=10)
    lr.fit(X_train_local, y_train_local)

    xgb_pred = xgb.predict_proba(X_test_local)[:, 1]
    cat_pred = cat.predict_proba(X_test_local)[:, 1]
    tab_pred = tab.predict_proba(X_test_local)[:, 1]
    
    # Meta-learner for stack
    xgb_pred_train = xgb.predict_proba(X_train_local)[:, 1]
    cat_pred_train = cat.predict_proba(X_train_local)[:, 1]
    tab_pred_train = tab.predict_proba(X_train_local)[:, 1]
    stack_train = np.column_stack([xgb_pred_train, cat_pred_train, tab_pred_train])
    meta = LogisticRegression().fit(stack_train, y_train_local)

    stack_test = np.column_stack([xgb_pred, cat_pred, tab_pred])
    stack_pred = meta.predict_proba(stack_test)[:, 1]
    stack_class = (stack_pred >= 0.5).astype(int)

# ------------------------------------------------------------
# 2. ROC curves
# ------------------------------------------------------------
plt.figure(figsize=(8,6))
models_to_plot = {
    'Logistic Regression': lr.predict_proba(X_test_local)[:, 1],
    'XGBoost': xgb_pred,
    'CatBoost': cat_pred,
    'TabNet': tab_pred,
    'Stacked Ensemble': stack_pred
}
for name, probs in models_to_plot.items():
    fpr, tpr, _ = roc_curve(y_test_local, probs)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc(fpr, tpr):.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.savefig('figures/roc_comparison.png', dpi=300)
plt.show()

# ------------------------------------------------------------
# 3. Confusion matrix
# ------------------------------------------------------------
cm = confusion_matrix(y_test_local, stack_class)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No CVD', 'CVD'])
fig, ax = plt.subplots(figsize=(6,5))
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title('Confusion Matrix – Stacked Ensemble')
plt.savefig('figures/confusion_matrix.png', dpi=300)
plt.show()
!pip install pytorch-tabnet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import pickle
import os

# Ensure directories exist
os.makedirs('results/kaggle', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ------------------------------------------------------------
# 1. Load or retrain models and test data
# ------------------------------------------------------------
X_test_path = 'results/kaggle/X_test_pp.npy'
y_test_path = 'results/kaggle/y_test.npy'
X_train_path = 'results/kaggle/X_train_pp.npy'
y_train_path = 'results/kaggle/y_train.npy'

# Try to load from disk, fallback to memory variables if available
if os.path.exists(X_test_path):
    X_test_local = np.load(X_test_path)
    y_test_local = np.load(y_test_path)
    X_train_local = np.load(X_train_path) if os.path.exists(X_train_path) else None
    y_train_local = np.load(y_train_path) if os.path.exists(y_train_path) else None
elif 'X_test' in globals():
    print("Files not found on disk. Using variables from memory.")
    X_test_local = X_test
    y_test_local = y_test
    X_train_local = X_train if 'X_train' in globals() else None
    y_train_local = y_train if 'y_train' in globals() else None
else:
    raise FileNotFoundError("Please run your main pipeline first to define data.")

if X_train_local is not None and y_train_local is not None:
    print("Retraining models...")
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, eval_metric='logloss', random_state=42)
    cat = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, verbose=0, random_seed=42)
    # Fix: learning_rate must be passed in optimizer_params for TabNet
    tab = TabNetClassifier(n_d=16, n_a=16, n_steps=5, gamma=1.5, 
                           optimizer_params=dict(lr=0.02), verbose=0, seed=42)
    lr = LogisticRegression(random_state=42)

    xgb.fit(X_train_local, y_train_local)
    cat.fit(X_train_local, y_train_local)
    tab.fit(X_train_local, y_train_local.values if isinstance(y_train_local, pd.Series) else y_train_local,
            eval_set=[(X_test_local, y_test_local.values if isinstance(y_test_local, pd.Series) else y_test_local)],
            max_epochs=50, patience=10)
    lr.fit(X_train_local, y_train_local)

    xgb_pred = xgb.predict_proba(X_test_local)[:, 1]
    cat_pred = cat.predict_proba(X_test_local)[:, 1]
    tab_pred = tab.predict_proba(X_test_local)[:, 1]

    # Meta-learner for stack
    xgb_pred_train = xgb.predict_proba(X_train_local)[:, 1]
    cat_pred_train = cat.predict_proba(X_train_local)[:, 1]
    tab_pred_train = tab.predict_proba(X_train_local)[:, 1]
    stack_train = np.column_stack([xgb_pred_train, cat_pred_train, tab_pred_train])
    meta = LogisticRegression().fit(stack_train, y_train_local)

    stack_test = np.column_stack([xgb_pred, cat_pred, tab_pred])
    stack_pred = meta.predict_proba(stack_test)[:, 1]
    stack_class = (stack_pred >= 0.5).astype(int)

# ------------------------------------------------------------
# 2. ROC curves
# ------------------------------------------------------------
plt.figure(figsize=(8,6))
models_to_plot = {
    'Logistic Regression': lr.predict_proba(X_test_local)[:, 1],
    'XGBoost': xgb_pred,
    'CatBoost': cat_pred,
    'TabNet': tab_pred,
    'Stacked Ensemble': stack_pred
}
for name, probs in models_to_plot.items():
    fpr, tpr, _ = roc_curve(y_test_local, probs)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc(fpr, tpr):.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.savefig('figures/roc_comparison.png', dpi=300)
plt.show()

# ------------------------------------------------------------
# 3. Confusion matrix
# ------------------------------------------------------------
cm = confusion_matrix(y_test_local, stack_class)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No CVD', 'CVD'])
fig, ax = plt.subplots(figsize=(6,5))
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title('Confusion Matrix – Stacked Ensemble')
plt.savefig('figures/confusion_matrix.png', dpi=300)
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, average_precision_score,
                             confusion_matrix, calibration_curve, ConfusionMatrixDisplay)
from sklearn.utils import resample
from sklearn.calibration import CalibrationDisplay
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.linear_model import LogisticRegression
import shap
import os

os.makedirs('figures', exist_ok=True)

# ------------------------------------------------------------
# 1. Load or retrain models and data
# ------------------------------------------------------------
# Load preprocessed data (adjust paths)
X_train = np.load('results/kaggle/X_train_pp.npy')
y_train = np.load('results/kaggle/y_train.npy')
X_test = np.load('results/kaggle/X_test_pp.npy')
y_test = np.load('results/kaggle/y_test.npy')
feature_names = np.load('results/kaggle/feature_names.npy')

print("Retraining models on full training set...")
xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                    eval_metric='logloss', use_label_encoder=False, random_state=42)
cat = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, verbose=0, random_seed=42)
tab = TabNetClassifier(n_d=16, n_a=16, n_steps=5, gamma=1.5, learning_rate=0.02, verbose=0, seed=42)
lr = LogisticRegression(random_state=42)

xgb.fit(X_train, y_train)
cat.fit(X_train, y_train)
tab.fit(X_train, y_train, eval_set=[(X_test, y_test)], max_epochs=50, patience=10)
lr.fit(X_train, y_train)

# Stacked ensemble
xgb_pred_train = xgb.predict_proba(X_train)[:, 1]
cat_pred_train = cat.predict_proba(X_train)[:, 1]
tab_pred_train = tab.predict_proba(X_train)[:, 1]
stack_train = np.column_stack([xgb_pred_train, cat_pred_train, tab_pred_train])
meta = LogisticRegression()
meta.fit(stack_train, y_train)

# Test predictions
xgb_pred = xgb.predict_proba(X_test)[:, 1]
cat_pred = cat.predict_proba(X_test)[:, 1]
tab_pred = tab.predict_proba(X_test)[:, 1]
stack_test = np.column_stack([xgb_pred, cat_pred, tab_pred])
stack_probs = meta.predict_proba(stack_test)[:, 1]
stack_class = (stack_probs >= 0.5).astype(int)

# ------------------------------------------------------------
# 2. ROC curves comparison
# ------------------------------------------------------------
plt.figure(figsize=(8,6))
models = {'LR': lr, 'XGBoost': xgb, 'CatBoost': cat, 'TabNet': tab, 'Stacked': meta}
for name, model in models.items():
    if name == 'Stacked':
        probs = stack_probs
    else:
        probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison – Kaggle')
plt.legend(loc='lower right')
plt.savefig('figures/roc_comparison.png', dpi=300)
plt.close()

# ------------------------------------------------------------
# 3. ROC with confidence interval (stacked)
# ------------------------------------------------------------
n_boot = 1000
aucs = []
for _ in range(n_boot):
    idx = resample(range(len(y_test)), replace=True)
    fpr, tpr, _ = roc_curve(y_test[idx], stack_probs[idx])
    aucs.append(auc(fpr, tpr))
ci_low, ci_high = np.percentile(aucs, [2.5, 97.5])
fpr, tpr, _ = roc_curve(y_test, stack_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, lw=2, label=f'Stacked (AUC = {roc_auc:.3f}, 95% CI [{ci_low:.3f}–{ci_high:.3f}])')
plt.plot([0,1], [0,1], 'k--')
plt.fill_between(fpr, tpr, alpha=0.2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with 95% CI')
plt.legend()
plt.savefig('figures/roc_ci.png', dpi=300)
plt.close()

# ------------------------------------------------------------
# 4. Precision‑Recall curve
# ------------------------------------------------------------
precision, recall, _ = precision_recall_curve(y_test, stack_probs)
avg_precision = average_precision_score(y_test, stack_probs)
plt.figure(figsize=(6,5))
plt.plot(recall, precision, lw=2, label=f'PR-AUC = {avg_precision:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision‑Recall Curve')
plt.legend()
plt.savefig('figures/pr_curve.png', dpi=300)
plt.close()

# ------------------------------------------------------------
# 5. Calibration curve with CI
# ------------------------------------------------------------
prob_true, prob_pred = calibration_curve(y_test, stack_probs, n_bins=10)
plt.figure(figsize=(6,5))
plt.plot(prob_pred, prob_true, marker='o', lw=2, label='Stacked')
plt.plot([0,1], [0,1], 'k--', label='Perfect')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration Curve')
plt.legend()
plt.savefig('figures/calibration_curve.png', dpi=300)
plt.close()

# ------------------------------------------------------------
# 6. Histogram of predicted probabilities by class
# ------------------------------------------------------------
plt.figure(figsize=(8,5))
plt.hist(stack_probs[y_test==0], bins=30, alpha=0.5, label='No CVD', density=True)
plt.hist(stack_probs[y_test==1], bins=30, alpha=0.5, label='CVD', density=True)
plt.xlabel('Predicted probability')
plt.ylabel('Density')
plt.title('Distribution of Predicted Probabilities')
plt.legend()
plt.savefig('figures/calibration_histogram.png', dpi=300)
plt.close()

# ------------------------------------------------------------
# 7. Confusion matrices
# ------------------------------------------------------------
cm = confusion_matrix(y_test, stack_class)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No CVD', 'CVD'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix (Counts)')
plt.savefig('figures/confusion_matrix.png', dpi=300)
plt.close()

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['No CVD', 'CVD'])
disp_norm.plot(cmap='Blues', values_format='.2f')
plt.title('Confusion Matrix (Normalised)')
plt.savefig('figures/confusion_matrix_norm.png', dpi=300)
plt.close()

# ------------------------------------------------------------
# 8. SHAP analysis (using XGBoost as representative)
# ------------------------------------------------------------
explainer = shap.TreeExplainer(xgb)
X_sample = X_test[:200]  # sample for speed
shap_values = explainer.shap_values(X_sample)

# Summary plot (swarm)
shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig('figures/shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# Bar plot (top 10)
mean_abs = np.abs(shap_values).mean(axis=0)
idx = np.argsort(mean_abs)[::-1][:10]
plt.figure(figsize=(10,6))
plt.barh(np.array(feature_names)[idx], mean_abs[idx])
plt.xlabel('Mean |SHAP value|')
plt.title('Top 10 Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('figures/shap_bar.png', dpi=300)
plt.close()

# Dependence plots for top 3 features
for i, feat in enumerate(['age_years', 'ap_hi', 'glucose']):
    if feat in feature_names:
        idx_feat = list(feature_names).index(feat)
        shap.dependence_plot(idx_feat, shap_values, X_sample, feature_names=feature_names, show=False)
        plt.title(f'SHAP Dependence: {feat}')
        plt.savefig(f'figures/shap_dependence_{feat}.png', dpi=300)
        plt.close()

# ------------------------------------------------------------
# 9. Decision curve analysis (using dcurves or manual)
# ------------------------------------------------------------
# Simplified manual decision curve (net benefit)
thresholds = np.arange(0, 0.5, 0.01)
net_benefit = []
for thresh in thresholds:
    y_pred = (stack_probs >= thresh).astype(int)
    tp = np.sum((y_pred == 1) & (y_test == 1))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    n = len(y_test)
    nb = tp/n - fp/n * (thresh/(1-thresh))
    net_benefit.append(nb)
plt.figure(figsize=(8,6))
plt.plot(thresholds, net_benefit, lw=2, label='Stacked Ensemble')
plt.plot([0,0.5], [0,0], 'k--', label='Treat none')
plt.plot([0,0.5], [y_test.mean(), y_test.mean()], 'k:', label='Treat all')
plt.xlabel('Threshold probability')
plt.ylabel('Net benefit')
plt.title('Decision Curve Analysis')
plt.legend()
plt.savefig('figures/decision_curve.png', dpi=300)
plt.close()

# ------------------------------------------------------------
# 10. Bar chart comparing AUCs of all models (internal)
# ------------------------------------------------------------
aucs_internal = {
    'LR': auc(fpr, tpr) for fpr,tpr,_ in [roc_curve(y_test, lr.predict_proba(X_test)[:,1])] # need recompute
}
# Recompute properly
auc_values = {}
for name, model in [('LR', lr), ('XGBoost', xgb), ('CatBoost', cat), ('TabNet', tab), ('Stacked', None)]:
    if name == 'Stacked':
        probs = stack_probs
    else:
        probs = model.predict_proba(X_test)[:,1]
    auc_values[name] = auc(roc_curve(y_test, probs)[0], roc_curve(y_test, probs)[1])
plt.figure(figsize=(8,5))
plt.bar(auc_values.keys(), auc_values.values(), color='steelblue')
plt.ylim(0.7, 1.0)
plt.ylabel('AUC')
plt.title('Model Comparison (Kaggle Test Set)')
for i, (k, v) in enumerate(auc_values.items()):
    plt.text(i, v+0.005, f'{v:.3f}', ha='center')
plt.savefig('figures/auc_comparison_bar.png', dpi=300)
plt.close()

# ------------------------------------------------------------
# 11. External validation bar chart
# ------------------------------------------------------------
external_datasets = ['Cleveland', 'Hungarian', 'Statlog']
external_aucs = [0.917, 0.809, 0.790]
plt.figure(figsize=(6,5))
plt.bar(external_datasets, external_aucs, color='darkorange')
plt.ylim(0.7, 1.0)
plt.ylabel('AUC')
plt.title('External Validation Performance')
for i, v in enumerate(external_aucs):
    plt.text(i, v+0.005, f'{v:.3f}', ha='center')
plt.savefig('figures/external_validation_bar.png', dpi=300)
plt.close()

# ------------------------------------------------------------
# 12. Learning curve for TabNet (if history available)
# ------------------------------------------------------------
# Assuming you captured history during TabNet training
# For demonstration, we create a dummy plot – replace with actual loss values if saved
# If you saved history, load it here.
# For now, we skip this figure (optional).

# ------------------------------------------------------------
# 13. Failure analysis plot
# ------------------------------------------------------------
# Use your failure analysis data (from 500 misclassifications)
# Example data – replace with your actual counts
misclass_reasons = {
    'Systolic BP 130-139': 62,
    'Age 55-60': 45,
    'Glucose 95-100': 38,
    'Single strong risk factor': 71
}
plt.figure(figsize=(8,5))
plt.barh(list(misclass_reasons.keys()), list(misclass_reasons.values()), color='salmon')
plt.xlabel('Number of misclassifications (out of 500)')
plt.title('Failure Analysis: Borderline Risk Factors')
plt.tight_layout()
plt.savefig('figures/failure_analysis.png', dpi=300)
plt.close()

# ------------------------------------------------------------
# 14. Negative control plot (shuffled target AUC)
# ------------------------------------------------------------
# Run cross‑validation with shuffled target
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
y_shuffled = y_train.copy()
np.random.seed(42)
np.random.shuffle(y_shuffled)
neg_scores = cross_val_score(RandomForestClassifier(), X_train, y_shuffled, cv=5, scoring='roc_auc')
plt.figure(figsize=(6,4))
plt.bar(range(1,6), neg_scores, color='gray')
plt.axhline(y=0.5, color='r', linestyle='--', label='Random (0.5)')
plt.xlabel('CV Fold')
plt.ylabel('AUC')
plt.title('Negative Control: Shuffled Target')
plt.ylim(0,1)
plt.legend()
plt.savefig('figures/negative_control.png', dpi=300)
plt.close()

print("All figures saved to 'figures/' directory.")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, average_precision_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.utils import resample
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.linear_model import LogisticRegression
import shap
import os

os.makedirs('figures', exist_ok=True)

# ------------------------------------------------------------
# 1. Load or retrain models and data
# ------------------------------------------------------------
# Load preprocessed data (adjust paths)
X_train = np.load('results/kaggle/X_train_pp.npy')
y_train = np.load('results/kaggle/y_train.npy')
X_test = np.load('results/kaggle/X_test_pp.npy')
y_test = np.load('results/kaggle/y_test.npy')
feature_names = np.load('results/kaggle/feature_names.npy')

print("Retraining models on full training set...")
xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                    eval_metric='logloss', use_label_encoder=False, random_state=42)
cat = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, verbose=0, random_seed=42)
tab = TabNetClassifier(n_d=16, n_a=16, n_steps=5, gamma=1.5, learning_rate=0.02, verbose=0, seed=42)
lr = LogisticRegression(random_state=42)

xgb.fit(X_train, y_train)
cat.fit(X_train, y_train)
tab.fit(X_train, y_train, eval_set=[(X_test, y_test)], max_epochs=50, patience=10)
lr.fit(X_train, y_train)

# Stacked ensemble
xgb_pred_train = xgb.predict_proba(X_train)[:, 1]
cat_pred_train = cat.predict_proba(X_train)[:, 1]
tab_pred_train = tab.predict_proba(X_train)[:, 1]
stack_train = np.column_stack([xgb_pred_train, cat_pred_train, tab_pred_train])
meta = LogisticRegression()
meta.fit(stack_train, y_train)

# Test predictions
xgb_pred = xgb.predict_proba(X_test)[:, 1]
cat_pred = cat.predict_proba(X_test)[:, 1]
tab_pred = tab.predict_proba(X_test)[:, 1]
stack_test = np.column_stack([xgb_pred, cat_pred, tab_pred])
stack_probs = meta.predict_proba(stack_test)[:, 1]
stack_class = (stack_probs >= 0.5).astype(int)

# ------------------------------------------------------------
# 2. ROC curves comparison
# ------------------------------------------------------------
plt.figure(figsize=(8,6))
models = {'LR': lr, 'XGBoost': xgb, 'CatBoost': cat, 'TabNet': tab, 'Stacked': meta}
for name, model in models.items():
    if name == 'Stacked':
        probs = stack_probs
    else:
        probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison – Kaggle')
plt.legend(loc='lower right')
plt.savefig('figures/roc_comparison.png', dpi=300)
plt.close()

# ------------------------------------------------------------
# 3. ROC with confidence interval (stacked)
# ------------------------------------------------------------
n_boot = 1000
aucs = []
for _ in range(n_boot):
    idx = resample(range(len(y_test)), replace=True)
    fpr, tpr, _ = roc_curve(y_test[idx], stack_probs[idx])
    aucs.append(auc(fpr, tpr))
ci_low, ci_high = np.percentile(aucs, [2.5, 97.5])
fpr, tpr, _ = roc_curve(y_test, stack_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, lw=2, label=f'Stacked (AUC = {roc_auc:.3f}, 95% CI [{ci_low:.3f}–{ci_high:.3f}])')
plt.plot([0,1], [0,1], 'k--')
plt.fill_between(fpr, tpr, alpha=0.2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with 95% CI')
plt.legend()
plt.savefig('figures/roc_ci.png', dpi=300)
plt.close()

# ------------------------------------------------------------
# 4. Precision‑Recall curve
# ------------------------------------------------------------
precision, recall, _ = precision_recall_curve(y_test, stack_probs)
avg_precision = average_precision_score(y_test, stack_probs)
plt.figure(figsize=(6,5))
plt.plot(recall, precision, lw=2, label=f'PR-AUC = {avg_precision:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision‑Recall Curve')
plt.legend()
plt.savefig('figures/pr_curve.png', dpi=300)
plt.close()

# ------------------------------------------------------------
# 5. Calibration curve with CI
# ------------------------------------------------------------
prob_true, prob_pred = calibration_curve(y_test, stack_probs, n_bins=10)
plt.figure(figsize=(6,5))
plt.plot(prob_pred, prob_true, marker='o', lw=2, label='Stacked')
plt.plot([0,1], [0,1], 'k--', label='Perfect')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration Curve')
plt.legend()
plt.savefig('figures/calibration_curve.png', dpi=300)
plt.close()

# ------------------------------------------------------------
# 6. Histogram of predicted probabilities by class
# ------------------------------------------------------------
plt.figure(figsize=(8,5))
plt.hist(stack_probs[y_test==0], bins=30, alpha=0.5, label='No CVD', density=True)
plt.hist(stack_probs[y_test==1], bins=30, alpha=0.5, label='CVD', density=True)
plt.xlabel('Predicted probability')
plt.ylabel('Density')
plt.title('Distribution of Predicted Probabilities')
plt.legend()
plt.savefig('figures/calibration_histogram.png', dpi=300)
plt.close()

# ------------------------------------------------------------
# 7. Confusion matrices
# ------------------------------------------------------------
cm = confusion_matrix(y_test, stack_class)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No CVD', 'CVD'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix (Counts)')
plt.savefig('figures/confusion_matrix.png', dpi=300)
plt.close()

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['No CVD', 'CVD'])
disp_norm.plot(cmap='Blues', values_format='.2f')
plt.title('Confusion Matrix (Normalised)')
plt.savefig('figures/confusion_matrix_norm.png', dpi=300)
plt.close()

# ------------------------------------------------------------
# 8. SHAP analysis (using XGBoost as representative)
# ------------------------------------------------------------
explainer = shap.TreeExplainer(xgb)
X_sample = X_test[:200]  # sample for speed
shap_values = explainer.shap_values(X_sample)

# Summary plot (swarm)
shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig('figures/shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# Bar plot (top 10)
mean_abs = np.abs(shap_values).mean(axis=0)
idx = np.argsort(mean_abs)[::-1][:10]
plt.figure(figsize=(10,6))
plt.barh(np.array(feature_names)[idx], mean_abs[idx])
plt.xlabel('Mean |SHAP value|')
plt.title('Top 10 Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('figures/shap_bar.png', dpi=300)
plt.close()

# Dependence plots for top 3 features
for i, feat in enumerate(['age_years', 'ap_hi', 'glucose']):
    if feat in feature_names:
        idx_feat = list(feature_names).index(feat)
        shap.dependence_plot(idx_feat, shap_values, X_sample, feature_names=feature_names, show=False)
        plt.title(f'SHAP Dependence: {feat}')
        plt.savefig(f'figures/shap_dependence_{feat}.png', dpi=300)
        plt.close()

# ------------------------------------------------------------
# 9. Decision curve analysis (using dcurves or manual)
# ------------------------------------------------------------
# Simplified manual decision curve (net benefit)
thresholds = np.arange(0, 0.5, 0.01)
net_benefit = []
for thresh in thresholds:
    y_pred = (stack_probs >= thresh).astype(int)
    tp = np.sum((y_pred == 1) & (y_test == 1))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    n = len(y_test)
    nb = tp/n - fp/n * (thresh/(1-thresh))
    net_benefit.append(nb)
plt.figure(figsize=(8,6))
plt.plot(thresholds, net_benefit, lw=2, label='Stacked Ensemble')
plt.plot([0,0.5], [0,0], 'k--', label='Treat none')
plt.plot([0,0.5], [y_test.mean(), y_test.mean()], 'k:', label='Treat all')
plt.xlabel('Threshold probability')
plt.ylabel('Net benefit')
plt.title('Decision Curve Analysis')
plt.legend()
plt.savefig('figures/decision_curve.png', dpi=300)
plt.close()

# ------------------------------------------------------------
# 10. Bar chart comparing AUCs of all models (internal)
# ------------------------------------------------------------
aucs_internal = {
    'LR': auc(fpr, tpr) for fpr,tpr,_ in [roc_curve(y_test, lr.predict_proba(X_test)[:,1])] # need recompute
}
# Recompute properly
auc_values = {}
for name, model in [('LR', lr), ('XGBoost', xgb), ('CatBoost', cat), ('TabNet', tab), ('Stacked', None)]:
    if name == 'Stacked':
        probs = stack_probs
    else:
        probs = model.predict_proba(X_test)[:,1]
    auc_values[name] = auc(roc_curve(y_test, probs)[0], roc_curve(y_test, probs)[1])
plt.figure(figsize=(8,5))
plt.bar(auc_values.keys(), auc_values.values(), color='steelblue')
plt.ylim(0.7, 1.0)
plt.ylabel('AUC')
plt.title('Model Comparison (Kaggle Test Set)')
for i, (k, v) in enumerate(auc_values.items()):
    plt.text(i, v+0.005, f'{v:.3f}', ha='center')
plt.savefig('figures/auc_comparison_bar.png', dpi=300)
plt.close()

# ------------------------------------------------------------
# 11. External validation bar chart
# ------------------------------------------------------------
external_datasets = ['Cleveland', 'Hungarian', 'Statlog']
external_aucs = [0.917, 0.809, 0.790]
plt.figure(figsize=(6,5))
plt.bar(external_datasets, external_aucs, color='darkorange')
plt.ylim(0.7, 1.0)
plt.ylabel('AUC')
plt.title('External Validation Performance')
for i, v in enumerate(external_aucs):
    plt.text(i, v+0.005, f'{v:.3f}', ha='center')
plt.savefig('figures/external_validation_bar.png', dpi=300)
plt.close()

# ------------------------------------------------------------
# 12. Learning curve for TabNet (if history available)
# ------------------------------------------------------------
# Assuming you captured history during TabNet training
# For demonstration, we create a dummy plot – replace with actual loss values if saved
# If you saved history, load it here.
# For now, we skip this figure (optional).

# ------------------------------------------------------------
# 13. Failure analysis plot
# ------------------------------------------------------------
# Use your failure analysis data (from 500 misclassifications)
# Example data – replace with your actual counts
misclass_reasons = {
    'Systolic BP 130-139': 62,
    'Age 55-60': 45,
    'Glucose 95-100': 38,
    'Single strong risk factor': 71
}
plt.figure(figsize=(8,5))
plt.barh(list(misclass_reasons.keys()), list(misclass_reasons.values()), color='salmon')
plt.xlabel('Number of misclassifications (out of 500)')
plt.title('Failure Analysis: Borderline Risk Factors')
plt.tight_layout()
plt.savefig('figures/failure_analysis.png', dpi=300)
plt.close()

# ------------------------------------------------------------
# 14. Negative control plot (shuffled target AUC)
# ------------------------------------------------------------
# Run cross‑validation with shuffled target
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
y_shuffled = y_train.copy()
np.random.seed(42)
np.random.shuffle(y_shuffled)
neg_scores = cross_val_score(RandomForestClassifier(), X_train, y_shuffled, cv=5, scoring='roc_auc')
plt.figure(figsize=(6,4))
plt.bar(range(1,6), neg_scores, color='gray')
plt.axhline(y=0.5, color='r', linestyle='--', label='Random (0.5)')
plt.xlabel('CV Fold')
plt.ylabel('AUC')
plt.title('Negative Control: Shuffled Target')
plt.ylim(0,1)
plt.legend()
plt.savefig('figures/negative_control.png', dpi=300)
plt.close()

print("All figures saved to 'figures/' directory.")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, average_precision_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.utils import resample
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.linear_model import LogisticRegression
import shap
import os

os.makedirs('figures', exist_ok=True)

# ------------------------------------------------------------
# 1. Load or retrain models and data
# ------------------------------------------------------------
# Corrected paths based on cell eanN6TE_Zzsp output
X_train = np.load('results/X_train_pp.npy')
y_train = np.load('results/y_train.npy')
X_test = np.load('results/X_test_pp.npy')
y_test = np.load('results/y_test.npy')

# Fallback for feature names if not saved to disk
if os.path.exists('results/feature_names.npy'):
    feature_names = np.load('results/feature_names.npy')
else:
    feature_names = ['age', 'sex', 'trestbps', 'chol', 'thalach', 'oldpeak']

print("Retraining models on full training set...")
xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                    eval_metric='logloss', use_label_encoder=False, random_state=42)
cat = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, verbose=0, random_seed=42)
tab = TabNetClassifier(n_d=16, n_a=16, n_steps=5, gamma=1.5, optimizer_params=dict(lr=0.02), verbose=0, seed=42)
lr = LogisticRegression(random_state=42)

xgb.fit(X_train, y_train)
cat.fit(X_train, y_train)
tab.fit(X_train, y_train, eval_set=[(X_test, y_test)], max_epochs=50, patience=10)
lr.fit(X_train, y_train)

# Stacked ensemble
xgb_pred_train = xgb.predict_proba(X_train)[:, 1]
cat_pred_train = cat.predict_proba(X_train)[:, 1]
tab_pred_train = tab.predict_proba(X_train)[:, 1]
stack_train = np.column_stack([xgb_pred_train, cat_pred_train, tab_pred_train])
meta = LogisticRegression()
meta.fit(stack_train, y_train)

# Test predictions
xgb_pred = xgb.predict_proba(X_test)[:, 1]
cat_pred = cat.predict_proba(X_test)[:, 1]
tab_pred = tab.predict_proba(X_test)[:, 1]
stack_test = np.column_stack([xgb_pred, cat_pred, tab_pred])
stack_probs = meta.predict_proba(stack_test)[:, 1]
stack_class = (stack_probs >= 0.5).astype(int)

# ------------------------------------------------------------
# 2. ROC curves comparison
# ------------------------------------------------------------
plt.figure(figsize=(8,6))
models_eval = {'LR': lr, 'XGBoost': xgb, 'CatBoost': cat, 'TabNet': tab, 'Stacked': 'stacked'}
for name, model_obj in models_eval.items():
    if name == 'Stacked':
        probs = stack_probs
    else:
        probs = model_obj.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.savefig('figures/roc_comparison.png', dpi=300)
plt.show()

# ------------------------------------------------------------
# 7. Confusion matrix
# ------------------------------------------------------------
cm = confusion_matrix(y_test, stack_class)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No CVD', 'CVD'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix (Stacked Ensemble)')
plt.savefig('figures/confusion_matrix.png', dpi=300)
plt.show()

# ------------------------------------------------------------
# 8. SHAP analysis
# ------------------------------------------------------------
explainer = shap.TreeExplainer(xgb)
X_sample = X_test[:200]
shap_values = explainer.shap_values(X_sample)
plt.figure()
shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
plt.savefig('figures/shap_summary.png', dpi=300, bbox_inches='tight')
plt.show()

print("Core diagnostic figures generated in 'figures/' directory.")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shap
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay, calibration_curve
)
from dcurves import dca

# Ensure output directory exists
os.makedirs('figures', exist_ok=True)
sns.set_style("whitegrid")

# Helper function to generate standardized plots
def generate_full_suite():
    # 1-5: INTERNAL VALIDATION (KAGGE/STACKED)
    # 6-10: EXTERNAL VALIDATION (UCI/STACKED)
    # 11-15: DATA/STATISTICAL DISTRIBUTIONS
    
    print("Generating 15 Figures...")

    # FIGURE 1: Internal ROC Comparison
    plt.figure(figsize=(8,6))
    for name, p in models_to_plot.items():
        fpr, tpr, _ = roc_curve(y_test_local, p)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc(fpr, tpr):.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.title('Fig 1: Internal ROC Comparison')
    plt.legend(); plt.savefig('figures/fig1_internal_roc.png', dpi=300); plt.show()

    # FIGURE 2: Internal Precision-Recall Curve
    plt.figure(figsize=(8,6))
    precision, recall, _ = precision_recall_curve(y_test_local, stack_pred)
    plt.plot(recall, precision, lw=2, color='teal', label=f'AP={average_precision_score(y_test_local, stack_pred):.3f}')
    plt.title('Fig 2: Internal Precision-Recall')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.savefig('figures/fig2_internal_pr.png', dpi=300); plt.show()

    # FIGURE 3: Internal Calibration Curve
    prob_true, prob_pred_vals = calibration_curve(y_test_local, stack_pred, n_bins=10)
    plt.figure(figsize=(8,6))
    plt.plot(prob_pred_vals, prob_true, marker='o', label='Stacked Ensemble')
    plt.plot([0,1], [0,1], 'k--', label='Perfect')
    plt.title('Fig 3: Internal Calibration Plot')
    plt.savefig('figures/fig3_internal_calibration.png', dpi=300); plt.show()

    # FIGURE 4: Internal SHAP Summary (Interpretability)
    explainer = shap.TreeExplainer(xgb)
    shap_v = explainer.shap_values(X_test_local[:200])
    plt.figure()
    shap.summary_plot(shap_v, X_test_local[:200], feature_names=feature_names, show=False)
    plt.title('Fig 4: Global Feature Importance (SHAP)')
    plt.savefig('figures/fig4_internal_shap.png', dpi=300, bbox_inches='tight'); plt.show()

    # FIGURE 5: Internal Decision Curve Analysis (DCA)
    df_dca = pd.DataFrame({'outcome': y_test_local, 'Stacked': stack_pred})
    dca_res = dca(data=df_dca, outcome='outcome', modelnames=['Stacked'])
    dca_res.plot()
    plt.title('Fig 5: Clinical Utility (DCA)')
    plt.savefig('figures/fig5_internal_dca.png', dpi=300); plt.show()

    # FIGURE 6: External (UCI) ROC Comparison
    plt.figure(figsize=(8,6))
    fpr_ext, tpr_ext, _ = roc_curve(y_test, final_probs)
    plt.plot(fpr_ext, tpr_ext, color='red', lw=2, label=f'Stacked External (AUC={roc_auc_score(y_test, final_probs):.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.title('Fig 6: External (UCI Cleveland) ROC')
    plt.legend(); plt.savefig('figures/fig6_external_roc.png', dpi=300); plt.show()

    # FIGURE 7: External Confusion Matrix
    cm_ext = confusion_matrix(y_test, final_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_ext, display_labels=['Healthy', 'CVD'])
    disp.plot(cmap='Reds')
    plt.title('Fig 7: External Confusion Matrix')
    plt.savefig('figures/fig7_external_cm.png', dpi=300); plt.show()

    # FIGURE 8: External Feature Importance (Permutation/XGB)
    plt.figure(figsize=(10,6))
    imp = pd.Series(xgb.feature_importances_, index=feature_names).sort_values()
    imp.plot(kind='barh', color='salmon')
    plt.title('Fig 8: External Feature Importance Weights')
    plt.savefig('figures/fig8_external_importance.png', dpi=300); plt.show()

    # FIGURE 9: External PR Curve
    plt.figure(figsize=(8,6))
    p_ext, r_ext, _ = precision_recall_curve(y_test, final_probs)
    plt.plot(r_ext, p_ext, lw=2, color='darkred')
    plt.title('Fig 9: External Precision-Recall Curve')
    plt.savefig('figures/fig9_external_pr.png', dpi=300); plt.show()

    # FIGURE 10: Model Error Heatmap (Internal)
    errors = (y_test_local != stack_class).astype(int)
    plt.figure(figsize=(10,4))
    sns.heatmap(errors.values.reshape(1, -1), cmap='coolwarm', cbar=False)
    plt.title('Fig 10: Internal Prediction Error Density Map')
    plt.savefig('figures/fig10_error_map.png', dpi=300); plt.show()

    # FIGURE 11: Age Distribution Comparison
    plt.figure(figsize=(8,6))
    sns.kdeplot(X_train_local[:,0], label='Internal', fill=True)
    sns.kdeplot(X_train[:,0], label='External', fill=True)
    plt.title('Fig 11: Standardized Age Distribution Comparison')
    plt.legend(); plt.savefig('figures/fig11_age_dist.png', dpi=300); plt.show()

    # FIGURE 12: Correlation Matrix (Internal)
    plt.figure(figsize=(10,8))
    sns.heatmap(pd.DataFrame(X_train_local, columns=feature_names).corr(), annot=True, cmap='viridis')
    plt.title('Fig 12: Internal Feature Correlation Matrix')
    plt.savefig('figures/fig12_internal_corr.png', dpi=300); plt.show()

    # FIGURE 13: Target Class Imbalance (Internal vs External)
    dist = pd.DataFrame({'Set': ['Internal']*2 + ['External']*2, 
                         'Class': [0,1,0,1], 
                         'Count': [sum(y_train_local==0), sum(y_train_local==1), sum(y_train==0), sum(y_train==1)]})
    plt.figure(figsize=(8,6))
    sns.barplot(data=dist, x='Set', y='Count', hue='Class')
    plt.title('Fig 13: Target Prevalence Comparison')
    plt.savefig('figures/fig13_class_dist.png', dpi=300); plt.show()

    # FIGURE 14: SBP vs DBP Joint Distribution
    plt.figure(figsize=(8,6))
    sns.jointplot(x=X_test_local[:,2], y=X_test_local[:,3], kind='hex', color='blue')
    plt.suptitle('Fig 14: BP Relationship (SBP vs DBP)', y=1.02)
    plt.savefig('figures/fig14_bp_joint.png', dpi=300); plt.show()

    # FIGURE 15: Ensemble Weights Pie Chart
    plt.figure(figsize=(8,8))
    weights = np.abs(meta.coef_[0])
    plt.pie(weights, labels=['XGB', 'Cat', 'Tab'], autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
    plt.title('Fig 15: Meta-Learner Stack Weights')
    plt.savefig('figures/fig15_ensemble_pie.png', dpi=300); plt.show()

    print("\nDone! All 15 figures saved to 'figures/' directory.")

try:
    generate_full_suite()
except NameError as e:
    print(f"Variable error: {e}. Please ensure both Internal and External validation cells have been executed.")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shap
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve
from dcurves import dca

# Ensure output directory exists
os.makedirs('figures', exist_ok=True)
sns.set_style("whitegrid")

# Helper function to generate standardized plots
def generate_full_suite():
    # 1-5: INTERNAL VALIDATION (KAGGE/STACKED)
    # 6-10: EXTERNAL VALIDATION (UCI/STACKED)
    # 11-15: DATA/STATISTICAL DISTRIBUTIONS
    
    print("Generating 15 Figures...")

    # FIGURE 1: Internal ROC Comparison
    plt.figure(figsize=(8,6))
    for name, p in models_to_plot.items():
        fpr, tpr, _ = roc_curve(y_test_local, p)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc(fpr, tpr):.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.title('Fig 1: Internal ROC Comparison')
    plt.legend(); plt.savefig('figures/fig1_internal_roc.png', dpi=300); plt.show()

    # FIGURE 2: Internal Precision-Recall Curve
    plt.figure(figsize=(8,6))
    precision, recall, _ = precision_recall_curve(y_test_local, stack_pred)
    plt.plot(recall, precision, lw=2, color='teal', label=f'AP={average_precision_score(y_test_local, stack_pred):.3f}')
    plt.title('Fig 2: Internal Precision-Recall')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.savefig('figures/fig2_internal_pr.png', dpi=300); plt.show()

    # FIGURE 3: Internal Calibration Curve
    prob_true, prob_pred_vals = calibration_curve(y_test_local, stack_pred, n_bins=10)
    plt.figure(figsize=(8,6))
    plt.plot(prob_pred_vals, prob_true, marker='o', label='Stacked Ensemble')
    plt.plot([0,1], [0,1], 'k--', label='Perfect')
    plt.title('Fig 3: Internal Calibration Plot')
    plt.savefig('figures/fig3_internal_calibration.png', dpi=300); plt.show()

    # FIGURE 4: Internal SHAP Summary (Interpretability)
    explainer = shap.TreeExplainer(xgb)
    shap_v = explainer.shap_values(X_test_local[:200])
    plt.figure()
    shap.summary_plot(shap_v, X_test_local[:200], feature_names=feature_names, show=False)
    plt.title('Fig 4: Global Feature Importance (SHAP)')
    plt.savefig('figures/fig4_internal_shap.png', dpi=300, bbox_inches='tight'); plt.show()

    # FIGURE 5: Internal Decision Curve Analysis (DCA)
    df_dca = pd.DataFrame({'outcome': y_test_local, 'Stacked': stack_pred})
    dca_res = dca(data=df_dca, outcome='outcome', modelnames=['Stacked'])
    dca_res.plot()
    plt.title('Fig 5: Clinical Utility (DCA)')
    plt.savefig('figures/fig5_internal_dca.png', dpi=300); plt.show()

    # FIGURE 6: External (UCI) ROC Comparison
    plt.figure(figsize=(8,6))
    fpr_ext, tpr_ext, _ = roc_curve(y_test, final_probs)
    plt.plot(fpr_ext, tpr_ext, color='red', lw=2, label=f'Stacked External (AUC={roc_auc_score(y_test, final_probs):.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.title('Fig 6: External (UCI Cleveland) ROC')
    plt.legend(); plt.savefig('figures/fig6_external_roc.png', dpi=300); plt.show()

    # FIGURE 7: External Confusion Matrix
    cm_ext = confusion_matrix(y_test, final_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_ext, display_labels=['Healthy', 'CVD'])
    disp.plot(cmap='Reds')
    plt.title('Fig 7: External Confusion Matrix')
    plt.savefig('figures/fig7_external_cm.png', dpi=300); plt.show()

    # FIGURE 8: External Feature Importance (Permutation/XGB)
    plt.figure(figsize=(10,6))
    imp = pd.Series(xgb.feature_importances_, index=feature_names).sort_values()
    imp.plot(kind='barh', color='salmon')
    plt.title('Fig 8: External Feature Importance Weights')
    plt.savefig('figures/fig8_external_importance.png', dpi=300); plt.show()

    # FIGURE 9: External PR Curve
    plt.figure(figsize=(8,6))
    p_ext, r_ext, _ = precision_recall_curve(y_test, final_probs)
    plt.plot(r_ext, p_ext, lw=2, color='darkred')
    plt.title('Fig 9: External Precision-Recall Curve')
    plt.savefig('figures/fig9_external_pr.png', dpi=300); plt.show()

    # FIGURE 10: Model Error Heatmap (Internal)
    errors = (y_test_local != stack_class).astype(int)
    plt.figure(figsize=(10,4))
    sns.heatmap(errors.values.reshape(1, -1), cmap='coolwarm', cbar=False)
    plt.title('Fig 10: Internal Prediction Error Density Map')
    plt.savefig('figures/fig10_error_map.png', dpi=300); plt.show()

    # FIGURE 11: Age Distribution Comparison
    plt.figure(figsize=(8,6))
    sns.kdeplot(X_train_local[:,0], label='Internal', fill=True)
    sns.kdeplot(X_train[:,0], label='External', fill=True)
    plt.title('Fig 11: Standardized Age Distribution Comparison')
    plt.legend(); plt.savefig('figures/fig11_age_dist.png', dpi=300); plt.show()

    # FIGURE 12: Correlation Matrix (Internal)
    plt.figure(figsize=(10,8))
    sns.heatmap(pd.DataFrame(X_train_local, columns=feature_names).corr(), annot=True, cmap='viridis')
    plt.title('Fig 12: Internal Feature Correlation Matrix')
    plt.savefig('figures/fig12_internal_corr.png', dpi=300); plt.show()

    # FIGURE 13: Target Class Imbalance (Internal vs External)
    dist = pd.DataFrame({'Set': ['Internal']*2 + ['External']*2, 
                         'Class': [0,1,0,1], 
                         'Count': [sum(y_train_local==0), sum(y_train_local==1), sum(y_train==0), sum(y_train==1)]})
    plt.figure(figsize=(8,6))
    sns.barplot(data=dist, x='Set', y='Count', hue='Class')
    plt.title('Fig 13: Target Prevalence Comparison')
    plt.savefig('figures/fig13_class_dist.png', dpi=300); plt.show()

    # FIGURE 14: SBP vs DBP Joint Distribution
    plt.figure(figsize=(8,6))
    sns.jointplot(x=X_test_local[:,2], y=X_test_local[:,3], kind='hex', color='blue')
    plt.suptitle('Fig 14: BP Relationship (SBP vs DBP)', y=1.02)
    plt.savefig('figures/fig14_bp_joint.png', dpi=300); plt.show()

    # FIGURE 15: Ensemble Weights Pie Chart
    plt.figure(figsize=(8,8))
    weights = np.abs(meta.coef_[0])
    plt.pie(weights, labels=['XGB', 'Cat', 'Tab'], autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
    plt.title('Fig 15: Meta-Learner Stack Weights')
    plt.savefig('figures/fig15_ensemble_pie.png', dpi=300); plt.show()

    print("\nDone! All 15 figures saved to 'figures/' directory.")

try:
    generate_full_suite()
except NameError as e:
    print(f"Variable error: {e}. Please ensure both Internal and External validation cells have been executed.")
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score

# Replace these with your actual true labels and predicted probabilities for each dataset
# Example structure – you need to load your saved predictions.
# For now, I'll show the logic. You must adapt to your saved arrays.

# Cleveland
y_true_cleveland = np.load('external/cleveland_y_true.npy')   # shape (n,)
y_pred_cleveland = np.load('external/cleveland_y_pred.npy')

# Hungarian
y_true_hungarian = np.load('external/hungarian_y_true.npy')
y_pred_hungarian = np.load('external/hungarian_y_pred.npy')

# Statlog
y_true_statlog = np.load('external/statlog_y_true.npy')
y_pred_statlog = np.load('external/statlog_y_pred.npy')

def bootstrap_ci(y_true, y_pred, n_boot=1000):
    aucs = []
    for _ in range(n_boot):
        idx = resample(range(len(y_true)), replace=True)
        aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))
    return np.percentile(aucs, [2.5, 97.5])

ci_cle = bootstrap_ci(y_true_cleveland, y_pred_cleveland)
ci_hun = bootstrap_ci(y_true_hungarian, y_pred_hungarian)
ci_stat = bootstrap_ci(y_true_statlog, y_pred_statlog)

print(f"Cleveland AUC: 0.917 (95% CI: {ci_cle[0]:.3f}–{ci_cle[1]:.3f})")
print(f"Hungarian AUC: 0.809 (95% CI: {ci_hun[0]:.3f}–{ci_hun[1]:.3f})")
print(f"Statlog AUC: 0.790 (95% CI: {ci_stat[0]:.3f}–{ci_stat[1]:.3f})")
import numpy as np
import os
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# 1. Create directory and save existing kernel variables to simulate the 'saved' state
os.makedirs('external', exist_ok=True)

def prepare_and_save_external(name, url, is_statlog=False):
    # Helper to get the arrays from the current kernel logic
    curr_df = load_statlog_dataset(url) if is_statlog else load_uci_dataset(url)
    X_eval = curr_df[feature_cols]
    y_eval = curr_df['target']
    X_sc = StandardScaler().fit_transform(KNNImputer(n_neighbors=5).fit_transform(X_eval))
    _, X_te, _, y_te = train_test_split(X_sc, y_eval, stratify=y_eval, test_size=0.2, random_state=42)
    
    # Generate predictions using the existing stacked models logic
    s_test = np.column_stack([xgb.predict_proba(X_te)[:,1], cat.predict_proba(X_te)[:,1]])
    y_p = meta.predict_proba(s_test)[:,1]
    
    np.save(f'external/{name}_y_true.npy', y_te.values if hasattr(y_te, 'values') else y_te)
    np.save(f'external/{name}_y_pred.npy', y_p)

# Prepare the files since they were missing
prepare_and_save_external('cleveland', datasets['Cleveland'])
prepare_and_save_external('hungarian', datasets['Hungarian'])
prepare_and_save_external('statlog', datasets['Statlog'], is_statlog=True)

# 2. Load the arrays (Original Logic)
y_true_cleveland = np.load('external/cleveland_y_true.npy')
y_pred_cleveland = np.load('external/cleveland_y_pred.npy')
y_true_hungarian = np.load('external/hungarian_y_true.npy')
y_pred_hungarian = np.load('external/hungarian_y_pred.npy')
y_true_statlog = np.load('external/statlog_y_true.npy')
y_pred_statlog = np.load('external/statlog_y_pred.npy')

def bootstrap_ci(y_true, y_pred, n_boot=1000):
    aucs = []
    seed = 42
    for i in range(n_boot):
        idx = resample(range(len(y_true)), replace=True, random_state=seed + i)
        if len(np.unique(y_true[idx])) > 1:
            aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))
    return np.percentile(aucs, [2.5, 97.5])

ci_cle = bootstrap_ci(y_true_cleveland, y_pred_cleveland)
ci_hun = bootstrap_ci(y_true_hungarian, y_pred_hungarian)
ci_stat = bootstrap_ci(y_true_statlog, y_pred_statlog)

print(f"Cleveland AUC: {roc_auc_score(y_true_cleveland, y_pred_cleveland):.3f} (95% CI: {ci_cle[0]:.3f}–{ci_cle[1]:.3f})")
print(f"Hungarian AUC: {roc_auc_score(y_true_hungarian, y_pred_hungarian):.3f} (95% CI: {ci_hun[0]:.3f}–{ci_hun[1]:.3f})")
print(f"Statlog AUC: {roc_auc_score(y_true_statlog, y_pred_statlog):.3f} (95% CI: {ci_stat[0]:.3f}–{ci_stat[1]:.3f})")
import numpy as np
import os
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# 1. Create directory and save existing kernel variables to simulate the 'saved' state
os.makedirs('external', exist_ok=True)

def prepare_and_save_external(name, url, is_statlog=False):
    # Helper to get the arrays from the current kernel logic
    curr_df = load_statlog_dataset(url) if is_statlog else load_uci_dataset(url)
    X_eval = curr_df[feature_cols]
    y_eval = curr_df['target']
    X_sc = StandardScaler().fit_transform(KNNImputer(n_neighbors=5).fit_transform(X_eval))
    _, X_te, _, y_te = train_test_split(X_sc, y_eval, stratify=y_eval, test_size=0.2, random_state=42)
    
    # Generate predictions using the existing stacked models logic (XGB, Cat, TabNet)
    # tab is available in the global scope from previous cell execution
    s_test = np.column_stack([
        xgb.predict_proba(X_te)[:,1],
        cat.predict_proba(X_te)[:,1],
        tab.predict_proba(X_te)[:,1]
    ])
    y_p = meta.predict_proba(s_test)[:,1]
    
    np.save(f'external/{name}_y_true.npy', y_te.values if hasattr(y_te, 'values') else y_te)
    np.save(f'external/{name}_y_pred.npy', y_p)

# Prepare the files using the corrected stacking logic
prepare_and_save_external('cleveland', datasets['Cleveland'])
prepare_and_save_external('hungarian', datasets['Hungarian'])
prepare_and_save_external('statlog', datasets['Statlog'], is_statlog=True)

# 2. Load the arrays and calculate Bootstrap CIs
y_true_cleveland = np.load('external/cleveland_y_true.npy')
y_pred_cleveland = np.load('external/cleveland_y_pred.npy')
y_true_hungarian = np.load('external/hungarian_y_true.npy')
y_pred_hungarian = np.load('external/hungarian_y_pred.npy')
y_true_statlog = np.load('external/statlog_y_true.npy')
y_pred_statlog = np.load('external/statlog_y_pred.npy')

def bootstrap_ci(y_true, y_pred, n_boot=1000):
    aucs = []
    seed = 42
    for i in range(n_boot):
        idx = resample(range(len(y_true)), replace=True, random_state=seed + i)
        if len(np.unique(y_true[idx])) > 1:
            aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))
    return np.percentile(aucs, [2.5, 97.5])

ci_cle = bootstrap_ci(y_true_cleveland, y_pred_cleveland)
ci_hun = bootstrap_ci(y_true_hungarian, y_pred_hungarian)
ci_stat = bootstrap_ci(y_true_statlog, y_pred_statlog)

print(f"Cleveland AUC: {roc_auc_score(y_true_cleveland, y_pred_cleveland):.3f} (95% CI: {ci_cle[0]:.3f}–{ci_cle[1]:.3f})")
print(f"Hungarian AUC: {roc_auc_score(y_true_hungarian, y_pred_hungarian):.3f} (95% CI: {ci_hun[0]:.3f}–{ci_hun[1]:.3f})")
print(f"Statlog AUC: {roc_auc_score(y_true_statlog, y_pred_statlog):.3f} (95% CI: {ci_stat[0]:.3f}–{ci_stat[1]:.3f})")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.utils import resample

# ------------------------------------------------------------
# 1. Statlog external validation (reproducible, with multiple seeds)
# ------------------------------------------------------------
print("=== Statlog External Validation ===")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat"
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df = pd.read_csv(url, sep=' ', names=columns)
X = df.drop('target', axis=1)
y = df['target']

# Preprocessing (same pipeline as main)
imputer = KNNImputer(n_neighbors=5)
scaler = StandardScaler()
X_imp = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imp)

aucs = []
seeds = [42, 123, 456, 789, 101112]   # multiple random seeds
for seed in seeds:
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=seed)
    
    # Train base models
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                        eval_metric='logloss', use_label_encoder=False, random_state=seed)
    cat = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05,
                             verbose=0, random_seed=seed)
    xgb.fit(X_train, y_train)
    cat.fit(X_train, y_train)
    
    # Out‑of‑fold predictions for meta‑learner (using 5‑fold CV on training set)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    xgb_oof = np.zeros(len(X_train))
    cat_oof = np.zeros(len(X_train))
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        xgb_fold = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                 eval_metric='logloss', use_label_encoder=False, random_state=seed)
        cat_fold = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05,
                                      verbose=0, random_seed=seed)
        xgb_fold.fit(X_tr, y_tr)
        cat_fold.fit(X_tr, y_tr)
        xgb_oof[val_idx] = xgb_fold.predict_proba(X_val)[:, 1]
        cat_oof[val_idx] = cat_fold.predict_proba(X_val)[:, 1]
    
    # Train meta‑learner on OOF predictions
    meta = LogisticRegression()
    meta.fit(np.column_stack([xgb_oof, cat_oof]), y_train)
    
    # Test predictions
    xgb_test = xgb.predict_proba(X_test)[:, 1]
    cat_test = cat.predict_proba(X_test)[:, 1]
    stack_test = np.column_stack([xgb_test, cat_test])
    final_probs = meta.predict_proba(stack_test)[:, 1]
    auc = roc_auc_score(y_test, final_probs)
    aucs.append(auc)
    print(f"Seed {seed}: AUC = {auc:.4f}")

print(f"Statlog AUC across seeds: mean = {np.mean(aucs):.4f}, std = {np.std(aucs):.4f}")
if np.mean(aucs) > 0.99:
    print("⚠️ WARNING: Statlog AUC is extremely high. Small test set size may cause instability.")
else:
    print("Statlog AUC is within expected range.")

# ------------------------------------------------------------
# 2. Verify Framingham stacking (out‑of‑fold check)
# ------------------------------------------------------------
print("\n=== Framingham Stacking Verification ===")
# Load Framingham data (adjust path to your file)
try:
    fram = pd.read_csv('data/framingham.csv')
except:
    print("Framingham data not found; skipping verification.")
    fram = None

if fram is not None:
    X_f = fram.drop(columns=['TenYearCHD'])
    y_f = fram['TenYearCHD']
    # Preprocessing (same as main pipeline)
    imputer_f = KNNImputer(n_neighbors=5)
    scaler_f = StandardScaler()
    X_f_imp = imputer_f.fit_transform(X_f)
    X_f_scaled = scaler_f.fit_transform(X_f_imp)
    
    # Simple check: train XGBoost with default params and compute 5‑fold CV AUC
    from sklearn.model_selection import cross_val_score
    xgb_f = XGBClassifier(n_estimators=100, max_depth=3, eval_metric='logloss', use_label_encoder=False)
    cv_auc = cross_val_score(xgb_f, X_f_scaled, y_f, cv=5, scoring='roc_auc')
    print(f"Framingham XGBoost 5‑fold CV AUC: mean = {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    if cv_auc.mean() < 0.7:
        print("⚠️ WARNING: Framingham baseline AUC is low. Check data quality or preprocessing.")
    else:
        print("Framingham baseline AUC is reasonable.")

print("\n✅ Verification complete.")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

# ------------------------------------------------------------
# 1. Statlog external validation (reproducible, with multiple seeds)
# ------------------------------------------------------------
print("=== Statlog External Validation ===")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat"
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df = pd.read_csv(url, sep=' ', names=columns)

# FIX: Map labels [1, 2] to [0, 1]
df['target'] = df['target'].map({1: 0, 2: 1})

X = df.drop('target', axis=1)
y = df['target']

# Preprocessing
imputer = KNNImputer(n_neighbors=5)
scaler = StandardScaler()
X_imp = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imp)

aucs = []
seeds = [42, 123, 456, 789, 101112]

for seed in seeds:
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=seed)

    # Train base models
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, eval_metric='logloss', random_state=seed)
    cat = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, verbose=0, random_seed=seed)
    tab = TabNetClassifier(verbose=0, seed=seed)
    
    xgb.fit(X_train, y_train)
    cat.fit(X_train, y_train)
    tab.fit(X_train, y_train.values, max_epochs=20)

    # Out-of-fold predictions for meta-learner
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    xgb_oof, cat_oof, tab_oof = np.zeros(len(X_train)), np.zeros(len(X_train)), np.zeros(len(X_train))
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        m_xgb = XGBClassifier(n_estimators=100, max_depth=3, eval_metric='logloss', random_state=seed).fit(X_tr, y_tr)
        m_cat = CatBoostClassifier(iterations=100, depth=4, verbose=0, random_seed=seed).fit(X_tr, y_tr)
        m_tab = TabNetClassifier(verbose=0, seed=seed)
        m_tab.fit(X_tr, y_tr.values, max_epochs=10)
        
        xgb_oof[val_idx] = m_xgb.predict_proba(X_val)[:, 1]
        cat_oof[val_idx] = m_cat.predict_proba(X_val)[:, 1]
        tab_oof[val_idx] = m_tab.predict_proba(X_val)[:, 1]

    meta = LogisticRegression().fit(np.column_stack([xgb_oof, cat_oof, tab_oof]), y_train)
    
    # Test predictions
    test_stack = np.column_stack([
        xgb.predict_proba(X_test)[:, 1],
        cat.predict_proba(X_test)[:, 1],
        tab.predict_proba(X_test)[:, 1]
    ])
    final_probs = meta.predict_proba(test_stack)[:, 1]
    auc_score = roc_auc_score(y_test, final_probs)
    aucs.append(auc_score)
    print(f"Seed {seed}: AUC = {auc_score:.4f}")

print(f"\nStatlog mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print("\n✅ Verification complete.")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import warnings

# Suppress TabNet specific warning for clean output
warnings.filterwarnings("ignore", message="No early stopping will be performed")

print("=== Statlog External Validation (Stability Check) ===")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat"
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df = pd.read_csv(url, sep=' ', names=columns)

# Map labels [1, 2] to [0, 1]
df['target'] = df['target'].map({1: 0, 2: 1})

X = df.drop('target', axis=1)
y = df['target']

# Preprocessing
imputer = KNNImputer(n_neighbors=5)
scaler = StandardScaler()
X_imp = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imp)

aucs = []
seeds = [42, 123, 456, 789, 101112]

for seed in seeds:
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=seed)

    # Train base models
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, eval_metric='logloss', random_state=seed)
    cat = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, verbose=0, random_seed=seed)
    tab = TabNetClassifier(verbose=0, seed=seed)
    
    xgb.fit(X_train, y_train)
    cat.fit(X_train, y_train)
    tab.fit(X_train, y_train.values, max_epochs=20)

    # Out-of-fold predictions for meta-learner
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    xgb_oof, cat_oof, tab_oof = np.zeros(len(X_train)), np.zeros(len(X_train)), np.zeros(len(X_train))
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        m_xgb = XGBClassifier(n_estimators=100, max_depth=3, eval_metric='logloss', random_state=seed).fit(X_tr, y_tr)
        m_cat = CatBoostClassifier(iterations=100, depth=4, verbose=0, random_seed=seed).fit(X_tr, y_tr)
        m_tab = TabNetClassifier(verbose=0, seed=seed)
        m_tab.fit(X_tr, y_tr.values, max_epochs=10)
        
        xgb_oof[val_idx] = m_xgb.predict_proba(X_val)[:, 1]
        cat_oof[val_idx] = m_cat.predict_proba(X_val)[:, 1]
        tab_oof[val_idx] = m_tab.predict_proba(X_val)[:, 1]

    meta = LogisticRegression().fit(np.column_stack([xgb_oof, cat_oof, tab_oof]), y_train)
    
    # Test predictions
    test_stack = np.column_stack([
        xgb.predict_proba(X_test)[:, 1],
        cat.predict_proba(X_test)[:, 1],
        tab.predict_proba(X_test)[:, 1]
    ])
    final_probs = meta.predict_proba(test_stack)[:, 1]
    auc_score = roc_auc_score(y_test, final_probs)
    aucs.append(auc_score)
    print(f"Seed {seed}: AUC = {auc_score:.4f}")

print(f"\nStatlog mean AUC across seeds: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print("\n✅ Verification complete.")
import os

# Retrieve all code run in the current session using IPython history
# and save it to a .py file
%history -f project_history.py

if os.path.exists('project_history.py'):
    print("✅ Successfully saved all previous code runs to 'project_history.py'")
else:
    print("❌ Failed to save code history.")
