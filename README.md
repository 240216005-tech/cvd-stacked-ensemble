# Stacked Ensemble for Cardiovascular Disease Prediction

[![ORCID](https://img.shields.io/badge/ORCID-0009--0009--1650--5787-green)](https://orcid.org/0009-0009-1650-5787)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19606370.svg)](https://doi.org/10.5281/zenodo.19606370)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A leakage-free stacked ensemble framework integrating **TabNet**, **XGBoost**, and **CatBoost** with a **Logistic Regression** meta-learner for cardiovascular disease (CVD) risk prediction — validated across five independent cohorts spanning four countries.

> **Paper:** *A Systematic Evaluation of a Stacked Ensemble Framework Integrating TabNet and Gradient Boosting for Cardiovascular Disease Prediction: Benchmarking, Interpretability, and Clinical Implementation*
>
> **Authors:** Akhil Tripathi · Dr. Imran Khan
>
> **Affiliation:** Department of Computer Science, Harcourt Butler Technical University, Kanpur, India

---

## Key Results

| Cohort      | N       | AUC           |
|-------------|---------|---------------|
| Kaggle CVD  | 68,783  | 0.910         |
| Framingham  | 4,238   | 0.731         |
| Cleveland   | 297     | 0.873         |
| Hungarian   | 261     | 0.856         |
| Statlog     | 270     | 0.889         |

**Negative stacking result:** Ablation analysis shows the stacked ensemble does not significantly outperform a single well-tuned XGBoost (DeLong's p = 0.660).

---

## Repository Structure

```
├── src/
│   ├── train.py                  # Main training pipeline
│   ├── evaluate.py               # ROC, calibration, SHAP, DCA plots
│   ├── external_validation.py    # UCI cohort validation
│   ├── optimise.py               # Bayesian hyperparameter tuning (Optuna)
│   └── delong.py                 # DeLong's test for AUC comparison
├── data/                         # Place your CSV datasets here
├── results/                      # Saved model outputs (.npy)
├── figures/                      # Generated plots
├── Dockerfile                    # Reproducible container environment
├── TRIPOD_AI_CHECKLIST.md        # TRIPOD+AI reporting compliance
├── CITATION.cff                  # GitHub "Cite this repository" metadata
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Bayesian hyperparameter optimisation (Optuna)

```bash
python src/optimise.py --data_path data/kaggle_cvd.csv --n_trials 50
```

### 3. Train on a dataset

```bash
python src/train.py --dataset kaggle --data_path data/kaggle_cvd.csv
```

### 4. Generate evaluation plots

```bash
python src/evaluate.py --dataset kaggle
```

### 5. Run DeLong's test (ablation study)

```bash
python src/delong.py --dataset kaggle
```

### 6. Run external validation (Cleveland, Hungarian, Statlog)

```bash
python src/external_validation.py
```

### 7. Docker (full reproducibility)

```bash
docker build -t cvd-ensemble .
docker run --rm cvd-ensemble
docker run --rm cvd-ensemble python src/train.py --data_path data/kaggle_cvd.csv
```

---

## Methodology

1. **Preprocessing:** Winsorization → KNN imputation (k=5) → feature engineering → StandardScaler — all fitted on training folds only (no leakage).
2. **Base Learners:** XGBoost, CatBoost, and TabNet trained with Bayesian hyperparameter optimisation (Optuna, 30–50 trials).
3. **Stacking:** Out-of-fold predictions from 5-fold stratified CV fed into a Logistic Regression meta-learner.
4. **Interpretability:** SHAP values for feature attribution; decision curve analysis vs. Framingham Risk Score and ASCVD.
5. **Reporting:** Follows TRIPOD+AI guidelines.

---

## Interpretability

SHAP analysis identifies clinically actionable thresholds:
- Age > 60 years
- Systolic BP > 140 mmHg
- Fasting glucose > 100 mg/dL

---

## Citation

If you use this code, please cite:

```bibtex
@article{tripathi2026stacked,
  title   = {A Systematic Evaluation of a Stacked Ensemble Framework
             Integrating TabNet and Gradient Boosting for Cardiovascular
             Disease Prediction: Benchmarking, Interpretability, and
             Clinical Implementation},
  author  = {Tripathi, Akhil and Khan, Imran},
  journal = {Big Data Mining and Analytics},
  year    = {2026},
  doi     = {10.5281/zenodo.19606370}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

- **Akhil Tripathi** — 240216005@hbtu.ac.in | [ORCID](https://orcid.org/0009-0009-1650-5787)
- **Dr. Imran Khan** — imran.k@hbtu.ac.in
