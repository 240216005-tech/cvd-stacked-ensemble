# TRIPOD+AI Reporting Checklist

**Paper:** A Systematic Evaluation of a Stacked Ensemble Framework Integrating TabNet and Gradient Boosting for Cardiovascular Disease Prediction

**Authors:** Akhil Tripathi, Dr. Imran Khan

**Reference:** Collins GS, et al. TRIPOD+AI statement: updated reporting guidelines for clinical prediction models. BMJ. 2024;385:e078378.

---

| # | Item | Section | Reported |
|---|------|---------|----------|
| **Title & Abstract** | | | |
| 1 | Identify as development/validation of a prediction model using AI | Title | Yes |
| 2 | Structured abstract with key elements | Abstract | Yes |
| **Introduction** | | | |
| 3a | Medical context and rationale | Sec I | Yes |
| 3b | Objectives including whether development, validation, or both | Sec I | Yes |
| **Methods** | | | |
| 4a | Source of data (cohort, registry, etc.) | Sec III | Yes |
| 4b | Dates of data collection/recruitment | Sec III | Yes |
| 5a | Key eligibility criteria | Sec III | Yes |
| 5b | Treatment received, if relevant | — | N/A |
| 6a | Outcome definition | Sec III | Yes |
| 6b | Blinding of outcome assessment | — | N/A |
| 7a | Candidate predictors and rationale | Sec III-A | Yes |
| 7b | Predictor assessment blinding | — | N/A |
| 8 | Sample size justification | Sec III | Yes |
| 9 | Missing data handling (KNN imputation, k=5) | Sec III-B | Yes |
| 10a | Statistical analysis and modelling | Sec III-C,D | Yes |
| 10b | Model type and learning algorithm | Sec III-C | Yes |
| 10c | Feature selection method | Sec III-A | Yes |
| 10d | Hyperparameter selection (Optuna, 30–50 trials) | Sec III-C | Yes |
| 10e | Model evaluation measures | Sec III-E | Yes |
| **AI-specific items** | | | |
| AI-1 | AI method justification (stacking rationale) | Sec III-C | Yes |
| AI-2 | Data preprocessing steps (winsorisation, scaling) | Sec III-B | Yes |
| AI-3 | Data splitting strategy (stratified 80/20, 5-fold CV) | Sec III-D | Yes |
| AI-4 | Software and versions (Python, XGBoost, CatBoost, TabNet) | Sec III | Yes |
| AI-5 | Reproducibility (code, Docker, Zenodo DOI) | Data Avail. | Yes |
| AI-6 | Leakage prevention (fit on train folds only) | Sec III-B | Yes |
| **Results** | | | |
| 11 | Participant flow diagram | Fig 1 | Yes |
| 12 | Participant characteristics | Table I | Yes |
| 13a | Discrimination (AUC with 95% CI) | Sec IV-A | Yes |
| 13b | Calibration (calibration curves, Brier score) | Sec IV-B | Yes |
| 14 | Model performance across subgroups | Sec IV | Yes |
| 15 | Full model specification for reproducibility | GitHub repo | Yes |
| 16 | Clinical utility (decision curve analysis) | Sec IV-D | Yes |
| **Discussion** | | | |
| 17 | Interpretation with reference to objectives | Sec V | Yes |
| 18 | Limitations (negative stacking result, cohort sizes) | Sec V | Yes |
| 19 | Implications for clinical use | Sec V | Yes |
| **Other** | | | |
| 20 | Source of funding | — | Yes |
| 21 | Competing interests declaration | — | Yes |
| 22 | Data and code availability (Zenodo DOI, GitHub) | Data Avail. | Yes |

---

## Notes

- **Negative result transparency (Item 18):** Ablation analysis explicitly reports that stacking does not significantly outperform XGBoost alone (DeLong's p = 0.660).
- **External validation (Items 13–14):** Five independent cohorts across four countries (India/Kaggle, USA/Framingham, USA/Cleveland, Hungary/Hungarian, UK/Statlog).
- **Reproducibility (AI-5):** Full code, trained models, and Docker container available at DOI: 10.5281/zenodo.19606370.
