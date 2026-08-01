# Explainable Credit Card Fraud Detection System using Machine Learning and SHAP

Automated, explainable fraud detection on credit card transactions, combining Logistic
Regression / Random Forest / XGBoost with SHAP-based interpretability and a Streamlit demo.

## Project Overview
- Goal: Build an end-to-end explainable fraud detection system that accurately identifies
  fraudulent transactions and explains *why* each prediction was made
- Dataset: Credit Card Fraud Detection dataset (ULB Machine Learning Group / Kaggle,
  `mlg-ulb/creditcardfraud`)
- Method: Logistic Regression (baseline), Random Forest, and XGBoost, with SMOTE +
  class-weighting for imbalance and SHAP for explainability
- Deliverable: Trained models + Streamlit web app with real-time SHAP explanations
- Timeline: 27 April 2026 – 18 July 2026 (Major Project 21CSA399A)

## Dataset Information
- Total transactions: 284,807
- Classes:
  - Legitimate (Class 0): 284,315 (99.828%)
  - Fraudulent (Class 1): 492 (0.172%)
- Features: `Time`, `V1`-`V28` (PCA-anonymized components), `Amount`, `Class`
- Split: 70% train / 15% validation / 15% test, stratified to preserve class ratio

> **Dataset note:** `data/creditcard.csv` is the real Kaggle file
> ([mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)), and
> all results below are the output of a full, real run of notebooks 01–05 with
> scikit-learn, XGBoost, imbalanced-learn, and SHAP — not a placeholder or reference
> implementation.

## Project Status — Complete

### Module 1: Data Preprocessing
- EDA: class distribution, amount distribution by class, time-of-day patterns, correlation
  heatmap (`notebooks/01`)
- Scaling (`StandardScaler` on `Time`/`Amount`), stratified 70/15/15 split, SMOTE on the
  training set only (`notebooks/02`, `src/preprocessing.py`)

### Module 2: Model Building & Comparison
- Logistic Regression baseline (`notebooks/03`)
- Random Forest + XGBoost, full 3-way comparison (`notebooks/04`)

### Module 3: SHAP Explainability
- `TreeExplainer` global summary + local waterfall + dependence plot (`notebooks/05`)

### Module 4: Streamlit Deployment
- Real-time prediction + live SHAP explanation (`app.py`)

## Results

These are the real numbers from the official pipeline (`scikit-learn` / `XGBoost` /
`imbalanced-learn` / `SHAP`) run on the real Kaggle dataset — notebooks 01–05, no
placeholder data or reference implementation involved.

**F1-optimal (balanced) operating point — test set, n=42,722:**

| Model | Threshold | Accuracy | Precision | Recall | F1-Score | ROC-AUC | AUPRC |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.99 | 99.84% | 53.0% | 83.8% | 0.649 | 0.9613 | 0.6866 |
| Random Forest | 0.76 | 99.95% | 93.3% | 75.7% | **0.836** | 0.9683 | **0.8375** |
| XGBoost | 0.92 | 99.94% | 84.3% | 79.7% | 0.819 | 0.9724 | 0.8249 |

AUPRC is the metric the Kaggle dataset card recommends given the 0.172% fraud rate, since
ROC-AUC stays high (0.96–0.97) for all three models even as precision varies a lot — the PR
curve (`results/pr_curves_comparison.png`) is the more discriminating comparison. On both
F1 and AUPRC, **Random Forest is the strongest model in this run**, narrowly ahead of
XGBoost.

> **Note on the deployed model:** `notebooks/04` hard-codes `XGBoost` as `FINAL_MODEL_NAME`,
> and `app.py` / `notebooks/05` load `xgboost_model.pkl` accordingly. Random Forest scores
> marginally higher on F1 (0.836 vs. 0.819) and AUPRC (0.8375 vs. 0.8249), but XGBoost is
> kept as the deployed model because it has the higher recall (79.7% vs. 75.7%) at a still-strong
> 84.3% precision — and the Interim Report's stated selection rule (Section 4.3) is "highest
> recall with acceptable precision," since missing a fraud case is costlier than an extra
> false alarm. See the Final Report (Section 9.8) for the fuller discussion.

**Logistic Regression — recall-priority ("screening") operating point, recall ≥ 90%**
(`notebooks/03` only; the ensembles weren't run at this operating point):

| Model | Threshold | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.15 | 1.3% | 90.5% | 0.025 |

This illustrates the real precision/recall trade-off on this dataset: pushing Logistic
Regression to catch 90%+ of fraud drives precision down to ~1%, i.e. almost all flagged
transactions would be false alarms at that threshold. Random Forest and XGBoost's
F1-optimal operating points above are a much better balance for this dataset.

Full numbers: `results/model_comparison_f1_optimal.csv`. Plots: `results/roc_curves_comparison.png`,
`results/pr_curves_comparison.png`, `results/pr_curve_logistic_regression.png`,
`results/confusion_matrices_all_models.png`.

## Explainability Highlights

Using `shap.TreeExplainer` (exact Shapley values) on the deployed XGBoost model over a
2,000-transaction test sample, the most fraud-predictive features are **V14, V4, V12, V8,
and V18** — see `results/shap_global_importance_bar.png` (full ranking),
`results/shap_summary_plot.png` (beeswarm), `results/shap_waterfall_example.png` (a single
fraud prediction explained), and `results/shap_dependence_V14.png` (dependence plot for the
top feature, V14).

## Repository Structure
```
credit-card-fraud-detection-project/
├── data/              # creditcard.csv (real Kaggle mlg-ulb/creditcardfraud file)
├── notebooks/         # 01 EDA, 02 preprocessing/SMOTE, 03 Logistic Regression,
│                      # 04 Random Forest + XGBoost + comparison, 05 SHAP
├── src/               # reusable preprocessing.py
├── models/            # trained models (see models/README.md)
├── results/           # plots, confusion matrices, ROC/PR curves, SHAP visuals, metrics CSVs
├── app.py             # Streamlit app (official pipeline, with reference-mode fallback)
├── tree_lib.py         # NumPy reference tree/ensemble implementation (fallback only)
├── models_lib.py       # NumPy reference Logistic Regression (fallback only)
└── requirements.txt
```

## Technologies Used
- Core ML: Python, Scikit-learn, XGBoost, Imbalanced-learn (SMOTE)
- Explainability: SHAP
- Data Science: NumPy, Pandas
- Visualization: Matplotlib, Seaborn
- Deployment: Streamlit
- Development: Jupyter Notebook (Colab-ready)

## Running This Project
```bash
pip install -r requirements.txt

# 1. Run notebooks/01 → 05 in order (Jupyter or Google Colab) — data/creditcard.csv is
#    already the real Kaggle file, so no dataset swap needed
# 2. Launch the app:
streamlit run app.py
```

## References
1. Bhattacharyya, S., Jha, S., Tharakunnel, K., & Westland, J. C. (2011). Data mining for
   credit card fraud: A comparative study. *Decision Support Systems*, 50(3), 602-613.
2. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic
   minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321-357.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*, 785-794.
4. Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating
   probability with undersampling for unbalanced classification. *IEEE SSCI 2015*, 159-166.
5. Fernandez, A., Garcia, S., Galar, M., Prati, R. C., Krawczyk, B., & Herrera, F. (2018).
   *Learning from Imbalanced Data Sets.* Springer.
6. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model
   predictions. *NeurIPS 2017*, 30.
7. ULB Machine Learning Group. (2018). Credit Card Fraud Detection Dataset. Kaggle.
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Author
Alisha Quaser (AA.SC.U3BCA2307101)
Amrita Vishwa Vidyapeetham — Major Project 21CSA399A

