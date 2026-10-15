# AI Engineering Capstone: ML + Gen AI

Two projects in one Flask web app. Each has a trained ML model and a Gemini-powered Gen AI layer (plain-language explanation of each prediction, plus a chatbot and a dashboard).

This submission is one combined application with a Banking page and a Healthcare page; all deliverables are shared between the two projects.

**Write-up:** see [WRITEUP.md](WRITEUP.md) for the answers to the five capstone questions for both projects.

| | Project 1: Banking | Project 2: Healthcare |
|---|---|---|
| Task | Customer churn prediction | Diabetes risk flag |
| Dataset | Iranian Churn Dataset (UCI) | CDC Diabetes Health Indicators (BRFSS 2015) |
| Model shipped | XGBoost | XGBoost |
| Gen AI | Explains the churn risk and top factors for a banker | Explains the result in a nurse's voice, never diagnoses or advises treatment |

## Datasets and licences

- **Project 1:** [Iranian Churn Dataset](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset), UCI. File: `Customer Churn.csv`. Licence: **CC BY 4.0**. About 15.7% churn, 3,150 rows.
- **Project 2:** [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset), Kaggle, from the CDC BRFSS 2015 survey. Licence: **CC0** on Kaggle (CC BY 4.0 on the [UCI copy](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)). About 14% positive, 253,680 rows.
  I used the original imbalanced file `diabetes_012_health_indicators_BRFSS2015.csv` (not the 50/50 file) and converted the 3-class target to binary (`Diabetes_012 == 2` is positive). Prediabetes is therefore merged into the "no diabetes" class.

The data files are not stored in this repo (see `.gitignore`). Download them from the links above to re-run the notebooks.

## What was done (see `Notebooks/`)

- EDA: duplicates (300 in churn, 24,206 in diabetes), outliers (BMI capped at 60), correlations and multicollinearity, a leakage check on `Status` (churn), target conversion (diabetes).
- Stratified train/test split. **Balancing is applied to the training set only**; the test set keeps the real class balance.
- Several models compared, each with class weights or SMOTE (churn) / undersampling (diabetes). Metrics: recall, precision, F1, PR-AUC, ROC-AUC. Accuracy is deliberately not reported because it is misleading on imbalanced data.
- XGBoost shipped in both projects. Scores are **risk scores**, not literal probabilities, because the training data was rebalanced.

### Churn: model comparison (cross-validation on the training set)

| Model | Recall | Precision | F1 | ROC-AUC | PR-AUC | PR-AUC std |
|---|---|---|---|---|---|---|
| XGBoost + class weight | 0.922 | 0.761 | 0.833 | 0.979 | **0.916** | 0.026 |
| RandomForest + class weight | 0.924 | 0.773 | 0.842 | 0.981 | 0.906 | 0.040 |
| XGBoost + SMOTE | 0.891 | 0.719 | 0.795 | 0.975 | 0.904 | 0.032 |
| RandomForest + SMOTE | 0.902 | 0.768 | 0.829 | 0.980 | 0.895 | 0.054 |
| LogReg + class weight | 0.863 | 0.500 | 0.632 | 0.935 | 0.751 | 0.022 |
| LogReg + SMOTE | 0.882 | 0.459 | 0.603 | 0.928 | 0.739 | 0.022 |

XGBoost + class weight was chosen: best PR-AUC (the main metric on imbalanced data), with a lower fold-to-fold spread than the Random Forests. Random Forest + class weight was a close second.

### Diabetes: model comparison (held-out test set, real class balance)

Models were trained on an undersampled training set and scored on the untouched test set.

| Model | Recall | Precision | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| XGBoost | 0.791 | 0.314 | 0.449 | 0.817 | **0.437** |
| Random Forest | 0.785 | 0.315 | 0.449 | 0.814 | 0.429 |
| Logistic Regression | 0.760 | 0.320 | 0.450 | 0.811 | 0.421 |

The three models are very close. XGBoost was chosen for the best recall, ROC-AUC and PR-AUC. For a health screening flag, recall matters most (missing a person is worse than flagging an extra one), and the low precision (about 0.31) is the cost of that.

### Shipped model results (final held-out test set, real class balance)

| Project | Recall | PR-AUC |
|---|---|---|
| Churn | 0.899 | 0.912 |
| Diabetes | 0.791 | 0.437 |

## Gen AI layer

- Model: Google Gemini through the `google-genai` SDK (default `gemini-3.5-flash-lite`, set via `GEMINI_MODEL`).
- **Healthcare safety:** the prompt forbids diagnoses, treatment and medication advice, and asks for a calm nurse-like description of the result. Output is also checked by a regex safety filter; on a violation it retries once, then falls back to a fixed template. A disclaimer is always appended. The chatbot declines diagnosis, treatment or medication questions and points to a doctor.
- If Gemini is unavailable or the quota is exhausted, the app still works with template explanations.

## Run it locally

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then put your own key in .env
python app.py
```

Open http://127.0.0.1:5000. Get a free API key from Google AI Studio and set `GEMINI_API_KEY` in `.env`. Never commit `.env`.

## Project layout

```
app.py                 Flask app (two projects, shared chatbot and dashboard)
templates/index.html   Single-page UI
models/                Trained XGBoost models and feature lists (.pkl)
Notebooks/             EDA, balancing and model comparison for both projects
requirements.txt
.env.example
```

## Limitations

- Project 2 is a **screening-style risk flag from survey answers**, not a diagnostic tool. It is trained on self-reported 2015 survey data, and its PR-AUC (0.437) is modest; many flagged people will not have diabetes.
- Prediabetes is merged with "no diabetes".
- Risk scores are not calibrated probabilities.
- Dashboard history is stored in a local SQLite file (`history.db`) and is not committed.
