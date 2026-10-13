# AI Capstone App (Banking churn + Healthcare diabetes flag)

One Flask app, two pages. Each page: input form -> XGBoost prediction -> Gemini explanation built from that
prediction -> dashboard (KPIs, charts, history) -> chatbot about the open case.

## Setup
```bash
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env           # Mac/Linux: cp .env.example .env  -> then paste your Gemini key
```
Copy the four model files from your notebooks into `models/`:
`xgb_churn_model.pkl`, `churn_feature_columns.pkl`, `xgb_diabetes_model.pkl`, `diabetes_feature_columns.pkl`.
Use the same xgboost / scikit-learn versions you trained with.

## Run
```bash
python app.py
```
Open http://127.0.0.1:5000. Without a key the app still works and uses a built-in template explanation.
