# Capstone Write-up: Churn and Diabetes Risk

One combined Flask application with two pages (Banking and Healthcare) sharing the chatbot and dashboard code, with one trained XGBoost model per project and a Gemini explanation layer. Code: github.com/HassanAbbas301/AI-Capstone-Churn-Diabetes (private). The README states that this is one combined app.

## Project 1: Banking, customer churn prediction

### 1. Which ML problem, and why this one?

I chose customer churn prediction over fraud, segmentation, credit risk and anomaly detection for three reasons. First, the data is readable: the Iranian Churn dataset has plain columns (complaints, call failures, subscription length), so the Gen AI layer can name real reasons. The best-known public fraud datasets are anonymised, which makes a plain-language "why was this flagged" impossible. Second, it has real labels, so I could measure whether the model works; segmentation and anomaly detection have no ground truth. Third, the output leads to a clear action: a retention call and offer copy.

### 2. Dataset and what the EDA showed

Source: [Iranian Churn Dataset, UCI](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset), file `Customer Churn.csv`, licence CC BY 4.0, 3,150 rows (I checked my file against the source row count). Before building anything the EDA showed: about 15.7% churn, so the classes are imbalanced; 300 duplicate rows; outliers in the usage columns; three highly correlated pairs (multicollinearity): Age Group and Age (0.96), Seconds of Use and Frequency of use (0.95), and Frequency of SMS and Customer Value (0.92); and a possible leakage risk in the Status column, which I tested by comparing results with and without it. Status is strongly linked to churn: 5.6% of active customers churned against 47.5% of non-active ones (121 of 2,166 and 325 of 684). I kept it as a normal account field and list it as a limitation, since some non-active customers may already be leaving. There were no missing values in any of the 14 columns (total missing: 0). I removed the duplicates (3,150 rows down to 2,850), dropped Age Group because it is a binned copy of Age (correlation 0.96, leaving 12 features); I kept the other two pairs because tree-based models such as XGBoost cope with correlated inputs, and made a stratified 80/20 train/test split (2,280 training rows, 570 test rows) with churn at 15.66% in training and 15.61% in test, and balanced the training set only, so the test set keeps the real churn rate.

### 3. Algorithms tried, the one shipped, and the metrics

I compared Logistic Regression, Random Forest and XGBoost, each with class weights and with SMOTE, using 5-fold stratified cross-validation on the training set.

| Model | Recall | Precision | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- |
| XGBoost + class weight (shipped) | 0.922 | 0.761 | 0.833 | 0.979 | 0.916 |
| Random Forest + class weight | 0.924 | 0.773 | 0.842 | 0.981 | 0.906 |
| XGBoost + SMOTE | 0.891 | 0.719 | 0.795 | 0.975 | 0.904 |
| Random Forest + SMOTE | 0.902 | 0.768 | 0.829 | 0.980 | 0.895 |
| Logistic Regression + class weight | 0.863 | 0.500 | 0.632 | 0.935 | 0.751 |
| Logistic Regression + SMOTE | 0.882 | 0.459 | 0.603 | 0.928 | 0.739 |

Tree-based models clearly beat Logistic Regression (PR-AUC about 0.90 to 0.92 against 0.74 to 0.75), so the churn signal is non-linear. Class weights did slightly better than SMOTE for both tree models and are simpler, because no synthetic rows are created. XGBoost and Random Forest with class weights are statistically close (differences under 0.01 are within fold-to-fold variation). I shipped XGBoost with class weights (scale\_pos\_weight): best PR-AUC, the main metric for imbalanced data, and steadier between folds (PR-AUC std 0.026 against 0.040 for the Random Forest). I did not report accuracy, because predicting "no churn" for everyone would already score about 84%. On the untouched test set it scored recall 0.899 and PR-AUC 0.912, close to the cross-validation result, so it is not overfitted. The score is a risk score, not a literal probability, because training used class weights.

### 4. How the Gen AI layer uses the model's output (real example)

The app sends Gemini the customer's inputs, the model's score and risk level, the flagged or not-flagged decision, and the top factors from XGBoost feature contributions. The wording changes with the result.

|  | High-risk case | Low-risk case |
| --- | --- | --- |
| Input | Complained: yes; account non-active; pay-as-you-go; 10 months; age 30; 26 call failures; 200 seconds of calls; 5 calls; 0 SMS; 2 distinct numbers | Complained: no; account active; pay-as-you-go; 30 months; age 30; 5 call failures; 3,000 seconds; 50 calls; 20 SMS |
| Prediction | 100 / 100, High risk, flagged: likely to churn | 6 / 100, Low risk, not flagged |
| Top factors | Complaint, 26 call failures, short call time (all raise the score) | 50 calls, active account, no complaint (all lower the score) |

Generated explanation for the high-risk case: "The customer has registered a complaint, which strongly increases their risk score. The account has experienced a very high number of call failures (26), contributing significantly to dissatisfaction." Recommended next action: "Call to retain: immediate personal outreach is necessary to address their complaints and resolve their high call failure issues before they formally churn." Offer copy: "We noticed you've been experiencing technical issues with your calls recently, and we want to make it right. Please give us a call today so our team can credit your account and ensure your service runs smoothly." For the low-risk case the same layer returned "No action: this customer is at a low risk level... no retention intervention is required at this time." The chatbot answers follow-up questions about the case on screen, for example "which factors mattered?".

### 5. What broke and what I changed

- **Duplicates and imbalance:** 300 duplicate rows and 15.7% churn. I removed the duplicates and balanced only the training set.
- **Probabilities were inflated** by class weights and SMOTE, so I stopped calling the output a probability and show it as a risk score out of 100.
- **The Gemini model I started with was retired** for new users (404). I switched to a current Flash-Lite model and set it in `.env`. The free tier then hit its quota (429), so the app falls back to a template explanation instead of crashing.
- **Raw `**` asterisks showed in the chat** because Gemini writes markdown. I asked for plain text and strip asterisks in the reply.
- **Dashboard colours were mismatched** (High shown green) because JSON sorted the keys alphabetically. I turned key sorting off.

## Project 2: Healthcare, diabetes risk flag

### 1. Which ML problem, and why this one?

I chose diabetes prediction over heart disease, readmission, general disease risk and patient segmentation. The CDC survey data is large (253,680 rows) and its inputs are plain answers a nurse recognises (blood pressure, cholesterol, BMI, general health), so the explanation can name them. It fits the brief's "a flag, not a diagnosis" framing, because the output is a screening flag from survey answers. The easily available heart-disease dataset is small (about 300 rows), readmission needs clinical records that are rarely public, and segmentation has no ground truth to evaluate against.

### 2. Dataset and what the EDA showed

Source: [Diabetes Health Indicators Dataset, Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset), from the CDC BRFSS 2015 survey, licence CC0 on Kaggle (CC BY 4.0 on the [UCI copy](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)). I used the original imbalanced file, not the 50/50 version. The EDA showed: 253,680 rows and 22 columns (21 inputs plus the target), with no missing values in any column, and 13.93% positive (35,346 diabetes against 218,334 without), so strongly imbalanced; a 3-class target (0 no diabetes, 1 prediabetes, 2 diabetes) that I had to convert to binary, with diabetes as the positive class, so prediabetes counts as "no diabetes"; 24,206 duplicate rows (9.54%), almost all from healthy respondents (the duplicated rows are 98.65% non-diabetic), which I removed (253,680 rows down to 229,474). This lifts the positive share from 13.93% to 15.29% (194,377 against 35,097, an imbalance ratio of 5.54); BMI outliers: 5,638 values (2.46%) sit above the IQR upper bound of 44 (max 98). Most are plausible, because high BMI is a known diabetes risk factor, so I kept the rows. But 805 rows (0.35%) are above 60 and 279 are above 80, and the jump from the 99th percentile (50) to the 99.9th (81.5) points to a small cluster of implausible self-reported values. I capped BMI at 60 (a fixed cutoff, not data-derived, so no train/test leakage) instead of deleting rows. MentHlth and PhysHlth are heavily skewed (most answers 0, a cluster at 30), which reflects how the survey question works, not bad data. Correlations with the target are modest: GenHlth 0.277, HighBP 0.254, BMI 0.216, DiffWalk 0.205, HighChol 0.195 and Age 0.177, with Income (-0.141), Education and PhysActivity negative. No feature is above 0.3, so no single variable predicts the target alone, and I kept the weak ones because correlation only measures linear association. I used a stratified split and undersampled the training set only, so the test set keeps the real class balance (about 15% positive).

### 3. Algorithms tried, the one shipped, and the metrics

I trained Logistic Regression, Random Forest and XGBoost on the undersampled training set and scored them on the untouched test set.

| Model | Recall | Precision | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- |
| XGBoost (shipped) | 0.791 | 0.314 | 0.449 | 0.817 | 0.437 |
| Random Forest | 0.785 | 0.315 | 0.449 | 0.814 | 0.429 |
| Logistic Regression | 0.760 | 0.320 | 0.450 | 0.811 | 0.421 |

The three models are very close. I shipped XGBoost for the best recall, ROC-AUC and PR-AUC. For a health screening flag a missed person costs more than an extra flag, so recall mattered most. The cost is low precision (about 0.31): roughly two in three flagged people will not have diabetes, which is why the app calls the result a risk flag for staff review. I did not report accuracy, because it hides the minority class. The score is a risk score, not a calibrated probability.

### 4. How the Gen AI layer uses the model's output (real example)

The app sends Gemini the answers, the score, the risk level, the flagged decision and the top factors. The prompt forbids diagnosis and treatment advice and asks for a nurse-style description. The wording changes with the result.

|  | High-risk case | Low-risk case |
| --- | --- | --- |
| Input | Female, 55-59; BMI 38; high blood pressure: yes; high cholesterol: yes; stroke: yes; general health 3 (good) | Female, 18-24; BMI 22; no high blood pressure or cholesterol; general health 3 (good) |
| Prediction | 79 / 100, Elevated risk, flagged for staff review | 7 / 100, Lower risk, not flagged |
| Top factors | High blood pressure, BMI 38, high cholesterol, stroke (all raise the score) | Age band 18-24, no high blood pressure, BMI 22 (all lower the score) |

For the high-risk case Gemini wrote: "This automated risk flag resulted in an elevated score of 0.79, which crosses the flag threshold of 0.5. This number is a ranking-style risk score rather than a literal real-world probability, and a clinician interprets it." It then listed the factors that raised the score, gave neutral background on them, and suggested questions for a doctor, such as "What steps or evaluations are appropriate given my BMI and past stroke history?" For the low-risk case it said the result "is currently not flagged, showing a lower risk score of 0.07". Every explanation ends with a fixed disclaimer that it is not a diagnosis or medical advice. In the chatbot, "Do I have diabetes?" got: "I cannot provide a diagnosis or determine if you have a condition. Please discuss this result and any health concerns with a doctor. I can explain how the survey answers contributed to this specific risk score." "What medicine should I take?" was declined the same way and named no medicine.

The hard line is enforced in four layers: the prompt forbids diagnosis, treatment, medication names and phrases like "you should"; a regex filter checks every output; a failed output is retried once with a stricter reminder and then replaced by a fixed template; and the disclaimer is always added.

### 5. What broke and what I changed

- **Wrong target column:** I first modelled `Diabetes_012` as if it were binary. I converted it to a binary target and documented that prediabetes is merged with "no diabetes".
- **Duplicates:** 24,206 duplicate rows, mostly healthy people. I removed them and noted that this slightly raises the positive share.
- **The safety filter missed phrases** such as "likely to develop diabetes". I widened the patterns and added the retry and template fallback.
- **The model copied "I am not a doctor and never replace one"** from the prompt word for word, which sounded defensive, not like a nurse. I reworded the prompt and left the disclaimer to be added separately.
- **Gemini limits:** a retired model (404) and the free-tier quota (429). The app now falls back to a template, so it never shows an error.
- **A confusing factor:** "smoked 100+ cigarettes: No" showed as raising the score. It comes from patterns in the survey data, not cause and effect, so I list it as a limitation.
