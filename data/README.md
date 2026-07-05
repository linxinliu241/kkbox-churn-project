# Data Description

## 1. Data Sources (Raw Data)

This project uses the KKBOX Churn Prediction dataset, which includes multiple tables:

- members.csv: user demographic information
- transactions.csv: payment and subscription history
- user_logs.csv: daily user activity logs
- train.csv: labeled training data (churn outcome)
- test.csv: unlabeled evaluation data

All datasets are provided by the Kaggle competition:
https://www.kaggle.com/c/kkbox-churn-prediction-challenge

---

## 2. Label Definition (Churn Construction Policy)

Churn labels are generated using a rule-based definition based on subscription expiration:

- A user is considered churned if no renewal occurs within the defined observation window
- The cutoff date is set at 2017-01-31 to prevent data leakage
- Label generation is implemented in the data preprocessing pipeline

The exact labeling logic is implemented in:
src/data/labeling.py

---

## 3. Dataset Versions

Two versions of the dataset were considered during development:

- Version 1: initial labeling logic (potential leakage from future information)
- Version 2: corrected version with strict time cutoff (final model uses this version)

Only Version 2 is used in final model training and evaluation.

---

## 4. Data Integration & Processing

Raw datasets are merged and transformed into a unified user-level dataset.

Key steps include:

- Aggregating transaction history at user level
- Summarizing user log activity
- Joining member information
- Feature engineering from temporal behavior

All processing is implemented in:
src/data/preprocess.py

---

## 5. Data Organization

、、、text
data/
├── raw/          # Original Kaggle datasets
├── processed/    # Cleaned and feature-engineered datasets
├── sample/       # Small subset for debugging and fast testing
、、、

---

## 6. Reproducibility

All datasets (except raw Kaggle data) are generated programmatically via scripts in src/.

To reproduce results:

1. Download raw data from Kaggle
2. Place in data/raw/
3. Run preprocessing scripts in src/
4. Train model using notebooks or src/train.py
