# Loyal Customers vs. Ghost Accounts
## A Machine Learning Framework for Predicting User Retention and Churn on KKbox

[Executive Summary](KKBox%20Churn%20Prediction%20Executive%20Summary.pdf)

A time-aware validation framework identifies high-risk users with XGBoost achieving strong predictive performance while maintaining model interpretability.

## Project Summary

| Component | Description |
|-----------|-------------|
| **Dataset** | ~17M user records across 25 monthly cohorts (2015.01–2017.02), with 28 engineered features |
| **Churn Definition** | Churn prediction within the future subscription window: churn = 1, non-churn = 0 |
| **Class Balance** | 7.44% churn vs. 92.56% non-churn |
| **Final Model** | XGBoost selected with PR-AUC = **0.542**, achieving **10×** improvement over random baseline (**0.055**) |



## Project Overview

Music streaming platforms acquire millions of users, yet retaining long-term subscribers remains a major challenge. This project develops a machine learning framework to predict user churn on KKBOX by leveraging subscription patterns, listening behaviors, and profile information. The analysis aims to uncover key drivers of customer retention and identify users at risk of leaving the platform.


## Data Source

| File / Source | Description |
|---|---|
| **transactions.csv** | User transaction records (~2.1M rows), including payment history, subscription plans, renewal status, and cancellation behavior. |
| **transactions_v2.csv** | Updated transaction records (~7.1M rows) containing additional subscription activities through 2017-03-31. |
| **user_logs.csv** | Daily user listening behavior logs (~104.9M rows), including song play counts, unique songs, and total listening duration. |
| **members.csv** | User demographic and account information (~3.4M users), including city, age, gender, registration method, and account dates. |


## Data Preprocessing Pipeline

## Data Pipeline

The data processing pipeline transforms raw KKBOX data sources into a user-cohort level modeling dataset. The pipeline consists of data filtering, integration, feature engineering, missing value handling, and cohort-based splitting.

![Data Pipeline](figures/data_pipeline.png)

### Pipeline Overview

**1. Raw Data Sources**

The project uses four raw data sources:

- **Transactions Data**: User subscription transactions, including payment records, renewal behavior, and cancellation information.
- **User Logs Data**: Daily listening activity records, including song plays, unique songs, and listening duration.
- **Members Data**: User demographic and registration information.
- **Train/Test Data**: Cohort-based churn labels and prediction periods.

---

**2. Churn Target Construction & Data Integration**

For each cohort month:

- Identify transactions before the cohort cutoff date.
- Select users whose membership expires within the cohort month as prediction targets.
- Retrieve corresponding user information from the member table.
- Identify listening activities occurring before the cutoff date for target users.
- Integrate transaction history, user profile information, and listening behaviors at the user-cohort level.

---

**3. Feature Engineering**

Multiple transaction and activity records are aggregated into user-level features within each cohort.

Generated features include:

- Number of transactions
- Total payment amount
- Average plan price
- Total auto-renew count
- Total cancellation count
- Last payment plan duration
- Last plan price
- Last auto-renew status
- Last cancellation status
- Days since last transaction
- Days to membership expiration
- Average payment per day

The final dataset is constructed at the **user-cohort level**, covering **25 monthly cohorts**.

Feature selection was performed by analyzing feature relationships with churn outcomes, resulting in **9 selected predictors** used for modeling.

---

**4. Data Preparation**

- Missing values were imputed for selected predictors.
- Cohort-based splitting was applied to simulate real-world prediction scenarios:

  - Training cohorts: earlier time periods
  - Validation cohorts: subsequent cohorts for model selection
  - Test cohorts: future unseen cohorts for final evaluation






























