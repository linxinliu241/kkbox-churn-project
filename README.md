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
```mermaid
%%{init: {
  "flowchart": {
    "nodeSpacing": 90,
    "rankSpacing": 70,
    "curve": "basis"
  },
  "themeVariables": {
    "fontSize": "20px"
  }
}}%%
flowchart LR
A["transactions.csv<br/>Payment & renewal"]
B["transactions_v2.csv<br/>Extended transactions"]
C["members.csv<br/>User profile"]
D["user_logs.csv<br/>Listening activity"]
E["Transaction Integration<br/>Merge transaction history"]
F["Target Users<br/>Cohort expiration filtering"]
G["Member Integration<br/>Merge user profile"]
H["Activity Integration<br/>Aggregate historical listening behavior by user and cohort"]

subgraph PIPELINE[" "]
direction TB
J["User-Cohort Aggregation<br/>Group historical records by user & cohort<br/>Generate user-level features"]
K["Feature Engineering<br/>Create subscription, payment, and engagement features"]
L["Feature Selection<br/>Analyze feature-churn relationships<br/>Select 9 predictive features"]
M["Final Dataset<br/>Missing value imputation and Data Split<br/>Cohort df_train / df_val / df_test"]
J --> K
K --> L
L --> M
end

A --> E
B --> E
E --> F
F --> G
C --> G
G --> H
D --> H
H --> J

classDef raw fill:#4F73B8,color:white,stroke:#333,font-size:20px;
classDef process fill:#63A46C,color:white,stroke:#333,font-size:20px;
class A,B,C,D raw;
class E,F,G,H,J,K,L,M process;
style PIPELINE fill:none,stroke:#666,stroke-width:2px,stroke-dasharray: 6 6

```


## Stage Summary

| Stage | Name | Key Transformations |
|---|---|---|
| 1 | **Transaction Data Generation** | Combined `transactions.csv` and `transactions_v2.csv`; removed duplicate transaction records; removed users with more than two transactions on the same day due to ambiguous ordering; sorted transaction history and generated 25 monthly cohort datasets based on membership expiration dates and cohort cutoff dates; constructed churn labels based on future renewal behavior. |
| 2 | **User Log Data Generation** | Filtered user listening logs to cohort users before each cutoff date; performed cohort-based aggregation on large-scale activity data; generated historical engagement features including listening activity, unique song counts, and usage velocity metrics; merged cohort-level activity information into transaction cohorts. |
| 3 | **User-Cohort Dataset Construction** | Merged transaction history, user activity, and member information by `msno` and cohort date; constructed user-cohort level observations where each row represents one user in one prediction month; removed unreliable demographic variables with inconsistent temporal interpretation. |
| 4 | **Feature Engineering & Selection** | Aggregated multiple historical records within each user-cohort into predictive features, including transaction statistics, payment behavior, renewal/cancellation patterns, recency metrics, and engagement trends; evaluated feature relationships with churn outcomes; selected 9 predictive features for modeling. |
| 5 | **Dataset Preparation** | Applied time-based cohort splitting to simulate future prediction scenarios; used earlier cohorts for training and validation and reserved future cohorts for final evaluation; performed missing value imputation on selected predictors before modeling. |



## Data Preparation Summary

| Step | Operation |
|---|---|
| 1 | Remove unreliable profile features (`bd`, `registration_init_time`) and unused raw fields; retain transaction and activity variables for cohort-based prediction. |
| 2 | Aggregate multiple transaction records within each user-cohort; generate subscription features including transaction count, payment statistics, renewal ratio, cancellation history, and latest subscription status. |
| 3 | Aggregate historical listening logs by user-cohort; transform daily activity records into engagement trend features and velocity metrics to capture changes in usage patterns over time rather than only absolute activity levels. |
| 4 | Create derived recency and behavioral features, including account tenure, days since last activity, payment intensity, and auto-renew behavior. |
| 5 | Handle missing values in selected predictors; missingness mainly occurred in activity-based features (`total_secs_velocity`, `num_unq_velocity`); apply imputation before modeling. |
| 6 | Remove redundant features based on correlation analysis (e.g., highly correlated usage velocity metrics); evaluate feature-churn relationships and select 9 predictors for final modeling. 

