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

```mermaid
flowchart LR

%% =====================
%% Raw Data Sources
%% =====================

subgraph RAW[" "]
direction TB

A["transactions.csv<br/>Payment records<br/>Renewal / cancellation behavior"]

B["transactions_v2.csv<br/>Extended transaction history"]

C["members.csv<br/>User demographic information"]

D["user_logs.csv<br/>Daily listening activity"]

end


%% =====================
%% Data Construction
%% =====================

subgraph CON[" "]
direction TB

E["Transaction Integration<br/>Combine transactions.csv<br/>and transactions_v2.csv"]

F["Target User Identification<br/>Select users expiring<br/>within cohort month"]

G["Member Integration<br/>Merge user profile<br/>information"]

H["Activity Integration<br/>Retrieve user logs<br/>before cutoff date"]

I["User-Cohort Dataset<br/>One row = one user<br/>one cohort month"]

end


%% =====================
%% Feature Engineering
%% =====================

subgraph FE[" "]
direction TB

J["User-Level Aggregation<br/>Multiple records → user features"]

K["Feature Engineering<br/>Payment patterns<br/>Renewal behavior<br/>Cancellation history<br/>Engagement trends"]

L["Feature Selection<br/>Feature-churn analysis<br/>Select 9 predictors"]

end


%% =====================
%% Model Preparation
%% =====================

subgraph PREP[" "]
direction TB

M["Missing Value Imputation"]

N["Cohort-based Split<br/>Train / Validation / Test"]

end


%% =====================
%% Final Dataset
%% =====================

O["Final Modeling Dataset<br/>25 monthly cohorts<br/>9 selected features"]


%% =====================
%% Connections
%% =====================

A --> E
B --> E

E --> F

C --> G
F --> G

G --> H
D --> H

H --> I

I --> J
J --> K
K --> L

L --> M
M --> N
N --> O


%% =====================
%% Styling
%% =====================

classDef raw fill:#4F73B8,color:white,stroke:#333;
classDef process fill:#63A46C,color:white,stroke:#333;
classDef final fill:#C94C4C,color:white,stroke:#333;

class A,B,C,D raw;
class E,F,G,H,I,J,K,L,M,N process;
class O final;

