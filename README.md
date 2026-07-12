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

%% =========================
%% Raw + Integration (Left)
%% =========================

A["transactions.csv<br/>Payment & renewal"]

B["transactions_v2.csv<br/>Extended transactions"]

C["members.csv<br/>User profile"]

D["user_logs.csv<br/>Listening activity"]


E["Transaction Integration<br/>Merge transactions"]

F["Target Users<br/>Cohort expiration filtering"]

G["Member Integration<br/>Merge profiles"]

H["Activity Integration<br/>Historical logs"]

I["User-Cohort Dataset<br/>User × Cohort"]


%% =========================
%% Right Vertical Pipeline
%% =========================

subgraph RIGHT[" "]
direction TB

J["Aggregation<br/>Generate user features"]

K["Feature Engineering<br/>Behavior & payment features"]

L["Feature Selection<br/>9 predictors"]

M["Imputation"]

N["Cohort Split<br/>Train / Val / Test"]

O["Final Dataset<br/>25 cohorts<br/>9 features"]

J --> K
K --> L
L --> M
M --> N
N --> O

end


%% =========================
%% Connections
%% =========================

A --> E
B --> E

E --> F

F --> G
C --> G

G --> H
D --> H

H --> I

I --> J


%% =========================
%% Style
%% =========================

classDef raw fill:#4F73B8,color:white,stroke:#333;
classDef process fill:#63A46C,color:white,stroke:#333;
classDef final fill:#C94C4C,color:white,stroke:#333;

class A,B,C,D raw;
class E,F,G,H,I,J,K,L,M,N process;
class O final;
