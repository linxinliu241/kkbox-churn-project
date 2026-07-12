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
%% Raw Data Sources
%% =========================

subgraph Raw["Raw Data Sources"]
    T["Transactions Data<br/>Payment records<br/>Renewal / cancellation behavior<br/>Membership expiration"]
    L["User Logs Data<br/>Daily listening activity<br/>Song plays & engagement"]
    M["Members Data<br/>Demographic information<br/>Registration details"]
    Y["Train/Test Data<br/>Cohort-based churn labels"]
end


%% =========================
%% Churn Target Construction
%% =========================

subgraph Target["Churn Target Construction"]
    C1["Cohort-based Filtering<br/>Select transactions before cutoff date<br/>Identify users expiring within cohort month"]
    
    C2["User Identification<br/>Match target users with member profiles"]
    
    C3["Activity Selection<br/>Retrieve user logs before cutoff date<br/>Select relevant listening history"]
end


%% =========================
%% Integration
%% =========================

subgraph Integration["Data Integration"]
    I["Merge Transaction, Profile,<br/>and Activity Data<br/>User × Cohort Level Dataset"]
end


%% =========================
%% Feature Engineering
%% =========================

subgraph Feature["Feature Engineering"]
    F1["Aggregate User Transactions<br/>Multiple records → user-level features"]
    
    F2["Behavioral Features<br/>Payment patterns<br/>Renewal behavior<br/>Cancellation history<br/>Listening engagement"]
    
    F3["Feature Selection<br/>Analyze feature–churn relationship<br/>Select 9 predictive features"]
end


%% =========================
%% Preparation
%% =========================

subgraph Preparation["Model Preparation"]
    P1["Missing Value Imputation"]
    
    P2["Cohort-based Split<br/>Train / Validation / Test"]
end


%% =========================
%% Final Dataset
%% =========================

Final["Final Modeling Dataset<br/>User-Cohort Level<br/>25 Monthly Cohorts<br/>9 Selected Features"]


%% Connections

T --> C1
L --> C3
M --> C2
Y --> C1

C1 --> C2
C2 --> C3
C3 --> I

I --> F1
F1 --> F2
F2 --> F3

F3 --> P1
P1 --> P2

P2 --> Final


%% Styling

classDef raw fill:#4F73B8,color:white,stroke:#333;
classDef process fill:#63A46C,color:white,stroke:#333;
classDef data fill:#D9864A,color:white,stroke:#333;
classDef final fill:#C94C4C,color:white,stroke:#333;

class T,L,M,Y raw;
class C1,C2,C3,F1,F2,F3,P1,P2 process;
class I data;
class Final final;
































