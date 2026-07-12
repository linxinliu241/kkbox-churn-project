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

%% ====================
%% Raw Sources
%% ====================

subgraph RAW["Raw Data Sources"]
direction TB

T1["transactions.csv<br/>Payment records<br/>Renewal / cancellation behavior"]

T2["transactions_v2.csv<br/>Extended transaction history"]

M["members.csv<br/>User demographic information"]

L["user_logs.csv<br/>Daily listening activity"]

end


%% ====================
%% Data Construction
%% ====================

subgraph CON["Data Construction"]
direction TB

TX["Transaction Integration<br/>Combine transaction history"]

TARGET["Cohort Target Construction<br/>Select users expiring within cohort month"]

PROFILE["Member Integration<br/>Merge user profile information"]

ACT["Activity Integration<br/>Retrieve logs before cutoff date"]

end


%% ====================
%% Feature Engineering
%% ====================

subgraph FE["Feature Engineering"]
direction TB

AGG["User-Cohort Aggregation<br/>Multiple records → user-level features"]

FEATURE["Behavioral Features<br/>Payment patterns<br/>Renewal behavior<br/>Cancellation history<br/>Engagement trends"]

SELECT["Feature Selection<br/>Select 9 predictive features"]

end


%% ====================
%% Preparation
%% ====================

subgraph PREP["Model Preparation"]
direction TB

IMP["Missing Value Imputation"]

SPLIT["Cohort-based Split<br/>Train / Validation / Test"]

end


FINAL["Final Modeling Dataset<br/>25 monthly cohorts<br/>9 selected features"]



%% Connections

T1 --> TX
T2 --> TX

TX --> TARGET

TARGET --> PROFILE
M --> PROFILE

PROFILE --> ACT
L --> ACT

ACT --> AGG

AGG --> FEATURE
FEATURE --> SELECT

SELECT --> IMP
IMP --> SPLIT
SPLIT --> FINAL


%% Style

classDef raw fill:#4F73B8,color:white,stroke:#333;
classDef process fill:#63A46C,color:white,stroke:#333;
classDef final fill:#C94C4C,color:white,stroke:#333;

class T1,T2,M,L raw;
class TX,TARGET,PROFILE,ACT,AGG,FEATURE,SELECT,IMP,SPLIT process;
class FINAL final;
