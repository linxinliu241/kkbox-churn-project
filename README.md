# Loyal Customers vs. Ghost Accounts
## A Machine Learning Framework for Predicting User Retention and Churn on KKbox

[Executive Summary](KKBox%20Churn%20Prediction%20Executive%20Summary.pdf)

A time-aware validation framework identifies high-risk users with XGBoost achieving strong predictive performance while maintaining model interpretability.

## Project Summary

| Component | Description |
|-----------|-------------|

| **Dataset** | 12.5M users, 25 corhot month from 2015.01 - 2017.02, integrate into 28 features |
| **Define Churn** | Labeling Churn within future subscription window: churn = 1, non-churn = 0 |
| **Class balance** | 7.44% churn, 92.56% non-churn |
| **Final Model** | XGBoost with PR-AUC of **0.542**, **10× improvement over random guessing 0.055**   |



# Project Overview

Music streaming platforms acquire millions of users, yet retaining long-term subscribers remains a major challenge. This project develops a machine learning framework to predict user churn on KKBOX by leveraging subscription patterns, listening behaviors, and profile information. The analysis aims to uncover key drivers of customer retention and identify users at risk of leaving the platform.
