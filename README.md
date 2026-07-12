# Loyal Customers vs. Ghost Accounts
## A Machine Learning Framework for Predicting User Retention and Churn on KKbox

[Executive Summary](KKBox%20Churn%20Prediction%20Executive%20Summary.pdf)

A time-aware validation framework identifies high-risk users with XGBoost achieving strong predictive performance while maintaining model interpretability.

## Project Summary

| Component | Description |
|-----------|-------------|
| **Objective** | Predict user churn on KKBOX, a music streaming platform, and identify key factors associated with customer retention. |
| **Dataset** | KKBOX user subscription, transaction, listening activity, and demographic information. The dataset contains highly imbalanced churn labels, with churn users representing a small minority of the population. |
| **Prediction Task** | Binary classification of whether a user will churn based on historical user information before the prediction period. |
| **Validation Strategy** | Time-series cross-validation across historical user cohorts was used to evaluate model robustness under realistic future prediction scenarios. |
| **Models Compared** | Logistic Regression, Random Forest, LightGBM, XGBoost, CatBoost, and a weighted ensemble model. |
| **Model Selection** | Models were compared based on validation performance using ROC-AUC, PR-AUC, and log loss. Although the ensemble achieved slightly improved validation performance, XGBoost was selected as the final model due to its competitive performance and better interpretability through SHAP analysis. |
| **Final Evaluation** | The selected XGBoost model was evaluated on three unseen future cohorts to assess generalization performance on users from future time periods. |
| **Key Result** | Achieved an overall PR-AUC of **0.542** on future cohorts compared with a random baseline of **0.055**, providing approximately a **10× improvement over random guessing**. |



# Project Overview

Music streaming platforms acquire millions of users, yet retaining long-term subscribers remains a major challenge. This project develops a machine learning framework to predict user churn on KKBOX by leveraging subscription patterns, listening behaviors, and profile information. The analysis aims to uncover key drivers of customer retention and identify users at risk of leaving the platform.
