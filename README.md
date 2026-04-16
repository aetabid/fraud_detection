# Fraud Detection & Loan Default Prediction
### Czech Financial Dataset Analysis

## Overview
A collaborative data science project analyzing the 1999 Czech 
banking dataset to predict loan defaults and detect financial 
risk patterns using machine learning.

## Projects

### Ayman's Work — Loan Default Prediction
Predicts whether a bank account will default on a loan using 
time-series behavioral features engineered from 1 million+ 
transaction records.

**Key Finding:** Balance volatility and erratic spending 
behavior are stronger predictors of loan default than 
average spending amount alone.

**Techniques Used:**
- Time-series feature engineering using np.polyfit 
  for balance trend detection
- SMOTE for class imbalance handling
- Random Forest Classifier (200 estimators)
- Confusion matrix and feature importance analysis

**Results:**
- Accuracy: 85%
- Recall on default class: 53%

### Utku's Work — Customer Segmentation
Customer segmentation and loan status classification using 
K-Means clustering and ensemble methods.

**Techniques Used:**
- K-Means clustering with elbow method
- Log transformation for outlier handling
- StandardScaler normalization
- GridSearchCV hyperparameter tuning
- XGBoost and Random Forest comparison

## Dataset
1999 Czech Banking Dataset — real financial transaction data 
including loans, accounts, and transaction history.

## Tech Stack
Python, Pandas, Scikit-learn, XGBoost, Imbalanced-learn, 
Matplotlib, Seaborn

## Authors
- Ayman Tabidi — github.com/aetabid
- Utku Seyithanoğlu
