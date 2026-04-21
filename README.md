# Fraud Detection & Loan Default Prediction
### Czech Financial Dataset Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Complete-green)

## Overview
A collaborative data science project analyzing the 1999 Czech banking dataset 
to predict loan defaults and detect financial risk patterns using machine 
learning. The dataset contains 1 million+ real financial transactions across 
4,500 unique bank accounts.

## Business Problem
Banks lose billions annually to loan defaults. Early identification of 
high-risk accounts allows lenders to intervene before default occurs. 
This project builds a predictive system using behavioral transaction 
patterns to flag at-risk accounts.

## Projects

### Ayman's Work — Loan Default Prediction
Predicts whether a bank account will default on a loan using time-series 
behavioral features engineered from 1 million+ transaction records.

**Key Finding:** Balance volatility and erratic spending behavior are 
stronger predictors of loan default than average spending amount alone.

**Techniques Used:**
- Time-series feature engineering using `np.polyfit` for balance trend detection
- SMOTE for class imbalance handling
- Random Forest Classifier (200 estimators)
- Confusion matrix and feature importance analysis

**Results:**
| Metric | Score |
|--------|-------|
| Accuracy | 85% |
| Recall (Default class) | 53% |
| Precision (Good class) | 94% |

### Utku's Work — Customer Segmentation
Customer segmentation and loan status classification using K-Means 
clustering and ensemble methods.

**Techniques Used:**
- K-Means clustering with elbow method
- Log transformation for outlier handling
- StandardScaler normalization
- GridSearchCV hyperparameter tuning
- XGBoost and Random Forest comparison

## Dataset
[1999 Czech Banking Dataset](https://sorry.vse.cz/~berka/challenge/pkdd.htm) — 
real financial transaction data including:
- 1,056,320 transactions
- 4,500 unique accounts
- Loan, client, card, and district data

## Tech Stack
`Python` `Pandas` `NumPy` `Scikit-learn` `XGBoost` `Imbalanced-learn` 
`Matplotlib` `Seaborn`

## How to Run
```bash
git clone https://github.com/aetabid/fraud_detection
cd fraud_detection
pip install -r requirements.txt
```
Open `Ayman_folder/loan_default_prediction.ipynb` in Jupyter or Google Colab.

## Authors
- **Ayman Tabidi** — [github.com/aetabid](https://github.com/aetabid)
- **Utku Seyithanoğlu** — 
[github.com/utkuseyithanoglu](https://github.com/utkuseyithanoglu)
