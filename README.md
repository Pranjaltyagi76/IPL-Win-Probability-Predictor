# IPL Win Probability Predictor

This project predicts the win probability of the batting team during the second innings of an IPL match using ball-by-ball data.

The goal of the project is to demonstrate an end-to-end machine learning workflow — from raw data processing and feature engineering to model training and deployment using a web interface.

# Project Overview

Built using historical IPL match and delivery data

Focuses on second innings chases, where the target is known

Calculates match-state features such as:

Runs left

Balls left

Wickets remaining

Current Run Rate (CRR)

Required Run Rate (RRR)

Uses a Logistic Regression model to estimate win probability

Deployed as an interactive Streamlit application

# Approach

Feature Engineering

Merged match-level and ball-by-ball data

Computed real-time match state features at each ball

Removed DLS-affected and inconsistent matches

Model Training

Used an sklearn Pipeline with:

One-Hot Encoding for categorical variables

Logistic Regression for probabilistic prediction

Evaluated performance using ROC-AUC

Deployment

Saved the trained pipeline as a serialized file

Built a Streamlit app for real-time prediction and visualization

# Project Structure
IPL WIN PREDICTOR/
│
├── app.py                  Streamlit app (inference + UI)
├── pipe.pkl                 Trained ML pipeline
├── README.md                Project documentation
├── requirements.txt         Project dependencies
│
├── data/
│   ├── matches.csv
│   └── deliveries.csv
│
└── src/
    ├── feature_engineering.py
    └── train_model.py


# Model Output

The app displays the win probability of the batting team

Probabilities are updated based on match state inputs

CRR and RRR are automatically calculated to avoid user inconsistency

# Evaluation Metric

ROC-AUC was used to evaluate the model’s ability to rank winning vs losing match states

# Tech Stack

Python

NumPy, Pandas

Scikit-learn

Streamlit

Git & GitHub

# Dataset

IPL dataset sourced from Kaggle:
https://www.kaggle.com/datasets/ramjidoolla/ipl-data-set

# Author

Pranjal Tyagi
B.Tech CSE (AI & DS)
