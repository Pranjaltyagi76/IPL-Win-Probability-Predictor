# IPL Win Probability Predictor

This project predicts the win probability of the batting team during the second innings of an IPL match based on the current match situation.

It demonstrates a complete end-to-end machine learning workflow, starting from raw IPL ball-by-ball data to a deployed interactive web application.

Project Overview

Uses historical IPL match and delivery data

Focuses on second-innings chases where the target score is known

Generates match-state features such as:

Runs left

Balls remaining

Wickets in hand

Current Run Rate (CRR)

Required Run Rate (RRR)

Trains a Logistic Regression model to estimate win probability

Deploys the trained model using a Streamlit web interface

Includes visualizations for better interpretation of predictions

# Approach
# Feature Engineering

Merged match-level and ball-by-ball datasets

Filtered inconsistent and DLS-affected matches

Computed real-time match features for each delivery

Created a clean training dataset focused on second innings

# Model Training

Built an sklearn Pipeline combining:

One-Hot Encoding for categorical features

Logistic Regression for probabilistic prediction

Evaluated the model using ROC-AUC (~0.88 on test data)

Saved the trained pipeline as a serialized file (pipe.pkl)

# Deployment

Loaded the trained model for inference only (no retraining)

Built an interactive Streamlit app

Automatically calculates CRR and RRR from user inputs

Displays win probabilities along with supporting visualizations

Application Features

Real-time win probability prediction

Automatic run rate calculations

Match summary and validation checks

# Visualizations:

CRR vs RRR comparison

Team-wise win probability bar chart

Match pressure snapshot (runs vs balls remaining)

# Project Structure
IPL WIN PREDICTOR/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── matches.csv
│   └── deliveries.csv
│
└── src/
    ├── feature_engineering.py
    └── train_model.py


# Dataset

IPL dataset sourced from Kaggle:
https://www.kaggle.com/datasets/ramjidoolla/ipl-data-set

# Author

Pranjal Tyagi
B.Tech CSE (AI & DS)
