"""Train the IPL win-probability model.

The split is season-based on purpose. A random split over ball-level rows
leaks: consecutive balls of the same match are nearly identical, so the same
match lands in both train and test and the model is scored partly on matches
it has already seen. Training on 2008-2017 and testing on 2018-2019 instead
mirrors real deployment -- predict future matches from past ones.
"""

import pickle

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from feature_engineering import prepare_dataset, FEATURE_COLUMNS

df = prepare_dataset('../data/matches.csv', '../data/deliveries.csv')

# Select features by name rather than position, so adding a column to the
# dataset can never silently change what the model trains on.
X = df[FEATURE_COLUMNS]
y = df['result']

TEST_SEASON_START = 2018

# Time-based split: past seasons train, future seasons test.
is_test = df['season'] >= TEST_SEASON_START

X_train, X_test = X[~is_test], X[is_test]
y_train, y_test = y[~is_test], y[is_test]

CATEGORICAL = ['batting_team','bowling_team','city']
NUMERIC = ['runs_left','balls_left','wickets','target','crr','rrr']

# Scaling the numeric features lets lbfgs converge within max_iter and keeps
# the logistic-regression probabilities well behaved. Previously the raw
# 'target' and 'crr' scales caused a ConvergenceWarning on every run.
preprocessor = ColumnTransformer([
    ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'),
     CATEGORICAL),
    ('scale', StandardScaler(), NUMERIC),
])

pipe = Pipeline([
    ('preprocess', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])

pipe.fit(X_train, y_train)

auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1])
print(f"Train seasons < {TEST_SEASON_START}  ({len(X_train)} balls)")
print(f"Test  seasons >= {TEST_SEASON_START} ({len(X_test)} balls)")
print(f"ROC AUC: {auc:.4f}")

pickle.dump(pipe, open('../pipe.pkl', 'wb'))
