"""Train the IPL win-probability model with a leakage-free split.

The original version used a random `train_test_split` on ball-level rows.
Because consecutive balls of the same match are almost identical, balls from
the same match ended up in BOTH train and test -- the model effectively
memorised matches, which inflated the reported ROC-AUC.

This version splits by *season* (train on 2008-2017, test on 2018-2019), which
mimics real deployment: predict future matches from past ones. For contrast it
also reports a group-random split (whole matches held out) and the old, leaky
row-random split so the gap is visible.
"""

import pickle

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from feature_engineering import prepare_dataset, FEATURE_COLUMNS

TEST_SEASON_START = 2018

CATEGORICAL = ['batting_team', 'bowling_team', 'city']
NUMERIC = ['runs_left', 'balls_left', 'wickets', 'target', 'crr', 'rrr']


def build_pipeline(estimator=None):
    # Scaling the numeric features lets lbfgs converge and keeps the
    # logistic-regression probabilities well behaved.
    preprocessor = ColumnTransformer([
        ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'),
         CATEGORICAL),
        ('scale', StandardScaler(), NUMERIC),
    ])
    if estimator is None:
        estimator = LogisticRegression(max_iter=1000)
    return Pipeline([
        ('preprocess', preprocessor),
        ('model', estimator),
    ])


def time_split(df):
    """Return X_train, X_test, y_train, y_test using the season-based split."""
    is_test = df['season'] >= TEST_SEASON_START
    X = df[FEATURE_COLUMNS]
    y = df['result']
    return X[~is_test], X[is_test], y[~is_test], y[is_test]


def evaluate(name, X_train, X_test, y_train, y_test):
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
    print(f"  {name:<28} ROC-AUC = {auc:.4f}  "
          f"(train={len(X_train):>6}, test={len(X_test):>6})")
    return pipe, auc


def main():
    df = prepare_dataset('../data/matches.csv', '../data/deliveries.csv')

    X = df[FEATURE_COLUMNS]
    y = df['result']
    groups = df['match_id']

    print("Comparing split strategies (same model, same features):\n")

    # 1) Leaky baseline: random split on individual balls.
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    evaluate("row-random (LEAKY)", Xr_tr, Xr_te, yr_tr, yr_te)

    # 2) Group-random: whole matches held out (no ball leaks across split).
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    evaluate(
        "match-grouped",
        X.iloc[tr_idx], X.iloc[te_idx], y.iloc[tr_idx], y.iloc[te_idx],
    )

    # 3) Time-based: train on past seasons, test on future seasons.
    is_test = df['season'] >= TEST_SEASON_START
    pipe, _ = evaluate(
        f"time-based (test >= {TEST_SEASON_START})",
        X[~is_test], X[is_test], y[~is_test], y[is_test],
    )

    print(
        "\nThe row-random number is optimistic because it leaks balls from the "
        "same\nmatch into train and test. The time-based split is the honest, "
        "deployment-\nrealistic number -- that's the model we ship.\n"
    )

    # Ship the time-based model (already fit on 2008-2017).
    pickle.dump(pipe, open('../pipe.pkl', 'wb'))
    print("Saved ../pipe.pkl (trained on seasons < %d)" % TEST_SEASON_START)


if __name__ == '__main__':
    main()
