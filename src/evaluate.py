"""Evaluate the win-probability model as a probability model, not a classifier.

The app shows a percentage, so ranking quality alone is not enough: a model can
order chases correctly (good ROC-AUC) and still be badly miscalibrated, e.g.
saying 80% for situations that are actually won 60% of the time. Brier score
measures that -- it is the mean squared error of the predicted probability,
so lower is better.

Run from the src/ directory:  python evaluate.py
"""

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

from feature_engineering import prepare_dataset
from train_model import build_pipeline, time_split, TEST_SEASON_START

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
}


def main():
    df = prepare_dataset('../data/matches.csv', '../data/deliveries.csv')
    X_train, X_test, y_train, y_test = time_split(df)

    print(f"Time-based split (train < {TEST_SEASON_START}, "
          f"test >= {TEST_SEASON_START})\n")

    print(f"{'Model':<26}{'ROC-AUC':>10}{'Brier':>10}")
    print("-" * 46)

    for name, estimator in MODELS.items():
        pipe = build_pipeline(estimator)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]

        print(f"{name:<26}"
              f"{roc_auc_score(y_test, proba):>10.4f}"
              f"{brier_score_loss(y_test, proba):>10.4f}")


if __name__ == '__main__':
    main()
