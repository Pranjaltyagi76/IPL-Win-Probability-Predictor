"""Evaluate the win-probability model as a *probability* model, not a classifier.

Because we output a win probability, accuracy is the wrong lens. This script:

  1. Compares Logistic Regression vs Gradient Boosting on the honest
     time-based split (ROC-AUC = ranking quality, Brier = probability quality).
  2. Wraps each model in CalibratedClassifierCV and shows the Brier improvement.
  3. Saves a reliability (calibration) diagram to reports/calibration.png.
  4. Reports per-phase Brier score (powerplay / middle / death overs) to show
     where the model is weakest.

Run from anywhere:  python src/evaluate.py
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless: just write the PNG
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss

from feature_engineering import prepare_dataset
from train_model import (
    build_pipeline, time_split, TEST_SEASON_START,
    DEFAULT_MATCHES, DEFAULT_DELIVERIES,
)

# Resolved relative to this file so the script runs from any directory.
REPORTS_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "reports")


def phase(balls_left):
    balls_bowled = 120 - balls_left
    if balls_bowled <= 36:
        return "powerplay (ov 1-6)"
    if balls_bowled <= 90:
        return "middle (ov 7-15)"
    return "death (ov 16-20)"


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df = prepare_dataset(DEFAULT_MATCHES, DEFAULT_DELIVERIES)
    X_train, X_test, y_train, y_test = time_split(df)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
    }

    print(f"Time-based split (train seasons < {TEST_SEASON_START}, "
          f"test >= {TEST_SEASON_START})\n")
    header = f"{'Model':<26}{'ROC-AUC':>10}{'Brier':>10}{'Brier(cal)':>12}"
    print(header)
    print("-" * len(header))

    curves = {}
    for name, est in models.items():
        pipe = build_pipeline(est)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, proba)
        brier = brier_score_loss(y_test, proba)

        # Calibrate on top of the fitted pipeline (isotonic, held-out CV).
        cal = CalibratedClassifierCV(
            build_pipeline(est), method="isotonic", cv=5
        )
        cal.fit(X_train, y_train)
        proba_cal = cal.predict_proba(X_test)[:, 1]
        brier_cal = brier_score_loss(y_test, proba_cal)

        print(f"{name:<26}{auc:>10.4f}{brier:>10.4f}{brier_cal:>12.4f}")
        curves[name] = proba

    # --- Reliability diagram -------------------------------------------------
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k:", label="perfectly calibrated")
    for name, proba in curves.items():
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)
        plt.plot(mean_pred, frac_pos, marker="o", label=name)
    plt.xlabel("Predicted win probability")
    plt.ylabel("Observed win frequency")
    plt.title("Reliability diagram (time-based test set)")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(REPORTS_DIR, "calibration.png")
    plt.savefig(out, dpi=120)
    print(f"\nSaved reliability diagram -> {out}")

    # --- Per-phase Brier (Logistic Regression) -------------------------------
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    phases = X_test["balls_left"].map(phase)

    print("\nPer-phase Brier score (Logistic Regression, lower = better):")
    for ph in ["powerplay (ov 1-6)", "middle (ov 7-15)", "death (ov 16-20)"]:
        mask = (phases == ph).values
        if mask.sum():
            b = brier_score_loss(y_test[mask], proba[mask])
            print(f"  {ph:<20} Brier = {b:.4f}  (n={int(mask.sum())})")


if __name__ == '__main__':
    main()
