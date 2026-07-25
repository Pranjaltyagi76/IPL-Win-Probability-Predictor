"""Reusable inference helpers.

Both the Streamlit form and the match-replay feature need to turn a match
situation into a win probability. Keeping that logic here means the terminal-
state rules (all out -> 0, target reached -> 1) are defined exactly once.
"""

import os
import pickle

import pandas as pd

from feature_engineering import FEATURE_COLUMNS

# pipe.pkl lives at the repo root, one level above this file.
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "pipe.pkl"
)


def load_model(path=DEFAULT_MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_features(batting_team, bowling_team, city,
                   runs_left, balls_left, wickets, target, crr, rrr):
    """Assemble a single-row feature frame in the exact training column order."""
    return pd.DataFrame(
        [[batting_team, bowling_team, city,
          runs_left, balls_left, wickets, target, crr, rrr]],
        columns=FEATURE_COLUMNS,
    )


def win_probability(pipe, features_row):
    """Win probability for the batting team.

    Returns (probability, message). Terminal match states are certainties, so
    they are resolved here rather than handed to the model, which is trained on
    live ball-by-ball rows and has no concept of a finished chase.
    """
    runs_left = int(features_row["runs_left"].iloc[0])
    wickets = int(features_row["wickets"].iloc[0])

    if runs_left <= 0:
        return 1.0, "Target already reached — chase won."
    if wickets == 0:
        return 0.0, "Batting team is all out — chase lost."

    prob = float(pipe.predict_proba(features_row[FEATURE_COLUMNS])[0][1])
    return prob, None
