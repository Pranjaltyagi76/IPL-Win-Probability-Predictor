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


# Human-readable names for the numeric features.
_PRETTY_NUMERIC = {
    "runs_left": "Runs left",
    "balls_left": "Balls left",
    "wickets": "Wickets in hand",
    "target": "Target",
    "crr": "Current run rate",
    "rrr": "Required run rate",
}


def _prettify(name):
    """Turn a transformed feature name into something a human can read."""
    if name.startswith("scale__"):
        return _PRETTY_NUMERIC.get(name[7:], name[7:])
    if name.startswith("ohe__"):
        body = name[5:]
        # Field names themselves contain underscores (e.g. batting_team), so
        # match the known field prefix rather than splitting on the first "_".
        for field, label in (("batting_team", "Batting team"),
                             ("bowling_team", "Bowling team"),
                             ("city", "City")):
            if body.startswith(field + "_"):
                return f"{label} = {body[len(field) + 1:]}"
        return body
    return name


def explain(pipe, features_row, top_n=6):
    """Break a single prediction into per-feature log-odds contributions.

    For a linear model each feature's contribution is coefficient * (encoded
    value), so this is an exact decomposition, not an approximation. Returns
    (label, contribution) pairs sorted by magnitude; positive pushes the win
    probability up, negative pushes it down.
    """
    pre = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]

    encoded = pre.transform(features_row[FEATURE_COLUMNS])
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()

    names = pre.get_feature_names_out()
    contributions = model.coef_[0] * encoded[0]

    pairs = [
        (_prettify(n), float(c))
        for n, c in zip(names, contributions)
        if abs(c) > 1e-9
    ]
    pairs.sort(key=lambda pair: abs(pair[1]), reverse=True)
    return pairs[:top_n]
