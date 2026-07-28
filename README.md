# IPL Win Probability Predictor

[![CI](https://github.com/Pranjaltyagi76/IPL-Win-Probability-Predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/Pranjaltyagi76/IPL-Win-Probability-Predictor/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)

Estimate the batting team's chance of winning at any point of an IPL second-innings
chase, and replay how that probability moved across a real match — ball by ball,
the way broadcast win-probability graphics do.

---

## Contents

- [What this does](#what-this-does)
- [Quickstart](#quickstart)
- [Features](#features)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Tests and CI](#tests-and-ci)
- [Dataset](#dataset)
- [Author](#author)

---

## What this does

Given a live match state — teams, venue, target, runs scored, balls remaining and
wickets in hand — the model returns the probability that the chasing side wins.

Two things separate this from a standard "fit a classifier, show a number" project:

1. **The evaluation is honest.** Splitting ball-by-ball rows at random leaks the
   same match into train and test and inflates the score. This project splits by
   season instead, and reports the lower, deployment-realistic number.
2. **Every prediction is explainable.** The app breaks each estimate into the
   individual factors pushing the win probability up or down.

## Quickstart

```bash
git clone https://github.com/Pranjaltyagi76/IPL-Win-Probability-Predictor.git
cd IPL-Win-Probability-Predictor
pip install -r requirements.txt
streamlit run app.py
```

The trained model (`pipe.pkl`) is not committed — it is a generated artifact.
The app trains it from the bundled data the first time it starts, so the clone
above is all you need.

Optional commands, all runnable from the repository root:

```bash
python src/train_model.py   # retrain, and print the split comparison
python src/evaluate.py      # ROC-AUC, Brier, calibration, per-phase scores
pytest -q                   # run the test suite
```

## Features

- **Match replay** — pick any of 625 historical chases and see the win-probability
  curve across the whole innings, annotated with wickets, fours and sixes.
- **Manual prediction** — enter a hypothetical match state and get an estimate.
- **Per-prediction explanations** — the top factors helping and hurting the
  batting team, as plain English plus a contribution chart.
- **Terminal-state handling** — an all-out side is shown 0%, a completed chase
  100%, instead of whatever the model happens to output.
- **Calibration reporting** — Brier score and reliability diagrams, not just
  ROC-AUC.

## How it works

### Feature engineering

`src/feature_engineering.py` merges the match and ball-by-ball tables, filters to
the eight canonical franchises and non-DLS matches, and derives the match state at
every delivery of the second innings:

| Feature | Meaning |
| --- | --- |
| `runs_left` | Runs still required |
| `balls_left` | Balls remaining in the innings |
| `wickets` | Wickets in hand |
| `target` | First-innings run total |
| `crr` | Current run rate |
| `rrr` | Required run rate |
| `batting_team`, `bowling_team`, `city` | Categorical context |

The label is whether the batting team went on to win.

### Model

A scikit-learn `Pipeline`: one-hot encoding for the categorical columns, standard
scaling for the numeric ones, and logistic regression on top. The pipeline is
serialised to `pipe.pkl`, which is generated rather than committed — the app
trains it automatically on first run.

## Project structure

```text
IPL-Win-Probability-Predictor/
│
├── app.py                     # Streamlit app (replay + manual modes)
├── requirements.txt
│
├── data/
│   ├── matches.csv
│   └── deliveries.csv
│
├── src/
│   ├── feature_engineering.py # Raw CSVs -> per-ball training rows
│   ├── train_model.py         # Split comparison + trains the shipped model
│   ├── predict.py             # Inference, terminal-state guards, explanations
│   ├── replay.py              # Ball-by-ball win-probability curves
│   └── evaluate.py            # ROC-AUC, Brier, calibration, per-phase scores
│
├── tests/                     # pytest suite
└── .github/workflows/ci.yml   # Runs the suite on every push and PR
```

## Tests and CI

A pytest suite covers data integrity (no negative `runs_left`, valid ranges, no
missing or infinite features), prediction validity (probabilities stay in
`[0, 1]`), cricketing sanity (win probability rises as runs needed fall and as
wickets in hand increase), and the terminal-state guards.

GitHub Actions runs the suite on every push and pull request to `main`.

## Dataset

IPL ball-by-ball data (2008–2019) from Kaggle:
<https://www.kaggle.com/datasets/ramjidoolla/ipl-data-set>

## Author

**Pranjal Tyagi** — B.Tech CSE (AI & DS)

If this project was useful to you, consider giving it a ⭐
