# IPL Win Probability Predictor

[![CI](https://github.com/Pranjaltyagi76/IPL-Win-Probability-Predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/Pranjaltyagi76/IPL-Win-Probability-Predictor/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)

Estimate the batting team's chance of winning at any point of an IPL second-innings
chase, and replay how that probability moved across a real match — ball by ball,
the way broadcast win-probability graphics do.

![Win probability across the 2008 IPL final](reports/replay_2008_final.png)

*The 2008 final, replayed by the model. Rajasthan Royals dip to 31% after early
wickets, recover, and win off the last ball — the curve crosses the 50% line 14
times. This is the app's default view.*

---

## Contents

- [What this does](#what-this-does)
- [Quickstart](#quickstart)
- [Features](#features)
- [How it works](#how-it-works)
- [Results](#results)
- [Why the reported accuracy dropped](#why-the-reported-accuracy-dropped)
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

## Results

All figures below are on the season-based test set (2018–2019), produced by
`python src/evaluate.py`.

Because the output is a probability, ROC-AUC alone is not enough — it only
measures ranking. **Brier score** (mean squared error of the predicted
probability, lower is better) measures whether the numbers themselves are
trustworthy.

| Model | ROC-AUC | Brier | Brier (calibrated) |
| --- | --- | --- | --- |
| **Logistic Regression** | **0.8228** | **0.1848** | **0.1771** |
| Gradient Boosting | 0.7836 | 0.2353 | 0.1992 |

Logistic regression wins on both ranking and probability quality, so the simpler
and more interpretable model is also the better one here — that is why it ships.
Isotonic calibration improves it further (0.1848 → 0.1771).

### Where the model is weakest

| Innings phase | Brier |
| --- | --- |
| Powerplay (ov 1–6) | 0.2409 |
| Middle (ov 7–15) | 0.1729 |
| Death (ov 16–20) | 0.1268 |

The model is least reliable early and sharpest at the death, which is expected:
genuine uncertainty collapses as a chase runs out of balls.

### Calibration reveals a distribution shift

![Reliability diagram](reports/calibration.png)

The reliability curve sits **above** the diagonal at the low end — when the model
says 15%, chasing teams actually won about 44% of the time. It is systematically
under-rating the chasing side.

The season split explains why. Chases succeeded **53.3%** of the time in the
training seasons (2008–2017) but **57.3%** in the test seasons (2018–2019), so a
model fitted on the earlier era carries its pessimism forward.

This is the point of a time-based split: a random split would have hidden the
shift by mixing both eras into training.

## Why the reported accuracy dropped

An earlier version of this project reported **ROC-AUC ≈ 0.887**. It now reports
**0.823**. Nothing about the model got worse — the earlier number was measured
wrongly, and this is the correction.

The dataset has one row per ball. Splitting those rows at random puts balls from
the *same match* on both sides of the split, and consecutive balls of a chase are
nearly identical: 42 needed off 30 with 5 wickets, then 40 off 29 with 5 wickets.
The model can effectively recognise matches it trained on, so the test score
measures memorisation as much as skill.

Running the same model and features under three different splits shows the size
of the effect:

| Split | ROC-AUC | What it measures |
| --- | --- | --- |
| Row-random (the leaky one) | 0.8872 | Balls from one match appear in train *and* test |
| Match-grouped | 0.8416 | Whole matches held out, seasons mixed |
| **Season-based (shipped)** | **0.8228** | Train 2008–2017, test 2018–2019 |

The season split is the one that matches how the model is actually used —
predicting matches that had not happened when it was trained — so that is the
number this project reports. `python src/train_model.py` prints all three.

**The 6.4-point gap is the cost of the leak.** A model chosen or tuned against
the 0.887 figure would have been optimised against a number that does not exist
in deployment.

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
