"""Behavioural tests for win-probability predictions."""

import itertools

from predict import build_features, win_probability


def _prob(pipe, runs_left, balls_left, wickets, target=180):
    overs_left = balls_left / 6
    balls_bowled = 120 - balls_left
    crr = ((target - runs_left) * 6 / balls_bowled) if balls_bowled else 0.0
    rrr = (runs_left * 6 / balls_left) if balls_left else 0.0
    features = build_features(
        "Mumbai Indians", "Chennai Super Kings", "Mumbai",
        runs_left, balls_left, wickets, target, crr, rrr,
    )
    prob, _ = win_probability(pipe, features)
    return prob


def test_probability_always_in_unit_range(pipe):
    grid = itertools.product(
        [5, 30, 60, 120, 200],   # runs_left
        [6, 30, 60, 90, 119],    # balls_left
        [1, 3, 6, 10],           # wickets
    )
    for runs_left, balls_left, wickets in grid:
        prob = _prob(pipe, runs_left, balls_left, wickets)
        assert 0.0 <= prob <= 1.0, (runs_left, balls_left, wickets, prob)
