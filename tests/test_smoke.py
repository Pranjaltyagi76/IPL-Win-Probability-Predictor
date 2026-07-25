"""Smoke tests: the model loads and the dataset builds."""

from predict import build_features, win_probability


def test_model_loads_and_predicts(pipe):
    features = build_features(
        "Mumbai Indians", "Chennai Super Kings", "Mumbai",
        runs_left=40, balls_left=30, wickets=6, target=180, crr=7.5, rrr=8.0,
    )
    prob, status = win_probability(pipe, features)
    assert status is None
    assert 0.0 <= prob <= 1.0


def test_dataset_is_non_empty(dataset):
    assert len(dataset) > 0
    assert "result" in dataset.columns
