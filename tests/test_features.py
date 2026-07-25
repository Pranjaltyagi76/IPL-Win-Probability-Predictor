"""Data-integrity tests for the prepared training dataset."""

import numpy as np

from feature_engineering import FEATURE_COLUMNS


def test_runs_left_never_negative(dataset):
    # A chase can end level (0) but never at a negative target.
    assert (dataset["runs_left"] >= 0).all()


def test_balls_left_in_valid_range(dataset):
    assert (dataset["balls_left"] > 0).all()
    assert (dataset["balls_left"] <= 120).all()


def test_wickets_between_zero_and_ten(dataset):
    assert dataset["wickets"].between(0, 10).all()


def test_result_is_binary(dataset):
    assert set(dataset["result"].unique()) <= {0, 1}


def test_no_missing_or_infinite_features(dataset):
    features = dataset[FEATURE_COLUMNS]
    assert not features.isna().any().any()
    numeric = features.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy()).all()
