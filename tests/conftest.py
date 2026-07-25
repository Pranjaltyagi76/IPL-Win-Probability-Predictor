"""Shared pytest fixtures.

Adds src/ to the import path and exposes the trained pipeline and the prepared
dataset once per test session, so individual tests stay fast and focused.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from predict import load_model  # noqa: E402
from feature_engineering import prepare_dataset  # noqa: E402


@pytest.fixture(scope="session")
def pipe():
    """The trained pipeline (auto-trained on first use if missing)."""
    return load_model()


@pytest.fixture(scope="session")
def dataset():
    """The prepared, leakage-free training dataset."""
    return prepare_dataset(
        os.path.join(ROOT, "data", "matches.csv"),
        os.path.join(ROOT, "data", "deliveries.csv"),
    )
