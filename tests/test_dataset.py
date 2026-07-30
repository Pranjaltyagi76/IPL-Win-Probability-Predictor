"""Schema and coverage checks for the canonical Cricsheet dataset.

These guard the data itself rather than the model. They would have caught the
blank-city bug that silently dropped 48 matches, so they assert coverage is
even across seasons rather than merely non-empty.
"""

import os

import pandas as pd
import pytest

from train_model import DEFAULT_MATCHES, DEFAULT_DELIVERIES

MATCH_COLUMNS = {
    "match_id", "season", "city", "venue", "team1", "team2",
    "winner", "target", "dls", "player_of_match",
}
DELIVERY_COLUMNS = {
    "match_id", "innings", "ball", "batting_team", "bowling_team",
    "batsman_runs", "total_runs", "is_legal", "is_wicket",
}


@pytest.fixture(scope="module")
def matches():
    return pd.read_csv(DEFAULT_MATCHES)


@pytest.fixture(scope="module")
def deliveries():
    return pd.read_csv(DEFAULT_DELIVERIES)


def test_data_files_exist():
    assert os.path.exists(DEFAULT_MATCHES)
    assert os.path.exists(DEFAULT_DELIVERIES)


def test_expected_columns_present(matches, deliveries):
    assert MATCH_COLUMNS <= set(matches.columns)
    assert DELIVERY_COLUMNS <= set(deliveries.columns)


def test_every_match_has_a_city(matches):
    # Blank cities previously survived parsing and then dropped whole matches
    # out of training when the feature frame was cleaned.
    assert matches["city"].isna().sum() == 0


def test_seasons_are_contiguous(matches):
    seasons = sorted(matches["season"].unique())
    assert seasons == list(range(seasons[0], seasons[-1] + 1))
    assert seasons[0] == 2008


def test_flags_are_binary(deliveries):
    assert set(deliveries["is_legal"].unique()) <= {0, 1}
    assert set(deliveries["is_wicket"].unique()) <= {0, 1}


def test_no_team_plays_itself(matches):
    assert (matches["team1"] == matches["team2"]).sum() == 0


def test_full_innings_has_120_legal_balls(deliveries):
    """A completed 20-over innings must contain exactly 120 legal deliveries.

    This is the check that pins the wide/no-ball handling: counting raw rows
    instead of legal ones would overshoot.
    """
    legal = (
        deliveries[deliveries["innings"] == 1]
        .groupby("match_id")["is_legal"].sum()
    )
    # Innings ending early (all out, or a chase completed) fall short, so the
    # floor is not fixed — but the common case must land exactly on 120.
    assert (legal == 120).sum() > 500

    # Exactly one innings exceeds 120: match 419155 (2010, CSK v DD) contains
    # an over of seven legal deliveries, an umpiring miscount that the source
    # records faithfully. Anything beyond that would mean wides or no-balls
    # are being counted as legal.
    over_limit = legal[legal > 120]
    assert set(over_limit.index) == {419155}
    assert over_limit.max() == 121


def test_season_coverage_is_even(dataset):
    """Every season should contribute a comparable number of training rows.

    The blank-city bug left 2020 with 2,415 rows against a ~6,500 norm; this
    fails loudly if anything similar silently drops matches again.
    """
    per_season = dataset.groupby("season").size()
    assert per_season.min() > 0.5 * per_season.median()
