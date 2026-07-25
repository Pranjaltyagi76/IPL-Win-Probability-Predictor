"""Reconstruct a real chase ball by ball and score each state with the model.

This powers the "match replay" feature: pick a historical match and watch the
win probability move across the whole second innings, the way broadcast
graphics do. The per-ball feature formulas mirror feature_engineering exactly,
so the replay is scored on the same inputs the model was trained on.
"""

import pandas as pd

from feature_engineering import FEATURE_COLUMNS

# Same normalisation the training pipeline applies, so team names match the
# categories the model learned.
TEAM_MAP = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Deccan Chargers': 'Sunrisers Hyderabad',
}

VALID_TEAMS = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals',
]


def load_data(matches_path='data/matches.csv',
              deliveries_path='data/deliveries.csv'):
    """Load the raw match and delivery tables once, for reuse across replays."""
    return pd.read_csv(matches_path), pd.read_csv(deliveries_path)


def list_matches(matches, deliveries):
    """Selectable, replayable matches with a human-readable label.

    Restricted to the eight canonical teams, non-DLS results, and matches that
    actually have a second innings to replay. Returned newest-season first.
    """
    m = matches.replace(TEAM_MAP)
    m = m[
        m['team1'].isin(VALID_TEAMS)
        & m['team2'].isin(VALID_TEAMS)
        & (m['dl_applied'] == 0)
    ].copy()

    chased = set(deliveries[deliveries['inning'] == 2]['match_id'].unique())
    m = m[m['id'].isin(chased)]

    m['season_year'] = m['Season'].str.extract(r'(\d{4})').astype(int)
    m['winner'] = m['winner'].replace(TEAM_MAP)
    m['label'] = (
        m['season_year'].astype(str) + '  ·  '
        + m['team1'] + ' vs ' + m['team2']
        + '  ·  ' + m['city'].fillna('Neutral venue')
        + '  —  ' + m['winner'].fillna('No result') + ' won'
    )

    out = m[['id', 'label', 'season_year']].sort_values(
        ['season_year', 'id'], ascending=[False, True]
    )
    return out.reset_index(drop=True)


def win_probability_curve(pipe, match_id, matches, deliveries):
    """Return a per-ball DataFrame of the second-innings chase for one match.

    Columns: ball_no, over, ball, runs_this_ball, is_wicket, is_boundary,
    current_score, runs_left, balls_left, wickets, crr, rrr, win_prob.
    win_prob is the batting team's probability, with terminal states forced.
    """
    match = matches[matches['id'] == match_id].iloc[0]

    d = deliveries[deliveries['match_id'] == match_id].copy()
    d.replace(TEAM_MAP, inplace=True)

    # target = first-innings run total, matching how training defines it.
    target = int(d[d['inning'] == 1]['total_runs'].sum())

    inn = d[d['inning'] == 2].sort_values(['over', 'ball']).reset_index(drop=True)
    if inn.empty:
        return pd.DataFrame()

    inn['ball_no'] = range(1, len(inn) + 1)
    inn['current_score'] = inn['total_runs'].cumsum()
    inn['balls_bowled'] = (inn['over'] - 1) * 6 + inn['ball']
    inn['balls_left'] = 120 - inn['balls_bowled']
    inn['runs_left'] = target - inn['current_score']
    inn['is_wicket'] = inn['player_dismissed'].notna().astype(int)
    inn['wickets'] = 10 - inn['is_wicket'].cumsum()
    inn['is_boundary'] = inn['batsman_runs'].isin([4, 6]).astype(int)
    inn['runs_this_ball'] = inn['total_runs']

    inn['crr'] = (inn['current_score'] * 6) / inn['balls_bowled']
    # balls_left can reach 0 on the final ball; guard the division.
    inn['rrr'] = (inn['runs_left'] * 6) / inn['balls_left'].clip(lower=1)

    inn['target'] = target
    # Some matches have no recorded city (NaN). Left as a float NaN it makes the
    # whole city column non-string and breaks the one-hot encoder's unknown
    # check, so coerce to a string the model treats as an unseen category.
    inn['city'] = match['city'] if pd.notna(match['city']) else 'Unknown'

    # Score every ball in one batched call, then force terminal states.
    features = inn[FEATURE_COLUMNS]
    proba = pipe.predict_proba(features)[:, 1]
    proba = proba.copy()
    proba[inn['runs_left'].values <= 0] = 1.0          # target reached
    proba[inn['wickets'].values == 0] = 0.0            # all out
    inn['win_prob'] = proba

    cols = ['ball_no', 'over', 'ball', 'runs_this_ball', 'is_wicket',
            'is_boundary', 'current_score', 'runs_left', 'balls_left',
            'wickets', 'crr', 'rrr', 'win_prob']
    return inn[cols]
