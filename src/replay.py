"""Reconstruct a real chase ball by ball and score each state with the model.

This powers the "match replay" feature: pick a historical match and watch the
win probability move across the whole second innings, the way broadcast
graphics do. The per-ball feature formulas mirror feature_engineering exactly,
so the replay is scored on the same inputs the model was trained on.

Reads the canonical dataset produced by build_dataset.py.
"""

import pandas as pd

from feature_engineering import FEATURE_COLUMNS

# Punjab Kings vs Royal Challengers Bengaluru, 2024. Chosen by ranking every
# chase on how often its win-probability curve crosses the 50% line: this one
# crosses 17 times, more than any other match in the dataset, and swings from
# 19% to a win. It opens the replay by default.
FEATURED_MATCH_ID = 1422124


def load_data(matches_path, deliveries_path):
    """Load the match and delivery tables once, for reuse across replays."""
    return pd.read_csv(matches_path), pd.read_csv(deliveries_path)


def list_matches(matches, deliveries):
    """Selectable, replayable matches with a human-readable label.

    Restricted to matches with an outright result, an unrevised target and a
    second innings to replay — the same conditions the model was trained on.
    Returned newest-season first.
    """
    m = matches[
        matches['winner'].notna()
        & (matches['dls'] == 0)
        & matches['target'].notna()
    ].copy()

    chased = set(deliveries[deliveries['innings'] == 2]['match_id'].unique())
    m = m[m['match_id'].isin(chased)]

    m['label'] = (
        m['season'].astype(str) + '  ·  '
        + m['team1'] + ' vs ' + m['team2']
        + '  ·  ' + m['city']
        + '  —  ' + m['winner'] + ' won'
    )

    out = m[['match_id', 'label', 'season']].sort_values(
        ['season', 'match_id'], ascending=[False, True]
    )
    return out.reset_index(drop=True)


def win_probability_curve(pipe, match_id, matches, deliveries):
    """Return a per-ball DataFrame of the second-innings chase for one match.

    Columns: ball_no, ball, runs_this_ball, is_wicket, is_boundary,
    current_score, runs_left, balls_left, wickets, crr, rrr, win_prob.
    win_prob is the batting team's probability, with terminal states forced.
    """
    match = matches[matches['match_id'] == match_id].iloc[0]

    inn = deliveries[
        (deliveries['match_id'] == match_id) & (deliveries['innings'] == 2)
    ].sort_values('ball').reset_index(drop=True)
    if inn.empty:
        return pd.DataFrame()

    inn = inn.copy()
    target = int(match['target'])

    inn['ball_no'] = range(1, len(inn) + 1)
    inn['current_score'] = inn['total_runs'].cumsum()
    # Legal deliveries only: wides and no-balls don't consume a ball.
    inn['balls_bowled'] = inn['is_legal'].cumsum()
    inn['balls_left'] = 120 - inn['balls_bowled']
    inn['runs_left'] = target - inn['current_score']
    inn['wickets'] = 10 - inn['is_wicket'].cumsum()
    inn['is_boundary'] = inn['batsman_runs'].isin([4, 6]).astype(int)
    inn['runs_this_ball'] = inn['total_runs']

    # An opening wide leaves balls_bowled at 0; clip both divisors so the run
    # rates stay finite on the first delivery and the last.
    inn['crr'] = (inn['current_score'] * 6) / inn['balls_bowled'].clip(lower=1)
    inn['rrr'] = (inn['runs_left'] * 6) / inn['balls_left'].clip(lower=1)

    inn['target'] = target
    inn['city'] = match['city']

    # Score every ball in one batched call, then force terminal states.
    proba = pipe.predict_proba(inn[FEATURE_COLUMNS])[:, 1].copy()
    proba[inn['runs_left'].values <= 0] = 1.0          # target reached
    proba[inn['wickets'].values == 0] = 0.0            # all out
    inn['win_prob'] = proba

    cols = ['ball_no', 'ball', 'runs_this_ball', 'is_wicket', 'is_boundary',
            'current_score', 'runs_left', 'balls_left', 'wickets',
            'crr', 'rrr', 'win_prob']
    return inn[cols]
