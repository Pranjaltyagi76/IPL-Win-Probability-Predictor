"""Turn the canonical dataset into per-ball training rows.

Reads the files produced by build_dataset.py (2008-2026) and reconstructs the
match state at every delivery of a second-innings chase.

Only *live* states are kept. A chase with the target reached, no balls left or
no wickets left is already decided, so those rows carry no information the model
needs to learn — predict.py resolves them with rules instead. Training on them
would just teach the model to restate certainties.
"""

import pandas as pd


def prepare_dataset(matches_path, deliveries_path):

    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)

    # Keep matches with an outright result and an unrevised target:
    #   - no winner  -> abandoned, or tied and settled by a super over, so
    #                   "did the chase succeed" has no clean answer
    #   - dls == 1   -> rain-revised, the target moved mid-innings
    matches = matches[
        matches['winner'].notna()
        & (matches['dls'] == 0)
        & matches['target'].notna()
    ]

    # Innings 2 only: innings 3+ are super overs, which are a different game.
    chase = deliveries[deliveries['innings'] == 2]

    df = chase.merge(
        matches[['match_id', 'season', 'city', 'winner', 'target']],
        on='match_id',
    )
    df = df.sort_values(['match_id', 'ball']).copy()

    grouped = df.groupby('match_id')
    df['current_score'] = grouped['total_runs'].cumsum()
    # Wides and no-balls don't consume a delivery, so count legal balls only.
    df['balls_bowled'] = grouped['is_legal'].cumsum()
    df['wickets'] = 10 - grouped['is_wicket'].cumsum()

    df['balls_left'] = 120 - df['balls_bowled']
    df['runs_left'] = df['target'] - df['current_score']

    # A chase is live only while runs are still needed, balls remain and
    # wickets remain. balls_bowled > 0 also protects the run-rate division,
    # which would otherwise blow up if the innings opens with a wide.
    df = df[
        (df['runs_left'] > 0)
        & (df['balls_left'] > 0)
        & (df['wickets'] > 0)
        & (df['balls_bowled'] > 0)
    ]

    df['crr'] = (df['current_score'] * 6) / df['balls_bowled']
    df['rrr'] = (df['runs_left'] * 6) / df['balls_left']

    df['result'] = (df['batting_team'] == df['winner']).astype(int)

    # 'match_id' and 'season' are metadata, not model features. They let the
    # training script split by match or by season so that balls from the same
    # match cannot leak into both train and test.
    final_df = df[
        ['match_id', 'season',
         'batting_team', 'bowling_team', 'city',
         'runs_left', 'balls_left', 'wickets',
         'target', 'crr', 'rrr', 'result']
    ].dropna()

    return final_df


# Columns the model actually trains on (excludes the target).
# Kept here so training and inference can never drift out of sync.
FEATURE_COLUMNS = [
    'batting_team', 'bowling_team', 'city',
    'runs_left', 'balls_left', 'wickets',
    'target', 'crr', 'rrr'
]
