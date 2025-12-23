import pandas as pd

def prepare_dataset(matches_path, deliveries_path):

    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)

    # First innings target
    first_innings = (
        deliveries
        .groupby(['match_id','inning'])['total_runs']
        .sum()
        .reset_index()
    )
    first_innings = first_innings[first_innings['inning'] == 1]
    first_innings.rename(columns={'total_runs':'target'}, inplace=True)

    matches = matches.merge(
        first_innings[['match_id','target']],
        left_on='id',
        right_on='match_id'
    )

    team_map = {
        'Delhi Daredevils': 'Delhi Capitals',
        'Deccan Chargers': 'Sunrisers Hyderabad'
    }
    matches.replace(team_map, inplace=True)
    deliveries.replace(team_map, inplace=True)

    valid_teams = [
        'Sunrisers Hyderabad','Mumbai Indians','Royal Challengers Bangalore',
        'Kolkata Knight Riders','Kings XI Punjab','Chennai Super Kings',
        'Rajasthan Royals','Delhi Capitals'
    ]

    matches = matches[
        (matches['team1'].isin(valid_teams)) &
        (matches['team2'].isin(valid_teams)) &
        (matches['dl_applied'] == 0)
    ]

    matches = matches[['match_id','city','winner','target']]
    df = matches.merge(deliveries, on='match_id')
    df = df[df['inning'] == 2].copy()

    df['current_score'] = df.groupby('match_id')['total_runs'].cumsum()
    df['balls_bowled'] = (df['over'] - 1) * 6 + df['ball']
    df['balls_left'] = 120 - df['balls_bowled']
    df['runs_left'] = df['target'] - df['current_score']

    df = df[(df['balls_left'] > 0) & (df['runs_left'] >= 0)]

    df['is_wicket'] = df['player_dismissed'].notna().astype(int)
    df['wickets_fallen'] = df.groupby('match_id')['is_wicket'].cumsum()
    df['wickets'] = 10 - df['wickets_fallen']

    df['crr'] = (df['current_score'] * 6) / df['balls_bowled']
    df['rrr'] = (df['runs_left'] * 6) / df['balls_left']

    df['result'] = (df['batting_team'] == df['winner']).astype(int)

    final_df = df[
        ['batting_team','bowling_team','city',
         'runs_left','balls_left','wickets',
         'target','crr','rrr','result']
    ].dropna()

    return final_df
