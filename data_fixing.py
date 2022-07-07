import pandas as pd


def fix_issue_1(df: pd.DataFrame) -> pd.DataFrame:
    """In several matches played by Lazio during the season 2007-08, the goalkeeper Marco Ballotta is missing in
    the lineup, resulting in data shifting and NULL values in column away_substitute_12."""

    def fix_issue_1_home(df: pd.DataFrame, missing_goalkeeper: str):
        """Fix matches where LAZIO is the home team"""
        to_fix_mask_home = (df['away_substitute_12'].isnull()) & (df['home_team'] == 'LAZIO')
        for index in df[to_fix_mask_home].index:
            # shift substitutes
            df.loc[index, 'home_substitute_1':'home_substitute_7'] = \
                df.loc[index, 'home_player_11':'home_substitute_6'].values
            # shift players
            df.loc[index, 'home_player_2':'home_player_11'] = \
                df.loc[index, 'home_player_1':'home_player_10'].values
            # set goalkeeper
            df.loc[index, 'home_player_1'] = missing_goalkeeper
            # right shift all away team data by one column
            df.loc[index, 'away_coach':'away_substitute_12'] = \
                df.loc[index, 'home_substitute_12':'away_substitute_11'].values
            df.loc[index, 'home_substitute_12'] = '-'
        return df

    def fix_issue_1_away(df: pd.DataFrame, missing_goalkeeper: str):
        """Fix matches where LAZIO is the away team"""
        to_fix_mask_away = (df['away_substitute_12'].isnull()) & (df['away_team'] == 'LAZIO')
        for index in df[to_fix_mask_away].index:
            # shift substitutes
            df.loc[index, 'away_substitute_1':'away_substitute_7'] = \
                df.loc[index, 'away_player_11':'away_substitute_6'].values
            # shift players
            df.loc[index, 'away_player_2':'away_player_11'] = \
                df.loc[index, 'away_player_1':'away_player_10'].values
            # set goalkeeper
            df.loc[index, 'away_player_1'] = missing_goalkeeper
            df.loc[index, 'away_substitute_12'] = '-'
        return df

    missing_goalkeeper = 'Marco Ballotta'
    df = fix_issue_1_home(df, missing_goalkeeper)
    df = fix_issue_1_away(df, missing_goalkeeper)
    still_to_fix_mask = (df['away_substitute_12'].isnull()) & (
            (df['home_team'] == 'LAZIO') | (df['away_team'] == 'LAZIO'))
    n_still_to_fix = len(df[still_to_fix_mask])
    if n_still_to_fix == 0:
        print('issue #1 FIXED')
    else:
        print('issue #1 NOT FIXED')
    return df


def fix_issue_2(df: pd.DataFrame) -> pd.DataFrame:
    """In round 37, season 2005-06, MESSINA-EMPOLI was suspended at 89’ with score 1-2.
    Then winner of the game was decided to be EMPOLI with a ‘by forfeit’ victory, i.e. 0-3 for EMPOLI.
    We keep the on-pitch score with all the data as the game was about to end when it was suspended."""
    to_fix_mask = (df['date'] == '<sus>')
    for index in df[to_fix_mask].index:
        df.loc[index, 'date'] = '07/05/2006'
        df.loc[index, 'time'] = '15:00'
        df.loc[index, 'referee'] = 'DIEGO PRESCHERN'
        df.loc[index, 'home_team'] = 'MESSINA'
        df.loc[index, 'away_team'] = 'EMPOLI'
        df.loc[index, 'home_score'] = '1'
        df.loc[index, 'away_score'] = '2'
    still_to_fix_mask = (df['date'] == '<sus>')
    n_still_to_fix = len(df[still_to_fix_mask])
    if n_still_to_fix == 0:
        print('issue #2 FIXED')
    else:
        print('issue #2 NOT FIXED')
    return df


def fix_issue_3(df: pd.DataFrame) -> pd.DataFrame:
    """In round 4, season 2012-13, CAGLIARI-ROMA was not played and a victory by forfeit was given to ROMA.
    We will discard this match as it does not bring information."""
    to_fix_mask = (df['date'] == '-') & (df['round'] == 4) & (df['season'] == '2012-13')
    df.drop(index=df[to_fix_mask].index.tolist(), inplace=True)
    still_to_fix_mask = (df['date'] == '-') & (df['round'] == 4) & (df['season'] == '2012-13')
    n_still_to_fix = len(df[still_to_fix_mask])
    if n_still_to_fix == 0:
        print('issue #3 FIXED')
    else:
        print('issue #3 NOT FIXED')
    return df
