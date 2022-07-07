import pandas as pd

from MatchResult import MatchResult

INPUT_CSV_NAME = 'raw.csv'
OUTPUT_CSV_NAME = 'data.csv'
# add player names as features
INCLUDE_PLAYERS = True


def data_fixing(df: pd.DataFrame):
    """Through manual inspection of the raw dataset, several matches with issues or inconsistent data were detected.
    Let’s fix them. We will use another source of Serie A matches to compare with."""

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
        df.loc[361, 'date'] = '07/05/2006'
        df.loc[361, 'time'] = '15:00'
        df.loc[361, 'referee'] = 'DIEGO PRESCHERN'
        df.loc[361, 'home_team'] = 'MESSINA'
        df.loc[361, 'away_team'] = 'EMPOLI'
        df.loc[361, 'home_score'] = '1'
        df.loc[361, 'away_score'] = '2'
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

    print('===> Phase 1: DATA FIXING ')
    df = fix_issue_1(df)
    df = fix_issue_2(df)
    df = fix_issue_3(df)
    print('===> Phase 1: DONE')
    return df


def data_manipulation(df: pd.DataFrame):
    def convert_date_str_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
        df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
        return df

    def sort_by_date_column(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(by='date')
        df = df.reset_index(drop=True)
        return df

    def cast_str_values_to_int(df: pd.DataFrame) -> pd.DataFrame:
        df['round'] = df['round'].astype(int)
        df['home_score'] = df['home_score'].astype(int)
        df['away_score'] = df['away_score'].astype(int)
        return df

    def add_target_column(df: pd.DataFrame) -> pd.DataFrame:
        def get_match_result_from_score(home_score: int, away_score: int) -> MatchResult:
            if home_score == away_score:
                return MatchResult.draw
            if home_score > away_score:
                return MatchResult.home
            return MatchResult.away

        results = {'result': []}
        for index, row in df.iterrows():
            results['result'] += [get_match_result_from_score(row['home_score'], row['away_score']).name]
        df.insert(loc=df.columns.get_loc('home_score'), column='result', value=results['result'])
        return df

    def explode_datetime_values(df: pd.DataFrame) -> pd.DataFrame:
        def get_exploded_datetime_values(df: pd.DataFrame) -> dict:
            data = {'year': [], 'month': [], 'day': [], 'hour': []}
            df['time'] = pd.to_datetime(df['time'], format="%H:%M")
            data['year'] += df['date'].map(lambda val: val.year).tolist()
            data['month'] += df['date'].map(lambda val: val.month).tolist()
            data['day'] += df['date'].map(lambda val: val.day).tolist()
            data['hour'] += df['time'].map(lambda val: val.hour).tolist()
            return data

        def insert_exploded_datetime_values(df, exploded):
            df.insert(loc=df.columns.get_loc('time'), column='year', value=exploded['year'])
            df.insert(loc=df.columns.get_loc('time'), column='month', value=exploded['month'])
            df.insert(loc=df.columns.get_loc('time'), column='day', value=exploded['day'])
            df.insert(loc=df.columns.get_loc('time'), column='hour', value=exploded['hour'])
            return df

        exploded = get_exploded_datetime_values(df)
        return insert_exploded_datetime_values(df, exploded)

    def drop_date_cols(df: pd.DataFrame):
        df = df.drop('date', axis=1, inplace=False)
        df = df.drop('time', axis=1, inplace=False)
        return df

    print('===> Phase 2: DATA MANIPULATION ')
    df = convert_date_str_to_datetime(df)
    df = sort_by_date_column(df)
    df = cast_str_values_to_int(df)
    df = add_target_column(df)
    df = explode_datetime_values(df)
    df = drop_date_cols(df)
    df = df.dropna()
    print('===> Phase 2: DONE ')
    return df


def data_encoding(df: pd.DataFrame):
    def get_column_names_containing_str(df: pd.DataFrame, substring: str) -> list[str]:
        return df.loc[:, df.columns.str.contains(substring)].columns.values.tolist()

    def encode_seasons(df: pd.DataFrame):
        season2index = {'20' + f'{i + 5}'.zfill(2) + '-' + f'{i + 6}'.zfill(2): i for i in range(16)}
        df['season'] = df['season'].map(season2index)
        print(f'Seasons encoded. Shape: {df.shape}')
        return df

    def encode_players(df: pd.DataFrame):
        """We need to manually one-hot encode the names of the players because we have to consider all the player columns
            to construct a list containing all the unique values. Moreover, the number of substitutes is inconsistent across
            seasons, rounds and matches."""
        player_columns = get_column_names_containing_str(df, 'player')
        player_columns += get_column_names_containing_str(df, 'substitute')
        home_player_columns = get_column_names_containing_str(df, 'home_player')
        home_player_columns += get_column_names_containing_str(df, 'home_substitute')
        away_player_columns = get_column_names_containing_str(df, 'away_player')
        away_player_columns += get_column_names_containing_str(df, 'away_substitute')
        all_unique_players = pd.concat([df[player_columns[i]] for i in range(len(player_columns))],
                                       axis=0).unique().tolist()
        all_unique_players.remove('-')
        result = []
        for index, row in df.iterrows():
            home_encoding = [0] * len(all_unique_players)
            away_encoding = [0] * len(all_unique_players)
            for idx in range(len(home_player_columns)):
                if row[home_player_columns[idx]] != '-':
                    home_player_idx = all_unique_players.index(row[home_player_columns[idx]])
                    home_encoding[home_player_idx] = 1
                if row[away_player_columns[idx]] != '-':
                    away_player_idx = all_unique_players.index(row[away_player_columns[idx]])
                    away_encoding[away_player_idx] = 1
            result += [home_encoding + away_encoding]
        home_away_player_columns = [f'home_player_{name}' for name in all_unique_players] + [f'away_player_{name}' for
                                                                                             name in all_unique_players]
        players_df = pd.DataFrame(result, columns=home_away_player_columns)
        df = df.drop(player_columns, axis=1)
        df = pd.concat([df, players_df.set_index(df.index)], axis=1)
        print(f'Players encoded. Shape: {df.shape}')
        return df

    def remove_players(df: pd.DataFrame):
        player_columns = get_column_names_containing_str(df, 'player')
        player_columns += get_column_names_containing_str(df, 'substitute')
        df = df.drop(player_columns, axis=1)
        return df

    def encode_remaining_feats(df: pd.DataFrame):
        df = pd.get_dummies(df)
        print(f'Remaining feats encoded. Shape: {df.shape}')
        return df

    def shift_home_team_cols_to_end(df: pd.DataFrame):
        cols_to_shift = get_column_names_containing_str(df, 'home_team')
        players_df = df[cols_to_shift].copy()
        df = df.drop(cols_to_shift, axis=1)
        df = pd.concat([df, players_df], axis=1)
        return df

    def shift_away_team_cols_to_end(df: pd.DataFrame):
        cols_to_shift = get_column_names_containing_str(df, 'away_team')
        players_df = df[cols_to_shift].copy()
        df = df.drop(cols_to_shift, axis=1)
        df = pd.concat([df, players_df], axis=1)
        return df

    def shift_referee_cols_to_end(df: pd.DataFrame):
        cols_to_shift = get_column_names_containing_str(df, 'referee')
        players_df = df[cols_to_shift].copy()
        df = df.drop(cols_to_shift, axis=1)
        df = pd.concat([df, players_df], axis=1)
        return df

    def shift_player_cols_to_end(df: pd.DataFrame):
        cols_to_shift = get_column_names_containing_str(df, 'player')
        players_df = df[cols_to_shift].copy()
        df = df.drop(cols_to_shift, axis=1)
        df = pd.concat([df, players_df], axis=1)
        return df

    print('===> Phase 3: DATA ENCODING ')
    df = encode_seasons(df)
    if INCLUDE_PLAYERS:
        df = encode_players(df)
    else:
        df = remove_players(df)
    df = encode_remaining_feats(df)
    df = shift_home_team_cols_to_end(df)
    df = shift_away_team_cols_to_end(df)
    df = shift_referee_cols_to_end(df)
    df = shift_player_cols_to_end(df)
    print('===> Phase 3: DONE ')
    return df


if __name__ == '__main__':
    df = pd.read_csv(INPUT_CSV_NAME)
    df = data_fixing(df)
    df = data_manipulation(df)
    df = data_encoding(df)
    df.to_csv(OUTPUT_CSV_NAME, index=False)
    print(f'DONE. Final shape: {df.shape}')
