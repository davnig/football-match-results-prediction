import pandas as pd
from sklearn.preprocessing import LabelEncoder

from data_encoding import encode_seasons, encode_remaining_feats, remove_lineup, remove_teams, remove_referees, \
    encode_players
from data_fixing import fix_issue_1, fix_issue_2, fix_issue_3
from data_manipulation import convert_date_str_to_datetime, sort_by_date_column, \
    cast_str_values_to_int, add_result_column, explode_datetime_values, drop_date_cols
from utils import add_historic_data_of_last_n_matches_as_features, MATCH_STATS_COLUMNS, \
    MATCH_LINEUP_COLUMNS

# If enabled, PLAYERS, COACHES, TEAMS and REFEREES will not be included
LESS_DATA = False
INPUT_CSV_NAME = 'raw.csv'
OUTPUT_CSV_NAME = 'data_rnn_simple.csv' if LESS_DATA else 'data_rnn.csv'

""" 
The goal of this pre-processing step is to clean the raw data and build a time series for every source match 
as illustrated below:

STARTING DATAFRAME:
-----------------------------
| idx | features |  target  |
-----------------------------
|  0  | MATCH_0  | result_0 |
-----------------------------
|  1  | MATCH_1  | result_1 |
-----------------------------
|  2  | MATCH_2  | result_2 |
-----------------------------

FINAL DATAFRAME:
------------------------------------------------------------------------
| idx | old_idx |                  features                 |  target  |
------------------------------------------------------------------------
|  1  |    0    | MATCH_0 home_hist_0 | MATCH_0 away_hist_0 | result_0 |
|  2  |         | MATCH_0 home_hist_1 | MATCH_0 away_hist_1 | result_0 |
|  3  |         | MATCH_0 home_hist_2 | MATCH_0 away_hist_2 | result_0 |
|  4  |         | MATCH_0 home_hist_3 | MATCH_0 away_hist_3 | result_0 |
|  5  |         | MATCH_0 home_hist_4 | MATCH_0 away_hist_4 | result_0 |
------------------------------------------------------------------------
|  7  |    1    | MATCH_1 home_hist_0 | MATCH_1 away_hist_0 | result_1 |
|  8  |         | MATCH_1 home_hist_1 | MATCH_1 away_hist_1 | result_1 |
|  9  |         | MATCH_1 home_hist_2 | MATCH_1 away_hist_2 | result_1 |
|  10 |         | MATCH_1 home_hist_3 | MATCH_1 away_hist_3 | result_1 |
|  11 |         | MATCH_1 home_hist_4 | MATCH_1 away_hist_4 | result_1 |
------------------------------------------------------------------------

Legend:
- 'MATCH_n' represent the data of a single match
- 'result_n' is the outcome of MATCH_n we want to predict
- 'MATCH_N home_hist_n' represent the data of the n-th previous game played by the home team of MATCH_n
- 'MATCH_N away_hist_n' represent the data of the n-th previous game played by the away team of MATCH_n

Prior to training, this structure will be converted to a nested array allowing for a fast retrieval of football 
match time series.

"""


def data_fixing(df: pd.DataFrame) -> pd.DataFrame:
    print('===> Phase 1: DATA FIXING ')
    df = fix_issue_1(df)
    df = fix_issue_2(df)
    df = fix_issue_3(df)
    print('===> Phase 1: DONE')
    return df


def data_manipulation(df: pd.DataFrame):
    def convert_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
        # keep just the historic data and target column
        not_hist_columns = df.filter(regex='^(?!(.*_[1-5]|result)$).*$', axis=1).columns.tolist()
        df = df.drop(columns=not_hist_columns, axis=1)
        # delete result features for historic matches
        df = df.drop(columns=df.filter(regex='^(?:.*result_[1-5]$).*$', axis=1).columns, axis=1)
        # delete all matches not having a complete history of 5 games
        df = df.drop(df[df['home_season_5'] == '-'].index)
        # insert 'id' columns
        df.insert(loc=0, column='id', value=df.index)
        # convert wide to long
        df = pd.wide_to_long(df, stubnames=[f'{home_or_away}_{col}' for col in not_hist_columns
                                            for home_or_away in ['home', 'away']], i=['id', 'result'], j='time',
                             sep='_',
                             suffix='\d+')
        df = df.reset_index()
        df = df.sort_values(by=['id', 'time'], ascending=[True, True])
        df = df.reset_index(drop=True)
        df = df.drop(columns=['id', 'time'])
        return df

    def fill_stat_values(df: pd.DataFrame) -> pd.DataFrame:
        for col in [f'{home_away}_{col}' for home_away in ['home', 'away'] for col in MATCH_STATS_COLUMNS]:
            df[col].replace(['-'], 0, inplace=True)
        return df

    def force_type(df: pd.DataFrame) -> pd.DataFrame:
        str_columns = MATCH_LINEUP_COLUMNS + ['season', 'year', 'home_team', 'away_team', 'referee']
        str_columns = [f'home_{col}' for col in str_columns] + [f'away_{col}' for col in str_columns]
        str_columns += ['result']
        int_columns = [x for x in df.columns if x not in str_columns]
        type_dict = {}
        for int_col in int_columns:
            type_dict[int_col] = 'int'
        for str_col in str_columns:
            type_dict[str_col] = 'str'
        return df.astype(type_dict)

    print('===> Phase 2: DATA MANIPULATION ')
    # if LESS_DATA:
    #     df = remove_lineup(df)
    #     df = remove_teams(df)
    #     df = remove_referees(df)
    df = df.dropna()
    df = convert_date_str_to_datetime(df)
    df = sort_by_date_column(df)
    df = cast_str_values_to_int(df)
    df = explode_datetime_values(df)
    df = drop_date_cols(df)
    df = add_result_column(df)
    df = add_historic_data_of_last_n_matches_as_features(df)
    df = convert_wide_to_long(df)
    # fill missing stats values with 0
    df = fill_stat_values(df)
    # delete group of 5 match having a NaN value
    df = df.drop(df.iloc[11550:11555, :].index).reset_index(drop=True)
    # force integer type for all columns except 'result'
    df = force_type(df)
    print('===> Phase 2: DONE ')
    return df


def data_encoding(df: pd.DataFrame):
    print('===> Phase 3: DATA ENCODING ')
    df = encode_seasons(df)
    if not LESS_DATA:
        df = encode_players(df)
        # COACHES will be encoded using get_dummies.
        # Same for TEAMS, preserving home / away distinction
    else:
        df = remove_lineup(df)
        df = remove_teams(df)
        df = remove_referees(df)
    # label encode 'year'
    le = LabelEncoder()
    df['home_year'] = le.fit_transform(df['home_year'])
    df['away_year'] = le.fit_transform(df['away_year'])
    df = encode_remaining_feats(df)
    print('===> Phase 3: DONE ')
    return df


if __name__ == '__main__':
    df = pd.read_csv('../' + INPUT_CSV_NAME)
    df = data_fixing(df)
    df = data_manipulation(df)
    df = data_encoding(df)
    df.to_csv(OUTPUT_CSV_NAME, index=False)
    print(f'DONE. Final shape: {df.shape}')
