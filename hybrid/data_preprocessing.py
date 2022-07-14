import pandas as pd
from sklearn.preprocessing import LabelEncoder

from data_encoding import encode_seasons, encode_players, remove_lineup, encode_remaining_feats, \
    remove_teams, remove_referees
from data_fixing import fix_issue_1, fix_issue_2, fix_issue_3
from data_manipulation import convert_date_str_to_datetime, sort_by_date_column, cast_str_values_to_int, \
    add_result_column, explode_datetime_values, drop_date_cols, force_type_in_hist_df, fill_stat_values_in_hist_df
from utils import add_historic_data_of_last_n_matches_as_features

# If enabled, PLAYERS, COACHES, TEAMS and REFEREES will not be included
LESS_DATA = True
INPUT_CSV_NAME = '../raw.csv'
OUTPUT_CSV_NAME = 'data_hybrid_simple.csv' if LESS_DATA else 'data_hybrid.csv'

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
|  0  |    0    | MATCH_0             | MATCH_0             | result_0 |
|  1  |         | MATCH_0 home_hist_1 | MATCH_0 away_hist_1 | result_0 |
|  2  |         | MATCH_0 home_hist_2 | MATCH_0 away_hist_2 | result_0 |
|  3  |         | MATCH_0 home_hist_3 | MATCH_0 away_hist_3 | result_0 |
|  4  |         | MATCH_0 home_hist_4 | MATCH_0 away_hist_4 | result_0 |
|  5  |         | MATCH_0 home_hist_5 | MATCH_0 away_hist_5 | result_0 |
------------------------------------------------------------------------
|  6  |    1    | MATCH_1             | MATCH_1             | result_1 |
|  7  |         | MATCH_1 home_hist_1 | MATCH_1 away_hist_1 | result_1 |
|  8  |         | MATCH_1 home_hist_2 | MATCH_1 away_hist_2 | result_1 |
|  9  |         | MATCH_1 home_hist_3 | MATCH_1 away_hist_3 | result_1 |
|  10 |         | MATCH_1 home_hist_4 | MATCH_1 away_hist_4 | result_1 |
|  11 |         | MATCH_1 home_hist_5 | MATCH_1 away_hist_5 | result_1 |
------------------------------------------------------------------------

Legend:
- 'MATCH_n' represent the data of a single match
- 'result_n' is the outcome of MATCH_n we want to predict
- 'MATCH_N home_hist_n' represent the data of the n-th previous game played by the home team of MATCH_n
- 'MATCH_N away_hist_n' represent the data of the n-th previous game played by the away team of MATCH_n

Prior to training, this structure will be converted to a nested array allowing for a fast retrieval of football 
match time series.

The difference with `rnn\\data_preprocessing.py` is that here, in the final dataset, we keep the data of the source 
match, of which we try to predict the result, as 6-th historic match in order to give extra context to the model.
"""


def data_fixing(df: pd.DataFrame):
    """Through manual inspection of the raw dataset, several matches with issues or inconsistent data were detected.
    Let’s fix them. We will use another source of Serie A matches to compare with."""
    print('===> Phase 1: DATA FIXING ')
    df = fix_issue_1(df)
    df = fix_issue_2(df)
    df = fix_issue_3(df)
    print('===> Phase 1: DONE')
    return df


def data_manipulation(df: pd.DataFrame):
    def convert_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
        not_hist_columns = df.filter(regex='^(?!.*_[1-5]$).*$', axis=1).columns.tolist()
        # delete all matches not having a complete history of 5 games
        df = df.drop(df[df['home_season_5'] == '-'].index)
        # rename columns of current match as historic with #0 index and duplicate
        home_current = pd.DataFrame(data=df.filter(regex='^(?!.*_[1-5]$).*$', axis=1).values.tolist(),
                                    columns=[f'home_{col}_0' for col in not_hist_columns]).set_index(df.index)
        away_current = pd.DataFrame(data=df.filter(regex='^(?!.*_[1-5]$).*$', axis=1).values.tolist(),
                                    columns=[f'away_{col}_0' for col in not_hist_columns]).set_index(df.index)
        df = df.drop(columns=df.filter(regex='^(?!.*_[1-5]$).*$', axis=1).columns, inplace=False)
        df = pd.concat([home_current, away_current, df], axis=1)
        # re-insert 'result' column
        df.insert(loc=1, column='result', value=df['home_result_0'])
        # insert 'id' columns
        df.insert(loc=0, column='id', value=df.index)
        # convert wide to long
        df = pd.wide_to_long(df, stubnames=[f'{home_or_away}_{col}' for col in not_hist_columns for home_or_away in
                                            ['home', 'away']], i=['id', 'result'], j='time_idx', sep='_', suffix='\d+')
        # clean unwanted columns
        df = df.drop(columns=['home_result', 'away_result'], axis=1)
        df = df.reset_index()
        df = df.sort_values(by=['id', 'time_idx'], ascending=[True, True])
        df = df.reset_index(drop=True)
        df = df.drop(columns=['id', 'time_idx'])
        return df

    print('===> Phase 2: DATA MANIPULATION ')
    df = convert_date_str_to_datetime(df)
    df = sort_by_date_column(df)
    df = cast_str_values_to_int(df)
    df = add_result_column(df)
    df = explode_datetime_values(df)
    df = drop_date_cols(df)
    df = add_historic_data_of_last_n_matches_as_features(df)
    df = convert_wide_to_long(df)
    df = fill_stat_values_in_hist_df(df)
    # delete group of 5 match having a NaN value
    df = df.drop(
        df.iloc[df[df['away_season'] == '-'].index.tolist()[0] - 5:df[df['away_season'] == '-'].index.tolist()[0] + 1,
        :].index).reset_index(drop=True)
    # force correct type for all columns
    df = force_type_in_hist_df(df)
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
    df = pd.read_csv(INPUT_CSV_NAME)
    df = data_fixing(df)
    df = data_manipulation(df)
    df = data_encoding(df)
    df.to_csv(OUTPUT_CSV_NAME, index=False)
    print(f'DONE. Final shape: {df.shape}')
