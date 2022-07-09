import pandas as pd
from sklearn.preprocessing import LabelEncoder

from data_encoding import encode_seasons, encode_remaining_feats, remove_lineup, remove_teams, remove_referees
from data_fixing import fix_issue_1, fix_issue_2, fix_issue_3
from data_manipulation import convert_date_str_to_datetime, sort_by_date_column, \
    cast_str_values_to_int, add_result_column, explode_datetime_values, drop_date_cols
from utils import add_historic_data_of_last_n_matches_as_features, MATCH_STATS_COLUMNS, \
    MATCH_COLUMNS_WITH_EXPL_DATETIME

INPUT_CSV_NAME = 'raw.csv'
OUTPUT_CSV_NAME = 'data_rnn.csv'
INCLUDE_LINEUP = False
INCLUDE_TEAMS = False
INCLUDE_REFEREES = False

"""
From input data, construct a dataset organized into 5-match long time series
"""


def data_fixing(df: pd.DataFrame) -> pd.DataFrame:
    print('===> Phase 1: DATA FIXING ')
    df = fix_issue_1(df)
    df = fix_issue_2(df)
    df = fix_issue_3(df)
    print('===> Phase 1: DONE')
    return df


def data_manipulation(df: pd.DataFrame):
    print('===> Phase 2: DATA MANIPULATION ')
    if not INCLUDE_LINEUP:
        df = remove_lineup(df)
    if not INCLUDE_TEAMS:
        df = remove_teams(df)
    if not INCLUDE_REFEREES:
        df = remove_referees(df)
    df = df.dropna()
    df = convert_date_str_to_datetime(df)
    df = sort_by_date_column(df)
    df = cast_str_values_to_int(df)
    df = explode_datetime_values(df)
    df = drop_date_cols(df)
    df = add_result_column(df)
    new_columns = df.columns.tolist()
    df = add_historic_data_of_last_n_matches_as_features(df)
    df.insert(loc=0, column='id', value=df.index)
    # keep just the historic data and target column
    df = df.drop(columns=MATCH_COLUMNS_WITH_EXPL_DATETIME, axis=1)
    # delete result features for historic matches
    df = df.drop(columns=[f'hist_{home_away}_result_{i}' for home_away in ['home', 'away'] for i in range(5)], axis=1)
    # delete all matches not having a complete history of 5 games
    df = df.drop(df[df['hist_home_season_4'] == '-'].index)
    # convert wide to long
    df = pd.wide_to_long(df, stubnames=[f'hist_{home_or_away}_{col}' for col in MATCH_COLUMNS_WITH_EXPL_DATETIME
                                        for home_or_away in ['home', 'away']], i=['id', 'result'], j='time', sep='_',
                         suffix='\d+')
    df = df.reset_index()
    df = df.sort_values(by=['id', 'time'], ascending=[True, True])
    df = df.reset_index(drop=True)
    df = df.drop(columns=['id', 'time'])
    # fill missing stats values with 0
    for col in [f'hist_{home_away}_{col}' for home_away in ['home', 'away'] for col in MATCH_STATS_COLUMNS]:
        df[col].replace(['-'], 0, inplace=True)
    # delete group of 5 match having a NaN value
    df = df.drop(df.iloc[11550:11555, :].index).reset_index(drop=True)
    # force integer type for all columns except 'result'
    df.loc[:, 'hist_home_season':'hist_away_hour'] = df.loc[:, 'hist_home_season':'hist_away_hour'].astype(int)
    print('===> Phase 2: DONE ')
    return df


def data_encoding(df: pd.DataFrame):
    print('===> Phase 3: DATA ENCODING ')
    df = encode_seasons(df)
    # label encode 'year'
    df[['hist_home_year', 'hist_away_year']] = df[['hist_home_year', 'hist_away_year']].astype(str)
    le = LabelEncoder()
    df['hist_home_year'] = le.fit_transform(df['hist_home_year'])
    df['hist_away_year'] = le.fit_transform(df['hist_away_year'])
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
