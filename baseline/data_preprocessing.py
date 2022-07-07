import pandas as pd

from data_encoding import encode_seasons, encode_players, remove_players, encode_remaining_feats, \
    shift_home_team_cols_to_end, shift_away_team_cols_to_end, shift_referee_cols_to_end, shift_player_cols_to_end
from data_fixing import fix_issue_1, fix_issue_2, fix_issue_3
from data_manipulation import convert_date_str_to_datetime, sort_by_date_column, cast_str_values_to_int, \
    add_result_column, explode_datetime_values, drop_date_cols

INPUT_CSV_NAME = 'raw.csv'
OUTPUT_CSV_NAME = 'data_baseline.csv'
# add player names as features
INCLUDE_PLAYERS = True


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
    print('===> Phase 2: DATA MANIPULATION ')
    df = convert_date_str_to_datetime(df)
    df = sort_by_date_column(df)
    df = cast_str_values_to_int(df)
    df = add_result_column(df)
    df = explode_datetime_values(df)
    df = drop_date_cols(df)
    # let's delete the features that are not available at pre-match time
    df = df.drop(columns=['home_score', 'away_score'])
    df = df.dropna()
    print('===> Phase 2: DONE ')
    return df


def data_encoding(df: pd.DataFrame):
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
