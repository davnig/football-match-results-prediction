import pandas as pd

from MatchResult import MatchResult
from utils import MATCH_LINEUP_COLUMNS, MATCH_STATS_COLUMNS


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


def add_result_column(df: pd.DataFrame) -> pd.DataFrame:
    def get_match_result_from_score(home_score: int, away_score: int) -> MatchResult:
        if home_score == away_score:
            return MatchResult.draw
        if home_score > away_score:
            return MatchResult.home
        return MatchResult.away

    results = {'result': []}
    for (index, home_score, away_score) in df[['home_score', 'away_score']].itertuples(name=None):
        results['result'] += [get_match_result_from_score(home_score, away_score).name]
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


def drop_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop('date', axis=1, inplace=False)
    df = df.drop('time', axis=1, inplace=False)
    return df


def fill_stat_values_in_hist_df(df: pd.DataFrame):
    """Fill missing stats values with 0 in historic dataframe"""
    for col in [f'{home_away}_{col}' for home_away in ['home', 'away'] for col in MATCH_STATS_COLUMNS]:
        df[col].replace(['-'], 0, inplace=True)
    return df


def fill_stat_values(df: pd.DataFrame):
    """Fill missing stats values with 0"""
    for col in MATCH_STATS_COLUMNS:
        df[col].replace(['-'], 0, inplace=True)
    return df


def force_type_in_hist_df(df: pd.DataFrame) -> pd.DataFrame:
    """Cast data to proper type in historic dataframe"""
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


def force_type(df: pd.DataFrame) -> pd.DataFrame:
    """Cast data to proper type"""
    str_columns = MATCH_LINEUP_COLUMNS + ['season', 'year', 'home_team', 'away_team', 'referee']
    str_columns += ['result']
    int_columns = [x for x in df.columns if x not in str_columns]
    type_dict = {}
    for int_col in int_columns:
        type_dict[int_col] = 'int'
    for str_col in str_columns:
        type_dict[str_col] = 'str'
    return df.astype(type_dict)
