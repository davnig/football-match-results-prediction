import pandas as pd

from MatchResult import MatchResult


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
