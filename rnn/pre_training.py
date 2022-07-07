import pandas as pd

from utils import add_historic_data_of_last_n_matches_as_features, MATCH_COLUMNS

if __name__ == '__main__':
    df = pd.read_csv('../raw.csv', nrows=400)
    df = add_historic_data_of_last_n_matches_as_features(df)
    df.insert(loc=0, column='id', value=df.index)
    # keep just the historic data
    df = df.drop(columns=MATCH_COLUMNS, axis=1)
    a = pd.wide_to_long(df, stubnames=[f'hist_{home_or_away}_{col}' for col in MATCH_COLUMNS for home_or_away in
                                       ['home', 'away']], i='id', j='time', sep='_', suffix='\d+')
    a = a.reset_index() \
        .sort_values(by=['id', 'time'], ascending=[True, True]) \
        .reset_index() \
        .drop(columns='index')
    # use numpy or first encode, then tensor.reshape
    # b = a.reshape(-1, 5, a.shape[-1])
    print('f')
