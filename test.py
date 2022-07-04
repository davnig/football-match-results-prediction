import pandas as pd

from utils import add_historic_data_of_last_n_matches_as_features

if __name__ == '__main__':
    df = pd.read_csv('raw1.csv')
    df = add_historic_data_of_last_n_matches_as_features(df)
