from random import randrange

import pandas as pd
import torch
from torch.utils.data import Dataset

from _MatchNotFoundException import MatchNotFoundException


class SerieAMatchesDataset(Dataset):
    def __init__(self, csv_file):
        self.dataframe = pd.read_csv(csv_file)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx):
        def show_error(index, error_x, error_y):
            print(f'error at index: {index}')
            print(f'x: {error_x}')
            print(f'x.shape: {error_x.shape}')
            print(f'y: {error_y}')
            print(f'y.shape: {error_y.shape}')

        def to_tensor(x: pd.DataFrame, y: list[int]):
            x_tensor = torch.flatten(torch.tensor(x.values))
            y_tensor = torch.tensor(y)
            return x_tensor, y_tensor

        x = self.dataframe.iloc[[idx]]  # df
        y = self.dataframe[['result_home', 'result_draw', 'result_away']].iloc[idx].values
        try:  # if we are not able to fetch at least one historical match, then we switch to another index
            x, y = to_tensor(x, y)
            exp_num_of_features = len(self.dataframe.columns)
            if x.shape[0] != exp_num_of_features:
                show_error(idx, x, y)
            if y.shape[0] != 3:
                show_error(idx, x, y)
            return x, y
        except MatchNotFoundException:
            new_idx = randrange(0, len(self.dataframe))
            # print(f'MatchNotFoundException for idx={idx}, switching to idx={new_idx}')
            return self.__getitem__(new_idx)
