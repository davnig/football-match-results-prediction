import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import HISTORY_LEN


class RNNSerieADataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        x_df = df.filter(regex='^(home|away).*$', axis=1)
        x_data = x_df.values.reshape(-1, HISTORY_LEN, x_df.shape[1])
        y_df = df.filter(regex='^.*(?:result).*$', axis=1)
        y_data = y_df.values.reshape(-1, HISTORY_LEN, y_df.shape[1])
        del df
        self.train_x = x_data
        self.train_y = y_data

    def __len__(self) -> int:
        return len(self.train_x)

    def __getitem__(self, idx):
        # print(f'\r\033[KPicked {idx}', end=' ')
        x = self.train_x[idx]
        y = self.train_y[idx, 4]
        x = torch.from_numpy(x.astype('float32'))
        y = torch.from_numpy(y.astype('float32'))
        return x, y
