import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import HISTORY_LEN


class RNNSerieADataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        data = df.values.reshape(-1, HISTORY_LEN, df.shape[1])
        del df
        self.train_x = data[:, :, :-3]
        self.train_y = data[:, :, -3:]

    def __len__(self) -> int:
        return len(self.train_x)

    def __getitem__(self, idx):
        # print(f'\r\033[KPicked {idx}', end=' ')
        x = self.train_x[idx]
        y = self.train_y[idx, 4]
        x = torch.from_numpy(x.astype('float32'))
        y = torch.from_numpy(y.astype('float32'))
        return x, y
