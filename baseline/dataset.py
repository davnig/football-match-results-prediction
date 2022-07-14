import pandas as pd
import torch
from torch.utils.data import Dataset


class SerieAMatchesDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        x_df = df.filter(regex='^(?!result).*$', axis=1)
        y_df = df.filter(regex='^(?:result).*$', axis=1)
        self.x_train = x_df.values
        self.y_train = y_df.values

    def __len__(self) -> int:
        return len(self.y_train)

    def __getitem__(self, idx):
        x = self.x_train[idx]
        y = self.y_train[idx]
        x = torch.from_numpy(x.astype('float32'))
        y = torch.from_numpy(y.astype('float32'))
        return x, y
