import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import HISTORY_LEN, MATCH_STATS_COLUMNS


class HybridSerieADataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        home_df = df.filter(regex='^home_.*?')
        away_df = df.filter(regex='^away_.*?')
        no_stats_no_scores_df = home_df.drop(
            columns=[f'home_{col}' for col in MATCH_STATS_COLUMNS + ['home_score', 'away_score']])
        target_df = df.filter(regex='^.*(?:result).*$', axis=1)
        home_data = home_df.values.reshape(-1, HISTORY_LEN + 1, home_df.shape[1])
        away_data = away_df.values.reshape(-1, HISTORY_LEN + 1, away_df.shape[1])
        no_stats_no_scores_data = no_stats_no_scores_df.values.reshape(-1, HISTORY_LEN + 1,
                                                                       no_stats_no_scores_df.shape[-1])
        target_data = target_df.values.reshape(-1, HISTORY_LEN + 1, target_df.shape[1])
        self.train_x_home = home_data
        self.train_x_away = away_data
        self.train_x_mlp = no_stats_no_scores_data
        self.train_y = target_data

    def __len__(self) -> int:
        return len(self.train_y)

    def __getitem__(self, idx):
        x_rnn_home = self.train_x_home[idx, 1:, :]
        x_rnn_away = self.train_x_away[idx, 1:, :]
        x_mlp = self.train_x_mlp[idx, :1, :]
        y = self.train_y[idx, 0]
        x_rnn_home = torch.from_numpy(x_rnn_home)
        x_rnn_away = torch.from_numpy(x_rnn_away)
        x_mlp = torch.from_numpy(x_mlp)
        y = torch.from_numpy(y)
        return x_rnn_home, x_rnn_away, x_mlp, y
