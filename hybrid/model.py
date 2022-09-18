import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split

from utils import accuracy, MATCH_STATS_COLUMNS


class HybridRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(HybridRNN, self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size + hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size * 2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, x, hidden, hidden_2):
        for i in range(x.shape[1]):
            input = torch.cat([x[:, i, :], hidden], dim=1)
            pre_hidden = self.linear(input)
            hidden = self.tanh(self.norm(pre_hidden))
            pre_hidden_2 = self.linear_2(torch.cat([hidden, hidden_2], dim=1))
            hidden_2 = self.tanh(self.norm(pre_hidden_2))
        return hidden_2

    def init_hidden(self, x_batch_size):
        return torch.zeros(x_batch_size, self.hidden_size)


class HybridMLP(nn.Module):
    def __init__(self, input_size):
        super(HybridMLP, self).__init__()
        self.input_size = input_size
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.Linear(64, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        output = self.layers(x)
        return output


class HybridNetwork(pl.LightningModule):
    def __init__(self, dataset, rnn_model: HybridRNN, mlp_model: HybridMLP,
                 learning_rate: float = 0.001, batch_size: int = 32):
        super(HybridNetwork, self).__init__()
        # self.save_hyperparameters()
        self.dataset = dataset
        self.rnn = rnn_model
        self.mlp = mlp_model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_set = Dataset()
        self.val_set = Dataset()
        self.test_set = Dataset()

    def standardize_and_normalize(self, train_set, val_set, test_set):
        standard_scaler = StandardScaler()
        min_max_scaler = MinMaxScaler()
        ored_list_of_statistics_feats = "|".join([i for i in MATCH_STATS_COLUMNS])
        columns_to_standardize = train_set.filter(regex=f'^.*(?:(${ored_list_of_statistics_feats})).*$',
                                                  axis=1).columns.tolist()
        columns_to_standardize += train_set.filter(regex='^.*(?:score).*$').columns.tolist()
        columns_to_normalize = columns_to_standardize
        columns_to_normalize += train_set.filter(regex='^.*(?:(season|round|year|month|day|hour)).*$').columns.tolist()
        # standardize
        standard_scaler.fit(train_set[columns_to_standardize])
        train_set[columns_to_standardize] = standard_scaler.transform(train_set[columns_to_standardize])
        val_set[columns_to_standardize] = standard_scaler.transform(val_set[columns_to_standardize])
        test_set[columns_to_standardize] = standard_scaler.transform(test_set[columns_to_standardize])
        # normalize
        min_max_scaler.fit(train_set[columns_to_normalize])
        train_set[columns_to_normalize] = min_max_scaler.transform(train_set[columns_to_normalize])
        val_set[columns_to_normalize] = min_max_scaler.transform(val_set[columns_to_normalize])
        test_set[columns_to_normalize] = min_max_scaler.transform(test_set[columns_to_normalize])
        return train_set, val_set, test_set

    def setup(self, stage):
        test_size = int(0.2 * len(self.dataset))
        val_size = int(0.2 * (len(self.dataset) - test_size))
        train_size = int(len(self.dataset) - test_size - val_size)
        self.train_set, self.val_set, self.test_set = random_split(self.dataset, (train_size, val_size, test_size))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)

    def forward(self, x_rnn_home, x_rnn_away, x_mlp):
        cuda_0 = torch.device('cuda:0')
        ''' === RNN HOME FORWARD === '''
        rnn_home_hidden = self.rnn.init_hidden(x_rnn_home.shape[0])
        rnn_home_hidden_2 = self.rnn.init_hidden(x_rnn_home.shape[0])
        rnn_home_hidden = rnn_home_hidden.to(cuda_0)
        rnn_home_hidden_2 = rnn_home_hidden_2.to(cuda_0)
        rnn_home_hidden = self.rnn(x_rnn_home, rnn_home_hidden, rnn_home_hidden_2)
        ''' === RNN AWAY FORWARD === '''
        rnn_away_hidden = self.rnn.init_hidden(x_rnn_away.shape[0])
        rnn_away_hidden_2 = self.rnn.init_hidden(x_rnn_away.shape[0])
        rnn_away_hidden = rnn_away_hidden.to(cuda_0)
        rnn_away_hidden_2 = rnn_away_hidden_2.to(cuda_0)
        rnn_away_hidden = self.rnn(x_rnn_away, rnn_away_hidden, rnn_away_hidden_2)
        ''' === MLP FORWARD === '''
        x_train = torch.cat([x_mlp.reshape(x_mlp.shape[0], x_mlp.shape[2]), rnn_home_hidden, rnn_away_hidden], dim=1)
        y_hat = self.mlp(x_train)
        return y_hat

    def training_step(self, batch, batch_idx):
        x_rnn_home, x_rnn_away, x_mlp, y = batch
        y_hat = self(x_rnn_home, x_rnn_away, x_mlp)
        loss = F.cross_entropy(y_hat, y.to(dtype=torch.float))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_rnn_home, x_rnn_away, x_mlp, y = batch
        y_hat = self(x_rnn_home, x_rnn_away, x_mlp)
        val_loss = F.cross_entropy(y_hat, y.to(dtype=torch.float))
        val_accuracy = accuracy(y, y_hat)
        self.log_dict({"val_accuracy": val_accuracy, "val_loss": val_loss}, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x_rnn_home, x_rnn_away, x_mlp, y = batch
        y_hat = self(x_rnn_home, x_rnn_away, x_mlp)
        test_loss = F.cross_entropy(y_hat, y.to(dtype=torch.float))
        test_accuracy = accuracy(y, y_hat)
        self.log_dict({"test_accuracy": test_accuracy, "test_loss": test_loss}, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x_rnn_home, x_rnn_away, x_mlp, y = batch
        return self(x_rnn_home, x_rnn_away, x_mlp)

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.learning_rate)
