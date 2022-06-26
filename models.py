import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split


class MLP(pl.LightningModule):
    def __init__(self, dataset, input_size: int, learning_rate: float = 0.001, batch_size: int = 32):
        super(MLP, self).__init__()
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_set = Dataset()
        self.val_set = Dataset()
        self.test_set = Dataset()
        self.input_size = input_size
        self.flatten = nn.Flatten()
        self.layers_stack = nn.Sequential(
            nn.Linear(input_size, 5120),
            nn.ReLU(),
            nn.Linear(5120, 1280),
            nn.ReLU(),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
            # nn.Softmax(dim=1) softmax is applied implicitly by CrossEntropyLoss
        )

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

    def forward(self, x):
        """Compute y_hat from dataloader input"""
        x = self.flatten(x)
        y_hat = self.layers_stack(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y.to(dtype=torch.float))
        tensorboard_logs = {"train_loss": loss}
        print(x[0])
        print(f'y: {y[0]}')
        print(f'y_hat: {y_hat[0]}')
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y.to(dtype=torch.float))
        val_accuracy = self.accuracy(y, y_hat)
        # print(f'y: {y} y_hat: {y_hat} accuracy: {val_accuracy}')
        return {"val_loss": val_loss, "val_accuracy": val_accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([el["val_loss"] for el in outputs]).mean()
        avg_accuracy = torch.tensor([el["val_accuracy"] for el in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss, "val_accuracy": avg_accuracy}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y.to(dtype=torch.float))
        test_accuracy = self.accuracy(y, y_hat)
        return {"test_loss": test_loss, "test_accuracy": test_accuracy}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([el["test_loss"] for el in outputs]).mean()
        avg_accuracy = torch.tensor([el["test_accuracy"] for el in outputs]).mean()
        tensorboard_logs = {"test_loss": avg_loss, "avg_accuracy": avg_accuracy}
        return {"test_loss": avg_loss, "log": tensorboard_logs, "progress_bar": tensorboard_logs}

    def predict_step(self, batch):
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.learning_rate)

    def accuracy(self, y, y_hat):
        # Assign each y_hat to its predicted class
        pred_classes = torch.where(y_hat < .5, 0, 1).squeeze().long()
        correct = (pred_classes == y).sum()
        return (correct / y.shape[0]).item()


class HybridRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(HybridRNN, self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size + hidden_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, input, hidden):
        combined = torch.cat([input, hidden], dim=0)
        pre_hidden = self.linear(combined)
        hidden = self.tanh(pre_hidden)
        return hidden

    def init_hidden(self, minibatch_size):
        return torch.zeros(minibatch_size, self.hidden_size)


class HybridMLP(nn.Module):
    def __init__(self, input_size):
        super(HybridMLP, self).__init__()
        self.input_size = input_size
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
            # nn.Softmax(dim=1) softmax is applied implicitly by CrossEntropyLoss
        )

    def forward(self, x):
        # 'x' is the combination of: 'x', 'x_historical_home', 'x_historical_away'
        # they all have size: minibatch_size x num_of_feats
        x = self.flatten(x)  # just in case x was not flattened
        output = self.layers(x)
        return output


class HybridNetwork(pl.LightningModule):
    def __init__(self, dataset, rnn_home_model: HybridRNN, rnn_away_model: HybridRNN, mlp_model: HybridMLP,
                 learning_rate: float = 0.001, batch_size: int = 32):
        super(HybridNetwork, self).__init__()
        self.dataset = dataset
        self.rnn_home = rnn_home_model
        self.rnn_away = rnn_away_model
        self.mlp = mlp_model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_set = Dataset()
        self.val_set = Dataset()
        self.test_set = Dataset()

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

    def forward(self, x, x_historical_home, x_historical_away):
        """Compute y_hat from dataloader input"""
        # 'x' comes in as:                minibatch_size x 1 x num_of_feats
        # 'x_historical_*' comes in as:   minibatch_size x 5 x num_of_feats
        # 'rnn_*_hidden' will be:         minibatch_size x num_of_feats
        batch_size = x.size(0)
        time_seq_len = x_historical_home.size(1)
        cuda_0 = torch.device('cuda:0')
        ''' === RNN HOME FORWARD === '''
        rnn_home_hidden = self.rnn_home.init_hidden(batch_size)
        rnn_home_hidden = rnn_home_hidden.to(cuda_0)
        for batch_idx in range(batch_size):
            for history_idx in range(time_seq_len):
                rnn_home_hidden[batch_idx] = self.rnn_home(
                    torch.flatten(x_historical_home[batch_idx, history_idx]),
                    rnn_home_hidden[batch_idx])
        ''' === RNN AWAY FORWARD === '''
        rnn_away_hidden = self.rnn_away.init_hidden(batch_size)
        rnn_away_hidden = rnn_away_hidden.to(cuda_0)
        for batch_idx in range(batch_size):
            for history_idx in range(time_seq_len):
                rnn_away_hidden[batch_idx] = self.rnn_away(
                    torch.flatten(x_historical_away[batch_idx, history_idx]),
                    rnn_away_hidden[batch_idx])
        ''' === MLP FORWARD === '''
        x_train = torch.cat([x, rnn_home_hidden, rnn_away_hidden], dim=1)
        y_hat = self.mlp(x_train)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, x_historical_home, x_historical_away, y = batch
        y_hat = self(x, x_historical_home, x_historical_away)
        loss = F.cross_entropy(y_hat, y.to(dtype=torch.float))
        tensorboard_logs = {"train_loss": loss}
        print(x[0])
        print(f'y: {y[0]}')
        print(f'y_hat: {y_hat[0]}')
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, x_historical_home, x_historical_away, y = batch
        y_hat = self(x, x_historical_home, x_historical_away)
        val_loss = F.cross_entropy(y_hat, y.to(dtype=torch.float))
        val_accuracy = self.accuracy(y, y_hat)
        # print(f'y: {y} y_hat: {y_hat} accuracy: {val_accuracy}')
        return {"val_loss": val_loss, "val_accuracy": val_accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([el["val_loss"] for el in outputs]).mean()
        avg_accuracy = torch.tensor([el["val_accuracy"] for el in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss, "val_accuracy": avg_accuracy}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, x_historical_home, x_historical_away, y = batch
        y_hat = self(x, x_historical_home, x_historical_away)
        test_loss = F.cross_entropy(y_hat, y.to(dtype=torch.float))
        test_accuracy = self.accuracy(y, y_hat)
        return {"test_loss": test_loss, "test_accuracy": test_accuracy}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([el["test_loss"] for el in outputs]).mean()
        avg_accuracy = torch.tensor([el["test_accuracy"] for el in outputs]).mean()
        tensorboard_logs = {"test_loss": avg_loss, "avg_accuracy": avg_accuracy}
        return {"test_loss": avg_loss, "log": tensorboard_logs, "progress_bar": tensorboard_logs}

    def predict_step(self, batch):
        x, x_historical_home, x_historical_away, y = batch
        return self(x, x_historical_home, x_historical_away)

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.learning_rate)

    def accuracy(self, y, y_hat):
        # Assign each y_hat to its predicted class
        pred_classes = torch.where(y_hat < .5, 0, 1).squeeze().long()
        correct = (pred_classes == y).sum()
        return (correct / y.shape[0]).item()
