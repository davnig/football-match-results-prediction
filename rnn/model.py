import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import random_split, DataLoader, Dataset

from utils import accuracy


class RNN(pl.LightningModule):
    def __init__(self, dataset, input_size, hidden_size, batch_size, learning_rate):
        super(RNN, self).__init__()
        self.dataset = dataset
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_set = Dataset()
        self.val_set = Dataset()
        self.test_set = Dataset()
        self.linear = nn.Linear(input_size + hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, 3)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def init_hidden(self, x_batch_size):
        return torch.zeros(x_batch_size, self.hidden_size, device=torch.device('cuda:0'))

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

    def forward(self, x, hidden):
        for i in range(x.shape[1]):
            input = torch.cat([x[:, i, :], hidden], dim=1)
            pre_hidden = self.linear(input)
            hidden = self.tanh(pre_hidden)
            output = self.softmax(self.linear_out(hidden))
        return hidden, output

    def training_step(self, batch, batch_idx):
        x, y = batch
        hidden = self.init_hidden(x.shape[0])
        _, y_hat = self(x, hidden)
        loss = F.cross_entropy(y_hat, y.to(dtype=torch.float))
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        hidden = self.init_hidden(x.shape[0])
        _, y_hat = self(x, hidden)
        val_loss = F.cross_entropy(y_hat, y.to(dtype=torch.float))
        val_accuracy = accuracy(y, y_hat)
        return {"val_loss": val_loss, "val_accuracy": val_accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([el["val_loss"] for el in outputs]).mean()
        avg_accuracy = torch.tensor([el["val_accuracy"] for el in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss, "val_accuracy": avg_accuracy}
        print(f'\nval_avg_accuracy: {avg_accuracy} val_avg_loss: {avg_loss}')
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        hidden = self.init_hidden(x.shape[0])
        _, y_hat = self(x, hidden)
        test_loss = F.cross_entropy(y_hat, y.to(dtype=torch.float))
        test_accuracy = accuracy(y, y_hat)
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
