import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import random_split, DataLoader, Dataset

from utils import accuracy


class RNN(pl.LightningModule):
    def __init__(self, dataset, input_size, hidden_size, batch_size, learning_rate):
        super(RNN, self).__init__()
        # self.save_hyperparameters()
        self.dataset = dataset
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_set = Dataset()
        self.val_set = Dataset()
        self.test_set = Dataset()
        self.linear = nn.Linear(input_size + hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size * 2, hidden_size)
        self.linear_out = nn.Linear(hidden_size, 3)
        self.norm = nn.LayerNorm(hidden_size)
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

    def forward(self, x, hidden, hidden_2):
        for i in range(x.shape[1]):
            input = torch.cat([x[:, i, :], hidden], dim=1)
            pre_hidden = self.linear(input)
            hidden = self.tanh(self.norm(pre_hidden))
            pre_hidden_2 = self.linear_2(torch.cat([hidden, hidden_2], dim=1))
            hidden_2 = self.tanh(self.norm(pre_hidden_2))
            output = self.softmax(self.linear_out(hidden_2))
        return hidden, output

    def training_step(self, batch, batch_idx):
        x, y = batch
        hidden = self.init_hidden(x.shape[0])
        hidden_2 = self.init_hidden(x.shape[0])
        _, y_hat = self(x, hidden, hidden_2)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        hidden = self.init_hidden(x.shape[0])
        hidden_2 = self.init_hidden(x.shape[0])
        _, y_hat = self(x, hidden, hidden_2)
        val_loss = F.cross_entropy(y_hat, y)
        val_accuracy = accuracy(y, y_hat)
        self.log_dict({"val_accuracy": val_accuracy, "val_loss": val_loss}, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        hidden = self.init_hidden(x.shape[0])
        hidden_2 = self.init_hidden(x.shape[0])
        _, y_hat = self(x, hidden, hidden_2)
        test_loss = F.cross_entropy(y_hat, y)
        test_accuracy = accuracy(y, y_hat)
        self.log_dict({"test_accuracy": test_accuracy, "test_loss": test_loss}, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.learning_rate)
