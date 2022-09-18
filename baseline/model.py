import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split

from utils import accuracy


class MLP(pl.LightningModule):
    def __init__(self, dataset, input_size: int, learning_rate: float = 0.001, batch_size: int = 32):
        super(MLP, self).__init__()
        # self.save_hyperparameters()
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_set = Dataset()
        self.val_set = Dataset()
        self.test_set = Dataset()
        self.input_size = input_size
        self.flatten = nn.Flatten()
        self.layers_stack = nn.Sequential(
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
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        val_accuracy = accuracy(y, y_hat)
        self.log_dict({"val_accuracy": val_accuracy, "val_loss": val_loss}, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        test_accuracy = accuracy(y, y_hat)
        self.log_dict({"test_accuracy": test_accuracy, "test_loss": test_loss}, prog_bar=True)

    def predict_step(self, batch):
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.learning_rate)
