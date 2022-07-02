import pytorch_lightning as pl
import torch
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
        x = x.to(dtype=torch.float)
        x = self.flatten(x)
        y_hat = self.layers_stack(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y.to(dtype=torch.float))
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
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
        y_hat = self(x)
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
