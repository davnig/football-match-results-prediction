import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from datasets import SerieAMatchesDataset
from models import MLP

learning_rate = 0.001
num_epochs = 20


def count_features(data_csv: str):
    df = pd.read_csv(data_csv, nrows=1)
    n_of_features = len(df.columns)
    del df
    return n_of_features


if __name__ == '__main__':
    csv_name = 'data1.csv'
    n_of_feats = count_features(csv_name)
    dataset = SerieAMatchesDataset(csv_file=csv_name)
    model = MLP(dataset=dataset, input_size=n_of_feats, learning_rate=learning_rate)
    logger = TensorBoardLogger("lightning_logs", name="baseline_results")
    trainer = Trainer(gpus=1, max_epochs=num_epochs, logger=logger)
    trainer.fit(model)
    trainer.test(model)
