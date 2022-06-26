import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from datasets import SerieAMatchesDataset
from models import MLP

learning_rate = 0.00001
num_epochs = 3

if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    tot_num_of_feats = len(df.columns)
    del df
    dataset = SerieAMatchesDataset(csv_file='data.csv')
    model = MLP(dataset=dataset, input_size=tot_num_of_feats, learning_rate=learning_rate)
    logger = TensorBoardLogger("lightning_logs", name="baseline_results")
    trainer = Trainer(gpus=1, max_epochs=num_epochs, logger=logger)
    trainer.fit(model)
    trainer.test(model)
