import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import SerieAFootballMatchesDataset
from models import RNN, NeuralNetwork, HybridNetwork

learning_rate = 0.00001
num_epochs = 3
hidden_size = 256

if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    tot_num_of_feats = len(df.columns)
    del df
    if torch.cuda.is_available():
        print(torch.cuda.current_device())
        print(torch.cuda.get_device_name(0))
        print(torch.cuda.memory_allocated(0))
        print(torch.cuda.memory_reserved(0))
        torch.cuda.set_device(0)
    dataset = SerieAFootballMatchesDataset(csv_file='data.csv')
    rnn_home = RNN(input_size=tot_num_of_feats, hidden_size=hidden_size)
    rnn_away = RNN(input_size=tot_num_of_feats, hidden_size=hidden_size)
    mlp = NeuralNetwork(hidden_size * 2 + tot_num_of_feats)
    model = HybridNetwork(dataset=dataset, rnn_home_model=rnn_home, rnn_away_model=rnn_away, mlp_model=mlp,
                          learning_rate=learning_rate)
    logger = TensorBoardLogger("lightning_logs", name="football_results")
    trainer = Trainer(gpus=1, max_epochs=num_epochs, logger=logger)
    trainer.fit(model)
    trainer.test(model)
