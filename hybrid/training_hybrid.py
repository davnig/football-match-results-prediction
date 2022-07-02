import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torchsummary import summary

from hybrid.dataset import SerieAMatchesWithHistoryDataset
from hybrid.model import HybridRNN, HybridMLP, HybridNetwork

learning_rate = 0.001
num_epochs = 20
hidden_size = 128


def count_features(data_csv: str):
    df = pd.read_csv(data_csv, nrows=1)
    n_of_features = len(df.columns)
    del df
    return n_of_features


if __name__ == '__main__':
    csv_name = 'data1.csv'
    tot_num_of_feats = count_features(csv_name)
    dataset = SerieAMatchesWithHistoryDataset(csv_file=csv_name)
    rnn_home = HybridRNN(input_size=tot_num_of_feats, hidden_size=hidden_size)
    rnn_away = HybridRNN(input_size=tot_num_of_feats, hidden_size=hidden_size)
    mlp = HybridMLP(hidden_size * 2 + tot_num_of_feats - 5)
    model = HybridNetwork(dataset=dataset, rnn_home_model=rnn_home, rnn_away_model=rnn_away, mlp_model=mlp,
                          learning_rate=learning_rate)
    print(summary(model))
    logger = TensorBoardLogger("../lightning_logs", name="hybrid_results")
    trainer = Trainer(fast_dev_run=False, gpus=1, max_epochs=num_epochs, logger=logger)
    trainer.fit(model)
    trainer.test(model)
