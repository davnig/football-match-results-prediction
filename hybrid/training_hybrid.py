import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torchsummary import summary

from hybrid.dataset import HybridSerieADataset
from hybrid.model import HybridRNN, HybridMLP, HybridNetwork
from utils import MATCH_STATS_COLUMNS

LEARNING_RATE = 0.001
NUM_EPOCHS = 20
HIDDEN_SIZE = 128
BATCH_SIZE = 32
CSV_NAME = 'data_hybrid.csv'


def count_features(data_csv: str):
    df = pd.read_csv(data_csv, nrows=1)
    x_rnn_features = df.columns
    x_mlp_features = set(df.columns.tolist()) - set(df.filter(regex='score', axis=1).columns.tolist())
    x_mlp_features = x_mlp_features - set(
        [f'{home_away}_{col}' for col in MATCH_STATS_COLUMNS for home_away in ['home', 'away']])
    x_mlp_features = list(x_mlp_features)
    return len(x_rnn_features) / 2, len(x_mlp_features)


if __name__ == '__main__':
    n_feats_rnn, n_feats_mlp = count_features(CSV_NAME)
    dataset = HybridSerieADataset(csv_file=CSV_NAME)
    rnn_home = HybridRNN(input_size=n_feats_rnn, hidden_size=HIDDEN_SIZE)
    rnn_away = HybridRNN(input_size=n_feats_rnn, hidden_size=HIDDEN_SIZE)
    mlp = HybridMLP(HIDDEN_SIZE * 2 + n_feats_mlp)
    model = HybridNetwork(dataset=dataset, rnn_home_model=rnn_home, rnn_away_model=rnn_away, mlp_model=mlp,
                          learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE)
    print(summary(model))
    logger = TensorBoardLogger("../lightning_logs", name="hybrid_results")
    trainer = Trainer(fast_dev_run=False, gpus=1, max_epochs=NUM_EPOCHS, logger=logger)
    trainer.fit(model)
    trainer.test(model)
