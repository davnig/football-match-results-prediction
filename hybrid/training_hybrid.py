import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torchsummary import summary

from hybrid.dataset import HybridSerieADataset
from hybrid.model import HybridRNN, HybridMLP, HybridNetwork
from utils import MATCH_STATS_COLUMNS

# If enabled, the model will not consider PLAYERS, COACHES, REFEREES and TEAMS features for training
SIMPLE_MODEL = True
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
HIDDEN_SIZE = 128
BATCH_SIZE = 32
CSV_NAME = 'data_hybrid_simple.csv' if SIMPLE_MODEL else 'data_hybrid.csv'


def count_features(data_csv: str):
    df = pd.read_csv(data_csv, nrows=1)
    num_x_rnn_home_features = len(df.filter(regex='^home_.*?').columns)
    num_x_rnn_away_features = len(df.filter(regex='^away_.*?').columns)
    num_x_mlp_features = len(df.filter(regex='^home_.*?').drop(
        columns=[f'home_{col}' for col in MATCH_STATS_COLUMNS + ['home_score', 'away_score']]).columns)
    return num_x_rnn_home_features, num_x_rnn_away_features, num_x_mlp_features


if __name__ == '__main__':
    n_feats_rnn_home, n_feats_rnn_away, n_feats_mlp = count_features(CSV_NAME)
    dataset = HybridSerieADataset(csv_file=CSV_NAME)
    rnn_home = HybridRNN(input_size=n_feats_rnn_home, hidden_size=HIDDEN_SIZE)
    rnn_away = HybridRNN(input_size=n_feats_rnn_away, hidden_size=HIDDEN_SIZE)
    mlp = HybridMLP(HIDDEN_SIZE * 2 + n_feats_mlp)
    model = HybridNetwork(dataset=dataset, rnn_home_model=rnn_home, rnn_away_model=rnn_away, mlp_model=mlp,
                          learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE)
    print(summary(model))
    logger = TensorBoardLogger("../training_logs", name="hybrid_results")
    trainer = Trainer(fast_dev_run=False, gpus=1, max_epochs=NUM_EPOCHS, logger=logger)
    trainer.fit(model)
    trainer.test(model)
