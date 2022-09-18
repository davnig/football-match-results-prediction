import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchsummary import summary

from hybrid.dataset import HybridSerieADataset
from hybrid.model import HybridRNN, HybridMLP, HybridNetwork
from utils import MATCH_STATS_COLUMNS

# If enabled, the model will not consider PLAYERS, COACHES, REFEREES and TEAMS features for training
SIMPLE_MODEL = False
CSV_NAME = 'data_hybrid_simple.csv' if SIMPLE_MODEL else 'data_hybrid.csv'
LOG_FOLDER_NAME = 'hybrid_simple_results' if SIMPLE_MODEL else 'hybrid_results'

# hyper parameters
LEARNING_RATE = 0.001
NUM_EPOCHS = 200
HIDDEN_SIZE = 128
BATCH_SIZE = 32
EARLY_STOP_DELTA = 0.000001
EARLY_STOP_PATIENCE = 20


def count_features(data_csv: str):
    df = pd.read_csv(data_csv, nrows=1)
    num_x_rnn_features = len(df.filter(regex='^home_.*?').columns)
    num_x_mlp_features = len(df.filter(regex='^home_.*?').drop(
        columns=[f'home_{col}' for col in MATCH_STATS_COLUMNS + ['home_score', 'away_score']]).columns)
    return num_x_rnn_features, num_x_mlp_features


if __name__ == '__main__':
    n_feats_rnn, n_feats_mlp = count_features(CSV_NAME)
    dataset = HybridSerieADataset(csv_file=CSV_NAME)
    rnn = HybridRNN(input_size=n_feats_rnn, hidden_size=HIDDEN_SIZE)
    mlp = HybridMLP(input_size=HIDDEN_SIZE * 2 + n_feats_mlp)
    model = HybridNetwork(dataset=dataset, rnn_model=rnn, mlp_model=mlp,
                          learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE)
    summary(model)
    logger = TensorBoardLogger("../training_logs", name=LOG_FOLDER_NAME)
    early_stop = EarlyStopping(monitor="val_loss", min_delta=EARLY_STOP_DELTA, patience=EARLY_STOP_PATIENCE, mode="min",
                               check_on_train_epoch_end=False)
    trainer = Trainer(fast_dev_run=False, gpus=1, max_epochs=NUM_EPOCHS, logger=logger, callbacks=[early_stop])
    trainer.fit(model)
    trainer.test(model)
