from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchsummary import summary

from rnn.dataset import RNNSerieADataset
from rnn.model import RNN
from utils import count_features

# If enabled, the model will not consider PLAYERS, COACHES, REFEREES and TEAMS features for training
SIMPLE_MODEL = False
CSV_NAME = 'data_rnn_simple.csv' if SIMPLE_MODEL else 'data_rnn.csv'
LOG_FOLDER_NAME = 'rnn_simple_results' if SIMPLE_MODEL else 'rnn_results'

# hyper parameters
LEARNING_RATE = 0.001
NUM_EPOCHS = 400
HIDDEN_SIZE = 128
BATCH_SIZE = 32
EARLY_STOP_DELTA = 0.000001
EARLY_STOP_PATIENCE = 15

if __name__ == '__main__':
    n_of_feats = count_features(CSV_NAME) - 3
    dataset = RNNSerieADataset(csv_file=CSV_NAME)
    model = RNN(dataset=dataset, input_size=n_of_feats, hidden_size=HIDDEN_SIZE, batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE)
    summary(model)
    logger = TensorBoardLogger("../training_logs", name=LOG_FOLDER_NAME)
    early_stop = EarlyStopping(monitor="val_loss", min_delta=EARLY_STOP_DELTA, patience=EARLY_STOP_PATIENCE, mode="min",
                               check_on_train_epoch_end=False)
    trainer = Trainer(gpus=1, max_epochs=NUM_EPOCHS, logger=logger, callbacks=[early_stop])
    trainer.fit(model)
    trainer.test(model)
