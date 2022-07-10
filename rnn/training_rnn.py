from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torchsummary import summary

from rnn.dataset import RNNSerieADataset
from rnn.model import RNN
from utils import count_features

# If enabled, the model will not consider PLAYERS, COACHES, REFEREES and TEAMS features for training
SIMPLE_MODEL = True
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
HIDDEN_SIZE = 128
BATCH_SIZE = 32
CSV_NAME = 'data_rnn_simple.csv' if SIMPLE_MODEL else 'data_rnn.csv'

if __name__ == '__main__':
    n_of_feats = count_features(CSV_NAME) - 3
    dataset = RNNSerieADataset(csv_file=CSV_NAME)
    model = RNN(dataset=dataset, input_size=n_of_feats, hidden_size=HIDDEN_SIZE, batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE)
    print(summary(model))
    logger = TensorBoardLogger("../training_logs", name="rnn_results")
    trainer = Trainer(gpus=1, max_epochs=NUM_EPOCHS, logger=logger)
    trainer.fit(model)
    trainer.test(model)
