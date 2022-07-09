from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from rnn.dataset import RNNSerieADataset
from rnn.model import RNN
from utils import count_features

LEARNING_RATE = 0.001
NUM_EPOCHS = 20
HIDDEN_SIZE = 128
BATCH_SIZE = 1
CSV_NAME = 'data_rnn.csv'

if __name__ == '__main__':
    n_of_feats = count_features(CSV_NAME) - 3
    dataset = RNNSerieADataset(csv_file=CSV_NAME)
    model = RNN(dataset=dataset, input_size=n_of_feats, hidden_size=HIDDEN_SIZE, output_size=3, batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE)
    logger = TensorBoardLogger("../lightning_logs", name="rnn_results")
    trainer = Trainer(gpus=1, max_epochs=NUM_EPOCHS, logger=logger)
    trainer.fit(model)
    trainer.test(model)
