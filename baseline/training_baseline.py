from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torchsummary import summary

from baseline.dataset import SerieAMatchesDataset
from baseline.model import MLP
from utils import count_features

# If enabled, the model will not consider PLAYERS, COACHES, REFEREES and TEAMS features for training
SIMPLE_MODEL = False
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
CSV_NAME = 'data_baseline_simple.csv' if SIMPLE_MODEL else 'data_baseline.csv'

if __name__ == '__main__':
    n_of_feats = count_features(CSV_NAME) - 3
    dataset = SerieAMatchesDataset(csv_file=CSV_NAME)
    model = MLP(dataset=dataset, input_size=n_of_feats, learning_rate=LEARNING_RATE)
    print(summary(model))
    logger = TensorBoardLogger("../training_logs", name="baseline_results")
    trainer = Trainer(gpus=1, max_epochs=NUM_EPOCHS, logger=logger)
    trainer.fit(model)
    trainer.test(model)
