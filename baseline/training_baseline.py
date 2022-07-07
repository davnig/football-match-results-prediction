from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from baseline.dataset import SerieAMatchesDataset
from baseline.model import MLP
from utils import count_features

LEARNING_RATE = 0.001
NUM_EPOCHS = 20

if __name__ == '__main__':
    csv_name = 'data_baseline.csv'
    n_of_feats = count_features(csv_name)
    dataset = SerieAMatchesDataset(csv_file=csv_name)
    model = MLP(dataset=dataset, input_size=n_of_feats, learning_rate=LEARNING_RATE)
    logger = TensorBoardLogger("../lightning_logs", name="baseline_results")
    trainer = Trainer(gpus=1, max_epochs=NUM_EPOCHS, logger=logger)
    trainer.fit(model)
    trainer.test(model)
