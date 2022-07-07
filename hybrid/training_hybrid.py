from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torchsummary import summary

from hybrid.dataset import SerieAMatchesWithHistoryDataset
from hybrid.model import HybridRNN, HybridMLP, HybridNetwork
from utils import count_features

LEARNING_RATE = 0.001
NUM_EPOCHS = 20
HIDDEN_SIZE = 128
CSV_NAME = 'data1.csv'

if __name__ == '__main__':
    tot_num_of_feats = count_features(CSV_NAME)
    dataset = SerieAMatchesWithHistoryDataset(csv_file=CSV_NAME)
    rnn_home = HybridRNN(input_size=tot_num_of_feats, hidden_size=HIDDEN_SIZE)
    rnn_away = HybridRNN(input_size=tot_num_of_feats, hidden_size=HIDDEN_SIZE)
    mlp = HybridMLP(HIDDEN_SIZE * 2 + tot_num_of_feats - 5)
    model = HybridNetwork(dataset=dataset, rnn_home_model=rnn_home, rnn_away_model=rnn_away, mlp_model=mlp,
                          learning_rate=LEARNING_RATE)
    print(summary(model))
    logger = TensorBoardLogger("../lightning_logs", name="hybrid_results")
    trainer = Trainer(fast_dev_run=False, gpus=1, max_epochs=NUM_EPOCHS, logger=logger)
    trainer.fit(model)
    trainer.test(model)
