import os
from unittest import TestCase
from si.io.csv_file import read_csv
from si.io.data_file import read_data_file
from si.metrics.mse import mse
from si.model_selection.split import train_test_split
from datasets import DATASETS_PATH
from si.neural_networks.activation import ReLUActivation
from si.neural_networks.layers import DenseLayer
from si.neural_networks.losses import MeanSquaredError
from si.neural_networks.neural_network import NeuralNetwork
from si.neural_networks.optimizers import SGD


class TestLosses(TestCase):

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_fit(self):
        net = NeuralNetwork(epochs=100, batch_size=16, optimizer=SGD, learning_rate=0.01, verbose=True,
                        loss=MeanSquaredError, metric=mse)
        n_features = self.train_dataset.X.shape[1]
        net.add(DenseLayer(6, (n_features,)))
        net.add(ReLUActivation())
        net.add(DenseLayer(4))
        net.add(ReLUActivation())
        net.add(DenseLayer(1))

        # train
        net.fit(self.train_dataset)

        # test
        out = net.predict(self.train_dataset)

        self.assertEqual(out.shape[0], self.train_dataset.shape()[0])