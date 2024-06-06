import unittest
import torch
from src.QNetwork import QNetwork

class test_QNetwork(unittest.TestCase):

    def test_create_q_network(self):
        """Test that the network is created accordingly to the
        expecifications provided.
        """

        n_obs = 32
        n_act = 4
        layers = [64, [64, 128], [128, 64], 64]
        device = "cpu"
        net = QNetwork(n_obs, n_act, layers, device)
        expected_layers = len(layers) * 2 -1
        actual_layers = len(net.layer_stack)
        self.assertEqual(expected_layers, actual_layers)
        input_layer = net.layer_stack[0]
        self.assertEqual(n_obs, input_layer.in_features)
        self.assertEqual(input_layer.out_features,
                        layers[0])
        for index in range(2, actual_layers-1, 2):
            actual_layer = net.layer_stack[index]
            expected_in_features = layers[index//2][0]
            expected_out_features = layers[index//2][1]
            self.assertEqual(actual_layer.in_features, expected_in_features)
            self.assertEqual(actual_layer.out_features, expected_out_features)

        output_layer = net.layer_stack[-1]
        self.assertEqual(output_layer.in_features, layers[-1])
        self.assertEqual(output_layer.out_features, n_act)

    def test_forward(self):
        """Test that the forward output is consistent with the action
        space.
        """
        n_obs = 32
        n_act = 4
        layers = [64, [64, 128], [128, 64], 64]
        device = "cpu"
        net = QNetwork(n_obs, n_act, layers, device)
        x = torch.tensor([i for i in range(32)], dtype=torch.float32)
        y = net(x)
        self.assertEqual(len(y), n_act)

    def test_batch_forward(self):
        """Test that the forward output is consistent with the action
        space when a batch is feeded.
        """
        n_obs = 32
        n_act = 4
        layers = [64, [64, 128], [128, 64], 64]
        device = "cpu"
        net = QNetwork(n_obs, n_act, layers, device)
        x = torch.tensor([i for i in range(32)], dtype=torch.float32)
        batch = x.repeat(128, 1)
        y = net(batch)
        self.assertEqual(y.shape[0], batch.shape[0])
        self.assertEqual(y.shape[1], n_act)

if __name__ == '__main__':
    unittest.main()