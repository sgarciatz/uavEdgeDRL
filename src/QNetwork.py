import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):


    """
    This network is used to estimate the Q function by means of
    reducing the temporal difference error during its trainning
    pahse
    """


    def __init__(self,
                 n_observations: int,
                 n_actions: int,
                 layers: list,
                 device: str = "cpu"):

        """Create the NN stacking the layers.

        Arguments:
        - n_observations: int = the number of observations and size of
          the input layer.
        - n_actions: int = the number of differenct actions and the
          size of the output layer.
        - layers: a list with the number of Linear layers and their
          number of neurons.
        - device: str = the device for pytorch (cuda or cpu).
        """

        super(QNetwork, self).__init__()
        self.n_actions = n_actions
        input_layer = nn.Linear(n_observations, layers[0])
        output_layer = nn.Linear(layers[-1],
                                      n_actions)
        hidden_layers = []
        for layer in layers[1:-1]:
            hidden_layers.append(nn.Linear(layer[0], layer[1]))
            hidden_layers.append(nn.ReLU())
        self.layer_stack = nn.Sequential(
            input_layer,
            nn.ReLU(),
            *hidden_layers,
            output_layer).to(device)

    def forward(self, x):

        """Feed input data into the Q-Network

        Parameters:
        - x = A minibatch of states.
        """

        logits = self.layer_stack(x)
        return logits

