import torch
import torch.nn as nn
import torch.nn.functional as F

class QDuelingNetwork(nn.Module):

    """
    This network is a variation of the traditional QNetwork where the
    output layer is divided into two parts.

    The Q value function is calculated as the sum of the V value
    function and the Advantaje value function for each actions minus
    the mean Advantaje value. In that way, the learning process is less
    sensible to meaningless states.
    """


    def __init__(self,
                 n_observations: int,
                 n_actions: int,
                 layers: list,
                 device: str = "cpu"):

        """Create the sequential NN stacking the layers.

        Arguments:
        - n_observations: int = the number of observation and size of
          the input layer.
        - n_actions int = the number of different actions and the size
          of the output lauer.
        - layers: list = a list with the number of Linear layers and
          their number of neurons.
        - device: str = the device for pytorch (cuda or cpu).
        """

        super(QDuelingNetwork, self).__init__()
        self.n_actions = n_actions
        input_layer = nn.Linear(n_observations, layers[0])
        hidden_layers = []
        for layer in layers[1:-1]:
            hidden_layers.append(nn.Linear(layer[0], layer[1]))
            hidden_layers.append(nn.ReLU())
        self.layer_stack = nn.Sequential(
            input_layer,
            nn.ReLU(),
            *hidden_layers
            ).to(device)
        self.value_output_layer = nn.Linear(layers[-1], 1).to(device)
        self.adv_output_layer = nn.Linear(layers[-1], n_actions).to(device)

    def forward(self, x):

        """Feed input data into the QDuelingNetwork

        Parameters:
        - x = A minibatch of states.
        """

        y = self.layer_stack(x)
        y = y.repeat(1,1)
        value = self.value_output_layer(y)
        adv = self.adv_output_layer(y)
        adv_mean = torch.mean(adv, dim=1, keepdim=True)
        output = value + adv - adv_mean
        return output
