import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):


    """
    This network is used to estimate the Q function by means of
    reducing the temporal difference error during its trainning
    pahse
    """


    def __init__(self, n_observations, n_actions):

        """
        Create the NN stacking the layers.
        
        Arguments:
        - n_observations: the number of observations and size of the
                          input layer.
        - n_actions: the number of differenct actions and the size of
                     the output layer
        """

        super(QNetwork, self).__init__()
        self.input_layer = nn.Linear(n_observations, 64)
        self.hidden_layer = nn.Linear(self.input_layer.out_features,
                                      64)
        self.output_layer = nn.Linear(self.hidden_layer.out_features,
                                      n_actions)

        self.layer_stack = nn.Sequential(
            self.input_layer,
            nn.ReLU(),
            self.hidden_layer,
            nn.ReLU(),
            self.output_layer
        )

    def forward(self, x):

        """
        Feed input data into the Q-Network
        """

        logits = self.layer_stack(x)
        return logits
