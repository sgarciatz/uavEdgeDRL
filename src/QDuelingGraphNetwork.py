import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class QDuelingGraphNetwork(nn.Module):


    """
    This network is a variation of the traditional QNetwork where the
    output layer is divided into two parts and graph convolutions are
    used to aggregate the input.
    
    The Q value function is calculated as the sum of the V value
    function and the Advantaje value function for each actions minus
    the mean Advantaje value. In that way, the learning process is less
    sensible to meaningless states.
    """

    def __init__(self,
                 n_observations: int,
                 n_obs_per_uav: int,
                 n_actions: int,
                 layers: list,
                 n_conv_layers: int = 2,
                 device: str = "cpu"):

        """Hola."""

        super(QDuelingGraphNetwork, self).__init__()
        self.n_actions = n_actions
        self.n_conv_layers = n_conv_layers
        self.g_conv_layer = GCNConv(n_obs_per_uav, n_obs_per_uav)
        first_linear_layer = nn.Linear(n_observations, layers[0])
        hidden_layers = []
        for layer in layers[1:-1]:
            hidden_layers.append(nn.Linear(layer[0], layer[1]))
            hidden_layers.append(nn.ReLU())
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(n_observations, layers[0]),
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

        uav_subgraph = x[1]
        y_uav = self.g_conv_layer(uav_subgraph.x, 
                                  uav_subgraph.edge_index)
        y_uav = torch.flatten(y_uav, start_dim = 0)

        x = torch.cat((y_uav, x[0][y_uav.shape[0]:]), 0)
        
        y = self.linear_layer_stack(x)
        value = self.value_output_layer(y)
        adv = self.adv_output_layer(y)
        adv_mean = torch.mean(adv, dim=0, keepdim=True)
        return value + adv - adv_mean
