import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

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
                 n_obs: int,
                 n_uavs: int,
                 n_mss: int,
                 layers: list,
                 device: str = "cpu"):

        """Hola."""

        super(QDuelingGraphNetwork, self).__init__()
        self.n_uavs = n_uavs
        self.dim = int(math.sqrt(self.n_uavs))
        self.n_mss = n_mss
        self.n_actions = n_uavs * n_mss
        self.device = device
        input_layer = nn.Linear(n_obs, layers[0])
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
        self.adv_output_layer = nn.Linear(layers[-1], self.n_actions).to(device)
        self._build_kernel(self.dim)

    def forward(self, x):

        """Feed input data into the QDuelingNetwork

        Parameters:
        - x = A minibatch of states.
        """

        x = self.apply_conv(x.repeat(1,1))
        y = self.layer_stack(x)
        value = self.value_output_layer(y)
        adv = self.adv_output_layer(y)
        adv_mean = torch.mean(adv, dim=1, keepdim=True)
        return value + adv - adv_mean
        return adv

    def apply_conv(self, x):

        """ Apply a non trainable convolutional cross correlation
        layer to aggregate all the info about node vecinity.
        """


        ms_heats = x[:, 0:self.n_mss * self.n_uavs]
        ms_heats = ms_heats.reshape((-1, self.n_mss, self.n_uavs))
        ms_heats = ms_heats.transpose(2, 1)
        ms_heats = ms_heats.reshape((-1, self.dim, self.dim, self.n_mss))
        y_conv = F.conv2d(ms_heats,
                         self.kernel,
                         stride=1,
                         groups=self.n_mss,
                         padding="same").to(self.device)
        y_conv = y_conv.reshape((-1, self.n_uavs, self.n_mss))
        y_conv = y_conv.transpose(2, 1)
        y_conv = y_conv.reshape((-1, self.n_mss * self.n_uavs))
        y_conv = torch.div(y_conv, self.max_cost)
        x[:, 0:self.n_mss * self.n_uavs] = y_conv
        return x

    def _build_kernel(self, diameter):

        """ Given the scenario's graph, create a kernel based on its
        diameter.

        Example: with a diameter of 4, the kernel would look like this
            4 4 4 4 4 4 4
            4 3 3 3 3 3 4
            4 3 2 2 2 3 4
            4 3 2 1 2 3 4
            4 3 2 2 2 3 4
            4 3 3 3 3 3 4
            4 4 4 4 4 4 4

        Parameters:
        - diameter = The graph diameter.
        """

        dim = (diameter * 2) - 1

        self.kernel = np.zeros((dim, dim), dtype=np.float32)
        for i in range(diameter):
                for j in range(i+1):
                    self.kernel[i][j] = diameter - j
        aux = self.kernel[0:diameter,0:diameter]
        aux_t_left = aux + aux.T - np.diag(np.diag(aux))
        aux_t_right = np.flip(aux_t_left, axis=1)[:,1:]
        aux_b_left = np.flip(aux_t_left[0:diameter-1,0:diameter], axis=0)
        aux_b_right = np.flip(aux_b_left, axis=1)[:,1:]
        self.kernel[0:diameter,0:diameter] = aux_t_left
        self.kernel[diameter:,0:diameter] = aux_b_left
        self.kernel[0:diameter,diameter:] = aux_t_right
        self.kernel[diameter:,diameter:] = aux_b_right
        self.kernel = torch.from_numpy(self.kernel)
        self.kernel = self.kernel.repeat(self.n_mss,1,1,1).to(self.device)
        self.max_cost = torch.tensor(self.kernel[0:diameter,0:diameter],
                                     dtype=torch.float32).to(self.device)
        self.max_cost = self.max_cost.sum()