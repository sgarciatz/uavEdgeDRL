import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Experience import Experience


class QEstimator(object):


    """
    This class' purpose is to manage the Q-estimator of a DRL agent. It
    may be implemented with a single Q-Network or with multiples, 
    depending on the variation.
    
    Attributes:
    - q_estimator: The Q-Network used for learning the Q-function.
    - n_actions: The size of the action space.
    - optimizer: The optimizer used to adjust the weights of q_estimator.
    - gamma: is the temporal discount factor. γ
    - loss_fn: The function used to calc the loss.
    - device: The CPU or GPU.
    - update_policy: The policy followed to update the second network.
    - second_q_estimator: The second Q-Network for DQL variations
      (target net, Double DQN, Dueling DQN...).
    """

    def __init__(self,
                 q_estimator: nn.Module,
                 optimizer: optim.Optimizer,
                 loss_fn,
                 gamma: float = 0.9,
                 device: str = "cpu",
                 update_policy: str = "replace",
                 second_q_estimator: nn.Module = None,
                 variation: str = "ddqn"
                 ):

        """
        Constructor of the Q-estimator updater.
        """

        self.device = device
        self.q_estimator = q_estimator
        self.n_actions = self.q_estimator.output_layer.out_features
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.gamma = torch.tensor(gamma)
        self.update_policy = update_policy
        self.second_q_estimator = second_q_estimator
        self.second_q_estimator.load_state_dict(self.q_estimator.state_dict())
        self.variation = variation

    def calculate_q_loss(self, batch):

        """
        Given a batch, calculate the loss using the given loss_fn.
        """ 

        states = torch.tensor(np.array([e.state for e in batch]),
                              dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array([[e.action] for e in batch]),
                               dtype=torch.int64).to(self.device)
        next_states = torch.tensor(np.array([e.next_state for e in batch]),
                                   dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array([e.reward for e in batch]),
                                   dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array([1 - e.done for e in batch]),
                             dtype=torch.float32).to(self.device)

        q_preds = torch.flatten(self.q_estimator(states).gather(1, actions))
        with (torch.no_grad()):
            if (self.variation == "ddqn"):
                q_tar_idx = self.second_q_estimator(next_states)\
                                    .max(dim=1).indices.unsqueeze(dim=1)
                q_tar_next = self.q_estimator(next_states)\
                                    .gather(1, q_target_idx).flatten()
            else:
                q_target_next = self.second_q_estimator(next_states)
                                        .max(dim=1).values
        q_tars = rewards + ( dones * self.gamma * q_target_next)
        loss = self.loss_fn(q_preds, q_tars)
        return loss

    def update_q_estimator(self, loss) -> None:

        """
        Updates the primary q_estimator given the loss and the 
        optimizer.
        """

        loss.backward()
        
        self.optimizer.step()
        self.optimizer.zero_grad()

    def update_second_q_estimator(self, polyak_param) -> None:

        """
        Updates the secondary q estimator (polyak or replace).
        """
        if (self.second_q_estimator is None):
            return
        if (self.update_policy == "replace"):
            self.second_q_estimator\
                    .load_state_dict(self.q_estimator.state_dict())
        if (self.update_policy == "polyak"):
            q_dict = self.q_estimator.state_dict()
            second_q_dict = self.second_q_estimator.state_dict()
            for key in q_dict:
                second_q_dict[key] = q_dict[key]*polyak_param\
                                     + second_q_dict[key]*(1-polyak_param)
            self.second_q_estimator.load_state_dict(second_q_dict)

