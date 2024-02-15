import torch
import torch.nn as nn
import torch.optim as optim
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
                 second_q_estimator: nn.Module = None
                 ):

        """
        Constructor of the Q-estimator updater.
        """

        self.q_estimator = q_estimator
        self.n_actions = self.q_estimator.output_layer.out_features
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.gamma = gamma
        self.device = device
        self.update_policy = update_policy
        
        self.second_q_estimator = second_q_estimator
        

    def get_q_target(self, experience: Experience) -> float:

        """
        Recieves an experience and calculates its target Q-value using
        the target network.
        """

#        state = torch.Tensor(self._flatten_state(experience.next_state))
        state = torch.Tensor(experience.next_state).to(self.device)
        reward = float(experience.reward)
        q_target = torch.full((self.n_actions,), reward).to(self.device)
        if (not experience.done):
            with (torch.no_grad()):
                if (self.second_q_estimator is not None):
                    q_target_next = self.second_q_estimator(state).max()
                else:
                    q_target_next = self.q_estimator(state).max()
            q_target += self.gamma * q_target_next
        return q_target

    def get_q_pred(self, experience: Experience) -> torch.Tensor:

        """
        Recieves an experience and estimates the Q-value of taking each
        possible action.
        """

#        state = torch.Tensor(self._flatten_state(experience.state))
        state = torch.Tensor(experience.state).to(self.device)
        q_preds = self.q_estimator(state)
        return q_preds

    def calculate_q_loss(self, batch):

        """
        Given a batch, calculate the loss using the given loss_fn.
        """ 

        q_preds = []
        q_tars = []
        for experience in batch:
            q_pred = self.get_q_pred(experience)
            q_preds.append(q_pred)
            q_tar = self.get_q_target(experience)
            q_tars.append(q_tar)
        q_preds = torch.cat(q_preds, dim=0).to(self.device)
        q_tars = torch.cat(q_tars, dim=0).to(self.device) 
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
            self.second_q_estimator.load_state_dict(self.q_estimator.state_dict())
        if (self.update_policy == "polyak"):
            q_dict = self.q_estimator.state_dict()
            second_q_dict = self.second_q_estimator.state_dict()
            for key in q_dict:
                second_q_dict[key] = q_dict[key]*polyak_param\
                                     + second_q_dict[key]*(1-polyak_param)
            self.second_q_estimator.load_state_dict(second_q_dict)
            
