import torch
import math
import numpy as np

class ActionSelector(object):


    """
    This class is responsible of interacting with the policies. It must
    know how to decrease the learning rate of each one.
    """

    def __init__(self,
                 policy: "Policy",
                 decay_strategy: str = 'linear',
                 start_exploration_rate: float = 1,
                 end_exploration_rate: float = 0.1,
                 decay_rate: float = 1.0) -> None:

        """
        Initialize the action selection policy, the initial and final,
        exploration rates and the decay strategy.
        """
        self.policy = policy
        self.decay_strategy = decay_strategy
        self.start_exploration_rate = start_exploration_rate
        self.end_exploration_rate = end_exploration_rate
        self.exploration_rate = start_exploration_rate
        self.decay_rate = decay_rate
        self.policy.update_exploration_rate(self.exploration_rate)

    def select_action(self, q_values: torch.Tensor) -> int:
        """Given a tensor of the Q-values associated to (state, action)
        tuples selects the action to perform following the policy.

        Args:
            q_values (torch.Tensor):  tensor of the Q-Values of the
            (state, action) tuples.

        Returns:
            int: The index of the action selected according to the
            policy and its Q-Value.
        """
        action = self.policy.select_action(q_values)
        return action

    def decay_exploration_rate(self, current_step: int, training_steps: int) -> None:
        """Decay the exploration rate according to the specified decay
        strategy.

        Args:
            current_step (int): The number of completed steps.
            training_steps (int): The number of step until the
            completion of training.
        """
        current_step += 1
        if (self.decay_strategy == "linear"):
            training_ratio = current_step / training_steps
            self.exploration_rate =\
                (1 - training_ratio) * self.start_exploration_rate\
                + training_ratio * self.end_exploration_rate
            self.policy.update_exploration_rate(self.exploration_rate)
        elif (self.decay_strategy == "freefall"):
            pass
        elif (self.decay_strategy == "exponential"):
            training_ratio = current_step / training_steps
            exponent = -1 * training_ratio * math.e * self.decay_rate
            self.exploration_rate =\
                self.start_exploration_rate * math.exp(exponent)
            self.policy.update_exploration_rate(self.exploration_rate)