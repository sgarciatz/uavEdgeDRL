import torch
from Policy import Policy


class ActionSelector(object):


    """
    This class is responsible of interacting with the policies. It must
    know how to decrease the learning rate of each one
    """
    
    def __init__(self, policy: Policy, decay_strategy: str = 'linear',
                 start_exploration_rate: float = 1, 
                 end_exploration_rate: float = 0.1) -> None:

        """
        Initialize the action selection policy, the initial and final,
        exploration rates and the decay strategy.
        """
        
        self.policy = policy
        self.decay_strategy = decay_strategy
        self.start_exploration_rate = start_exploration_rate
        self.end_exploration_rate = end_exploration_rate
        self.exploration_rate = start_exploration_rate
        self.policy.update_exploration_rate(self.exploration_rate)

    def select_action(self, q_values: torch.Tensor) -> int:

        """
        Recieves a tensor of the Q-values associated to selecting each
        action.
        
        Parameters:
        - q_values: A tensor of the Q-Values of the (state, action)
         tuples.
         
        Returns:
        - int: The index of the action selected according to the policy
        and its Q-Value.
        """

        action = self.policy.select_action(q_values)
        return action

    def decay_exploration_rate(self, training_done_ratio: int) -> None:

        """
        This function decays the exploration rate according to the 
        specified decay strategy.
        
        Parameters:
        - training_done_ratio: The percentage of the training process
         done expressed as a ratio.
        """

        if (self.decay_strategy == "linear"):
            self.exploration_rate =\
                (1 - training_done_ratio) * self.start_exploration_rate\
                + training_done_ratio * self.end_exploration_rate
        elif (self.decay_strategy == "freefall"):
            pass
        self.policy.update_exploration_rate(self.exploration_rate)

if __name__ == "__main__":
    print("BoltzmannPolicy")
    from BoltzmannPolicy import BoltzmannPolicy
    start_tau = 5
    end_tau = 0.1
    decay_strat = "linear"
    myPolicy = BoltzmannPolicy(start_tau)
    
    myActionSelector = ActionSelector(myPolicy, decay_strat, 
                                      start_tau, end_tau)
    q_values = torch.tensor ([1, 2, 3])
    myActionSelector.select_action(q_values)
    myActionSelector.decay_exploration_rate(0.2)
    myActionSelector.select_action(q_values)
    myActionSelector.decay_exploration_rate(0.4)
    myActionSelector.select_action(q_values)
    myActionSelector.decay_exploration_rate(0.6)
    myActionSelector.select_action(q_values)
    myActionSelector.decay_exploration_rate(0.8)
    myActionSelector.select_action(q_values)
    myActionSelector.decay_exploration_rate(1.0)
    myActionSelector.select_action(q_values)
    
    print("EpsilonGreedyPolicy")
    from EpsilonGreedyPolicy import EpsilonGreedyPolicy
    start_tau = 1
    end_tau = 0.1
    decay_strat = "linear"
    myPolicy = EpsilonGreedyPolicy(start_tau)
    
    myActionSelector = ActionSelector(myPolicy, decay_strat, 
                                      start_tau, end_tau)
    q_values = torch.tensor ([1, 2, 3])
    myActionSelector.select_action(q_values)
    myActionSelector.decay_exploration_rate(0.2)
    myActionSelector.select_action(q_values)
    myActionSelector.decay_exploration_rate(0.4)
    myActionSelector.select_action(q_values)
    myActionSelector.decay_exploration_rate(0.6)
    myActionSelector.select_action(q_values)
    myActionSelector.decay_exploration_rate(0.8)
    myActionSelector.select_action(q_values)
    myActionSelector.decay_exploration_rate(1.0)
    myActionSelector.select_action(q_values)
