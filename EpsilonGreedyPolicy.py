import torch
from Policy import Policy


class EpsilonGreedyPolicy(Policy):


    """
    The Εpsilon Greedy (E-Greedy) policy provides a mechanism to select greedy actions
    with a probability of (1 - E) and select actions randomly with a E
    probability. 
    
    E is decayed as the training advances to favor exploitation over
    exploration.
    """
    
    
    def __init__(self, e: float = 1.0) -> None:

        """
        Initializes the epsilon parameter
        """
        self.epsilon = e

    def select_action(self, q_values: torch.Tensor):

        """
        Choose the action to apply to the enviroment.
        
        Arguments:
        - q_values: A tensor is expected with the Q-values of taking an
         action in the current enviroment's state.
        
        Returns:
        - int: The index that refers to the selected action
        """
        
        random_number = torch.rand(1).item()

        condition = (random_number - self.epsilon) > 0.0
        if (condition):
            action = torch.argmax(q_values).item()
            
        else:
            probabilities = torch.ones(q_values.shape, dtype= torch.float64)
            action = torch.multinomial(probabilities, 1).item()

        return action
        
    def update_exploration_rate(self, new_value) -> None:

        """
        Update the current value for epsilon
        """
        
        self.epsilon = new_value
if __name__ == "__main__":
    epsilon = 0.1
    myPolicy = EpsilonGreedyPolicy(epsilon)
    q_values = torch.tensor ([3, 2, 3, 1])
    myPolicy.select_action(q_values)
