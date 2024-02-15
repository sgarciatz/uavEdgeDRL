import torch
from Policy import Policy


class BoltzmannPolicy(Policy):


    """
    The Boltzmann Policy tries to improve over random exploration by 
    selecting action using their relative Q-values. The action with
    higher Q-values will be chosen with a higher probability.
    
    Attributes:
    
    - temperature (τ): The temperature attribute controls how uniform or
     concentrated the probability distribution is. High values of τ
     push the distribution to become more uniform, low values make it
     more concentrated. 
    """
    
    def __init__(self, temperature: float = 1) -> None:

        """
        Initializes the temperature parameters
        """
        self.temperature = temperature
        
    def select_action(self, q_values) -> int:

        """
        Choose the action to apply to the enviroment.
        
        Parameters:
        - q_values: A tensor is expected with the Q-values of taking an
         action in the current enviroment's state.
        
        Returns:
        - int: The index that refers to the selected action
        """

        exp_values = torch.exp(q_values / self.temperature)
        probabilities = exp_values / torch.sum(exp_values)
        sampled_action = torch.multinomial(probabilities, 1).item()
        return sampled_action

    def update_exploration_rate(self, new_value) -> None:

        """
        Update the temperature parameter
        """

        self.temperature = new_value

if __name__ == "__main__":
    
    temperature = 1
    myPolicy = BoltzmannPolicy(temperature)
    q_values = torch.tensor ([1, 2, 3])
    myPolicy.select_action(q_values)
