import torch
from .Policy import Policy


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
        """Create the policy and and initializes the
        temperature parameter

        Args:
            temperature (float, optional): Defaults to 1.
        """
        self.temperature = temperature

    def select_action(self, q_values: "torch.tensor") -> int:
        """Given a a batch of q_values associated to (state, action)
        tuples, select the one to perform.

        Args:
            q_values (torch.tensor): The batch of q_values

        Returns:
            int: The index of the action to carry out according to the
            policy.
        """

        try:
            exp_values = torch.exp(q_values / self.temperature)
            probabilities = exp_values / torch.sum(exp_values)
            sampled_action = torch.multinomial(probabilities, 1).item()
            return sampled_action
        except:
            self.select_action = lambda q_values: torch.argmax(q_values).item()
            return self.select_action(q_values)


    def update_exploration_rate(self, new_value: float) -> None:
        """Update the exploration rate (decreasing it most of the
        times).

        Args:
            new_value (float): the new value of the exploration rate.
        """

        self.temperature = new_value
