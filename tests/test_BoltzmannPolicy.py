import unittest
import torch
from src.BoltzmannPolicy import BoltzmannPolicy

class test_BoltzmannPolicy(unittest.TestCase):

    def test_initialization(self):
        """Test the initialization of the Bolztmann Policy.
        """
        temp = 5
        boltzmannPolicy: BoltzmannPolicy = BoltzmannPolicy(temp)

        self.assertEqual(temp, boltzmannPolicy.temperature)

    def test_update_exploration_rate(self):
        """Test that the exploration rate decreases
        """
        temp = 5
        boltzmannPolicy: BoltzmannPolicy = BoltzmannPolicy(temp)
        new_temp = temp - 0.1
        boltzmannPolicy.update_exploration_rate(new_temp)
        self.assertEqual(new_temp, boltzmannPolicy.temperature)

    def test_action_selection_base(self):
        """Test that the actions are selected according to a stochastic
        distribution when the the temperature is > 1.
        """
        temp = 5
        boltzmannPolicy: BoltzmannPolicy = BoltzmannPolicy(temp)
        q_values = torch.tensor([5.0, 1.0, 1.0, 1.0])
        trials = 1000
        best_action_count = 0
        for _ in range(trials):
            if (boltzmannPolicy.select_action(q_values) == 0):
                best_action_count += 1
        minimum_expectation = int(1 / len(q_values) * trials)
        self.assertGreaterEqual(best_action_count, minimum_expectation)


    def test_action_selection_1(self):
        """Test that the actions are selected according to a stochastic
        distribution when the the temperature is 1.
        """
        temp = 1
        boltzmannPolicy: BoltzmannPolicy = BoltzmannPolicy(temp)
        q_values = torch.tensor([5.0, -10.0, 1.0, 1.0])
        trials = 1000
        best_action_count = 0
        for _ in range(trials):
            if (boltzmannPolicy.select_action(q_values) == 0):
                best_action_count += 1
        minimum_expectation = int(q_values[0] / q_values.sum(dim=0) * trials)

        self.assertGreater(best_action_count, minimum_expectation)

    def test_action_selection(self):
        """Check that lower temperature values yield a higher change to
        select the best possible action
        """
        temp_low = 0.05
        temp_high = 2
        boltzmann_p1: BoltzmannPolicy = BoltzmannPolicy(temp_low)
        boltzmann_p2: BoltzmannPolicy = BoltzmannPolicy(temp_high)
        q_values = torch.tensor([2.0, 1.5, 1.0, 0.5])
        trials = 1000
        best_action_count_1 = 0
        best_action_count_2 = 0
        for _ in range(trials):
            if (boltzmann_p1.select_action(q_values) == 0):
                best_action_count_1 += 1
            if (boltzmann_p2.select_action(q_values) == 0):
                best_action_count_2 += 1

        self.assertGreater(best_action_count_1, best_action_count_2)

if __name__ == '__main__':
    unittest.main()