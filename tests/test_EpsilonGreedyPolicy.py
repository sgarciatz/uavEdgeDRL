import unittest
import torch
import statistics
import numpy as np
from src.EpsilonGreedyPolicy import EpsilonGreedyPolicy

class test_EpsilonGreedyPolicy(unittest.TestCase):


    def test_initialization(self):
        """Create the EpsilonGreedyPolicy with a given exploration rate.
        Check that the exploration rate is consistent.
        """
        e = 1.0
        epsilonG_policy = EpsilonGreedyPolicy(e)
        self.assertEqual(e, epsilonG_policy.epsilon)

    def test_update_exploration_rate(self):
        """Test that the exploration rate is updated.
        """
        e = 1.0
        epsilonG_policy = EpsilonGreedyPolicy(e)
        new_e = e - 0.05
        epsilonG_policy.update_exploration_rate(new_e)
        self.assertEqual(new_e, epsilonG_policy.epsilon)

    def test_action_selection_1(self):
        """Test that with a exploration rate of 1, the action selection
        is random.
        """
        e = 1.0
        epsilonG_policy = EpsilonGreedyPolicy(e)
        q_values = torch.tensor([2.0, 1.5, 1.0, 0.5, 3.0])
        trials = 1000
        actions_selected = []
        for _ in range(trials):
            action = epsilonG_policy.select_action(q_values)
            actions_selected.append(action)

        actions_selected = np.array(actions_selected)
        mean = actions_selected.mean() / (len(q_values) - 1)
        expected_mean = 0.5
        mean_tolerance = 0.05
        std_dev = actions_selected.std() / (len(q_values) - 1)
        expected_std_dev = 0.3
        std_dev_tolerance = 0.05
        self.assertLessEqual(abs(expected_mean - mean) ,
                             abs(expected_mean - mean_tolerance))
        self.assertLessEqual(abs(expected_std_dev - std_dev) ,
                             abs(expected_std_dev - std_dev_tolerance))


    def test_action_selection_0(self):
        """Test that when the exploration rate is 0, the best action is
        always chosen.
        """
        e = 0.0
        epsilonG_policy = EpsilonGreedyPolicy(e)
        q_values = torch.tensor([1.0, 1.5, 3.0, 0.5, 2.0])
        trials = 1000
        best_action_count = 0
        for _ in range(trials):
            if (epsilonG_policy.select_action(q_values) == q_values.argmax()):
                best_action_count += 1
        self.assertEqual(trials, best_action_count)

    def test_action_selection_n(self):
        """Test that when the exploration rate is 0 < n < 1, the best
        action is frequently chosen.
        """

        e = 0.3
        epsilonG_policy = EpsilonGreedyPolicy(e)
        q_values = torch.tensor([1.0, 1.5, 3.0, 0.5, 2.0])
        trials = 1000
        best_action_count = 0
        for _ in range(trials):
            if (epsilonG_policy.select_action(q_values) == q_values.argmax()):
                best_action_count += 1
        minimum_expectation = 1/len(q_values) * trials
        best_action_count = float(best_action_count)
        self.assertGreater(best_action_count, minimum_expectation)

if __name__ == '__main__':
    unittest.main()