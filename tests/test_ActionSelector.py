import unittest
import torch
from src.ActionSelector import ActionSelector
from src.BoltzmannPolicy import BoltzmannPolicy
from src.EpsilonGreedyPolicy import EpsilonGreedyPolicy

class test_ActionSelector(unittest.TestCase):


    def test_bolztmann_action_selector(self):
        """Check that the ActionSelector is initialized correctly.
        """
        start_exploration_rate = 5.0
        end_explotarion_rate = 0.1
        decay_rate = 1.0
        decay_strat = "exponential"
        boltzmannPolicy = BoltzmannPolicy(start_exploration_rate)
        actionSelector = ActionSelector(
            boltzmannPolicy,
            decay_strategy=decay_strat,
            start_exploration_rate=start_exploration_rate,
            end_exploration_rate=end_explotarion_rate,
            decay_rate=decay_rate)
        q_values = torch.tensor([5.0, 4.0, 1.0, 1.0, 1.0, 1.0])
        trials = 100
        best_action_count = 0
        second_best_action_count = 0
        other_action_count = 0
        for trial in range(trials):
            action = actionSelector.select_action(q_values)
            if (action == 0):
                best_action_count += 1
            if (action == 1):
                second_best_action_count += 1
            if (action == 2):
                other_action_count += 1
            actionSelector.decay_exploration_rate(trial, trials)
        self.assertGreater(best_action_count, second_best_action_count)
        self.assertGreater(second_best_action_count, other_action_count)

    def test_epsilon_action_selector(self):
        """Check that the agent selects actions according to the epsilon
        greedy policy.
        """
        start_exploration_rate = 1.0
        end_explotarion_rate = 0.1
        decay_strat = "linear"
        epsilonGreedyPolicy = EpsilonGreedyPolicy(start_exploration_rate)
        actionSelector = ActionSelector(
            epsilonGreedyPolicy,
            decay_strategy=decay_strat,
            start_exploration_rate=start_exploration_rate,
            end_exploration_rate=end_explotarion_rate)
        q_values = torch.tensor([5.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5])
        trials = 1000
        best_action_count = 0
        other_action_count = 0
        for trial in range(trials):
            if (actionSelector.select_action(q_values) == 0):
                best_action_count += 1
            else:
                other_action_count += 1
            actionSelector.decay_exploration_rate(trial, trials)
        self.assertGreater(best_action_count, other_action_count)

    def test_exponential_decay(self):
        """Test that the exploration rate is decayed.
        """
        start_exploration_rate = 3.0
        end_explotarion_rate = 0.1
        decay_rate = 2.5
        decay_strat = "exponential"
        boltzmannPolicy = BoltzmannPolicy(start_exploration_rate)
        actionSelector_b = ActionSelector(
            boltzmannPolicy,
            decay_strategy=decay_strat,
            start_exploration_rate=start_exploration_rate,
            end_exploration_rate=end_explotarion_rate,
            decay_rate=decay_rate)
        steps = 1000
        prev_e = actionSelector_b.exploration_rate
        actionSelector_b.decay_exploration_rate(0, steps)
        e = actionSelector_b.exploration_rate
        diff = prev_e - e
        self.assertGreater(diff, 0)
        for step in range(1,steps):
            prev_diff = diff
            actionSelector_b.decay_exploration_rate(step, steps)
            prev_e = e
            e = actionSelector_b.exploration_rate
            diff = prev_e - e
            self.assertGreater(prev_diff, diff)

    def test_linear_decay(self):
        """Test that the exploration rate is decayed.
        """
        start_exploration_rate = 3.0
        end_explotarion_rate = 0.1
        decay_rate = 1000.0
        decay_strat = "linear"
        boltzmannPolicy = BoltzmannPolicy(start_exploration_rate)
        actionSelector_b = ActionSelector(
            boltzmannPolicy,
            decay_strategy=decay_strat,
            start_exploration_rate=start_exploration_rate,
            end_exploration_rate=end_explotarion_rate,
            decay_rate=decay_rate)
        steps = 1000
        prev_e = actionSelector_b.exploration_rate
        actionSelector_b.decay_exploration_rate(0, steps)
        e = actionSelector_b.exploration_rate
        diff = prev_e - e
        self.assertGreater(diff, 0)
        for step in range(1,steps):
            prev_diff = diff
            actionSelector_b.decay_exploration_rate(step, steps)
            prev_e = e
            e = actionSelector_b.exploration_rate
            diff = prev_e - e
            self.assertAlmostEqual(prev_diff, diff)

if __name__ == '__main__':
    unittest.main()