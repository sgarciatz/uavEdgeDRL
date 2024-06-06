import unittest
import torch
import random
from src.QEstimator import QEstimator
from src.QDuelingNetwork import QDuelingNetwork
from src.Experience import Experience


class test_QNetwork(unittest.TestCase):

    def test_initialization(self):
        """check that the QEstimator is initialized correctly.
        """
        n_obs = 32
        n_act = 5
        layers = [32, [32, 64], [64, 128], [128, 64], [64, 32], 32]
        gamma = 0.9
        device = "cpu"
        update_policy = "replace"
        update_freq = 50
        variation = "dqn"
        policy_net = QDuelingNetwork(n_obs, n_act, layers, device)
        target_net = QDuelingNetwork(n_obs, n_act, layers, device)
        target_net.load_state_dict(policy_net.state_dict())
        loss_fn = torch.nn.HuberLoss()
        learning_rate = 1e-4
        optimizer = torch.optim.AdamW(policy_net.parameters(),
                                      learning_rate,
                                      amsgrad=True)
        q_estimator = QEstimator(policy_net,
                                 optimizer,
                                 loss_fn,
                                 gamma,
                                 device,
                                 update_policy,
                                 update_freq,
                                 target_net,
                                 variation)
        self.assertEqual(policy_net, q_estimator.q_estimator)
        self.assertEqual(target_net, q_estimator.second_q_estimator)
        self.assertEqual(optimizer, q_estimator.optimizer)
        self.assertEqual(loss_fn, q_estimator.loss_fn)
        self.assertEqual(gamma, q_estimator.gamma)
        self.assertEqual(device, q_estimator.device)
        self.assertEqual(update_policy, q_estimator.update_policy)
        self.assertEqual(variation, q_estimator.variation)

        n_experiences = 100
        random.seed(24)
        batch = []
        for index in range(n_experiences):
            state = [float(random.randrange(0, 9)) for _ in range(n_obs)]
            action = random.randint(0, 4)
            reward = random.randrange(-1, 1)
            next_state = [float(random.randrange(0, 9)) for _ in range(n_obs)]
            done = False
            priority = 1000
            experience = Experience(state,
                                    action,
                                    reward,
                                    next_state,
                                    done,
                                    priority)
            batch.append(experience)
        q_estimator.calculate_q_loss(batch)
        self.assertEqual(policy_net, q_estimator.q_estimator)
        self.assertEqual(target_net, q_estimator.second_q_estimator)
        self.assertEqual(optimizer, q_estimator.optimizer)
        self.assertEqual(loss_fn, q_estimator.loss_fn)
        self.assertEqual(gamma, q_estimator.gamma)
        self.assertEqual(device, q_estimator.device)
        self.assertEqual(update_policy, q_estimator.update_policy)
        self.assertEqual(variation, q_estimator.variation)


    def test_replace_update(self):
        """Check that when update_second_q_estimator is called,
        both state dicts are the same.
        """
        n_obs = 32
        n_act = 5
        layers = [32, [32, 64], [64, 128], [128, 64], [64, 32], 32]
        gamma = 0.9
        device = "cpu"
        update_policy = "replace"
        update_freq = 50
        variation = "dqn"
        policy_net = QDuelingNetwork(n_obs, n_act, layers, device)
        target_net = QDuelingNetwork(n_obs, n_act, layers, device)
        target_net.load_state_dict(policy_net.state_dict())
        loss_fn = torch.nn.HuberLoss()
        learning_rate = 1e-4
        optimizer = torch.optim.AdamW(policy_net.parameters(),
                                      learning_rate,
                                      amsgrad=True)
        q_estimator = QEstimator(policy_net,
                                 optimizer,
                                 loss_fn,
                                 gamma,
                                 device,
                                 update_policy,
                                 update_freq,
                                 target_net,
                                 variation)
        n_experiences = 100
        random.seed(24)
        batch = []
        for index in range(n_experiences):
            state = [float(random.randrange(0, 9)) for _ in range(n_obs)]
            action = random.randint(0, 4)
            reward = random.randrange(-1, 1)
            next_state = [float(random.randrange(0, 9)) for _ in range(n_obs)]
            done = False
            priority = 1000
            experience = Experience(state,
                                    action,
                                    reward,
                                    next_state,
                                    done,
                                    priority)
            batch.append(experience)
        loss, _ = q_estimator.calculate_q_loss(batch)
        self.assertEqual(
            q_estimator.q_estimator.state_dict().__str__(),
            q_estimator.second_q_estimator.state_dict().__str__())
        q_estimator.update_q_estimator(loss)
        self.assertNotEqual(
            q_estimator.q_estimator.state_dict().__str__(),
            q_estimator.second_q_estimator.state_dict().__str__())
        q_estimator.update_second_q_estimator(update_freq)
        self.assertEqual(
            q_estimator.q_estimator.state_dict().__str__(),
            q_estimator.second_q_estimator.state_dict().__str__())

    def test_polyak_update(self):
        """Check that when update_second_q_estimator is called,
        both state dicts are the same.
        """
        n_obs = 32
        n_act = 5
        layers = [32, [32, 64], [64, 128], [128, 64], [64, 32], 32]
        gamma = 0.9
        device = "cpu"
        update_policy = "polyak"
        update_freq = 50
        variation = "dqn"
        policy_net = QDuelingNetwork(n_obs, n_act, layers, device)
        target_net = QDuelingNetwork(n_obs, n_act, layers, device)
        target_net.load_state_dict(policy_net.state_dict())
        loss_fn = torch.nn.HuberLoss()
        learning_rate = 1e-4
        optimizer = torch.optim.AdamW(policy_net.parameters(),
                                      learning_rate,
                                      amsgrad=True)
        q_estimator = QEstimator(policy_net,
                                 optimizer,
                                 loss_fn,
                                 gamma,
                                 device,
                                 update_policy,
                                 update_freq,
                                 target_net,
                                 variation)
        n_experiences = 100
        random.seed(24)
        batch = []
        for index in range(n_experiences):
            state = [float(random.randrange(0, 9)) for _ in range(n_obs)]
            action = random.randint(0, 4)
            reward = random.randrange(-1, 1)
            next_state = [float(random.randrange(0, 9)) for _ in range(n_obs)]
            done = False
            priority = 1000
            experience = Experience(state,
                                    action,
                                    reward,
                                    next_state,
                                    done,
                                    priority)
            batch.append(experience)
        loss, _ = q_estimator.calculate_q_loss(batch)
        self.assertEqual(
            q_estimator.q_estimator.state_dict().__str__(),
            q_estimator.second_q_estimator.state_dict().__str__())
        q_estimator.update_q_estimator(loss)
        self.assertNotEqual(
            q_estimator.q_estimator.state_dict().__str__(),
            q_estimator.second_q_estimator.state_dict().__str__())
        q_estimator.update_second_q_estimator(update_freq)
        q_estimator.update_second_q_estimator(update_freq)
        self.assertNotEqual(
            q_estimator.q_estimator.state_dict().__str__(),
            q_estimator.second_q_estimator.state_dict().__str__())

if __name__ == '__main__':
    unittest.main()