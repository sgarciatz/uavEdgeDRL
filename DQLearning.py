import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from itertools import count
from gymnasium.spaces.utils import flatten_space
import random
import math
from QNetwork import QNetwork
from QEstimator import QEstimator
from ExperienceSampler import ExperienceSampler
from ActionSelector import ActionSelector
from Experience import Experience
from EpsilonGreedyPolicy import EpsilonGreedyPolicy
from BoltzmannPolicy import BoltzmannPolicy
import numpy as np
import random
import sys

class DQLearning(object):


    """
    This class represents the DQN agent and holds all the
    hyperparameters relative to the training process.

    Attributes:
    - max_episodes: the number of episodes of the training phase.
    - batch_size: is the number of transitions sampled from the replay
       buffer. B
    - gamma: is the temporal discount factor. γ
    - exploration_start: holds the start value of ε for the ε-greedy
     policy.
    - exploration_end: holds the final value of ε for the ε-greedy
     policy.
    - exploration_decay: controls the rate of exploration decay. Higher
     means a slower decay.
    - soft_update_rate: controls the soft update between the QNetwork
     and the target QNetwork. τ
    - learning_rate: is the learning rate of the optimizer.
    """


    def __init__(self, parameters):

        """
        Initialize the agent using the characteristic of the environment
        and load the training hyperparameters.

        Arguments:
        - parameters: the dictionary with all the information relevant
          for training.
        """

        #Set hyperparameters
        self.training_steps = parameters["training_steps"]
        self.h = parameters["h"]
        self.batches = parameters["batches"]
        self.batch_size = parameters["batch_size"]
        self.updates_per_batch = parameters["updates_per_batch"]

        #Set the device for torch
        self.device = parameters["device"]

        # Set Components
        self.environment = parameters["env"]
        self.q_estimator = parameters["q_estimator"]
        self.experience_sampler = parameters["memory"]
        self.action_selector = parameters["action_selector"]

        #Prepare the logging class TrainLogger
        self.logger = parameters["logger"]

    def _flatten_state(self, state) -> list:

        """
        Helper function used to flatten the state dict in order to feed
        it to other methods.
        """
        raw_state = []
        for value in list(state.values()):
            if (isinstance(value, np.ndarray)):
                for value2 in value:
                    raw_state.append(value2)
            else:
                raw_state.append(value)
        return raw_state

    def _gather_experiences(self):

        """
        Using the current policy (policy_net), fill the replay memory
        buffer. This is the first step of the loop of the DQN
        Algorithm.
        """

        n_experiences = self.h
        done = True
        experience = None
        for i in range(n_experiences):
            if (done):
                seed = random.randint(0, sys.maxsize)
                state, info = self.environment.reset(seed=seed)
            else:
                state = next_state
#            raw_state = torch.Tensor(self._flatten_state(state))
            raw_state = torch.Tensor(state).to(self.device)
            with (torch.no_grad()):
                q_tar = self.q_estimator.q_estimator(raw_state)
            action = self.action_selector.select_action(q_tar)
            next_state, reward, terminated, truncated, info =\
                self.environment.step(action)
            done = terminated or truncated
            experience = Experience(state, action, reward, next_state, done, 99)
            self.experience_sampler.add_experience(experience)

    def validate_learning(self, n_validations: int):

        """Use the Q estimator network with its current weights to
        check how well the agent performs in a enviroment during an
        episode.

        Parameters:
        - n_validations: int = The number of episodes to execute.
        """

        rewards = []
        ep_lengths = []
        for _ in range(n_validations):
            done = False
            ep_length = 0
            ep_reward = 0
            state, info = self.environment.reset()
#            state = torch.Tensor(self._flatten_state(state))
            state = torch.Tensor(state).to(self.device)
            while (not done):
                with (torch.no_grad()):
                    q_estimate = self.q_estimator.q_estimator(state)
                action = self.action_selector.select_action(q_estimate)
                next_state, reward, terminated, truncated, info =\
                    self.environment.step(action)
#                state = torch.Tensor(self._flatten_state(next_state))
                state = torch.Tensor(next_state).to(self.device)
                done = terminated or truncated
                ep_length += 1
                ep_reward += reward
            rewards.append(ep_reward)
            ep_lengths.append(ep_length)

        return sum(rewards) / n_validations, sum(ep_lengths) / n_validations

    def scripted_validate_learning(self, action_sequence: list[int]):

        """Use a predefined action sequence to carry out an episode.
        This method is intented for testing purposes.

        Parameters:
        - action_squence: list[int] = The sequence of action sto carry out.
        """

        state, _ = self.environment.reset(seed = 0)

        for action in action_sequence:
            self.environment.step(action)

    def _sample_experience_batch(self):

        """
        Using the ExperienceSampler, sample batch_size experiences
        """

        batch = self.experience_sampler.sample_experience(self.batch_size)
        return batch

    def train(self):

        """
        The maing training loop of the DQN agent. The training consists
        of the execution of the following steps:

        For each step of training:
            generate h experiences with the current policy
            gather B batches, For each bacth:
                For each u in batch_updates:
                    calculate Qtar of all experiences
                    estimate Q^ of all experiences
                    calc the loss between Qtar and Q^
                    update Qfunc estimator's parameters
            decrease exploring rate
        """

        self.logger.print_training_header()
        for step in range(self.training_steps):
            step_losses = []
            self._gather_experiences()
            for b in range(self.batches):
                batch = self._sample_experience_batch()
                for update in range(self.updates_per_batch):
                    loss, td_error = self.q_estimator.calculate_q_loss(batch)
                    self.q_estimator.update_q_estimator(loss)
                    self.experience_sampler.update_batch_priorities(
                        batch,
                        td_error)
                    step_losses.append(loss.item())
            reward, ep_length = self.validate_learning(10)
            expl_rate = self.action_selector.exploration_rate
            self.logger.add_training_step(step,
                                          expl_rate,
                                          sum(step_losses) / len(step_losses),
                                          reward,
                                          ep_length)
            self.logger.print_training_step()
            self.action_selector.decay_exploration_rate(step,
                                                        self.training_steps)
            self.q_estimator.update_second_q_estimator(step)
        self.logger.print_training_footer()
        self.q_estimator.pickle_model()

if __name__ == "__main__":
    import gymnasium as gym
    import gym_network
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
#    myEnv = gym.make('NetworkEnv-v0', input_file='/home/santiago/Documents/Trabajo/Workspace/GLOMIM/glomim_v1/InputScenarios/paper2_small_01.json')
    myEnv = gym.make("CartPole-v1")
#    myEnv = gym.make("TestEnv-v0")
    agent = DQLearning(myEnv)
    agent.train()
    agent.logger.plot_rewards()
