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
from TrainLogger import TrainLogger
import numpy as np
import random


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


    def __init__(self, environment):

        """ 
        Initialize the agent using the characteristic of the environment
        and load the training hyperparameters.
        
        Arguments:
        - environment: the instance of the environment where the agent
         lives.
        """

        #Set hyperparameters
        self.training_steps = 5000
        self.memory_size = 32768
        self.h = 1024
        self.batches = 8
        self.batch_size = 128
        self.updates_per_batch = 8

        self.gamma = 0.9
        self.learning_rate = 0.00001
        
        #Read the env
        self.environment = environment
        self.n_actions = self.environment.action_space.n
        self.n_observations =\
            flatten_space(self.environment.observation_space).shape[0]

        #Set the device for torch
        self.device = torch.device("cpu")
        if (torch.cuda.is_available()):
            self.device = torch.device("cuda")

        #Prepare the QEstimator
        policy_net = QNetwork(self.n_observations,
                                   self.n_actions).to(self.device)
        target_net = QNetwork(self.n_observations,
                                   self.n_actions).to(self.device)
        optimizer = optim.AdamW(policy_net.parameters(),
                                     lr=self.learning_rate, amsgrad=True)
        loss_fn = torch.nn.MSELoss().to(self.device)
        update_policy = "replace"
        self.polyak_param = 50
        variation = "ddqn"
        self.q_estimator = QEstimator(policy_net,
                                      optimizer,
                                      loss_fn,
                                      self.gamma,
                                      self.device,
                                      update_policy,
                                      target_net,
                                      variation)
        #Prepare the ExpercienceSampler
        self.experience_sampler = ExperienceSampler(self.memory_size)

        #Prepare the ActionSelector
        self.start_epsilon = 0.9
        self.end_epsilon = 0.05
        self.decay_rate = 1.2
        self.action_policy = EpsilonGreedyPolicy(self.start_epsilon)
#        self.action_policy = BoltzmannPolicy(self.start_epsilon)
        self.action_selector = ActionSelector(
            self.action_policy,
            decay_strategy = "exponential",
            start_exploration_rate = self.start_epsilon,
            end_exploration_rate = self.end_epsilon,
            decay_rate = self.decay_rate)

        #Prepare the logging class TrainLogger
        self.logger = TrainLogger()

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
        Using the inicial policy (policy_net), fill the replay memory
        buffer. This is the first step of the loop of the DQN
        Algorithm.
        """ 

        n_experiences = self.h
        
        done = True
        experience = None
        for i in range(n_experiences):
            if (done):
                state, info = self.environment.reset()
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
            experience = Experience(state, action, reward, next_state, done, 9999)  
            self.experience_sampler.add_experience(experience)
    
    def _validate_learning(self):

        """
        Use the Q estimator network with its current weights to check
        how well the agent performs in a enviroment during an episode.
        """

        rewards = []
        ep_lengths = []
        for _ in range(10):
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
        return sum(rewards) / 10, sum(ep_lengths) / 10

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
                    loss = self.q_estimator.calculate_q_loss(batch)
                    self.q_estimator.update_q_estimator(loss)
                    step_losses.append(loss.item())
            reward, ep_length = self._validate_learning()
            expl_rate = self.action_selector.exploration_rate
            self.logger.add_training_step(step, 
                                          expl_rate,
                                          sum(step_losses) / len(step_losses),
                                          reward,
                                          ep_length)
            self.logger.print_training_step()
            self.action_selector.decay_exploration_rate(step, self.training_steps)
            if (step % self.polyak_param == 0):
                self.q_estimator.update_second_q_estimator(self.polyak_param)
        self.logger.print_training_footer()
                
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
