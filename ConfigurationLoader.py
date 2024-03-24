import os
import random
import json
import argparse
from pathlib import Path
from QEstimator import QEstimator
from QNetwork import QNetwork
from QDuelingNetwork import QDuelingNetwork
from ExperienceSampler import ExperienceSampler
from ActionSelector import ActionSelector
from TrainLogger import TrainLogger
from EpsilonGreedyPolicy import EpsilonGreedyPolicy
from BoltzmannPolicy import BoltzmannPolicy
from DQLearning import DQLearning
import gymnasium as gym
import numpy as np
import torch
import gym_network


class ConfigurationLoader(object):


    """
    ConfigurationLoader class reads the json file and prepares the
    training parameters.
    """

    def __init__(self, input_path):

        """
        Reads a json file and loads the parameters for setup
        the DRL agent training.

        Arguments:
        - input_path: The path where the input file is located.
        """

        self.configuration = json.load(open(input_path))

        #Set the device for torch
        self.device = torch.device("cpu")
        if (torch.cuda.is_available()):
            self.device = torch.device("cuda")
        self.set_env()

    def _flatten_state(self, state) -> list:

        """
        Helper function used to flatten the state dict in order to feed
        it to other methods.

        Parameters:
        - state: the state to flatten.
        """

        raw_state = []
        for value in list(state.values()):
            if (isinstance(value, np.ndarray)):
                for value2 in value:
                    raw_state.append(value2)
            else:
                raw_state.append(value)
        return raw_state


    def set_env(self) -> gym.Env:

        """
        Sets the gym enviroment where the agent will be trained.
        """

        config = self.configuration["env"]
        env_name = config["name"]
        env_params = None
        if ("params" in config):
            env_params = config["params"]
            self.env = gym.make(env_name, input_file=env_params)#, render_mode="human")
            return
        self.env = gym.make(env_name)


    def get_optimizer(self, 
                      optim_id, policy_net,
                      learning_rate) -> torch.optim.Optimizer:

        """
        Maps a configuration parameter to a PyTorch optimizer.
        """

        if (optim_id == "adamw"):
            optimizer = torch.optim.AdamW(policy_net.parameters(),
                                          lr=learning_rate,
                                          amsgrad=True)
        elif (optim_id == "adam"):
            optimizer = torch.optim.Adam(policy_net.parameters(),
                                         lr=learning_rate,
                                         amsgrad=True)
        else:
            optimizer = torch.optim.AdamW(policy_net.parameters(),
                                          lr=learning_rate,
                                          amsgrad=True)
        return optimizer

    def get_loss_fn(self, loss_fn_id):

        """
        Maps a configuration parameter to a PyTorch optimizer.
        """

        if (loss_fn_id == "mse"):
            loss_fn = torch.nn.MSELoss()
        elif (loss_fn_id == "crossentropy"):
            loss_fn = torch.nn.CrossEntropyLoss()
        elif (loss_fn_id == "huber"):
            loss_fn = torch.nn.HuberLoss()
        return loss_fn

    def get_q_estimator(self) -> QEstimator:

        """
        Given the configuration loaded, create and return a QEstimator.
        """

        n_obs = gym.spaces.utils.flatten_space(
                    self.env.observation_space).shape[0]
        n_act = gym.spaces.utils.flatten_space(
                    self.env.action_space).shape[0]
        config = self.configuration["hyperparameters"]["q_estimator"]
        network_type = "dueling"
        if ("type" in config):
            network_type = config["type"]
        layers = config["layers"]
        if (network_type == "dueling"):
            policy_net = QDuelingNetwork(n_obs,
                                         n_act,
                                         layers,
                                         device = self.device)
        elif (network_type == "simple"):
            policy_net = QNetwork(n_obs,
                                  n_act,
                                  layers,
                                  device = self.device)
        optim_id = config["optimizer"]
        learning_rate = config["learning_rate"]
        optim = self.get_optimizer(optim_id,
                                   policy_net,
                                   learning_rate)
        loss_fn_id = config["loss_fn"]
        loss_fn = self.get_loss_fn(loss_fn_id)
        update_policy = config["update_policy"]
        update_param = config["update_param"]
        gamma = config["gamma"]
        variation = self.configuration["variation"]
        target_net = None
        if (variation == "ddqn" or variation == "target"):
            if (network_type == "dueling"):
                target_net = QDuelingNetwork(n_obs,
                                             n_act,
                                             layers,
                                             device = self.device)
            elif (network_type == "simple"):
                target_net = QNetwork(n_obs,
                                      n_act,
                                      layers,
                                      device = self.device)
            target_net.load_state_dict(policy_net.state_dict())
        if (not os.path.exists("models")):
            os.makedirs("models")
        output_path = "models/" + self.configuration["id"] + ".pt"
        q_estimator = QEstimator(policy_net,
                                 optim,
                                 loss_fn,
                                 gamma,
                                 self.device,
                                 update_policy,
                                 update_param,
                                 target_net,
                                 variation,
                                 output_path)
        return q_estimator

    def get_action_selector(self) -> ActionSelector:

        """
        Given the configuration, create and return an ActionSelector
        """

        config = self.configuration["hyperparameters"]["policy"]
        policy_type = config["type"]
        decay_strat = config["decay_strat"]
        start_expl_rate = config["start_expl_rate"]
        end_expl_rate = config["end_expl_rate"]
        decay_rate = config["decay_rate"]
        if (policy_type == "epsilon"):
            policy = EpsilonGreedyPolicy(start_expl_rate)
        elif (policy_type == "boltzmann"):
            policy = BoltzmannPolicy(start_expl_rate)
        actionSelector = ActionSelector(
            policy,
            decay_strategy=decay_strat,
            start_exploration_rate=start_expl_rate,
            end_exploration_rate=end_expl_rate,
            decay_rate= decay_rate)

        return actionSelector

    def get_memory(self) -> ExperienceSampler:

        """
        Given the configuration, create and return an ExperienceSampler.
        """

        config = self.configuration["hyperparameters"]["memory"]
        memory_type = config["type"]
        memory_size = config["memory_size"]
        epsilon = config["epsilon"]
        alpha = config["alpha"]
        memory = ExperienceSampler(memory_size,
                                   memory_type,
                                   self.device,
                                   epsilon,
                                   alpha)
        return memory

    def get_parameters(self) -> dict:

        """
        Given the configuration, create a dictionary with the training
        parameters. Also setup the TrainLogger.
        """

        parameters = {}
        parameters["id"] = self.configuration["id"]
        parameters["env"] = self.env
        parameters["device"] = self.device

        parameters["logger"] = TrainLogger(parameters["id"],
                                           json.dumps(self.configuration,
                                                      indent=4))
        config = self.configuration["hyperparameters"]
        parameters["training_steps"] = config["training_steps"]
        parameters["batches"] = config["batches"]
        parameters["batch_size"] = config["batch_size"]
        parameters["updates_per_batch"] = config["updates_per_batch"]
        parameters["h"] = config["h"]
        parameters["q_estimator"] = self.get_q_estimator()
        parameters["action_selector"] = self.get_action_selector()
        parameters["memory"] = self.get_memory()
        return parameters

    def get_agent(self):

        """
        Get the agent ready to train.
        """
        parameters = self.get_parameters()
        agent = DQLearning(parameters)
        return agent

def parse_arguments ():
    args_parser = argparse.ArgumentParser(
        description= ("Create DRL Agent to learn how "
                      "to perform in a given enviroment."))
    args_parser.add_argument("Filename",
                             metavar="F",
                             type=Path,
                             nargs=1,
                             help=("The path of the input file "
                                   "with the agent parameters."))
    args = args_parser.parse_args()
    return args.Filename[0]

def show_validation(env):

    """ Once a validation is carried out, show the deployment scheme.

    Parameters:
    - env = The environment.
    """

    import matplotlib.pyplot as plt
    import networkx as nx

    G = env.network_graph.graph
    deployment_schme = plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    # Show the deployment scheme and all the heatmaps.

    xMax = max([node.position[0] for node in G])
    positions_dict = {}
    for node in G:
        positions_dict[node] = [node.position[1], xMax - node.position[0]]

    colors = []
    for index in range(len(list(G.nodes)[0].microservices)):
         colors.append([node.microservices[index] * (index + 1) for node in G])
    colors = np.sum(colors, axis=0)
    print(colors)
    nx.draw_networkx(G,
                     pos=positions_dict, 
                     node_color=colors,
                     cmap="Pastel1",
                     with_labels=False)

    heatmaps = plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax = axes.flatten()
    for ms_index in range(len(list(G.nodes)[0].microservices)):
        colors = [uav.microservicesHeat[ms_index] for uav in G]
        nx.draw_networkx(G,
                         pos=positions_dict,
                         node_color=colors,
                         cmap="inferno",
                         vmin=0,
                         vmax=5,
                         with_labels=False,
                         ax=ax[ms_index])

    vmin, vmax = 0, 5
    sm = plt.cm.ScalarMappable(cmap="inferno", 
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax)
    plt.show()

def print_validation(env):

    G = env.network_graph.graph
    heatmaps = []
    deployment = []

    cost_fn_value = 0
    for uav in G:
        print(uav)
        for cost in uav.microservicesCosts:
            cost_fn_value += cost

def main(args):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    agent = ConfigurationLoader(args).get_agent()
    agent.train()
    
    agent.q_estimator.load_model()
    agent.validate_learning(1, testing = True)
    show_validation(agent.environment)
    print_validation(agent.environment)

if __name__ == "__main__":
    main(parse_arguments())
