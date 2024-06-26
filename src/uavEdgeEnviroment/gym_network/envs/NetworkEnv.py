import json
import gymnasium as gym
from gymnasium.spaces import Tuple, Box, Discrete, MultiDiscrete, Dict
import networkx as nx
from gym_network.envs.UAV import UAV
from gym_network.envs.Microservice import Microservice
from gym_network.envs.NetworkGraph import NetworkGraph
import numpy as np
import math


class NetworkEnv(gym.Env):


    """
    A class that extends Gymnasium's Env to hold the information
    about the networks.
    """

    def __init__(self, input_file, seed = 0):

        """
        Constructor for the network enviroment.
        """

        self.observation_space = None
        self.action_space = None
        self.microservices = []
        self.msToDeploy = []
        self.worstCaseSolution = 0
        self.input_file = input_file
        self.build_env(input_file)
        self.build_observation_space()
        self.build_action_space()
        self.episode_step = 0

    def _get_info(self):

        """
        Returns a dictionary with all the info about of serving the
        UAVs each microservice.
        """

        uav_observation = self.network_graph.get_uav_info()
        ms_observation = self.network_graph.get_ms_info()
        return uav_observation | ms_observation

    def _get_obs(self):

        """
        Return the cost of serving the UAVs each microservice.
        """


        uav_observation = self.network_graph.get_uav_info()
        uav_ms_cost_obs = []
        for uav in uav_observation.values():
            [uav_ms_cost_obs.append(value / self.max_uav_ms_cost) \
             for value in uav['ms_costs']]
        uav_ram_left_obs = []
        for uav in uav_observation.values():
            uav_ram_left_obs.append(uav['ram_left'] / self.max_ram_cap)
        uav_cpu_left_obs = []
        for uav in uav_observation.values():
            uav_cpu_left_obs.append(uav['cpu_left'] / self.max_cpu_cap)

        ms_observation = self.network_graph.get_ms_info()
        ms_ram_left_obs = []
        for ms in ms_observation.values():
            ms_ram_left_obs.append(ms['ram_req'] / self.max_ram_req)
        ms_cpu_req_obs = []
        for ms in ms_observation.values():
            ms_cpu_req_obs.append(ms['cpu_req'] / self.max_cpu_req)
        ms_repl_index = []
        for ms in ms_observation.values():
            ms_repl_index.append(ms['replic_left'] / self.max_repl_index)

        return np.concatenate((uav_ms_cost_obs, uav_ram_left_obs,
                               uav_cpu_left_obs, ms_ram_left_obs,
                               ms_cpu_req_obs, ms_repl_index),
                              dtype=np.float32)

    def _get_reward(self) -> float:

        """
        Get the reward of the end of the episode.

        reward = (1 - s/w_s) * p
        """

        steps = sum([ms.replic_index for ms in self.network_graph.ms_list])
        extra_steps = self.episode_step - steps
        extra_steps_penalty = math.pow(0.95, extra_steps)
        current_solution = self.network_graph.get_total_cost()
        solution_ratio = current_solution / self.worstCaseSolution
        reward = (1 - solution_ratio) * extra_steps_penalty
#        print("Episode Reward:  ", reward,
#              "\n  -Extra steps:  ", extra_steps,
#              "\n  -Current Hops: ", current_solution,
#              "\n  -Worst Hops:  ", self.worstCaseSolution)
        return reward

    def build_env(self, input_file):

        """
        Read the configuration from a file and load it to a NetworkX
        graph.

        The microservices are read first to know the maximum size of
        the lists in the UAVs.
        """

        input_data = json.load(open(input_file))
        ms_heatmaps = []

        for ms in input_data['microserviceList']:
                ms_heatmaps.append(ms['heatmap'])
        ms_heatmaps = np.asarray(ms_heatmaps)
        uav_list = []
        for uav in input_data['uavList']:
            ms_heats = ms_heatmaps[:,uav['position'][0],uav['position'][1]]
            uav_list.append(UAV(uav['uavId'],
                            uav['position'],
                            uav['ramCapacity'],
                            uav['ramCapacity'],
                            uav['cpuCapacity'],
                            uav['cpuCapacity'],
                            ms_heats))

        ms_list = []
        for ms in input_data['microserviceList']:
                ms_list.append(Microservice(ms['microserviceId'],
                                            ms['ramRequirement'],
                                            ms['cpuRequirement'],
                                            float(ms['replicationIndex'])))

        self.network_graph = NetworkGraph(uav_list,
                                          ms_list,
                                          input_data['shape'])
        self.network_graph.generate_heatmaps()

    def build_action_space(self):

        """Create the action space, the seed is not really important
        because it is used for sampling randomly.
        """

        actions = [len(self.network_graph.ms_list),
                   len(self.network_graph.uav_list)]

        self.action_space = MultiDiscrete(actions,
                                          start=[0,0],
                                          seed=42)

    def build_observation_space(self):

        """Create the observation space of the agent. It is composed of
        the data that the UAV swarm provides and the queue of
        microservices to deploy.
        """


        uav_ms_cost = len(self.network_graph.uav_list)\
                      * len(self.network_graph.ms_list)
        self.max_uav_ms_cost = 5 * (self.network_graph.diameter+1)
        uav_ram_left = len(self.network_graph.uav_list)
        self.max_ram_cap = max([uav.ram_cap for uav in self.network_graph.uav_list])

        uav_cpu_left = len(self.network_graph.uav_list)
        self.max_cpu_cap = max([uav.cpu_cap for uav in self.network_graph.uav_list])

        ms_ram_req = len(self.network_graph.ms_list)
        self.max_ram_req = max([ms.ram_req for ms in self.network_graph.ms_list])
        ms_cpu_req = len(self.network_graph.ms_list)
        self.max_cpu_req = max([ms.cpu_req for ms in self.network_graph.ms_list])


        ms_repl_index = len(self.network_graph.ms_list)
        self.max_repl_index =\
            max([ms.replic_index for ms in self.network_graph.ms_list])

        uav_ms_cost_obs = Box(low=0,
                              high=1,
                              shape=(uav_ms_cost,),
                              dtype=np.float32)
        uav_ram_left_obs = Box(low=0,
                               high=1,
                               shape=(uav_ram_left,),
                               dtype=np.float32)
        uav_cpu_left_obs = Box(low=0,
                               high=1,
                               shape=(uav_cpu_left,),
                               dtype=np.float32)
        ms_ram_req_obs = Box(low=0,
                              high=1,
                              shape=(ms_ram_req,),
                              dtype=np.float32)
        ms_cpu_req_obs = Box(low=0,
                              high=1,
                              shape=(ms_cpu_req,),
                              dtype=np.float32)
        ms_repl_index = Box(low=0,
                            high=1,
                            shape=(ms_repl_index,),
                            dtype=np.float32)

        self.observation_space = Tuple(
            (uav_ms_cost_obs, uav_ram_left_obs, uav_cpu_left_obs,
             ms_ram_req_obs, ms_cpu_req_obs, ms_repl_index),
            seed=42)

        self.observation_space = gym.spaces.utils.flatten_space(
            self.observation_space)

    def is_terminal_state(self):

        """ Check if all the microservice instances has been
        deployed.
        """
        is_terminal = True
        for ms in self.network_graph.ms_list:
            degree = self.network_graph.graph.degree(ms)
            is_terminal = is_terminal and (degree == ms.replic_index)
        return is_terminal

    def reset(self, seed=None, options=None, testing: bool = False):
        """
        Deallocate all resources in the UAVs. Generate new random
        heatmaps for each microservice. Reset the deployment queue.
        Calculate the worst possible solution.
        """

        super().reset(seed=seed)

        self.episode_step = 0
        if (testing):
            self.build_env(self.input_file)
        else:
            self.network_graph.reset()
        self.worstCaseSolution = self.network_graph.get_baseline_estimation()
#        self.network_graph.graph.add_edge(self.network_graph.uav_list[1], self.network_graph.ms_list[0])
#        self.network_graph.graph.add_edge(self.network_graph.uav_list[27], self.network_graph.ms_list[1])
#        self.network_graph.graph.add_edge(self.network_graph.uav_list[48], self.network_graph.ms_list[2])
#        self.network_graph.graph.add_edge(self.network_graph.uav_list[15], self.network_graph.ms_list[3])
#        self.network_graph.draw_graph()
#        for ms in self.network_graph.ms_list:
#            self.network_graph.draw_heatmap(ms)
#            self.network_graph.draw_path_cost(ms)
        observation = self._get_obs()
        info = {}

        return observation, info

    def step(self, action):

        """
        Deploy a microservice in an UAV.

        NO - The agent can select from the 5 best destination UAVs. Right
        after taking the action, the agent shall observe the resulting
        network state, i.e., for each node, its costs and remaining
        capacity. The agent is only rewarded at the end of the episode
        acording to its improvement with respect to the worst solution.
        The end of the episode is reached when all instances are
        deployed.

        The agents chooses the uav to deploy the ms.
        """

        ms = self.network_graph.ms_list[action[0]]
        uav = self.network_graph.uav_list[action[1]]
        reward = 0
        self.episode_step += 1
        if (self.network_graph.graph.has_edge(ms, uav)\
            or (not uav.ms_fits(ms)) or (ms.replic_left) == 0):
            reward = -1
        else:
            uav.deploy_ms(ms)
            self.network_graph.graph.add_edge(ms, uav)
        observation = self._get_obs()
        info = self._get_info()
        terminated = self.is_terminal_state()
        if (terminated):
            reward = self._get_reward()
#        self.network_graph.draw_graph()
#        self.network_graph.draw_path_costs()
        return observation, reward, terminated, False, info



