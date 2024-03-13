import json
import gymnasium as gym
from gymnasium.spaces import Tuple, Box, Discrete, Dict
import networkx as nx
from gym_network.envs.UAV import UAV
from gym_network.envs.Microservice import Microservice
from gym_network.envs.NetworkGraph import NetworkGraph
import numpy as np
from perlin_noise import PerlinNoise
import random
import math
import copy

class NetworkEnv(gym.Env):


    """
    A class that extends Gymnasium's Env to hold the information
    about the networks.
    """

    def __init__(self, input_file, seed = 0):

        """
        Constructor for the network enviroment.
        """

        self.graph_state = None
        self.observation_space = None
        self.action_space = None
        self.microservices = []
        self.msToDeploy = []
        self.worstCaseSolution = 0
        self.build_env(input_file)
        self.build_observation_space()
        self.build_action_space()
        self.episode_step = 0

    def _get_info(self):

        """
        Returns a dictionary with all the info about of serving the
        UAVs each microservice.
        """

        info: dict = {}
        for uav in list(self.network_graph.graph.nodes):
            info[f"{uav.id}-microservicesCosts"] =\
                np.array(uav.microservicesCosts)
            info[f"{uav.id}-cpuAvailable"] =\
                np.array([uav.cpuCapacity - uav.cpuAllocated])
            info[f"{uav.id}-ramAvailable"] =\
                np.array([uav.ramCapacity - uav.ramAllocated])

        for i, ms in enumerate(self.msToDeploy):
            info[f"next_{i}-msId"] = float(ms.idToInt())
            info[f"next_{i}-cpuReq"] = np.array([ms.cpuRequirement])
            info[f"next_{i}-ramReq"] = np.array([ms.ramRequirement])
        return info

    def _get_obs(self):

        """
        Return the cost of serving the UAVs each microservice.
        """


        uav_observation = []
        for uav in self.network_graph.graph:
            for ms_index in range(len(self.microservices)):
                uav_observation.append(int(uav.microservicesCosts[ms_index]))
            uav_observation.append(
                int(uav.ms_fits(self.msToDeploy[0],
                                self.msToDeploy[0].idToInt())))
        ms_observation = []
        for ms in self.msToDeploy:
            ms_observation.append(ms.idToInt())
        return np.concatenate((uav_observation, ms_observation), dtype=np.int32)

    def _reset_deployment_queue(self):
        
        """
        Reset the queue of deployment of microservices.
        """
        
        self.msToDeploy = []
        msListCopy = copy.deepcopy(self.microservices)
        done = False
        while (not done):
            done = True
            for msCopy, microservice in zip(msListCopy, self.microservices):
                if (msCopy.replicationIndex > 0):
                    self.msToDeploy.append(microservice)
                    msCopy.replicationIndex -= 1
                    done = False

    def _pop_deployment_queue(self) -> Microservice:

        """
        Pop the first element from the deployment list and append an
        \"empty microservice\" at the end.
        """
        ms = self.msToDeploy.pop(0)
        self.msToDeploy.append(Microservice('empty', -1, -1, -1))
        return ms

    def _reinsert_deployment_queue(self, ms: Microservice) -> None:

        """Reinsert a microservice that could not be deployed.
        
        Arguments:
        - ms: Microservice = The microservice that could not be
            deployed and are going to be reinserted.
        """

        self.msToDeploy.insert(0, ms)
        self.msToDeploy.pop(-1)

    def _get_reward(self) -> float:
    
        """
        Get the reward of the end of the episode.
        
        reward = (1 - s/w_s)^τ
        """
        
        extra_steps = self.episode_step - len(self.msToDeploy)
        extra_steps_penalty = math.pow(0.95, extra_steps)
        currentSolution = self.network_graph.get_total_cost()
        reward = math.pow(1 - (currentSolution / self.worstCaseSolution), math.pi) * extra_steps_penalty
        return reward
        
        
    def build_env(self, input_file):

        """
        Read the configuration from a file and load it to a NetworkX
        graph.
        
        The microservices are read first to know the maximum size of
        the lists in the UAVs.
        """
        
        inputData = json.load(open(input_file))
        msList: list[Microservice] = []
        for ms in inputData['microserviceList']:
                msList.append(Microservice(ms['microserviceId'], 
                                           ms['ramRequirement'],
                                           ms['cpuRequirement'],
                                           float(ms['replicationIndex']),
                                           np.array(ms['heatmap'])))
        self.microservices = sorted(msList, 
                                    key=lambda ms: ms.replicationIndex)

        self._reset_deployment_queue()
        
        self.network_graph = NetworkGraph(inputData['uavList'], 
                                          self.microservices)

    def build_action_space(self):

        """
        
        """

        self.action_space = Discrete(len(self.network_graph.graph),
                                     start=0, 
                                     seed=42)

    def build_observation_space(self):

        """
        
        """

        uav_obs_length = len(self.network_graph.graph.nodes) 
        uav_obs_length *= (len(self.microservices) + 1)

        uav_observation = Box(low=0,
                              high=5*len(self.network_graph.graph.nodes),
                              shape=(uav_obs_length,),
                              dtype= np.intc)
        ms_queue_observation = Box(low=-1,
                                    high= len(self.microservices) - 1,
                                    shape=(len(self.msToDeploy),),
                                    dtype=np.intc)
        
        self.observation_space = Tuple(
            (uav_observation, ms_queue_observation),
            seed=42)
        self.observation_space = gym.spaces.utils.flatten_space(self.observation_space)

    def reset(self, seed=None, options=None):
        """
        Deallocate all resources in the UAVs. Generate new random
        heatmaps for each microservice. Reset the deployment queue.
        Calculate the worst possible solution.
        """
        
        super().reset(seed=seed)
        self.episode_step = 0
        self.network_graph.reset()
        self._reset_deployment_queue()
        self.worstCaseSolution = self.network_graph.get_total_cost()
        observation = self._get_obs()
        info = {}
#        print(self.microservices)
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
        
        ms = self._pop_deployment_queue()
        msIndex = self.microservices.index(ms)
        dstUav = list(self.network_graph.graph.nodes)[action]
        is_deployed = dstUav.deployMicroservice(ms, msIndex)
        if (is_deployed):
            self.network_graph.calculate_serving_cost(msIndex)
        else:
            self._reinsert_deployment_queue(ms)
        observation = self._get_obs()
        terminated = False
        reward = 0
        self.episode_step += 1
        if (self.msToDeploy[0].id == "empty"): 
            terminated = True
            reward = self._get_reward()

        info = self._get_info()
        return observation, reward, terminated, False, info

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    myEnv = NetworkEnv('/home/santiago/Downloads/paper2_large_01.json')
    observation, info = myEnv.reset()
    steps = len(myEnv.msToDeploy)
    for index in range(steps):
        obs, reward, terminated, truncated, info = myEnv.step(index*3)

        
