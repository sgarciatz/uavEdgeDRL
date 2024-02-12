import json
import gymnasium as gym
from gymnasium.spaces import Graph, Box, Discrete, Dict, Text
import networkx as nx
from gym_network.envs.UAV import UAV
from gym_network.envs.Microservice import Microservice
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

    def __init__(self, input_file):

        """
        Constructor for the network enviroment.
        """

        self.graph = nx.Graph()
        self.graph_state = None
        self.observation_space = None
        self.action_space = None
        self.microservices = []
        self.msToDeploy = []
        self.worstCaseSolution = 0
        self.build_env(input_file)

    def _get_info(self):

        """
        Returns a dictionary with all the info about of serving the
        UAVs each microservice
        """

        info: dict = {}
        for uav in list(self.graph.nodes):
            info[f"{uav.id}-microservicesCosts"] =\
                np.array(uav.microservicesCosts)
            info[f"{uav.id}-cpuAvailable"] =\
                np.array([uav.cpuCapacity - uav.cpuAllocated])
            info[f"{uav.id}-ramAvailable"] =\
                np.array([uav.ramCapacity - uav.ramAllocated])

        for i, ms in enumerate(self.msToDeploy):
            info[f"next_{i}-msId"] = ms.idToInt()
            info[f"next_{i}-cpuReq"] = np.array([ms.cpuRequirement])
            info[f"next_{i}-ramReq"] = np.array([ms.ramRequirement])
        return info

    def _get_obs(self):

        """
        Return the cost of serving the UAVs each microservice
        """

        observation = self._get_info()

        return observation
        

    def _connect_nodes(self):
    
        """
        Create an auxiliar data structure to hold the info about the
        paths
        """
         
        adjMatrix : np.ndarray = np.zeros(
            (len(self.graph.nodes), len(self.graph.nodes)))
                    
        uav : UAV = None
        neighbour  : UAV = None
        for row in range(adjMatrix.shape[0]):
            for value in range(adjMatrix.shape[1]):
                uav = list(self.graph.nodes)[row]
                neighbour = list(self.graph.nodes)[value]

                if (abs(uav.position[0] - neighbour.position[0]) <= 1
                    and abs(uav.position[1] - neighbour.position[1]) <= 1):
                    # Is an adjacent matrix element 
                    adjMatrix[row][value] = 1
                    
        uavList = list(self.graph.nodes)
        
        for srcUav in uavList:
            for dstUav in list(set(uavList) - set([srcUav])):
                if (abs(srcUav.position[0] - dstUav.position[0]) <= 1
                    and abs(srcUav.position[1] - dstUav.position[1]) <= 1):
                    self.graph.add_edge(srcUav, dstUav)

    def _generate_heatmap(self, msIndex: int):

        """
        Use Perlin noise to generate a random heatmap
        """
        
        random_seed = random.randint(0, 999999)
        random_octaves = random.uniform(3.5, 6.0)
        noise = PerlinNoise(octaves= random_octaves, seed = random_seed)
        xMax = max([node.position[0] for node in self.graph.nodes])
        yMax = max([node.position[1] for node in self.graph.nodes])
        noiseValues = []
        for uav in list(self.graph.nodes):
            coords = [uav.position[0] / xMax, uav.position[1] / yMax]
            noiseValue = (noise(coords) + 1) / 2
            noiseValues.append(noiseValue)

        maxNoise, minNoise = max(noiseValues), min(noiseValues)

        finalNoises = []
        for nv in noiseValues:
            nv = (nv - minNoise) / (maxNoise -  minNoise)
            nv = math.floor((nv) * 10)
            if   (nv < 2): nv = 0
            elif (nv < 3): nv = 1
            elif (nv < 5.5): nv = 2
            elif (nv < 7): nv = 3
            elif (nv < 9): nv = 4
            else: nv = 5
            finalNoises.append(nv)
        for uav, noiseValue in zip(list(self.graph.nodes), finalNoises):
            uav.microservicesHeat[msIndex] = noiseValue

    def _reset_deployment_queue(self):
        
        """
        Reset the queue of deployment of microservices
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

    def _recalculate_costs(self, srcUav: UAV, msIndex: int):

        """
        Calculate the new cost to serve the service to all users
        
        Arguments:
         - srcUav: the UAV on which the new microsercice instance has
                   been deployed.
         - msIndex: the index of the microservice that has been 
                    deployed.
        """
        
        for dstUav in list(self.graph.nodes):
            pathLength = nx.shortest_path_length(self.graph, srcUav, dstUav)
            heat = dstUav.microservicesHeat[msIndex]
            newMsCost = pathLength * heat
            if (newMsCost < dstUav.microservicesCosts[msIndex]):
                dstUav.microservicesCosts[msIndex] = newMsCost

    def _get_reward(self) -> float:
    
        """
        Get the reward of the end of the episode.
        """
        
        currentSolution = 0
        for uav in self.graph:
            for msCost in uav.microservicesCosts:
                currentSolution += msCost
        
        reward = 1 - (currentSolution / self.worstCaseSolution)
        print("Episode Reward: 1 - (", 
              currentSolution,
              "/",
              self.worstCaseSolution,
              ") = ",
              reward)
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
                                           ms['replicationIndex'],
                                           np.array(ms['heatmap'])))
        self.microservices = sorted(msList, 
                                    key=lambda ms: ms.replicationIndex)
        self._reset_deployment_queue()
        for uav in inputData['uavList']:
                    self.graph.add_node(UAV(uav['uavId'], 
                                            uav['position'],
                                            uav['ramCapacity'], 
                                            uav['ramAllocated'],
                                            uav['cpuCapacity'],
                                            uav['cpuAllocated'],
                                            self.microservices,
                                            len(msList)-1))

        # Connect the nodes of the graph
        self._connect_nodes()
        # Define the observation and action spaces
        observations = {}

        for uav in list(self.graph.nodes):
            maxPathLength =\
                nx.single_source_shortest_path_length(self.graph, uav)
            maxPathLength = max(maxPathLength.values())
            observations[f"{uav.id}-microservicesCosts"] =\
                Box(0.0,
                    maxPathLength * 5,
                    shape=(len(self.microservices),),
                    dtype=np.float64)
                
            observations[f"{uav.id}-cpuAvailable"] =\
                Box(0.0,
                    uav.cpuCapacity,
                    shape=(1,),
                    dtype=np.float64)
                
            observations[f"{uav.id}-ramAvailable"] =\
                Box(0.0,
                    uav.cpuCapacity,
                    shape=(1,),
                    dtype=np.float64)
                
                
        for i, ms in enumerate(self.msToDeploy):

            observations[f"next_{i}-msId"] =\
                Discrete(1,
                         start=-1)
            observations[f"next_{i}-cpuReq"] =\
                Box(0.0,
                    ms.cpuRequirement,
                    shape=(1,),
                    dtype=np.float64)
                    
            observations[f"next_{i}-ramReq"] =\
                Box(0.0,
                    ms.ramRequirement,
                    shape=(1,),
                    dtype=np.float64)
                                
        self.observation_space = Dict(observations, seed=42)
        self.action_space = Discrete(len(list(self.graph.nodes)),
                                     start=0, 
                                     seed=42)

    def reset(self, seed=None, options=None):
        """
        Deallocate all resources in the UAVs. Generate new random
        heatmaps for each microservice. Reset the deployment queue.
        Calculate the worst possible solution.
        """
        
        super().reset(seed=seed)
        n_microservices = len(list(self.graph.nodes)[0].microservices)
        for index in range(n_microservices):
            self._generate_heatmap(index)

        for uav in list(self.graph.nodes):
            for ms in range(len(uav.microservices)):
                longestPath =\
                    nx.single_source_shortest_path_length(self.graph, uav)
                longestPath = max(longestPath.values())
                uav.microservices[ms] = 0
                uav.microservicesCosts[ms] =\
                    uav.microservicesHeat[ms] * longestPath

        self._reset_deployment_queue()
        self.worstCaseSolution = 0
        for uav in self.graph:
            for msCost in uav.microservicesCosts:
                self.worstCaseSolution += msCost
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
        
        ms = self._pop_deployment_queue()
#        print("Current ms:", ms.id, f"\nList of ms: {[mss.id for mss in self.microservices]}")
        msIndex = self.microservices.index(ms)

        dstUav = list(self.graph.nodes)[action]
        dstUav.deployMicroservice(ms, msIndex)

        self._recalculate_costs(dstUav, msIndex)
    
        observation = self._get_obs()
        
        terminated = False        
        reward = 0
        if (self.msToDeploy[0].id == "empty"): 
            terminated = True
            reward = self._get_reward()

        info = self._get_info()
        
        return observation, reward, terminated, False, info
        

            
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    myEnv = NetworkEnv('/home/santiago/Downloads/paper2_large_01.json')
    observation, info = myEnv.reset()
    #visualize_graph(myEnv.graph)
    steps = len(myEnv.msToDeploy)
    for index in range(steps):
        obs, reward, terminated, truncated, info = myEnv.step(index*3)

        
