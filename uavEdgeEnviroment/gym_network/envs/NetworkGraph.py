import networkx as nx
from gym_network.envs.HeatmapGenerator import HeatmapGenerator
from gym_network.envs.UAV import UAV
from gym_network.envs.Microservice import Microservice

class NetworkGraph(object):


    """
    
    """

    def __init__(self, uav_list: list[UAV], microservices: list[Microservice]):

        """
        
        """

        self.heatmap_generator = HeatmapGenerator()
        self.graph = nx.Graph()
        self.n_microservices = len(microservices)

        for uav in uav_list:
            self.graph.add_node(UAV(uav['uavId'], 
                                    uav['position'],
                                    uav['ramCapacity'], 
                                    uav['ramAllocated'],
                                    uav['cpuCapacity'],
                                    uav['cpuAllocated'],
                                    microservices,
                                    len(uav_list)))
            
        for src_uav in self.graph:
            src_x = src_uav.position[0]
            src_y = src_uav.position[1]
            for dst_uav in self.graph:
                distance_x = abs(src_x - dst_uav.position[0])
                distance_y = abs(src_y - dst_uav.position[1])
                if (((distance_x == 1) or (distance_y == 1))
                     and (distance_x + distance_y <= 2)):
                    self.graph.add_edge(src_uav, dst_uav)
        self.longest_paths_costs = []
        for uav in self.graph:
            path_cost = nx.single_source_shortest_path_length(self.graph, uav)
            path_cost = max(path_cost.values())
            self.longest_paths_costs.append(path_cost)
            for ms_index in range(self.n_microservices):
                uav.microservicesCosts[ms_index] =\
                    path_cost * uav.microservicesHeat[ms_index]

    def generate_heatmaps(self):

        """
        
        """

        xMax = max([uav.position[0] for uav in self.graph.nodes])
        yMax = max([uav.position[1] for uav in self.graph.nodes])
        for ms_index in range(self.n_microservices):
            noise_map = self.heatmap_generator.generate_heatmap(
                (xMax, yMax),
                [uav.position for uav in list(self.graph.nodes)])
            for uav, noiseValue in zip(list(self.graph.nodes), noise_map):
                uav.microservicesHeat[ms_index] = noiseValue

    def calculate_serving_cost(self, ms_index: int):

        """
        Calculate the cost to serve the specified service to all users.
        
        Arguments:
         - srcUav: the UAV on which the new microsercice instance has
                   been deployed.
         - msIndex: the index of the microservice that has been 
        """

        server_uavs = list(
                        filter(
                            lambda uav: uav.microservices[ms_index] == 1,
                            self.graph))
        for uav in self.graph:
            cost = uav.microservicesHeat[ms_index]

            if (cost != 0):
                
                path_length = [nx.shortest_path_length(
                                  self.graph,
                                  dst_uav,
                                  uav) for dst_uav in server_uavs]
                path_length = min(path_length)
                cost *= path_length
            uav.microservicesCosts[ms_index] = cost

    def get_total_cost(self):

        """
        
        """

        cost = 0
        
        for uav in self.graph:
            for ms_index in range(self.n_microservices):
                cost += uav.microservicesCosts[ms_index]
        return cost

    def reset(self):

        """
        Deallocate all resources in the UAVs. Generate new random
        heatmaps for each microservice. Calculate the worst cost
        """

        #self.generate_heatmaps()
        for i, uav in enumerate(self.graph):
            uav.ramAllocated = 0
            uav.cpuAllocated = 0
            for ms_index in range(self.n_microservices):
                uav.microservices[ms_index] = 0
                cost = uav.microservicesHeat[ms_index]\
                       * self.longest_paths_costs[i]
                uav.microservicesCosts[ms_index] = cost

