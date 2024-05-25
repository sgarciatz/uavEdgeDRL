import networkx as nx
import numpy as np
import math
from gym_network.envs.HeatmapGenerator import HeatmapGenerator
from gym_network.envs.UAV import UAV
from gym_network.envs.Microservice import Microservice
import matplotlib.pyplot as plt
from matplotlib import cm


class NetworkGraph(object):


    """
    
    """

    def __init__(self, uav_list, ms_list, scenario_shape):

        """
        
        """

        self.graph = nx.Graph()
        self.uav_list = uav_list
        [self.graph.add_node(uav) for uav in uav_list]
        self.ms_list = ms_list
        [self.graph.add_node(ms) for ms in ms_list]
        self.scenario_shape = scenario_shape
        self.ms_offset = self.scenario_shape[1] + 5
        self.heatmap_generator = HeatmapGenerator()

        for src_uav in self.uav_list:
            for dst_uav in set(self.uav_list) - set([src_uav]):
                hor_distance = abs(src_uav.position[0] - dst_uav.position[0])
                ver_distance = abs(src_uav.position[1] - dst_uav.position[1])
                if ((hor_distance < 2) and (ver_distance < 2) ):
                    self.graph.add_edge(src_uav, dst_uav)

        node_graph = self.graph.subgraph(self.uav_list)
        self.diameter = nx.diameter(node_graph)
        self.mean_path_length = []
        for src_uav in self.uav_list:
            for dst_uav in set(self.uav_list) - set([src_uav]):
                path_length = nx.shortest_path_length(self.graph,
                                                      source=src_uav,
                                                      target=dst_uav)
                self.mean_path_length.append(path_length)
        self.mean_path_length =\
             sum(self.mean_path_length) / len(self.mean_path_length)
        self.mean_path_length = math.ceil(self.mean_path_length) + 1

    def draw_graph(self):

        """ Draw the network graph """

        positions = {}
        for uav in self.uav_list:
            pos_0 = self.scenario_shape[0]-1 - uav.position[0]
            positions[uav] = [uav.position[1], pos_0]
        vertical_pos = self.scenario_shape[0] / (len(self.ms_list))
        for index, ms in enumerate(self.ms_list):
            positions[ms] = [self.ms_offset,
                             (index*vertical_pos) + (vertical_pos/2)]
        node_colors = []
        node_labels = {uav: uav.id for uav in self.uav_list}
        for uav in self.uav_list:
            node_colors.append('#a0d6e8')
        for ms in self.ms_list:
            node_colors.append('#ff8a00')
            node_labels[ms] = ms.id


        edge_colors = []
        for edge in self.graph.edges:
            if(isinstance(edge[0], Microservice)\
               or isinstance(edge[1], Microservice)):
                edge_colors.append('#f44336')
            else:
                edge_colors.append('#16537e')

        nx.draw_networkx(self.graph, 
                         pos=positions,
                         node_color=node_colors,
                         edge_color=edge_colors,
                         labels=node_labels,
                         linewidths=2.0)
                         
        plt.tight_layout()
        plt.get_current_fig_manager().window.showMaximized()
        plt.show()

    def draw_heightmap(self):

        node_graph = nx.Graph()
        [node_graph.add_node(uav) for uav in self.uav_list]

        x_lim = (max([uav.position[0] for uav in self.uav_list])+1) * 3
        x = np.arange(0, x_lim, 1)
        y_lim = (max([uav.position[1] for uav in self.uav_list])+1) * 3
        y = np.arange(0, y_lim, 1)
        x,y = np.meshgrid(x, y)

        z = np.zeros((x_lim, y_lim))

        for uav in self.uav_list:
            for i in range(0, 3):
                for j in range(0, 3):
                    z[(uav.position[0]*3)+i][(uav.position[1]*3)+j] = uav.ms_heats[0]
        
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        surf = ax.plot_surface(x, y, z.T, cmap=cm.inferno, vmin=0, vmax=5,
                               rcount = 20, ccount = 10, linewidth=1,
                               antialiased=False)
        plt.show()

    def draw_heatmap(self, microservice):


        node_graph = nx.Graph()
        [node_graph.add_node(uav) for uav in self.uav_list]
        positions = {}
        for uav in self.uav_list:
            pos_0 = self.scenario_shape[0]-1 - uav.position[0]
            positions[uav] = [uav.position[1], pos_0]

        node_colors = []
        ms_index = self.ms_list.index(microservice)
        node_labels = {uav: uav.id for uav in self.uav_list}
        for uav in self.uav_list:
            node_colors.append(uav.ms_heats[ms_index])
        
        
        nx.draw_networkx_nodes(node_graph, 
                 pos=positions,
                 cmap='inferno',
                 vmin=0,
                 vmax=5,
                 node_color=node_colors,)
        plt.show()

    def draw_heatmaps(self):

        """ Draw all the heatmaps in a single window."""

        for i, ms in enumerate(self.ms_list):
            plt.subplot(221+i)
            node_graph = nx.Graph()
            [node_graph.add_node(uav) for uav in self.uav_list]
            positions = {}
            for uav in self.uav_list:
                pos_0 = self.scenario_shape[0]-1 - uav.position[0]
                positions[uav] = [uav.position[1], pos_0]

            node_colors = []
            node_labels = {uav: uav.id for uav in self.uav_list}
            for uav in self.uav_list:
                node_colors.append(uav.ms_heats[i])
            
            
            nx.draw_networkx_nodes(node_graph, 
                     pos=positions,
                     cmap='inferno',
                     vmin=0,
                     vmax=5,
                     node_color=node_colors,)
        plt.tight_layout()
        plt.get_current_fig_manager().window.showMaximized()
        plt.show()

    def draw_path_cost(self, microservice):

        """Use networkx to draw the path cost of uav_i-microservice for
        each uav.

        Parameters:
        - microservice = The microservice, i.e., the target node used to
          evaluate the path cost.
        """

        uavs_ms_list = [uav for uav in self.uav_list]
        uavs_ms_list.append(microservice)
        subgraph = self.graph.subgraph(uavs_ms_list).copy()
        positions = {}
        for uav in self.uav_list:
            pos_0 = self.scenario_shape[0]-1 - uav.position[0]
            positions[uav] = [uav.position[1], pos_0]
        positions[microservice] = [self.ms_offset, self.scenario_shape[0]/2]
        node_colors = []
        node_labels = {}
        ms_index = self.ms_list.index(microservice)
        for uav in self.uav_list:
            try:
                node_labels[uav] = nx.shortest_path_length(subgraph, 
                                                           uav,
                                                           microservice) - 1
                node_labels[uav] *= uav.ms_heats[ms_index]
            except:
                node_labels[uav] = '∞'
        node_labels[microservice] = microservice.id
        for uav in self.uav_list:
            if (microservice in subgraph.neighbors(uav)):
                node_colors.append('#cc0000')
            else:
                node_colors.append('#a0d6e8')
        node_colors.append('#ff8a00')
        edge_colors = []
        for edge in subgraph.edges:
            if(isinstance(edge[0], Microservice)\
               or isinstance(edge[1], Microservice)):
                edge_colors.append('#f44336')
            else:
                edge_colors.append('#16537e')
        nx.draw_networkx(subgraph, 
                         pos=positions,
                         node_color=node_colors,
                         edge_color=edge_colors,
                         labels=node_labels,
                         linewidths=2.0)
        plt.show()


    def draw_path_costs(self):

        """Draw all the path costs of uav to microservice for each uav"""

        for i, ms in enumerate(self.ms_list):
            plt.subplot(221+i)
            uavs_ms_list = [uav for uav in self.uav_list]
            uavs_ms_list.append(ms)
            subgraph = self.graph.subgraph(uavs_ms_list).copy()
            positions = {}
            for uav in self.uav_list:
                pos_0 = self.scenario_shape[0]-1 - uav.position[0]
                positions[uav] = [uav.position[1], pos_0]
            positions[ms] = [self.ms_offset, self.scenario_shape[0]/2]
            node_colors = []
            node_labels = {}
            for uav in self.uav_list:
                try:
                    node_labels[uav] = nx.shortest_path_length(subgraph, 
                                                               uav,
                                                               ms) - 1
                    node_labels[uav] *= uav.ms_heats[i]
                except:
                    node_labels[uav] = '∞'
            node_labels[ms] = ms.id
            for uav in self.uav_list:
                if (ms in subgraph.neighbors(uav)):
                    node_colors.append('#cc0000')
                else:
                    node_colors.append('#a0d6e8')
            node_colors.append('#ff8a00')
            edge_colors = []
            for edge in subgraph.edges:
                if(isinstance(edge[0], Microservice)\
                   or isinstance(edge[1], Microservice)):
                    edge_colors.append('#f44336')
                else:
                    edge_colors.append('#16537e')
            nx.draw_networkx(subgraph, 
                             pos=positions,
                             node_color=node_colors,
                             edge_color=edge_colors,
                             labels=node_labels,
                             linewidths=2.0)
        plt.tight_layout()
        plt.get_current_fig_manager().window.showMaximized()
        plt.show()

    def generate_heatmaps(self):

        """
        
        """

        xMax = max([uav.position[0] for uav in self.uav_list])
        yMax = max([uav.position[1] for uav in self.uav_list])
        for ms_index in range(len(self.ms_list)):
            noise_map = self.heatmap_generator.generate_heatmap(
                (xMax, yMax),
                [uav.position for uav in list(self.uav_list)])
            for uav, noiseValue in zip(list(self.graph.nodes), noise_map):
                uav.ms_heats[ms_index] = noiseValue


    def get_total_cost(self):

        """
        Compute the actual cost of the graph. For each uav, calculate
        the length of the path to the closest replica of each 
        microservice weighted by the microservice heat.
        """

        cost = 0
        
        for uav in self.uav_list:
            for ms_index, ms in enumerate(self.ms_list):
                try:
                    path_length = nx.shortest_path_length(self.graph,
                                                          source=uav,
                                                          target=ms)
                except:
                    path_length = self.mean_path_length
                heat = uav.ms_heats[ms_index]
                cost += (path_length * heat)
        
        return cost

    def reset(self):

        """
        Deallocate all resources in the UAVs. Generate new random
        heatmaps for each microservice. Calculate the worst cost
        """

        self.generate_heatmaps()
        for uav in self.uav_list:
            uav.ram_left = uav.ram_cap
            uav.cpu_left = uav.cpu_cap
            for ms in self.ms_list:
                ms.replic_left = ms.replic_index
                try:
                    self.graph.remove_edge(ms, uav)
                except:
                    pass

    def get_uav_info(self):

        """Obtain the cost to serve each ms to each uav and the left
        computing capabilities of each node uav.
        """

        uav_info_dict = {}
        
        for uav in self.uav_list:
            uav_ms_costs = []
            for ms_index, ms in enumerate(self.ms_list):
                try: 
                    path_length = nx.shortest_path_length(self.graph,
                                                          source=uav,
                                                          target=ms)
                except:
                    path_length = self.diameter
                heat = uav.ms_heats[ms_index]
                uav_ms_costs.append(int(path_length * heat))
            uav_info_dict[uav.id] = {'ms_costs': uav_ms_costs,
                                     'ram_left': int(uav.ram_left),
                                     'cpu_left': int(uav.cpu_left)}
        return uav_info_dict

    def get_ms_info(self):

        """Obtain the requeriments of each microservice and its replication
        index.
        """

        ms_info_dict = {}

        for ms in self.ms_list:
            ms_info_dict[ms.id] = {'ram_req':  int(ms.ram_req),
                                     'cpu_req': int(ms.cpu_req),
                                     'replic_index': int(ms.replic_index),
                                     'replic_left': int(ms.replic_left)}
        return ms_info_dict

    def get_baseline_estimation(self):

        """Get an estimation of a solution to surpass."""

        cost_acc = 0
        for uav in self.uav_list:
            for ms_index, ms in enumerate(self.ms_list):
                path_length =\
                    math.floor(self.mean_path_length / ms.replic_index)
                cost_acc += path_length * uav.ms_heats[ms_index]
        return cost_acc
