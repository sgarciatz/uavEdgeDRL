import numpy as np
import math
import copy
import gurobipy as gp
import gymnasium as gym
import networkx as nx



class GLOMIP(object):


    """A Linear Programming approach to solve the placement of
    multi-instanced microservices under computation constraints.
    
    Attributes:
    - env = The placement problem actual environment.
    - model = The Linear Programming problem especification.
    """

    def __init__(self, environment):

        """Create the model using the given environment"""

        self.env = environment

        self.gurobi_env = gp.Env(empty=True)
        self.gurobi_env.setParam("OutputFlag",0)
        self.gurobi_env.start()
        self.model = gp.Model(env=self.gurobi_env)
        self.max_distance = []
        self.uav_list = self.get_uav_list()
        self.path_cost_matrix = self.get_path_cost_matrix()
        self.ms_list = self.get_ms_list()
        self.build_model()

    def get_uav_list(self) -> np.ndarray:

        """Create a numpy array with the list of UAVs within the
        enviroment. The UAVs are ordered by their position.
        
        Return: np.ndarray = The array of UAVs.
        """

        uav_list = self.env.get_wrapper_attr("network_graph").graph.nodes
        uav_list = sorted(list(uav_list), key = lambda uav: uav.position[1])
        uav_list = sorted(uav_list, key = lambda uav:uav.position[0])
        return np.array(uav_list)

    def get_path_cost_matrix(self) -> np.ndarray:

        """Create a 2D numpy array with the cost of every simple path
        within the network graph.
        
        Return: np.ndarray = The matrix of path costs.
        """

        path_cost_matrix = []
        G = self.env.get_wrapper_attr("network_graph").graph
        for uav in self.uav_list:
            for uav2 in self.uav_list:
                path_cost = nx.shortest_path_length(G, uav, uav2)
                path_cost_matrix.append(path_cost)
        path_cost_matrix = np.array(path_cost_matrix)
        path_cost_matrix = path_cost_matrix.reshape((len(self.uav_list),
                                                     len(self.uav_list)))
        return np.array(path_cost_matrix)

    def get_ms_list(self) -> np.ndarray:

        """Create a numpy array with the list of the microservices to
        be deployed. They are ordered by the replication index.
        
        Return: np.ndarray = The array of microservices.
        """

        ms_list = self.env.get_wrapper_attr("microservices")
        ms_list = sorted(ms_list, key = lambda ms: ms.replicationIndex)
        return np.array(ms_list)

    def build_model(self):

        """Given the enviroment, add the corresponding decision
        variables, constraints and the objective function.
        
        Variables:
        - self.X[i,j]: bool = Wether the UAV self.uav_list[i] deploys
          microservice self.ms_list[j].
        - self.Y[i, j, k]: bool = Wether UAV self.uav_list[k] is the closest
          UAV that can serve microservice self.ms_list[j] to the UAV
          self.uav_list[i]. 
        - self.Z[i,j]: integer = The actual cost of serving the microservice
          self.ms_list[j] to the UAV self.uav_list[i].
        """

        self.model = gp.Model(env=self.gurobi_env)
        self.X = 0
        
        max_heat = 0
        for uav in self.uav_list:
            for heat in uav.microservicesHeat:
                if (heat > max_heat):
                    max_heat = heat
        longest_path = self.path_cost_matrix.max()
        big_M = max_heat * longest_path + 1
        self.big_M = big_M
        self.X, self.Y, self.Z = [], [], []
        # Add variables
        for uav in self.uav_list:
            for ms in self.ms_list:
                x = self.model.addVar(name=(f"X uav_{uav.position[0]},"
                                            f"{uav.position[1]} ms_{ms.id}"),
                                      vtype="B")
                self.X.append(x)
                z = self.model.addVar(name=(f"Z uav_{uav.position[0]},"
                                            f"{uav.position[1]} ms_{ms.id}"),
                                       vtype="I",
                                       lb=-big_M,
                                       ub=max_heat * longest_path + big_M)
                self.Z.append(z)
                for uav2 in self.uav_list:
                    y = self.model.addVar(name= (f".Y uav_{uav.position[0]},"
                                                 f"{uav.position[1]} "
                                                 f"ms_{ms.id}"
                                                 f" uav_{uav2.position[0]},"
                                                 f"{uav2.position[1]}"),
                                         vtype="B")
                    self.Y.append(y)
        self.X = np.array(self.X).reshape((len(self.uav_list), 
                                           len(self.ms_list)))
        self.Y = np.array(self.Y).reshape((len(self.uav_list),
                                           len(self.ms_list),
                                           len(self.uav_list)))
        self.Z = np.array(self.Z).reshape((len(self.uav_list),
                                           len(self.ms_list)))
        # Add constraints
        repli_index_constraints = []
        for j, ms in enumerate(self.ms_list):
            constraint = ms.replicationIndex - gp.quicksum(self.X[:,j]) == 0
            constraint = self.model.addConstr(
                constraint,
                name = f"Repl. of {ms.id} >= {ms.replicationIndex}")
            repli_index_constraints.append(constraint)
        cpu_cap_constraints = []
        ram_cap_constraints = []
        for i, uav in enumerate(self.uav_list):
            constraint = uav.cpuCapacity >= \
                gp.quicksum([self.X[i,j] * ms.cpuRequirement \
                      for j, ms in enumerate(self.ms_list)])
            constraint = self.model.addConstr(
                constraint,
                name = f"CPU cap. of {uav.id} not surpased")
            cpu_cap_constraints.append(constraint)
            constraint = uav.ramCapacity >= \
                gp.quicksum([self.X[i,j] * ms.ramRequirement \
                      for j, ms in enumerate(self.ms_list)])
            constraint = self.model.addConstr(
                constraint,
                name = f"RAM cap. of {uav.id} not surpased")
            ram_cap_constraints.append(constraint)
        # Add self.Y activation artificial constraints
        y_activation_constraints = []
        for i, uav in enumerate(self.uav_list):
            for j, ms in enumerate(self.ms_list):
                constraint = (len(self.uav_list) - 1) == gp.quicksum(self.Y[i,j,:])
                constraint = self.model.addConstr(
                    constraint,
                    name = f"Only one y activated for i={i} and j={j}")
                y_activation_constraints.append(constraint)
        z_cost_constraint = []
        for i, uav in enumerate(self.uav_list):
            for j, ms in enumerate(self.ms_list):
                    for k, uav2 in enumerate(self.uav_list):
                        ms_heat = self.uav_list[i].microservicesHeat[j]
                        path_cost = self.path_cost_matrix[i,k]
                        y_activation = self.Y[i,j,k] * big_M
                        x = self.X[k,j]
                        z = self.Z[i,j]
                        constraint = z >= \
                            (path_cost * x * ms_heat) \
                            + (1 - x) * big_M \
                            - y_activation
    #                        + (1 - self.X[k,j]) * big_M \

                        constraint = self.model.addConstr(
                            constraint,
                            name = f"Cost of {ms.id} for {uav.id}")
                        z_cost_constraint.append(constraint)
        self.model.setObjective(gp.quicksum(self.Z.flatten()), gp.GRB.MINIMIZE)

    def solve(self):

        """Once the problem especification is built, attempt to solve
        it.
        """

        status = self.model.optimize()
        deployment_dict = { ms.idToInt():[] for ms in self.ms_list}
        for i, x_row in enumerate(self.X):
            for j, x in enumerate(x_row):
                if (abs(int(x.x)) == 1):
                    deployment_dict[self.ms_list[j].idToInt()].append(i)
        return deployment_dict

    def test(self, n_validations):

        """Use the solver to check the best possible outcome of
        n_validations episodes.

        Parameters:
        - n_validations: int = The number of episodes to execute.
        """

        rewards = []
        ep_lengths = []
        jumps = []
        ms_replicas = int(sum([ms.replicationIndex for ms in self.ms_list]))
        for _ in range(n_validations):
            done = False
            ep_length = 0
            ep_reward = 0
            state, info = self.env.reset(testing=True)
            self.build_model()
            solution_dict = self.solve()
            jumps.append(sum([z.x for z in self.Z.flatten()]))
            while (not done):
                action = solution_dict[state[-ms_replicas]].pop(0)
                next_state, reward, terminated, truncated, info =\
                    self.env.step(action)
                action = solution_dict[state[-ms_replicas]]
                state = next_state
                done = terminated or truncated
                ep_length += 1
                ep_reward += reward
            rewards.append(ep_reward)
            ep_lengths.append(ep_length)
        return sum(rewards) / n_validations, sum(ep_lengths) / n_validations, sum(jumps) / n_validations

