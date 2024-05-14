import numpy as np

class UAV(object):


    """
    The UAV is a class that hold the following information:
    
    - id: The id of the UAV as a way to identify it. string
    - position: It's position within the enviroment to plot it. 
      int, int
    - ramCapacity: Its whole RAM capacity. float
    - ramAllocated: The RAM that has been allocated to deploy 
      microservices. float
    - cpuCapacity: Its whole cpu capacity (cores). float
    - cpuAllocated: The cpu (cores) that has been allocated to deploy
      microservices. float
    - microservices: a one hot encoding list that specifies the 
      which microservices has been deployed. Ej: [0, 1, 0, 1]
    - microservicesHeat: an array holding the heatmap value of
      of each microservice for this UAV. Ej: [1, 4, 0, 5]
    - compoundCost: the result of multiplying each microservice's heat
      by the path length to the closest instance.
    """
    
    def __init__(self, uavId: str,
                 position: list[int],
                 ram_cap: float = 4.0,
                 ram_left: float = 4.0,
                 cpu_cap: float = 4.0,
                 cpu_left: float = 4.0,
                 ms_heats: list = []) -> None:

        self.id = uavId
        self.position = position
        self.ram_cap = ram_cap
        self.ram_left = ram_left
        self.cpu_cap = cpu_cap
        self.cpu_left = cpu_left
        self.ms_heats = ms_heats


    def ms_fits(self, ms):

        """Check wether a microservice fits or not.
        
        Parameters:
        - ms: the microservice to deploy.
        """

        if ((ms.ram_req <= self.ram_left) and (ms.cpu_req <= self.cpu_left)):
            return True
        else:
            return False

    def deploy_ms(self, ms):

        """Reduce the ram_left and the cpu_left because of the
        deployment of ms.
        """

        self.ram_left -= ms.ram_req
        self.cpu_left -= ms.cpu_req
        ms.replic_left -= 1


