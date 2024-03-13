from gym_network.envs.Microservice import Microservice
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
                 ramCapacity: float = 4.0,
                 ramAllocated: float = 0.0,
                 cpuCapacity: float = 4.0,
                 cpuAllocated: float = 0.0,
                 microservices: list[Microservice] = [],
                 longestPath: int = 100 ) -> None:

        self.id = uavId
        self.position = position
        self.ramCapacity = ramCapacity
        self.ramAllocated = ramAllocated
        self.cpuCapacity = cpuCapacity
        self.cpuAllocated = cpuAllocated
        self.microservices = np.zeros(len(microservices))
        self.microservicesHeat = np.zeros(len(microservices))
        for index, row in enumerate(self.microservicesHeat):
            self.microservicesHeat[index] = \
                microservices[index]\
                    .heatmap[self.position[0]][self.position[1]]
        self.microservicesCosts = []
        for heatValue in self.microservicesHeat:
            self.microservicesCosts.append(heatValue * longestPath)

    def deployMicroservice(self, ms: Microservice, msIndex: int) -> bool:

        """Deploy a microservice if it fits"""

        if (self.ms_fits(ms, msIndex)):
            self.ramAllocated += ms.ramRequirement
            self.cpuAllocated += ms.cpuRequirement
            self.microservices[msIndex] = 1
            return True
        else:
            return False


    def ms_fits(self, ms: Microservice, msIndex: int) -> bool:

        ramRemaining: float = \
            self.ramCapacity - (self.ramAllocated + ms.ramRequirement)
        cpuRemaining: float = \
            self.cpuCapacity - (self.cpuAllocated + ms.cpuRequirement)
        if ((ramRemaining < 0) or (cpuRemaining < 0)):
            return False
        else:
            return True

    def __str__(self) -> str:
        output =   f'UAV id: {self.id}'\
                 + f'\n\t-Position: {self.position}'\
                 + f'\n\t-RAM: {self.ramCapacity} (capacity)'\
                 + f' {self.ramAllocated} (allocated)'\
                 + f'\n\t-CPU: {self.cpuCapacity} (capacity)'\
                 + f' {self.cpuAllocated} (allocated)'\
                 + f'\n\t-Microservices: {self.microservices}'\
                 + f'\n\t-Microservices\'s heat: {self.microservicesHeat}'\
                 + f'\n\t-Microservices cost: {self.microservicesCosts}'
        return output
                
    def toJSON(self) -> dict:
        json = {
            'uavId': self.id,
            'position': self.position,
            'ramCapacity': self.ramCapacity,
            'ramAllocated': self.ramAllocated,
            'cpuCapacity': self.cpuCapacity,
            'cpuAllocated': self.cpuAllocated,
            'microservices': [ms for ms in self.microservices],
            'microservicesHeat': [heat for heat in self.microservicesHeat],
            'microservicesCost': [cost for cost in self.microservicesCosts]
            }
        return json
