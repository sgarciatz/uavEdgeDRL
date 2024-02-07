import numpy as np

class Microservice(object):

    def __init__(self, msId: str, ramRequirement: float = 1, cpuRequirement: float = 1, replicationIndex: int = 1,  heatmap: np.array = None) -> None:
        self.id: str               = msId
        self.ramRequirement: float = ramRequirement
        self.cpuRequirement: float = cpuRequirement
        self.replicationIndex: int = replicationIndex
        self.heatmap: np.array     = heatmap

    def idToInt(self) -> int:
    
        """
        Converts the ID into a number.
        """
        if (self.id == "empty"): return -1
        
        return int(self.id[-1])

    def __str__(self) -> str:
        heatmapString = ''
        if (self.heatmap is not None):
            for row in self.heatmap:
                heatmapString += f'\n\t{row}'
              
        return f'Microservice: {self.id}\n\t-RAM requirement: {self.ramRequirement}\n\t-CPU requirement: {self.cpuRequirement}\n\t-Heatmap:{heatmapString}\n\t-Replication index: {self.replicationIndex}'
        
    def toJSON(self) -> dict:
        json = {
            'microserviceId' : self.id,
            'ramRequirement' : self.ramRequirement,
            'cpuRequirement' : self.cpuRequirement,
            'replicationIndex' : self.replicationIndex,            
            'heatmap'        :  [[heatValue for heatValue in row] for row in self.heatmap]
            }
        return json
