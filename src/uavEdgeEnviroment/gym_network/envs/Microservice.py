import numpy as np

class Microservice(object):

    def __init__(self,
                 msId: str,
                 ram_req: float = 1,
                 cpu_req: float = 1,
                 replic_index: int = 1) -> None:

        self.id = msId
        self.ram_req = ram_req
        self.cpu_req = cpu_req
        self.replic_index = replic_index
        self.replic_left = self.replic_index
