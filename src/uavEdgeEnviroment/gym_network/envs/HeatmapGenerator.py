from perlin_noise import PerlinNoise
import random
import numpy as np
import torch
from time import time
import math


class HeatmapGenerator(object):


    """ HeatmapGenerator uses perlim noise to create realistic heatmaps
    for microservices.
    
    Attributes:
    - noise_generator: The perlin noise generator.
    """

    def __init__(self, 
                 min_octave: float = 3.5,
                 max_octave:float = 6.0):

        """
        Initialize the perlin noise generator.
        """

        self.device = torch.device("cpu")
        if (torch.cuda.is_available()):
            self.device = torch.device("cuda")
        self.min_octave = min_octave
        self.max_octave = max_octave

    def generate_heatmap_torch(self,
                               heatmap_size: tuple,
                               positions: list) -> torch.tensor:

        """
        Use the perlin noise generator to produce a Heatmap for a
        microservice. Operations are paralelized with torch.

        Parameters:
        - heatmap_size: 2D size of the scenario.
        - positions: The position of the uavs within the scenario.
        """

        start_time = time()
        octave = random.uniform(self.min_octave, self.max_octave)
        noise_generator = PerlinNoise(octaves = octave, seed = random.random())
        x_coords = torch.tensor(np.array([p[0] for p in positions]),
                                dtype=torch.float).to(self.device)
        y_coords = torch.tensor(np.array([p[1] for p in positions]),
                                dtype=torch.float).to(self.device)
        x_coords /= heatmap_size[0]
        y_coords /= heatmap_size[1]
        coords = zip(x_coords, y_coords)
        noise_map = [noise_generator([x, y]) for x, y in coords]
        noise_map = (torch.tensor(np.array(noise_map)) + 1.0) / 2.0
        max_noise, min_noise = torch.max(noise_map), torch.min(noise_map)
        noise_map = (noise_map - min_noise) / (max_noise- min_noise)
        mask5 = torch.where(noise_map > 0.9, 1.0, 0.0)
        mask4 = torch.where(noise_map > 0.7, 1.0, 0.0) - mask5
        mask3 = torch.where(noise_map > 0.55, 1.0, 0.0) - mask4 - mask5
        mask2 = torch.where(noise_map > 0.3, 1.0, 0.0) - mask3 - mask4 - mask5
        mask1 = torch.where(noise_map > 0.2, 1.0, 0.0) - mask2 - mask3 - mask4 - mask5
        noise_map = mask5 * 5 + mask4 * 4 + mask3 * 3 + mask2 * 2 + mask1
        return noise_map

    def generate_heatmap(self,
                         heatmap_size: tuple,
                         positions: list) -> list:

        """
        Use the perlin noise generator to produce a Heatmap for a
        microservice.

        Parameters:
        - heatmap_size: 2D size of the scenario.
        - positions: The position of the uavs within the scenario.
        """

        start_time = time()
        octave = random.uniform(self.min_octave, self.max_octave)
        noise_generator = PerlinNoise(octaves = octave, seed = random.random())
        noiseValues = []
        for uav in positions:
            coords = [uav[0] / heatmap_size[0], uav[1] / heatmap_size[1]]
            noiseValue = (noise_generator(coords) + 1) / 2
            noiseValues.append(noiseValue)
        maxNoise, minNoise = max(noiseValues), min(noiseValues)
        finalNoises = []
        for nv in noiseValues:
            nv = (nv - minNoise) / (maxNoise -  minNoise)
            if   (nv < .2): nv = 0
            elif (nv < .3): nv = 1
            elif (nv < .55): nv = 2
            elif (nv < .7): nv = 3
            elif (nv < .9): nv = 4
            else: nv = 5
            finalNoises.append(nv)
        return finalNoises

