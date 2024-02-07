from setuptools import setup

setup(
    name="gym_network",
    version="0.0.1",
    install_requires=["gymnasium==0.29.1", 
                      "networkx==3.2.1",
                      "numpy==1.26.3",
                      "perlin-noise==1.12"],
)
