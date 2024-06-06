# UAV-EDGE-DRL

This repository contains a modular implementation of a DQN with multiple modifications:

 - Prioritized Experience replay (PER).
 - Boltzmann Policy.
 - Double DQN (DDQN).
 - Dueling Networks.
 - Polyak update (soft updating).

It is implemented in a modular fashion so that the developer can choose the combination of componentes that better fits its needs.

Alognside the implementation of the DQN, a Gymnasium enviroment is also provided to train the agent to solve an specific placement problem. This enviroment an UAV-based Flying Ad Hoc Network (FANET) with edge computing capabilities. The agent learns to deploy microservice instances in order to provide them to ground users and IoT devices while minimizing the latency.

To automate the generation of syntethic enviroment Perlin Noise is employed.
