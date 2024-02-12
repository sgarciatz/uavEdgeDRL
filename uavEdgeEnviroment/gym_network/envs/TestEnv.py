import gymnasium as gym
import numpy as np


class TestEnv(gym.Env):


    """
    TestEnv is a simple enviroment that is meant to test the learning
    capability of different DRL algorithms and variations.
    
    Attributes:
    - observation_space: A gym space that holds the position of the 
     goal and the current position of the agent.
    - action_space: A gym space that holds two possible actions: 0 is
     going downwards and 1 is going upwards.
    - agent_pos: The position of the agent.
    - goal_pos: The position of the goal.
    """

    def __init__(self):

        """
        Constructor for the NetworkEnv object. It is concerned with how
        complex the enviroment should be.
        """

        self.observation_space = gym.spaces.Box(low=-1,
                                                high=7,
                                                shape = (2, ),
                                                dtype = np.int8,
                                                seed = 42)
        self.action_space = gym.spaces.Discrete(2,
                                                start = 0,
                                                seed = 42)
        self.agent_pos = 0
        self.goal_pos = 0

    def _get_obs(self):

        """
        Returns the observation of the current state of the env.
        """

        obs = np.array([self.agent_pos, self.goal_pos], dtype = np.int8)
        return obs

    def _get_info(self):

        """
        Returns information about the state of the env
        """
        obs = self._get_obs()
        info = {"agent_pos": obs[0], "goal_pos": obs[1]}
        return info

    def reset(self, seed = None, options = None):

        """
        Resets the position of the agent and the position of the goal.
        """

        super().reset(seed=seed)
        self.agent_pos = 1
        self.goal_pos = 7
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def step(self, action):

        """
        Takes the action recieved by parameter and applies the changes
        in the enviroment.
        """

        if (action == 1):
            self.agent_pos += 1
        else:
            self.agent_pos -= 1
        observation = self._get_obs()
        info = self._get_info()
        reward = 0
        terminated = False
        if (self.agent_pos == self.goal_pos):
            reward = 1
            terminated = True
        elif (self.agent_pos < 0):
            reward = -1
            terminated = True
        return observation, reward, terminated, False, info


if __name__ == "__main__":
    myEnv = TestEnv()
    terminated = False
    observation, info = myEnv.reset(seed = 42)
    step_count = 0
    for i in range(2):
        print(f"Episode {i}")
        while (not terminated):
            observation, reward, terminated, _, info = myEnv.step(1)
            print(f"Step {step_count} taken")
            print(f"\t-Agent position: {observation[0]}")
            print(f"\t-Goal position: {observation[1]}")
            print(f"\t-Reward: {reward}")
            if (not terminated):
                step_count += 1
        print("Goal reached")
        print()
        terminated = False
        observation, info = myEnv.reset()

