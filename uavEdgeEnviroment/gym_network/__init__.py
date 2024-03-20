from gymnasium.envs.registration import register

register(
     id="NetworkEnv-v0",
     entry_point="gym_network.envs:NetworkEnv",
     max_episode_steps=50,
     reward_threshold=None,
     nondeterministic=False,
     order_enforce=True,
     autoreset=False, 
)

register(
     id="TestEnv-v0",
     entry_point="gym_network.envs:TestEnv",
     max_episode_steps=100,
     reward_threshold=None,
     nondeterministic=False,
     order_enforce=True,
     autoreset=False, 
)
