from gymnasium.envs.registration import register

register(
     id="NetworkEnv-v0",
     entry_point="gym_network.envs:NetworkEnv",
     max_episode_steps=None,
     reward_threshold=None,
     nondeterministic=False,
     order_enforce=True,
     autoreset=False, 
)
