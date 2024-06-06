from gymnasium import Wrapper, Env


class ActionUnfolderWrapper(Wrapper):


    def __init__(self, env: Env, dim_uav: int):
        super().__init__(env)
        self.dim_uav = dim_uav

    def step(self, action: "Any") -> "Any":
        action_uav = action // self.dim_uav
        action_ms = action % self.dim_uav
        return super().step([action_uav, action_ms])
