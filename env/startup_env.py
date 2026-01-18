import gymnasium
from pettingzoo import ParallelEnv

class StartupEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "startup_v0"}

    def __init__(self):
        pass

    def reset(self, seed=None, options=None):
        pass

    def step(self, actions):
        pass

    def render(self):
        pass

    def close(self):
        pass
