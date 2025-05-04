import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class Environment(gym.Env):
    def __init__(self, marketData):
        pass

    # https://gymnasium.farama.org/api/env/#gymnasium.Env.reset
    # Resets the environment to an initial internal state, returning an initial observation and info
    def reset(self, seed=None, options=None):
        pass
        # return observation, info_dict

    # https://gymnasium.farama.org/api/env/#gymnasium.Env.step
    # Run one timestep of the environment’s dynamics using the agent actions
    def step(self, action):
        pass
        # return observation, reward, terminated, truncated, info_dict

    # Returns the current state of the environment as an np array
    def getState(self):
        pass

    # Compute the render frames as specified by render_mode during the initialization of the environment
    def render(self):
        pass