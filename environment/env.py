import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class Environment(gym.Env):
    def __init__(self, marketData):
        # https://gymnasium.farama.org/api/spaces/
        
        # Three possible actions (0,1,2)
        # Buy = 0, Sell = 1, Hold = 2
        self.action_space = spaces.Discrete(3)

        # 5D vector stored as np array, holding what is observable in each state:
        # ETF price, cash held, shares held, F&G index value, timestep
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,))

        self.current_step = 0
        self.starting_cash = 10000
        self.monthly_contribution = 500
        self.cash = self.starting_cash
        self.shares_held = 0

    # https://gymnasium.farama.org/api/env/#gymnasium.Env.reset
    # Resets the environment to an initial internal state, returning an initial observation and info
    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.cash = self.starting_cash
        self.shares_held = 0
        
        # return observation, info_dict
        return self.get_observation(), {}

    # https://gymnasium.farama.org/api/env/#gymnasium.Env.step
    # Run one timestep of the environment’s dynamics using the agent actions
    def step(self, action):
        pass
        # Get current ETF price

        # Apply action

        # Increase timestep

        # Calculate reward

        # Check if episode is over

        # return observation, reward, terminated, truncated, info_dict

    # Returns the current state of the environment as an np array
    def get_observation(self):
        pass

    # Compute the render frames as specified by render_mode during the initialization of the environment
    def render(self):
        pass