import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf

class Environment(gym.Env):
    def __init__(self):
        # https://gymnasium.farama.org/api/spaces/
        
        # Three possible actions (0,1,2)
        # Buy = 0, Sell = 1, Hold = 2
        self.action_space = spaces.Discrete(3)

        # 5D vector stored as np array, holding what is observable in each state:
        # ETF price, cash held, shares held, F&G index value, timestep
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,))


        self.data = yf.download("VOO", start="2010-01-01", end="2023-12-31")
        self.current_step = 0
        self.starting_cash = 10000
        self.monthly_contribution = 500
        self.cash = self.starting_cash
        self.shares_held = 0
        self.prev_value = 0

    # https://gymnasium.farama.org/api/env/#gymnasium.Env.reset
    # Resets the environment to an initial internal state, returning an initial observation and info
    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.cash = self.starting_cash
        self.shares_held = 0
        self.prev_value = self.cash + self.shares_held * self.data.iloc[self.current_step]["Close"].iloc[0]
        
        # return observation, info_dict
        return self.get_observation(), {}

    # https://gymnasium.farama.org/api/env/#gymnasium.Env.step
    # Run one timestep of the environment’s dynamics using the agent actions
    def step(self, action):
        # Get current ETF price
        data_row = self.data.iloc[self.current_step]
        price = float(data_row["Close"].iloc[0])
        
        # Apply action
        # FOR NOW, when buying/selling we either buy with all the cash or sell all shares (to simplify)
        if action == 0: # BUY
            print("BUY")
            shares_to_buy = self.cash // price
            self.shares_held += shares_to_buy
            self.cash -= shares_to_buy * price
        elif action == 1: # SELL
            print("SELL")
            self.cash += self.shares_held * price
            self.shares_held = 0
        elif action == 2: # HOLD
            print("HOLD")
            pass

        # Increase timestep
        self.current_step += 1

        # Get current portfolio value
        new_portfolio_value = self.cash + self.shares_held * price

        # Calculate reward
        reward = new_portfolio_value - self.prev_value
        self.prev_value = new_portfolio_value

        # Check if episode is over
        # print("Current step:", self.current_step)
        # print("Len data:", len(self.data))
        terminated = self.current_step >= len(self.data) - 1

        return self.get_observation(), reward, terminated, False, {}

    # Returns the current state of the environment as an np array
    def get_observation(self):
        data_row = self.data.iloc[self.current_step]
        price = float(data_row["Close"].iloc[0])
        fear_and_greed = 50

        # price, cash, shares held, F&G, 
        return np.array([price, self.cash, self.shares_held, fear_and_greed, self.current_step], dtype=np.float32)

    # Compute the render frames as specified by render_mode during the initialization of the environment
    def render(self):
        pass

    def test_data(self):
        observation, _ = env.reset()
        print("Initial state:", observation)
        print("TESTING DATA:")
        print(env.data.head())

        # CHECK IF THE VOO SHARE PRICE HELD HERE IS ACTUALLY VALID
        print("STEP 1:", env.step(0))
        print("STEP 2:", env.step(1))
        print("STEP 3:", env.step(2))
        print("STEP 4:", env.step(2))
        print("STEP 5:", env.step(0))

env = Environment()
env.test_data()
