import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
from scraper import get_fear_and_greed_index


class Environment(gym.Env):
    def __init__(self):
        # https://gymnasium.farama.org/api/spaces/
        
        # Three possible actions (0,1,2)
        # Buy = 0, Sell = 1, Hold = 2
        # self.action_space = spaces.Discrete(3)

        # Continous actions
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # 5D vector stored as np array, holding what is observable in each state:
        # ETF price, cash held, shares held, F&G index value, timestep
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,))


        self.data = yf.download("VOO", start="2010-01-01", end="2023-12-31")
        self.current_step = 0
        self.starting_cash = 10000
        self.monthly_contribution = 500
        self.cash = self.starting_cash
        self.shares_held = 0
        self.prev_value = 0.0

        self.fear_and_greed = 50

    # https://gymnasium.farama.org/api/env/#gymnasium.Env.reset
    # Resets the environment to an initial internal state, returning an initial observation and info
    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.cash = self.starting_cash
        self.shares_held = 0
        self.prev_value = self.cash + self.shares_held * self.data.iloc[self.current_step]["Close"].iloc[0]
        
        self.fear_and_greed = get_fear_and_greed_index()

        # return observation, info_dict
        return self.get_observation(), {}

    # https://gymnasium.farama.org/api/env/#gymnasium.Env.step
    # Run one timestep of the environment’s dynamics using the agent actions
    def step(self, action):
        # Get current ETF price
        data_row = self.data.iloc[self.current_step]
        price = float(data_row["Close"])

        # Determines asset allocation as action (range between 0 and 1)
        target_allocation = float(np.clip(action[0], 0, 1))
        target_value_in_voo = self.prev_value * target_allocation
        current_value_in_voo = self.shares_held * price
        delta = target_value_in_voo - current_value_in_voo
        shares_to_trade = int(delta // price)

        # Transaction cost = 0.05%, as per the article "Leveraging LLM-based sentiment
        # analysis for portfolio optimization with proximal policy optimization"
        trade_value = abs(shares_to_trade * price)
        transaction_cost = 0.0005 * trade_value

        # Continous portfolio management/allocation
        if shares_to_trade > 0:
            print("BUY")
            cost = shares_to_trade * price + transaction_cost
            if cost <= self.cash:
                self.shares_held += shares_to_trade
                self.cash -= cost
        elif shares_to_trade < 0:
            print("SELL")
            shares_to_sell = abs(shares_to_trade)
            if shares_to_sell <= self.shares_held:
                self.shares_held -= shares_to_sell
                self.cash += shares_to_sell * price - transaction_cost
        elif shares_to_trade == 0:
            print("HOLD")
            pass

        # Apply action
        # FOR NOW, when buying/selling we either buy with all the cash or sell all shares (to simplify)
        # if action == 0: # BUY
        #     print("BUY")
        #     shares_to_buy = self.cash // price
        #     self.shares_held += shares_to_buy
        #     self.cash -= shares_to_buy * price

        #     trade_value = shares_to_buy * price
        #     transaction_cost = 0.0005 * trade_value
        #     self.cash -= transaction_cost
        # elif action == 1: # SELL
        #     print("SELL")
        #     self.cash += self.shares_held * price

        #     trade_value = self.shares_held * price
        #     transaction_cost = 0.0005 * trade_value
        #     self.cash -= transaction_cost
            
        #     self.shares_held = 0
        # elif action == 2: # HOLD
        #     print("HOLD")
        #     pass

        # print(self.fear_and_greed)



        # Increase timestep
        self.current_step += 1

        # Monthly contribution aprox every 21 trading days = 1 month
        if self.current_step % 21 == 0:
            self.cash += self.monthly_contribution

        # Get current portfolio value
        new_portfolio_value = self.cash + self.shares_held * price

        # Calculate reward, also as per the above-mentioned article
        reward = np.log(new_portfolio_value / self.prev_value)
        self.prev_value = new_portfolio_value

        # Check if episode is over
        # print("Current step:", self.current_step)
        # print("Len data:", len(self.data))
        terminated = self.current_step >= len(self.data) - 1

        return self.get_observation(), reward, terminated, False, {}

    # Returns the current state of the environment as an np array
    def get_observation(self):
        data_row = self.data.iloc[self.current_step]
        price = float(data_row["Close"])
        fear_and_greed = self.fear_and_greed

        # price, cash, shares held, F&G, 
        return np.array([price, self.cash, self.shares_held, fear_and_greed, self.current_step], dtype=np.float32)

    # Compute the render frames as specified by render_mode during the initialization of the environment
    def render(self):
        pass

#     def test_data(self):
#         observation, _ = env.reset()
#         print("Initial state:", observation)
#         print("TESTING DATA:")
#         print(env.data.head())
#         print("Scraped Fear & Greed Index:", self.fear_and_greed)

#         # CHECK IF THE VOO SHARE PRICE HELD HERE IS ACTUALLY VALID
#         print("STEP 1:", env.step(0))
#         print("STEP 2:", env.step(1))
#         print("STEP 3:", env.step(2))
#         print("STEP 4:", env.step(2))
#         print("STEP 5:", env.step(0))

# env = Environment()
# env.test_data()
