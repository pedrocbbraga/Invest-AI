import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
from scraper import load_historical_fgi



class Environment(gym.Env):
    def __init__(self):
        # https://gymnasium.farama.org/api/spaces/
        
        # Three possible actions (0,1,2)
        # Buy = 0, Sell = 1, Hold = 2
        # self.action_space = spaces.Discrete(3)

        # Continous actions
        self.action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

        # 5D vector stored as np array, holding what is observable in each state:
        # ETF price, cash held, shares held, F&G index value, timestep
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,))


        # self.data_voo = yf.download("VOO", start="2014-01-03", end="2024-12-31")
        # self.data_1 = yf.download("GOOG", start="2014-01-03", end="2024-12-31")
        # self.data_2 = yf.download("VUG", start="2014-01-03", end="2024-12-31")
        # self.data_3 = yf.download("AAPL", start="2014-01-03", end="2024-12-31")
        self.data_voo = yf.download("VOO", start="2011-01-03", end="2023-12-31")
        self.data_1 = yf.download("VUG", start="2011-01-03", end="2023-12-31")
        self.data_2 = yf.download("VT", start="2011-01-03", end="2023-12-31")
        self.data_3 = yf.download("QQQ", start="2011-01-03", end="2023-12-31")
        self.current_step = 0
        self.starting_cash = 10000
        self.monthly_contribution = 500
        self.cash = self.starting_cash

        self.shares_held_1 = 0
        self.shares_held_2 = 0
        self.shares_held_3 = 0

        self.prev_value = 0.0

        self.fgi_data = load_historical_fgi("fear-greed-2011-2023.csv")


    # https://gymnasium.farama.org/api/env/#gymnasium.Env.reset
    # Resets the environment to an initial internal state, returning an initial observation and info
    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.cash = self.starting_cash
        self.shares_held_1 = 0
        self.shares_held_2 = 0
        self.shares_held_3 = 0
        self.prev_value = self.cash + self.shares_held_1 * self.data_1.iloc[self.current_step]["Close"].iloc[0] + self.shares_held_2 * self.data_2.iloc[self.current_step]["Close"].iloc[0] + self.shares_held_3 * self.data_3.iloc[self.current_step]["Close"].iloc[0]
        
        

        # return observation, info_dict
        return self.get_observation(), {}

    # https://gymnasium.farama.org/api/env/#gymnasium.Env.step
    # Run one timestep of the environment’s dynamics using the agent actions
    def step(self, action):
        # First apply action based on current prices
        row_1 = self.data_1.iloc[self.current_step]
        price_1 = float(row_1["Close"]) if not isinstance(row_1["Close"], pd.Series) else float(row_1["Close"].iloc[0])

        row_2 = self.data_2.iloc[self.current_step]
        price_2 = float(row_2["Close"]) if not isinstance(row_2["Close"], pd.Series) else float(row_2["Close"].iloc[0])

        row_3 = self.data_3.iloc[self.current_step]
        price_3 = float(row_3["Close"]) if not isinstance(row_3["Close"], pd.Series) else float(row_3["Close"].iloc[0])

        target_allocation = np.clip(action, 0, 1)
        if np.sum(target_allocation) > 1:
            target_allocation /= np.sum(target_allocation)

        current_portfolio_value = (
            self.cash +
            self.shares_held_1 * price_1 +
            self.shares_held_2 * price_2 +
            self.shares_held_3 * price_3
        )

        for i, (price, shares_attr) in enumerate([
            (price_1, "shares_held_1"),
            (price_2, "shares_held_2"),
            (price_3, "shares_held_3")
        ]):
            target_value = current_portfolio_value * target_allocation[i]
            current_value = getattr(self, shares_attr) * price
            delta = target_value - current_value
            shares_to_trade = int(delta // price)

            trade_value = abs(shares_to_trade * price)
            transaction_cost = 0.0005 * trade_value

            if shares_to_trade > 0:
                # print("BUY")
                cost = shares_to_trade * price + transaction_cost
                if cost <= self.cash:
                    setattr(self, shares_attr, getattr(self, shares_attr) + shares_to_trade)
                    self.cash -= cost
            elif shares_to_trade < 0:
                # print("SELL")
                shares_to_sell = abs(shares_to_trade)
                if shares_to_sell <= getattr(self, shares_attr):
                    setattr(self, shares_attr, getattr(self, shares_attr) - shares_to_sell)
                    self.cash += shares_to_sell * price - transaction_cost

        self.current_step += 1
        if self.current_step % 21 == 0:
            self.cash += self.monthly_contribution

        if self.current_step < len(self.data_1):
            next_row_1 = self.data_1.iloc[self.current_step]
            next_price_1 = float(next_row_1["Close"]) if not isinstance(next_row_1["Close"], pd.Series) else float(next_row_1["Close"].iloc[0])

            next_row_2 = self.data_2.iloc[self.current_step]
            next_price_2 = float(next_row_2["Close"]) if not isinstance(next_row_2["Close"], pd.Series) else float(next_row_2["Close"].iloc[0])

            next_row_3 = self.data_3.iloc[self.current_step]
            next_price_3 = float(next_row_3["Close"]) if not isinstance(next_row_3["Close"], pd.Series) else float(next_row_3["Close"].iloc[0])

            new_value = (
                self.cash +
                self.shares_held_1 * next_price_1 +
                self.shares_held_2 * next_price_2 +
                self.shares_held_3 * next_price_3
            )
            reward = np.log(new_value / current_portfolio_value)
        else:
            reward = 0.0

        done = self.current_step >= len(self.data_1) - 1

        return self.get_observation(), reward, done, False, {}


    def get_observation(self):
        # Use prices from the previous step, NOT the current step
        prev_step = max(self.current_step - 1, 0)

        row_1 = self.data_1.iloc[prev_step]
        price_1 = float(row_1["Close"]) if not isinstance(row_1["Close"], pd.Series) else float(row_1["Close"].iloc[0])

        row_2 = self.data_2.iloc[prev_step]
        price_2 = float(row_2["Close"]) if not isinstance(row_2["Close"], pd.Series) else float(row_2["Close"].iloc[0])

        row_3 = self.data_3.iloc[prev_step]
        price_3 = float(row_3["Close"]) if not isinstance(row_3["Close"], pd.Series) else float(row_3["Close"].iloc[0])

        date = pd.to_datetime(row_1.name).normalize()

        fg_row = self.fgi_data[self.fgi_data["Date"] == date]
        fear_and_greed = float(fg_row["FearGreedIndex"].iloc[0]) if not fg_row.empty else 50

        return np.array([
            self.cash,
            price_1, self.shares_held_1,
            price_2, self.shares_held_2,
            price_3, self.shares_held_3,
            fear_and_greed,
            self.current_step
        ], dtype=np.float32)

    # Compute the render frames as specified by render_mode during the initialization of the environment
    def render(self):
        pass

#     def test_data(self):
#         observation, _ = env.reset()
#         print("Initial state:", observation)
#         print("TESTING DATA:")
#         print(env.data.head())
#         print("Scraped Fear & Greed Index:", self.fear_and_greed)

#         # CHECK IF THE 1 SHARE PRICE HELD HERE IS ACTUALLY VALID
#         print("STEP 1:", env.step(0))
#         print("STEP 2:", env.step(1))
#         print("STEP 3:", env.step(2))
#         print("STEP 4:", env.step(2))
#         print("STEP 5:", env.step(0))

# env = Environment()
# env.test_data()
