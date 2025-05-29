from stable_baselines3 import PPO
from env import Environment
import pandas as pd
import matplotlib.pyplot as plt

def test_model(model_path="ppo_invest_ai.zip", steps=3000):
    env = Environment()
    model = PPO.load(model_path)

    obs, _ = env.reset()
    done = False
    total_reward = 0
    step = 0

    dates = []
    agent_values = []
    asset_allocation_weights = []
    buy_and_hold_values = []

    print("Starting evaluation...\\n")

    # Buy and hold strategy for comparison purposes (does not include transaction costs f that)
    buy_and_hold_cash = env.starting_cash
    buy_and_hold_data = env.data_voo
    buy_and_hold_shares = buy_and_hold_cash // buy_and_hold_data.iloc[0]["Close"]
    buy_and_hold_cash -= buy_and_hold_shares * buy_and_hold_data.iloc[0]["Close"]
    # for i in range(1, steps):
    #     if i % 21 == 0:
    #         buy_and_hold_cash += env.monthly_contribution
    #         price = float(buy_and_hold_data.iloc[i]["Close"])
    #         shares_bought = buy_and_hold_cash // price
    #         buy_and_hold_cash -= shares_bought * price
    #         buy_and_hold_shares += shares_bought

    while not done and step < steps:
        action, _states = model.predict(obs)
        obs, reward, done, _, _ = env.step(action) 
        total_reward += reward

        date = env.data_1.index[min(env.current_step, len(env.data_1)-1)]
        dates.append(date)

        price_1 = float(env.data_1.iloc[min(step, len(buy_and_hold_data)-1)]["Close"])
        price_2 = float(env.data_2.iloc[min(step, len(buy_and_hold_data)-1)]["Close"])
        price_3 = float(env.data_3.iloc[min(step, len(buy_and_hold_data)-1)]["Close"])

        agent_portfolio_value_timestep = (env.cash + env.shares_held_1 * price_1 + env.shares_held_2 * price_2 + env.shares_held_3 * price_3)
        agent_values.append(agent_portfolio_value_timestep)
        current_buy_and_hold_price = float(buy_and_hold_data.iloc[min(step, len(buy_and_hold_data)-1)]["Close"])
        current_buy_and_hold_value = float(buy_and_hold_cash + buy_and_hold_shares * current_buy_and_hold_price)
        buy_and_hold_values.append(current_buy_and_hold_value)

        total_value = agent_portfolio_value_timestep
        w1 = (env.shares_held_1 * price_1) / total_value
        w2 = (env.shares_held_2 * price_2) / total_value
        w3 = (env.shares_held_3 * price_3) / total_value
        asset_allocation_weights.append([w1, w2, w3])

        if step % 21 == 0:
            buy_and_hold_cash += env.monthly_contribution
            price = float(buy_and_hold_data.iloc[step]["Close"])
            shares_bought = buy_and_hold_cash // price
            buy_and_hold_cash -= shares_bought * price
            buy_and_hold_shares += shares_bought

        step += 1

    final_price = float(buy_and_hold_data.iloc[min(step, len(buy_and_hold_data)-1)]["Close"])

    final_price_1 = float(env.data_1.iloc[min(step, len(buy_and_hold_data)-1)]["Close"])
    final_price_2 = float(env.data_2.iloc[min(step, len(buy_and_hold_data)-1)]["Close"])
    final_price_3 = float(env.data_3.iloc[min(step, len(buy_and_hold_data)-1)]["Close"])

    buy_and_hold_value = float(buy_and_hold_cash + buy_and_hold_shares * final_price)
    agent_portfolio_value = (env.cash + env.shares_held_1 * final_price_1 + env.shares_held_2 * final_price_2 + env.shares_held_3 * final_price_3)

    print(f"\\nEvaluation complete.")
    print(f"Total steps taken: {step}")
    print(f"Total reward earned: {total_reward:.2f}")
    print(f"Agent final portfolio value: {agent_portfolio_value:.2f}")
    print(f"Buy and Hold S&P final value: {buy_and_hold_value:.2f}")

    df = pd.DataFrame({"Date": dates, "Agent": agent_values, "Buy & Hold S&P 500": buy_and_hold_values}).set_index("Date")
    df.plot(title="Cumulative Return Agent vs S&P Buy & Hold", ylabel="Portfolio Value", grid=True)
    plt.tight_layout()
    plt.savefig("cumulative_returns.png")
    plt.close()

    weights_df = pd.DataFrame(asset_allocation_weights, columns=["VUG", "VT", "QQQ"], index=dates)
    weights_df = weights_df.rolling(window=100, min_periods=1).mean()
    plt.figure(figsize=(10, 6))
    for col in weights_df.columns:
        plt.plot(weights_df.index, weights_df[col], label=col)
    plt.title("Portfolio Weights Over Time")
    plt.ylabel("Weight")
    plt.xlabel("Date")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("portfolio_weights.png")
    plt.close()

test_model()