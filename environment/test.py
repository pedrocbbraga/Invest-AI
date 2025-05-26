from stable_baselines3 import PPO
from env import Environment

def test_model(model_path="ppo_invest_ai.zip", steps=3000):
    env = Environment()
    model = PPO.load(model_path)

    obs, _ = env.reset()
    done = False
    total_reward = 0
    step = 0

    print("Starting evaluation...\\n")

    # Buy and hold strategy for comparison purposes (does not include transaction costs f that)
    buy_and_hold_cash = env.starting_cash
    buy_and_hold_data = env.data_voo
    buy_and_hold_shares = buy_and_hold_cash // buy_and_hold_data.iloc[0]["Close"]
    buy_and_hold_cash -= buy_and_hold_shares * buy_and_hold_data.iloc[0]["Close"]
    for i in range(1, steps):
        if i % 21 == 0:
            buy_and_hold_cash += env.monthly_contribution
            price = float(buy_and_hold_data.iloc[i]["Close"])
            shares_bought = buy_and_hold_cash // price
            buy_and_hold_cash -= shares_bought * price
            buy_and_hold_shares += shares_bought

    while not done and step < steps:
        action, _states = model.predict(obs)
        obs, reward, done, _, _ = env.step(action) 
        total_reward += reward
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

test_model()