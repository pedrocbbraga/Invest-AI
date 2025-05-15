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

    buy_and_hold_cash = env.starting_cash
    buy_and_hold_data = env.data
    buy_and_hold_shares = buy_and_hold_cash // buy_and_hold_data.iloc[0]["Close"]
    buy_and_hold_cash -= buy_and_hold_shares * buy_and_hold_data.iloc[0]["Close"]

    while not done and step < steps:
        action, _states = model.predict(obs)
        obs, reward, done, _, _ = env.step(action) 
        total_reward += reward
        step += 1

    final_price = float(buy_and_hold_data.iloc[min(step, len(buy_and_hold_data)-1)]["Close"])
    buy_and_hold_value = float(buy_and_hold_cash + buy_and_hold_shares * final_price)

    print(f"\\nEvaluation complete.")
    print(f"Total steps taken: {step}")
    print(f"Total reward earned: {total_reward:.2f}")
    print(f"Agent final portfolio value: {env.cash + env.shares_held * final_price:.2f}")
    print(f"Buy and Hold final value: {buy_and_hold_value:.2f}")

test_model()