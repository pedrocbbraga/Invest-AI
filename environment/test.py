from stable_baselines3 import PPO
from env import Environment

def test_model(model_path="ppo_invest_ai.zip", steps=1000):
    env = Environment()
    model = PPO.load(model_path)

    obs, _ = env.reset()
    done = False
    total_reward = 0
    step = 0

    print("Starting evaluation...\\n")

    while not done and step < steps:
        action, _states = model.predict(obs)
        obs, reward, done, _, _ = env.step(action) 
        total_reward += reward
        step += 1

    print(f"\\nEvaluation complete.")
    print(f"Total steps taken: {step}")
    print(f"Total reward earned: {total_reward:.2f}")

test_model()