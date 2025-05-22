from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env import Environment

env = Environment()

check_env(env, warn=True)

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=2000000)

model.save("ppo_invest_ai")

print("Training complete and model saved.")
