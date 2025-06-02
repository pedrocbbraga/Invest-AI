from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env import Environment

env = Environment()

check_env(env, warn=True)

model = PPO("MlpPolicy",
            env,
            learning_rate=0.01,
            n_steps=2500,
            n_epochs=15,
            gamma=0.99,
            clip_range=0.1,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1)

model.learn(total_timesteps=1000000)

model.save("ppo_invest_ai")

print("Training complete and model saved.")
