# run_env_test.py
from env import Environment

env = Environment()
obs, _ = env.reset()

done = False
step = 0

while not done and step < 10:  # test first 10 steps
    action = [0.5]  # neutral allocation (50%)
    obs, reward, done, _, _ = env.step(action)
    step += 1
