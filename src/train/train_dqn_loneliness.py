
import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from loneliness_env import LonelinessEnv  # فرض بر این که فایل در همان پوشه است

# 1. ساخت محیط
env = LonelinessEnv()

# 2. بررسی صحت محیط
check_env(env, warn=True)

# 3. آموزش مدل DQN
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-3,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    tensorboard_log="./dqn_loneliness_tensorboard/"
)

# 4. شروع آموزش
model.learn(total_timesteps=10000)

# 5. ذخیره مدل آموزش‌دیده
model.save("dqn_loneliness_model")

# 6. تست عامل آموزش‌دیده
env = LonelinessEnv()
obs = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward

print("Total reward:", total_reward)
