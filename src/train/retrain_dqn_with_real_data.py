
import pandas as pd
import gym
import numpy as np
from stable_baselines3 import DQN
from loneliness_env import LonelinessEnv

# بارگذاری داده واقعی مصنوعی
df = pd.read_csv("loneliness_dataset.csv")

# بارگذاری مدل آموزش‌دیده قبلی
model = DQN.load("dqn_loneliness_model")

# تعریف محیط
env = LonelinessEnv()

# تقویت یادگیری با داده‌های موجود (درون محیط)
# به‌صورت تجربه‌های محیطی مجدد اجرا می‌کنیم
for episode in range(500):  # آموزش مجدد در ۵۰۰ اپیزود دیگر
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

# ذخیره مدل آموزش‌دیده نهایی
model.save("dqn_loneliness_model_finetuned")
