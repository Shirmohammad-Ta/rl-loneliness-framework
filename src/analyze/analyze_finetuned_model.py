
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from loneliness_env import LonelinessEnv

# بارگذاری مدل آموزش‌دیده فاین‌تیون‌شده
model = DQN.load("dqn_loneliness_model_finetuned")

# ساخت محیط
env = LonelinessEnv()

# تحلیل رفتار مدل نهایی در 1000 اپیزود
n_episodes = 1000
episode_rewards = []
all_H = []
all_T = []
all_FL = []

for ep in range(n_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    H_vals, T_vals, FL_vals = [], [], []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        H_vals.append(obs[0])
        T_vals.append(obs[1])
        S = 1 / (1 + np.exp(-env.alpha * obs[1] * (1 - obs[0])))
        FL = S * env.weighted_sum
        FL_vals.append(FL)

    episode_rewards.append(total_reward)
    all_H.append(H_vals)
    all_T.append(T_vals)
    all_FL.append(FL_vals)

# میانگین‌گیری برای نمودارهای نهایی
max_len = max(len(h) for h in all_H)
avg_H = np.zeros(max_len)
avg_T = np.zeros(max_len)
avg_FL = np.zeros(max_len)
counts = np.zeros(max_len)

for h, t, fl in zip(all_H, all_T, all_FL):
    for i in range(len(h)):
        avg_H[i] += h[i]
        avg_T[i] += t[i]
        avg_FL[i] += fl[i]
        counts[i] += 1

avg_H /= counts
avg_T /= counts
avg_FL /= counts

# رسم نمودار reward
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, marker='o', color='green')
plt.title("Total Reward per Episode (Fine-Tuned Model)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig("finetuned_reward_plot.png")
plt.close()

# رسم نمودارهای H, T, FL
plt.figure(figsize=(12, 6))
plt.plot(avg_H, label="Average H (Identity)", color='blue')
plt.plot(avg_T, label="Average T (Loneliness)", color='red')
plt.plot(avg_FL, label="Average FL (Loneliness Poverty)", color='purple')
plt.title("Average Dynamics Over Time (Fine-Tuned Model)")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("finetuned_dynamics_plot.png")
plt.close()
