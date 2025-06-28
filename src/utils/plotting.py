import matplotlib.pyplot as plt

def plot_episode_rewards(rewards, save_path="reward_plot.png", title="Total Reward per Episode"):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, color='green')
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_dynamics(H, T, FL, save_path="dynamics_plot.png", title="Dynamics of H, T, FL Over Time"):
    plt.figure(figsize=(12, 6))
    plt.plot(H, label="Identity (H)", color='blue')
    plt.plot(T, label="Loneliness (T)", color='red')
    plt.plot(FL, label="Loneliness Poverty (FL)", color='purple')
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
