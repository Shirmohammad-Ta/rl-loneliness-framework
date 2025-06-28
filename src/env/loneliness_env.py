
import gym
import numpy as np

class LonelinessEnv(gym.Env):
    def __init__(self):
        super(LonelinessEnv, self).__init__()

        # fix Parameters
        self.alpha = 1.0
        self.lambda_ = 0.5
        self.gamma = 0.3
        self.delta = 0.05
        self.weighted_sum = 0.68  #  sum(w_i * V_i)

        # : [H, T] State space 0 و 1
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        # Action Space: 5
        self.action_space = gym.spaces.Discrete(5)  # 0 → 0.0, ..., 4 → 1.0

        self.max_steps = 50
        self.reset()

    def reset(self):
        self.H = 0.9
        self.T = 0.6
        self.steps = 0
        return np.array([self.H, self.T], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        sacrifice_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        sacrifice = sacrifice_levels[action]

        S = 1 / (1 + np.exp(-self.alpha * self.T * (1 - self.H)))
        C = S * sacrifice
        self.H = max(0.0, self.H - self.lambda_ * self.T * self.weighted_sum)
        self.T = min(1.0, self.T - self.gamma * C + self.delta)

        FL = S * self.weighted_sum
        reward = -FL  # Reward for FL

        done = self.steps >= self.max_steps or self.H <= 0.01
        state = np.array([self.H, self.T], dtype=np.float32)

        return state, reward, done, {}
