# Modeling Existential Loneliness with Reinforcement Learning

This repository contains the code, environment, and models for the paper:
**"A Reinforcement Learning Framework Bridging Psychology, Mathematics, and AI"**

##  Overview

This project converts a psychological theory (Existential Loneliness Poverty) into a mathematical model, then an algorithm, and finally trains a reinforcement learning agent to optimize human identity and loneliness balance.

### Key Components:
- Psychological theory: Identity, Loneliness, Sacrifice
- Mathematical model: FL = S * Weighted Sum
- Environment: Custom Gym environment (`LonelinessEnv`)
- RL Agent: DQN using Stable-Baselines3
- Output: Interpretable dynamics of H, T, and FL over time

---

##  Project Structure

- `src/env/` - Custom Gym environment
- `src/train/` - Training and fine-tuning scripts
- `src/analyze/` - Post-training analysis and plots
- `data/` - Synthetic dataset
- `models/` - Pretrained and fine-tuned models
- `results/` - Reward and dynamic plots
- `docs/` - Paper summary or PDF version

---

##  How to Run

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Train model:
```
python src/train/train_agent.py
```

3. Fine-tune with real data:
```
python src/train/retrain_with_real_data.py
```

4. Analyze results:
```
python src/analyze/analyze_finetuned_model.py
```


---


