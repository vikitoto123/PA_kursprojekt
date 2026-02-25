import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import numpy as np
import optuna

def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical('n_steps', [256, 512, 1024, 2048])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    n_epochs = trial.suggest_int('n_epochs', 5, 20)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    
    env = Monitor(gym.make("LunarLander-v3"))
    model = PPO(
        "MlpPolicy", env, 
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size, 
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=0
    )
    model.learn(total_timesteps=50000)
    
    rewards = env.get_episode_rewards()
    mean_reward = np.mean(rewards[-20:])
    env.close()
    
    return mean_reward

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print("Best hyperparameters:", study.best_params)
print("Best reward:", study.best_value)

env = Monitor(gym.make("LunarLander-v3"))
model = PPO("MlpPolicy", env, **study.best_params, verbose=1)
model.learn(total_timesteps=100000)

rewards = env.get_episode_rewards()
for i, reward in enumerate(rewards):
    print(f"Episode {i+1}: Reward = {reward:.2f}")

window = 10
moving_avg = [np.mean(rewards[max(0, i-window+1):i+1]) for i in range(len(rewards))]

plt.figure(figsize=(10, 6))
plt.plot(rewards, alpha=0.3, label='Episode Reward')
plt.plot(moving_avg, linewidth=2, label=f'Moving Average (window={window})')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('LunarLander PPO Learning Curve with Optuna Tuning')
plt.legend()
plt.grid(True)
plt.savefig('learning_curve_ppo_optuna.png')
plt.show()

env.close()