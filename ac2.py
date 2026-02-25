import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import numpy as np
import optuna

def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical('n_steps', [5, 10, 20, 50, 100])
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    ent_coef = trial.suggest_float('ent_coef', 1e-8, 0.1, log=True)
    vf_coef = trial.suggest_float('vf_coef', 0.1, 0.9)
    
    env = Monitor(gym.make("LunarLander-v3"))
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        verbose=0
    )
    model.learn(total_timesteps=50000)
    
    # Evaluate
    rewards = env.get_episode_rewards()
    mean_reward = np.mean(rewards[-20:]) 
    env.close()
    
    return mean_reward

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print("Best hyperparameters:", study.best_params)
print("Best reward:", study.best_value)

env = Monitor(gym.make("LunarLander-v3"))
model = A2C("MlpPolicy", env, **study.best_params, verbose=1)
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
plt.title('LunarLander A2C Learning Curve with Optuna Tuning')
plt.legend()
plt.grid(True)
plt.savefig('a2c_learning_curve_optuna.png')
plt.show()

env.close()