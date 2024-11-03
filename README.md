##### Name: SHIVAAY DHONDIYAL
##### Company: CODTECH IT SOLUTIONS PVT.LTD
##### ID: CT08DS8845
##### Domain: ARTIFICIAL INTELLIGENCE
##### Duration: OCTOBER 5th, 2024 to NOVEMBER 5th, 2024

# CartPole Reinforcement Learning Project

## Table of Contents
- [CartPole Reinforcement Learning Project](#cartpole-reinforcement-learning-project)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Code Walkthrough](#code-walkthrough)
    - [Imports](#imports)
    - [Environment Setup](#environment-setup)
    - [Training \& Setting Up the Agent](#training--setting-up-the-agent)
    - [Running the Agent](#running-the-agent)
    - [Visualizing Steps Over Time](#visualizing-steps-over-time)
    - [Evaluating Agent Multiple Times](#evaluating-agent-multiple-times)

## Introduction
This project demonstrates the implementation of a reinforcement learning agent using the PPO (Proximal Policy Optimization) algorithm to solve the CartPole environment from OpenAI Gym. The objective is to keep a pole balanced on a moving cart for as long as possible.

## Prerequisites
To run this project, you will need:
- Python 3.7 or higher
- Libraries: `matplotlib`, `numpy`, `gymnasium`, `stable-baselines3`, `IPython`

## Installation
Install the required libraries using pip:

```bash
pip install matplotlib numpy gymnasium stable-baselines3 ipython
```

## Code Walkthrough

### Imports
The following libraries are imported for various functionalities:
```python

import matplotlib.pyplot as plt     # For plotting graphs
import numpy as np                  # For mathematical calculations
import gymnasium as gym             # For creating and managing environments
from stable_baselines3 import PPO    # For implementing the PPO algorithm
from IPython.display import clear_output    # For clearing output in visualizations
from gymnasium.wrappers import TimeLimit    # For setting a time limit on episodes
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv  # For enhanced model and enviroment creation
```

### Environment Setup

Setting up the CartPole environment with a time limit:

```python

env = gym.make('CartPole-v1', render_mode="rgb_array")  # Create the environment
env = TimeLimit(env, max_episode_steps=800)  # Set a time limit for episodes

# Wrapping the environment for vectorized training
env = DummyVecEnv([lambda: env])  
env = VecNormalize(env, norm_obs=True, norm_reward=False)  # Normalize observations
```

### Training & Setting Up the Agent

Creating and training the PPO agent:

```python
model = PPO("MlpPolicy", env, learning_rate=0.005, gamma=0.999, n_steps=2048, verbose=1, seed=42)
model.learn(total_timesteps=50000)  # Train the agent
```

### Running the Agent

Resetting the environment and running the trained agent over a few episodes:

```python
reset_output = env.reset()  # Reset environment
obs = reset_output[0]  # Get the initial observation

episode_rewards = []  # Store rewards for each episode
step_rewards = []     # Store step rewards

for _ in range(5):  # Run over 5 episodes
    obs = env.reset()  # Reset environment for a new episode
    total_reward = 0
    done = False

    while not done:  
        action = model.predict(obs)[0]  # Get action from the model
        obs, reward, done, info = env.step(action)  # Step in the environment
        total_reward += reward  # Accumulate rewards
        
        plt.imshow(env.render(mode="rgb_array"))  # Render the environment
        plt.axis('off')
        clear_output(wait=True)
        plt.show()
        step_rewards.append(reward)

    episode_rewards.append(total_reward[0])  # Store total reward for the episode
```

### Visualizing Steps Over Time

Calculating and plotting the cumulative rewards over the steps:

```python
mean_reward = np.mean(episode_rewards)  # Calculate mean reward
print(f"Mean Reward over 5 episodes: {mean_reward}") 
print(f"List of episode rewards: {episode_rewards}") 

plt.plot(np.cumsum(step_rewards), label='Total Reward')  # Cumulative sum of rewards
plt.title('Total Rewards Over Steps')
plt.xlabel('Step')
plt.ylabel('Cumulative Reward')
plt.legend()
plt.show()
```

### Evaluating Agent Multiple Times

Evaluating the agent's performance over multiple runs to check stability:

```python
def evaluate_agent(model, env, num_eval_episodes=10):
    total_rewards = []
    
    for _ in range(num_eval_episodes):
        obs = env.reset()  # Reset environment at start of each episode
        done = False
        episode_reward = 0
        
        while not done:  # Continue until episode ends
            action = model.predict(obs)[0]
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
        total_rewards.append(episode_reward)
    
    average_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    return average_reward, std_reward

# Run evaluations multiple times and store results
num_runs = 10 
all_avg_rewards = []
all_std_rewards = []

for i in range(num_runs):
    avg_reward, std_reward = evaluate_agent(model, env, num_eval_episodes=10)
    all_avg_rewards.append(avg_reward)
    all_std_rewards.append(std_reward)
    print(f"Run {i + 1}: Average reward: {avg_reward}, Standard deviation: {std_reward}")

# Overall average and variance across all runs
print(f"Overall Average Reward: {np.mean(all_avg_rewards)}")
print(f"Overall Standard Deviation in Reward: {np.mean(all_std_rewards)}")
```
