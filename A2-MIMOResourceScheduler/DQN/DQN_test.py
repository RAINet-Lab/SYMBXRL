'''
The following code is part of "SymbXRL: Symbolic Explainable Deep Reinforcement Learning for Mobile Networks"
Copyright - RESILIENT AI NETWORK LAB, IMDEA NETWORKS

DISCLAIMER: THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
'''

import sys
sys.path.insert(0, '../')
import numpy as np
import gymnasium as gym
from DQNAgent import *
import h5py
import time
import torch
from custom_mimo_env import MimoEnv
import matplotlib.pyplot as plt

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
H_file = h5py.File('/home/abhishek/data/A2/Datasets/LOS_highspeed2_64_7.hdf5', 'r')
H = np.array(H_file.get('H'))
se_max_ur = np.array(H_file.get('se_max'))

# Initialize environment and agent
env = MimoEnv(H, se_max_ur)
print('Environment initialized')
agent = DQNAgent(alpha=0.0003, input_dims=21, n_actions=127, batch_size=256, device=device)

# Load pre-trained model
agent.load_model('/home/abhishek/data/A2/models/DQN_956.59_300_dtLOS_HS2_final.pth')

# Evaluate the model
observation, info = env.reset()
done = False
score = 0
step_rewards = []
mean_rew = []

# Testing loop
while not done:
    action = agent.choose_action(np.squeeze(observation))
    next_obs, reward, done, _, info = env.step(action)
    score += reward
    step_rewards.append(reward)

    mean_reward = np.mean(step_rewards)
    mean_rew.append(mean_reward)
    
    # Printing episode information
    test_print = f'Step: {info["current_step"]} | Step Reward: {reward} | Mean Reward: {mean_reward:.3f} | Score: {score:.3f}\n'
    print(test_print)
    
    observation = next_obs

# Visualization
plt.figure(figsize=(10, 5))
plt.plot(step_rewards, label='Step Reward')
plt.plot(mean_rew, label='Mean Reward')
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.title('DQN Agent Performance')
plt.legend()
plt.show()
