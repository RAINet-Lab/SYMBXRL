'''
The following code is part of "SymbXRL: Symbolic Explainable Deep Reinforcement Learning for Mobile Networks"
Copyright - RESILIENT AI NETWORK LAB, IMDEA NETWORKS

DISCLAIMER: THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
'''

import sys
import os
# Get the directory two levels up from the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, project_root)
from constants import PROJ_ADDR
import numpy as np
import gymnasium as gym
from DQNAgent import *
import h5py
import time
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)  # goes up one level to A2-MIMOResourceScheduler
sys.path.insert(0, parent_dir)
from custom_mimo_env import MimoEnv
import torch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
H_file = h5py.File(f'{PROJ_ADDR}/A2-MIMOResourceScheduler/Datasets/LOS_lowspeed_64_7.hdf5', 'r')
H = np.array(H_file.get('H'))
se_max_ur = np.array(H_file.get('se_max'))

# Split data into training and testing sets
train_ratio = 0.8
num_samples = H.shape[0]
num_train = int(train_ratio * num_samples)
num_test = num_samples - num_train

H_train = H[:num_train]
H_test = H[num_train:]
se_max_train = se_max_ur[:num_train]
se_max_test = se_max_ur[num_train:]

print(f"Training samples: {num_train}, Testing samples: {num_test}")

# Initialize environment and agent
env = MimoEnv(H_train, se_max_train)
print('Environment initialized')
agent = DQNAgent(alpha=0.0003, input_dims=21, n_actions=127, batch_size=256, device=device)

# Get number of epochs from user input
try:
    epochs = int(input("Enter the number of epochs to train (default is 300): "))
except ValueError:
    epochs = 300  # Default to 300 if input is not valid
print(f"Training for {epochs} epochs")

beta = 1  # Weight for replay buffer

# Lists to store training metrics
step_rewards = []
mean_rew = []
losses = []
best_score = 0 
best_model_path = None  

start_time = time.time()  # Start timer

# Training loop
for epoch in range(epochs):
    observation, info = env.reset()
    done = False
    score = np.zeros(epochs)
    
    # Episode loop
    while not done:
        action = agent.choose_action(np.squeeze(observation))
        next_obs, reward, done, _, info = env.step(action)
        agent.remember(np.squeeze(observation), action, reward, np.squeeze(next_obs), done)
        score[epoch] += reward
        step_rewards.append(reward)

        # Learning from experiences
        replay_buffer_flag = np.random.choice([True, False], p=[beta, 1 - beta])
        if replay_buffer_flag:
            total_loss = agent.learn(observation, action, reward, next_obs, done, replay_buffer=True)
        else:
            total_loss = agent.learn(observation, action, reward, next_obs, done, replay_buffer=False)

        losses.append(total_loss)
        mean_reward = np.mean(step_rewards)
        mean_rew.append(mean_reward)
        
        # Printing episode information
        log_print = f'Episode: {epoch+1} | Step: {info["current_step"]} | Step Reward: {reward} | Mean Reward: {mean_reward:.3f} | Score: {score[epoch]:.3f}\n'
        print(log_print)
        
        observation = next_obs

end_time = time.time()
training_time = end_time - start_time

# Calculate training time in hours, minutes, and seconds
hours = int(training_time) // 3600
minutes = int(training_time) % 3600 // 60
seconds = int(training_time) % 60

print(f"Training time: {hours} hours, {minutes} minutes, {seconds} seconds")

# Save the final model
final_model_path = f"models/DQN_{score[epoch]:.2f}_{epoch + 1}_dtLOS_HS2_final.pth"
agent.save_model(final_model_path)
print("Final model saved successfully.")

# Evaluate the model
print("############################################################### EVALUATION STARTS ############################################################################################################")
print("Evaluation started...")

# Initialize environment
env = MimoEnv(H_test, se_max_test)
print('Environment initialized')

# Evaluate the model
observation, info = env.reset()
done = False
score = 0

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
