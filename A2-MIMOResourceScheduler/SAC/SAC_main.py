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
import h5py
from SACArgs import SACArgs
from sac import SAC
from replay_memory import ReplayMemory
from smartfunc import sel_ue
import torch
import time
from custom_mimo_env import MimoEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load data
H_file = h5py.File('/home/abhishek/data/A2/Datasets/LOS_highspeed2_64_7.hdf5','r')
H = np.array(H_file.get('H'))
se_max_ur = np.array(H_file.get('se_max'))
print('Data loaded successfully')

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

# Define SAC arguments
try:
    max_episode = int(input("Enter the number of epochs to train (default is 300): "))
except ValueError:
    max_episode = 300  # Default to 300 if input is not valid
print(f"Training for {max_episode} epochs")
args = SACArgs(H_train, max_episode=max_episode)

# Set random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Initialize environment
env = MimoEnv(H_train, se_max_train)
print('Environment initialized')

# Get environment parameters
num_states = env.observation_space.shape[0]
num_actions = len([env.action_space.sample()])
max_actions = env.action_space.n

# Initialize SAC agent
agent = SAC(num_states, num_actions, max_actions, args, args.lr, args.alpha_lr)
memory = ReplayMemory(args.replay_size, args.seed)
print('SAC build finished')

print("###############################################################TRAINING STARTS ############################################################################################################") 

updates = 0
random = 1
epsilon = 20000
step_rewards = []
acn_str = []
grp_str = []
mean_rew = []
best_score = 0 
best_model_path = None 

start_time = time.time()  # Start timer
print("Training started...")
# Training loop
for i_episode in range(args.max_episode):
    observation, info = env.reset()
    done = False
    grp_str.append(observation[14:])
    score = np.zeros(args.max_episode_steps)
    
    # Episode loop
    while not done:
        if random > np.random.rand(1):
            action, final_action = agent.random_action()
        else:
            action, final_action = agent.select_action(observation)
        random -= 1/epsilon
        ue_select, idx = sel_ue(final_action[0])
        acn_str.append(ue_select)
        next_obs, reward, done, _, info = env.step(final_action[0])
        grp_str.append(next_obs[14:])

        # Update SAC parameters
        if len(memory) > args.batch_size:
            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, log_pi_mean = agent.update_parameters(memory, args.batch_size, updates)
            updates += 1

        # Update scores and rewards
        score[i_episode] += reward
        step_rewards.append(reward)
        mask = 1 if info['current_step'] >= args.max_episode_steps - 1 else float(not done)
        memory.push(observation, action, reward, next_obs, mask) 
        mean_reward = np.mean(step_rewards)
        mean_rew.append(mean_reward)
        log_print = f'Episode: {i_episode+1} | Step: {info["current_step"]} |Action taken: {ue_select} | Step Reward: {reward} | Mean Reward: {mean_reward:.3f} | Score: {score[i_episode]:.3f}\n'
        print(log_print)        
        observation = next_obs
     


end_time = time.time()
training_time = end_time - start_time

# Calculate training time in hours, minutes, and seconds
hours = int(training_time) // 3600
minutes = int(training_time) % 3600 // 60
seconds = int(training_time) % 60

print(f"Training time: {hours} hours, {minutes} minutes, {seconds} seconds")

# # Save the final model
# final_model_path = f"models/SACG_{score[i_episode]:.2f}_{i_episode + 1}_dtLOS_HS2_final.pth"
# agent.save_checkpoint(final_model_path)
# print("Final model saved successfully.")

#Evaluate the model
print("###############################################################EVALUATION STARTS ############################################################################################################")
print("Evaluation started...")

# Initialize environment
env = MimoEnv(H_test, se_max_test)
print('Environment initialized')

observation, info = env.reset()
done = False
score = 0
# Episode loop
while not done:
    action, final_action = agent.select_action(observation)
    ue_select, idx = sel_ue(final_action[0])
    next_obs, reward, done, _, info = env.step(final_action[0])

    # Update scores and rewards
    score += reward
    step_rewards.append(reward)
    mean_reward = np.mean(step_rewards)
    mean_rew.append(mean_reward)
    test_print = f'Step: {info["current_step"]} |Action taken: {ue_select} | Step Reward: {reward} | Mean Reward: {mean_reward:.3f} | Score: {score:.3f}\n'
    print(test_print)        
    observation = next_obs
