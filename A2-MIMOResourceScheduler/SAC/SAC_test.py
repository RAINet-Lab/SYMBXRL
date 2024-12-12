
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
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
# Load data

H_file = h5py.File('/home/abhishek/data/A2/Datasets/LOS_highspeed2_64_7.hdf5','r')
H = np.array(H_file.get('H'))
se_max_ur = np.array(H_file.get('se_max'))
print('Data loaded successfully')

max_episode = None  # Default to 300 if input is not valid
args = SACArgs(H, max_episode=max_episode)

# Set random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Initialize environment
env = MimoEnv(H, se_max_ur)
print('Environment initialized')

# Get environment parameters
num_states = env.observation_space.shape[0]
num_actions = len([env.action_space.sample()])
max_actions = env.action_space.n

# Initialize SAC agent
agent = SAC(num_states, num_actions, max_actions, args, args.lr, args.alpha_lr)
memory = ReplayMemory(args.replay_size, args.seed)

# Load the model
agent.load_checkpoint('/home/abhishek/data/A2/models/SACG_884.53_551_dtLOS_HS2_checkpointed.pth_')
print('SAC build finished')


#Evaluate the model
print("###############################################################EVALUATION STARTS ############################################################################################################")
print("Evaluation started...")

step_rewards = []
acn_str = []
grp_str = []
mean_rew = []

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