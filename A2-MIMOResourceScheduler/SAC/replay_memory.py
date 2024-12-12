'''
The following code is part of "SymbXRL: Symbolic Explainable Deep Reinforcement Learning for Mobile Networks" 
[reference] Use and modified code from https://github.com/jimkon/Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces
[reference] Use and modified code from https://github.com/qinganrice/SMART
[reference] Use and modified code from https://github.com/renew-wireless/RENEWLab
[reference] Qing An, Chris Dick, Santiago Segarra, Ashutosh Sabharwal, Rahman Doost-Mohammady, ``A Deep Reinforcement Learning-Based Resource Scheduler for Massive MIMO Networks'', arXiv:2303.00958, 2023


DISCLAIMER: THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
'''

import random
import os
import pickle
import numpy as np

class ReplayMemory:
    def __init__(self, capacity, seed):
        """
        Initialize the replay memory.

        Parameters:
        capacity (int): The maximum number of experiences the buffer can hold.
        seed (int): Random seed for reproducibility.
        """
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []  # Initialize the buffer to store experiences
        self.position = 0  # Initialize the position pointer

    def push(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.

        Parameters:
        state (object): The current state.
        action (object): The action taken.
        reward (float): The reward received.
        next_state (object): The next state.
        done (bool): Whether the episode is done.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)  # Expand the buffer if under capacity
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity  # Circular buffer

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.

        Parameters:
        batch_size (int): The number of experiences to sample.

        Returns:
        tuple: A tuple of numpy arrays (state, action, reward, next_state, done).
        """
        batch = random.sample(self.buffer, batch_size)  # Randomly sample a batch
        state, action, reward, next_state, done = map(np.stack, zip(*batch))  # Unzip and stack the batch
        return state, action, reward, next_state, done

    def __len__(self):
        """
        Return the current size of the buffer.
        """
        return len(self.buffer)

    def save_buffer(self, suffix="", save_path=None):
        """
        Save the buffer to a file.

        Parameters:
        suffix (str): Optional suffix for the filename.
        save_path (str): Optional save path. If None, a default path is used.
        """
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')  # Create checkpoints directory if it doesn't exist

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}".format(suffix)  # Default save path
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)  # Save the buffer to a file

    def load_buffer(self, save_path):
        """
        Load the buffer from a file.

        Parameters:
        save_path (str): The path to the file to load.
        """
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)  # Load the buffer from a file
            self.position = len(self.buffer) % self.capacity  # Set the position pointer

    def view_buffer(self):
        """
        Print the contents of the buffer for debugging.
        """
        for idx, experience in enumerate(self.buffer):
            if experience is not None:
                state, action, reward, next_state, done = experience
                print(f"Index {idx}:")
                print(f"  State: {state}")
                print(f"  Action: {action}")
                print(f"  Reward: {reward}")
                print(f"  Next State: {next_state}")
                print(f"  Done: {done}")
                print("-" * 20)
