'''
The following code is part of "SymbXRL: Symbolic Explainable Deep Reinforcement Learning for Mobile Networks"
Copyright - RESILIENT AI NETWORK LAB, IMDEA NETWORKS

DISCLAIMER: THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
'''

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQNNetwork(nn.Module):
    """
    A Deep Q-Network (DQN) architecture for estimating state-action values.
    This network takes an input state and outputs Q-values for all possible actions.
    It uses a fully-connected (FC) architecture with ReLU activations between layers.
    Attributes:
        fc1 (nn.Linear): First fully-connected layer.
        fc2 (nn.Linear): Second fully-connected layer.
        fc3 (nn.Linear): Third fully-connected layer.
        fc4 (nn.Linear): Final fully-connected layer that outputs Q-values.
    Args:
        input_dims (int): Dimensionality of the input state.
        n_actions (int): Number of available actions.
        fc1_dims (int, optional): Number of units in the first FC layer. Defaults to 1024.
        fc2_dims (int, optional): Number of units in the second FC layer. Defaults to 512.
        fc3_dims (int, optional): Number of units in the third FC layer. Defaults to 256.
    """
    def __init__(self, input_dims, n_actions, fc1_dims=1024, fc2_dims=512, fc3_dims=256):
        super(DQNNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        self.fc4 = nn.Linear(fc3_dims, n_actions)

    def forward(self, state):
        """
        Forward pass of the DQN network.
        Args:
            state (torch.Tensor): Input state tensor.
        Returns:
            torch.Tensor: Q-values for all possible actions.
        """

        x = torch.relu(self.fc1(state))  # Apply ReLU activation to fc1 output
        x = torch.relu(self.fc2(x))      # Apply ReLU activation to fc2 output
        x = torch.relu(self.fc3(x))      # Apply ReLU activation to fc3 output
        q_values = self.fc4(x)             # No activation for final layer (Q-values)
        return q_values
    

class ReplayBuffer:
    """
    A replay buffer for storing transitions experienced by an agent during training.
    This buffer samples random batches of transitions for training the DQN network.
    It ensures efficient memory usage by overwriting experiences once the buffer is full.
    Attributes:
        mem_size (int): Maximum capacity of the replay buffer.
        mem_cntr (int): Current counter for storing transitions (index in the buffer).
        state_memory (np.ndarray): Buffer to store states.
        new_state_memory (np.ndarray): Buffer to store new states after taking an action.
        action_memory (np.ndarray): Buffer to store actions taken.
        reward_memory (np.ndarray): Buffer to store rewards received.
        terminal_memory (np.ndarray): Buffer to store done flags (0 for not done, 1 for done).
        device (str): Device to store tensors on (e.g., "cpu" or "cuda").
    """

    def __init__(self, max_size, input_shape, device):
        """
        Initializes the replay buffer.
        Args:
            max_size (int): Maximum capacity of the replay buffer.
            input_shape (tuple): Shape of the input state.
            device (str): Device to store tensors on (e.g., "cpu" or "cuda").
        """
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.device = device

    def store_transition(self, state, action, reward, state_, done):
        """
        Stores a transition (experience) in the replay buffer.
        Args:
            state (np.ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            state_ (np.ndarray): New state after taking the action.
            done (bool): Flag indicating if the episode ended after taking the action.
        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1.0 - done  # Convert done flag (0 for not done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        """
        Samples a random batch of transitions from the replay buffer.
        Args:
            batch_size (int): Number of transitions to sample.
        Returns:
            tuple: A tuple containing the sampled states, actions, rewards,
                   next states, and terminal flags as PyTorch tensors on the specified device.
        """

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False if max_mem >= batch_size else True)

        states = torch.tensor(self.state_memory[batch], dtype=torch.float32).to(self.device)
        actions = torch.tensor(self.action_memory[batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor(self.reward_memory[batch], dtype=torch.float32).to(self.device)
        states_ = torch.tensor(self.new_state_memory[batch], dtype=torch.float32).to(self.device)
        terminal = torch.tensor(self.terminal_memory[batch], dtype=torch.float32).to(self.device)

        return states, actions, rewards, states_, terminal
    
class CustomLoss(nn.Module):
    '''
    Loss Function for DQN. It is a custom loss function that calculates the loss between the Q-values of the network and the target Q-values.
    '''
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, q_selected, target_q_values):
        mse_loss = nn.MSELoss()(q_selected, target_q_values)
        total_loss = mse_loss        
        return total_loss
    
class DQNAgent:
    def __init__(self, input_dims, n_actions, alpha=0.0001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=0.001, mem_size=100000, batch_size=20, device=torch.device("cpu")):
        """
        Initialize the DQN Agent.
        Args:
            input_dims (tuple): Dimensions of the input state.
            n_actions (int): Number of possible actions.
            alpha (float, optional): Learning rate. Defaults to 0.0001.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            epsilon_start (float, optional): Initial exploration rate. Defaults to 1.0.
            epsilon_end (float, optional): Minimum exploration rate. Defaults to 0.01.
            epsilon_decay (float, optional): Rate of exploration decay. Defaults to 0.001.
            mem_size (int, optional): Size of the replay memory. Defaults to 100000.
            batch_size (int, optional): Batch size for training. Defaults to 20.
            device (torch.device, optional): Device for computation. Defaults to torch.device("cpu").
        """
        self.gamma = gamma
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        self.q_network = DQNNetwork(input_dims, n_actions).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.memory = ReplayBuffer(mem_size, input_dims, device)
        self.loss_function = CustomLoss()
        self.device = device

    def choose_action(self, observation):
        """
        Choose an action based on the current observation.
        Args:
            observation (array_like): Current observation/state.
        Returns:
            int: Selected action.
        """
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = torch.tensor(np.array([observation])).float().to(self.device) 
            q_values = self.q_network(state)
            action = torch.argmax(q_values).item()
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > \
                self.epsilon_end else self.epsilon_end
        return action

    def remember(self, state, action, reward, new_state, done):
        """
        Store a transition in the replay memory.
        Args:
            state (array_like): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            new_state (array_like): Next state.
            done (bool): Whether the episode is finished.
        """
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_model(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load_model(self, path):
        self.q_network.load_state_dict(torch.load(path))
        self.q_network.eval()

    def encode_to_multibinary(self, integer):
        """
        Encode an integer into a multibinary representation.
        Args:
            integer (int): Integer to encode.
        Returns:
            list: List representing the multibinary encoding.
        """
        binary_string = bin(integer)[2:]
        binary_string_padded = binary_string.zfill(7)
        binary_array = [int(bit) for bit in binary_string_padded]
        return binary_array

    def learn(self, state, action, reward, next_state, done, replay_buffer=False):
        """
        Update the Q-network based on a transition or a batch of transitions.
        Args:
            state (array_like or tensor): Current state or batch of states.
            action (int or tensor): Action taken or batch of actions.
            reward (float or tensor): Reward received or batch of rewards.
            next_state (array_like or tensor): Next state or batch of next states.
            done (bool or tensor): Whether the episode is finished or batch of done flags.
            replay_buffer (bool, optional): Whether to use replay buffer. Defaults to False.

        Returns:
            float: Loss value of the training step.
        """
        if replay_buffer == False:
            state = torch.tensor([state], dtype=torch.float32).to(self.device)  
            next_state = torch.tensor([next_state], dtype=torch.float32).to(self.device)
            reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
            self.optimizer.zero_grad()
            q_values = self.q_network(state)
            next_q_values = self.q_network(next_state)
            target_q_values = reward + self.gamma * torch.max(next_q_values) * (1 - done)
            action_mask = torch.zeros(self.n_actions).to(self.device)
            action_mask[action] = 1
            q_selected = torch.sum(q_values * action_mask)
            total_loss = self.loss_function(q_selected, target_q_values)
            total_loss.backward()
            self.optimizer.step()
        else:
            states, actions, rewards, next_states, terminals = self.memory.sample_buffer(self.batch_size)
            self.optimizer.zero_grad()
            q_values = self.q_network(states)
            next_q_values = self.q_network(next_states)
            target_q_values = rewards + self.gamma * torch.max(next_q_values, dim=1)[0] * (1 - terminals)
            target_q_values = target_q_values.unsqueeze(1)
            action_mask = torch.zeros((self.batch_size, self.n_actions)).to(self.device)
            action_mask.scatter_(1, actions.unsqueeze(1), 1)
            q_selected = torch.sum(q_values * action_mask, dim=1)
            q_selected = q_selected.unsqueeze(1)
            total_loss = self.loss_function(q_selected, target_q_values)
            total_loss.backward()
            self.optimizer.step()

        return total_loss.item()