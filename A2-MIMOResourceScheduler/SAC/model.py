'''
The following code is part of "SymbXRL: Symbolic Explainable Deep Reinforcement Learning for Mobile Networks" 
[reference] Use and modified code from https://github.com/jimkon/Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces
[reference] Use and modified code from https://github.com/qinganrice/SMART
[reference] Use and modified code from https://github.com/renew-wireless/RENEWLab
[reference] Qing An, Chris Dick, Santiago Segarra, Ashutosh Sabharwal, Rahman Doost-Mohammady, ``A Deep Reinforcement Learning-Based Resource Scheduler for Massive MIMO Networks'', arXiv:2303.00958, 2023


DISCLAIMER: THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2  # Maximum value for log standard deviation
LOG_SIG_MIN = -20  # Minimum value for log standard deviation
epsilon = 1e-6  # Small value to prevent division by zero

# Initialize weights of the policy networks
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        """
        Initialize the Value Network for estimating state values.

        Parameters:
        num_inputs (int): Number of input features.
        hidden_dim (int): Number of hidden units in each layer.
        """
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        """
        Forward pass through the Value Network.

        Parameters:
        state (torch.Tensor): The input state.

        Returns:
        torch.Tensor: Estimated state value.
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        """
        Initialize the Q Network for estimating Q-values for action-value function.

        Parameters:
        num_inputs (int): Number of input features.
        num_actions (int): Number of actions.
        hidden_dim (int): Number of hidden units in each layer.
        """
        super(QNetwork, self).__init__()

        # Architecture for Q1
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Architecture for Q2
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        """
        Forward pass through the Q Network.

        Parameters:
        state (torch.Tensor): The input state.
        action (torch.Tensor): The input action.

        Returns:
        tuple: Q-values for the first and second Q-function.
        """
        xu = torch.cat([state, action], dim=-1)  # Concatenate state and action

        # Q1 network
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        # Q2 network
        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        """
        Initialize the Gaussian Policy for continuous action spaces.

        Parameters:
        num_inputs (int): Number of input features.
        num_actions (int): Number of actions.
        hidden_dim (int): Number of hidden units in each layer.
        action_space (gym.Space, optional): Action space to scale actions.
        """
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # Action rescaling based on the action space
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        """
        Forward pass through the Gaussian Policy Network.

        Parameters:
        state (torch.Tensor): The input state.

        Returns:
        tuple: Mean and log standard deviation of the action distribution.
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)  # Clamping log_std
        return mean, log_std

    def sample(self, state):
        """
        Sample an action from the Gaussian Policy.

        Parameters:
        state (torch.Tensor): The input state.

        Returns:
        tuple: Sampled action, log probability of the action, and mean of the action distribution.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)  # Gaussian distribution
        x_t = normal.rsample()  # Sample with reparameterization trick
        y_t = torch.tanh(x_t)  # Apply Tanh function to bound the action
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Adjust log probability for the action transformation
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        """
        Move the Gaussian Policy to a specified device.

        Parameters:
        device (torch.device): The target device.

        Returns:
        nn.Module: The policy network on the specified device.
        """
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        """
        Initialize the Deterministic Policy for continuous action spaces.

        Parameters:
        num_inputs (int): Number of input features.
        num_actions (int): Number of actions.
        hidden_dim (int): Number of hidden units in each layer.
        action_space (gym.Space, optional): Action space to scale actions.
        """
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)  # Noise for exploration

        self.apply(weights_init_)

        # Action rescaling based on the action space
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        """
        Forward pass through the Deterministic Policy Network.

        Parameters:
        state (torch.Tensor): The input state.

        Returns:
        torch.Tensor: The deterministic action.
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        """
        Sample an action from the Deterministic Policy.

        Parameters:
        state (torch.Tensor): The input state.

        Returns:
        tuple: Sampled action, log probability (zero in this case), and the mean action.
        """
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)  # Clamping noise to a small range
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        """
        Move the Deterministic Policy to a specified device.

        Parameters:
        device (torch.device): The target device.

        Returns:
        nn.Module: The policy network on the specified device.
        """
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
