'''
The following code is part of "SymbXRL: Symbolic Explainable Deep Reinforcement Learning for Mobile Networks" 
[reference] Use and modified code from https://github.com/jimkon/Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces
[reference] Use and modified code from https://github.com/qinganrice/SMART
[reference] Use and modified code from https://github.com/renew-wireless/RENEWLab
[reference] Qing An, Chris Dick, Santiago Segarra, Ashutosh Sabharwal, Rahman Doost-Mohammady, ``A Deep Reinforcement Learning-Based Resource Scheduler for Massive MIMO Networks'', arXiv:2303.00958, 2023


DISCLAIMER: THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
'''

import os
import torch
import torch.nn.functional as F
import action_space
from smartfunc import soft_update, hard_update
import numpy as np
import time
import torch.nn as nn
from torch.optim import Adam
from model import GaussianPolicy, QNetwork, DeterministicPolicy


class SAC(object):
    def __init__(self, num_inputs, num_actions, max_actions, args, lr, alpha_lr):
        """
        Initialize the Soft Actor-Critic (SAC) agent.

        Parameters:
        num_inputs (int): Number of input features.
        num_actions (int): Number of actions.
        max_actions (int): Maximum number of actions.
        args (argparse.Namespace): Arguments containing hyperparameters and configurations.
        lr (float): Learning rate for the critic networks.
        alpha_lr (float): Learning rate for the alpha optimizer.
        """
        self.gamma = args.gamma  # Discount factor
        self.tau = args.tau  # Soft update parameter
        self.alpha = args.alpha  # Entropy regularization coefficient
        self.action_space = action_space.Discrete_space(max_actions)  # Action space object
        self.k_nearest_neighbors = 1  # Number of nearest neighbors for action selection
        self.eval = args.eval  # Evaluation mode flag

        self.gpu_ids = [0] if args.cuda and args.gpu_nums > 1 else [-1]  # GPU configuration

        self.num_actions = num_actions
        self.num_inputs = num_inputs
        self.max_actions = max_actions

        self.policy_type = args.policy  # Type of policy (Gaussian or Deterministic)
        self.target_update_interval = args.target_update_interval  # Interval for updating target network
        self.automatic_entropy_tuning = args.automatic_entropy_tuning  # Flag for automatic entropy tuning

        self.device = torch.device("cuda:0" if args.cuda else "cpu")  # Device configuration

        # Initialize critic networks
        if len(self.gpu_ids) == 1:
            self.critic = QNetwork(num_inputs, num_actions, args.hidden_size).to(self.device)
        if len(self.gpu_ids) > 1:
            self.critic = QNetwork(num_inputs, num_actions, args.hidden_size)
            self.critic = nn.DataParallel(self.critic, device_ids=self.gpu_ids).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        # Initialize target critic network
        if len(self.gpu_ids) == 1:
            self.critic_target = QNetwork(num_inputs, num_actions, args.hidden_size).to(self.device)
        if len(self.gpu_ids) > 1:
            self.critic_target = QNetwork(num_inputs, num_actions, args.hidden_size)
            self.critic_target = nn.DataParallel(self.critic_target, device_ids=self.gpu_ids).to(self.device)
        hard_update(self.critic_target, self.critic)  # Hard update of target network

        # Initialize policy network
        if self.policy_type == "Gaussian":
            # Target entropy based on the action space dimension
            if self.automatic_entropy_tuning:
                self.target_entropy = -torch.prod(torch.Tensor(num_actions).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=alpha_lr)

            if len(self.gpu_ids) == 1:
                self.policy = GaussianPolicy(num_inputs, num_actions, args.hidden_size).to(self.device)
            if len(self.gpu_ids) > 1:
                self.policy = GaussianPolicy(num_inputs, num_actions, args.hidden_size)
                self.policy = nn.DataParallel(self.policy, device_ids=self.gpu_ids).to(self.device)
            
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            if len(self.gpu_ids) == 1:
                self.policy = DeterministicPolicy(num_inputs, num_actions, args.hidden_size).to(self.device)
            if len(self.gpu_ids) > 1:
                self.policy = DeterministicPolicy(num_inputs, num_actions, args.hidden_size)
                self.policy = nn.DataParallel(self.policy, device_ids=self.gpu_ids).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)

    def knn_action(self, s_t, proto_action):
        """
        Find the k-nearest neighbors of the given proto_action and select the best action based on the critic evaluation.

        Parameters:
        s_t (torch.Tensor): The current state.
        proto_action (np.ndarray): The prototype action to search for nearest neighbors.

        Returns:
        tuple: The best raw action and the best action after evaluation.
        """
        # Get the k-nearest neighbors of the prototype action
        raw_actions, actions = self.action_space.search_point(proto_action, self.k_nearest_neighbors)

        if not isinstance(s_t, np.ndarray):
            s_t = s_t.detach().cpu().numpy()
        # Expand state for each action
        s_t = np.tile(s_t, [raw_actions.shape[1], 1])

        # Reshape state and actions for the critic evaluation
        s_t = s_t.reshape(len(raw_actions), raw_actions.shape[1], s_t.shape[1]) if self.k_nearest_neighbors > 1 \
            else s_t.reshape(raw_actions.shape[0], s_t.shape[1])
        raw_actions = torch.FloatTensor(raw_actions).to(self.device)
        s_t = torch.FloatTensor(s_t).to(self.device)

        # Evaluate each action through the critic network
        actions_evaluation, _ = self.critic(s_t, raw_actions)
        # Find the index of the action with the maximum value
        max_index = np.argmax(actions_evaluation.detach().cpu().numpy(), axis=1)
        max_index = max_index.reshape(len(max_index),)

        raw_actions = raw_actions.detach().cpu().numpy()
        # Return the best action
        if self.k_nearest_neighbors > 1:
            return raw_actions[[i for i in range(len(raw_actions))], max_index, [0]].reshape(len(raw_actions), 1), \
                   actions[[i for i in range(len(actions))], max_index, [0]].reshape(len(actions), 1)
        else:
            return raw_actions[max_index], actions[max_index]

    def random_action(self):
        """
        Sample a random action from the action space.

        Returns:
        tuple: The raw action and the selected action.
        """
        proto_action = np.random.uniform(-1., 1., self.num_actions)  # Random prototype action
        raw_action, action = self.action_space.search_point(proto_action, 1)
        raw_action = raw_action[0]
        action = action[0]
        assert isinstance(raw_action, np.ndarray)
        return raw_action, action[0]

    def select_action(self, state):
        """
        Select an action based on the current state using the policy network.

        Parameters:
        state (np.ndarray): The current state.

        Returns:
        tuple: The raw action and the selected action.
        """
        state = torch.FloatTensor(state).to(self.device)
        if not self.eval:
            if len(self.gpu_ids) == 1:
                proto_action, _, _ = self.policy.sample(state)
            if len(self.gpu_ids) > 1:
                proto_action, _, _ = self.policy.module.sample(state)
        else:
            if len(self.gpu_ids) == 1:
                _, _, proto_action = self.policy.sample(state)
            if len(self.gpu_ids) > 1:
                _, _, proto_action = self.policy.module.sample(state)
        
        proto_action = proto_action.detach().cpu().numpy()[0].astype('float64')

        raw_action, action = self.knn_action(state, proto_action)
        assert isinstance(raw_action, np.ndarray)
        action = action[0]
        raw_action = raw_action[0]
        return raw_action, action[0]

    def update_parameters(self, memory, batch_size, updates):
        """
        Update the parameters of the critic and policy networks based on a batch of experiences.

        Parameters:
        memory (ReplayMemory): The replay memory containing experiences.
        batch_size (int): The size of the batch to sample from memory.
        updates (int): The number of updates that have been performed.

        Returns:
        tuple: Losses for Q-functions, policy, and alpha, as well as alpha value and log_pi mean.
        """
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = np.reshape(action_batch, (batch_size, 1, -1))
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = np.reshape(reward_batch, (batch_size, 1, -1))
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        mask_batch = np.reshape(mask_batch, (batch_size, 1, -1))
        mask_batch = torch.FloatTensor(mask_batch).to(self.device)

        with torch.no_grad():
            # Compute target values
            if len(self.gpu_ids) == 1:
                next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            if len(self.gpu_ids) > 1:
                next_state_action, next_state_log_pi, _ = self.policy.module.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        # Compute Q-function loss
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # Compute policy loss
        if len(self.gpu_ids) == 1:
            pi, log_pi, _ = self.policy.sample(state_batch)
        if len(self.gpu_ids) > 1:
            pi, log_pi, _ = self.policy.module.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Update alpha if automatic entropy tuning is enabled
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        # Soft update target network
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), log_pi.mean()

    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        """
        Save the model parameters to a checkpoint file.

        Parameters:
        env_name (str): Name of the environment.
        suffix (str): Optional suffix for the checkpoint filename.
        ckpt_path (str): Optional path to save the checkpoint file.
        """
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    def load_checkpoint(self, ckpt_path):
        """
        Load model parameters from a checkpoint file.

        Parameters:
        ckpt_path (str): Path to the checkpoint file.
        """
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            # Set the model to evaluation or training mode
            if self.eval:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()


