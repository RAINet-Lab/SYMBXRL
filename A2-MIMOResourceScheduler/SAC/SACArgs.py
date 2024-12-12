'''
The following code is part of "SymbXRL: Symbolic Explainable Deep Reinforcement Learning for Mobile Networks" 
[reference] Use and modified code from https://github.com/jimkon/Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces
[reference] Use and modified code from https://github.com/qinganrice/SMART
[reference] Use and modified code from https://github.com/renew-wireless/RENEWLab
[reference] Qing An, Chris Dick, Santiago Segarra, Ashutosh Sabharwal, Rahman Doost-Mohammady, ``A Deep Reinforcement Learning-Based Resource Scheduler for Massive MIMO Networks'', arXiv:2303.00958, 2023


DISCLAIMER: THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
'''

class SACArgs:
    def __init__(self, H, max_episode=30):
        # Default values for arguments
        self.policy = "Gaussian"  # Policy name (Gaussian, Deterministic)
        self.eval = False # False : Train the agent, True : Evaluate the agent
        self.gamma = 0.99 # Discount factor
        self.tau = 0.005 # Target smoothing coefficient (Critic to Critic target network)
        self.lr = 0.0003 # Learning rate for the critic network
        self.alpha_lr = 0.0003 # Learning rate for the actor network
        self.alpha = 0.2 # Entropy coefficient (0.0 = no entropy, 1.0 = maximum entropy)
        self.automatic_entropy_tuning = True
        self.seed = 1 # Random seed
        self.batch_size = 256 # Batch size for Replay Memory
        self.max_episode_steps = len(H) # Maximum number of steps for each episode
        self.max_episode = max_episode  # Maximum number of episodes
        self.hidden_size = 512 # Hidden size for the networks
        self.updates_per_step = 1 # Number of updates per step
        # self.save_per_epochs = 15 
        self.start_steps = 3000 # Number of steps for uniform-random action selection, before running real policy. Helps exploration 
        self.target_update_interval = 1 # Value update interval for the Critic target networks 
        self.replay_size = 1000000 # Size of the replay buffer
        self.cuda = 1 # Cuda ID to use
        self.gpu_nums = 1 # Number of GPUs to use