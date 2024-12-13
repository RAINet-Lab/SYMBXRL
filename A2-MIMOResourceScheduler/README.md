COPYRIGHT: RESILIENT AI NETWORKS LAB, IMDEA NETWORKS INSTITUTE, SPAIN 

# A2 : MIMO RESOURCE SCHEDULER

Includes the following:
1. The MIMO Resource Scheduler Environment and DRL Agents A2 trained on this environment.
2. Action Steering Implementation Notebook and Codes for Agent A2.
3. Plots shown in the paper.

## Environment Overview

The Environment (MIMO Resource Scheduler) is an OpenAI Gym-based simulator designed for training Reinforcement Learning (RL) agents to optimize user scheduling in a Massive MIMO environment. It is compatible with the **Stable Baselines3** library, enabling a wide range of RL algorithms to be used for training. This simulator was designed and used for research experiments for our paper published in **IEEE Infocom 2025** - SYMBXRL: Symbolic Explainable Deep Reinforcement Learning for Mobile Networks.

## Usage and Acknowledgements

For using this environment, please refer to the LICENSE and acknowledge our publication and the reference Publication [1], from where the datasets have been sourced and processed.

## Dataset Source

- Dataset: [Indoor Mobility Channel Measurement Dataset, RENEW Labs, Rice University](https://renew-wireless.org/dataset-indoor-channel.html)
- GitHub: [RENEW Labs GitHub](https://github.com/RENEW-Wireless/RENEWLab)
- Reference Publication [1]:  
  Qing An, Santiago Segarra, Chris Dick, Ashutosh Sabharwal, Rahman Doost-Mohammady,  
  "A Deep Reinforcement Learning-Based Resource Scheduler for Massive MIMO Networks,"  
  IEEE Transactions on Machine Learning in Communications and Networking, 2023.  
  DOI: 10.1109/TMLCN.2023.3313988


## Installation and Setup

1. Clone the repository:
   ```bash
   git clone <repository-link>
   cd A2
2. Create and activate a virtual environment:
    ```bash
    python3 -m venv env
    source env/bin/activate
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
## Agent Implementations

### 1. Soft Actor-Critic (SAC) trained Agent (Based on [1])
- Navigate to the `SAC/` directory.
- Run `SAC_test.py`. 

### 2. Deep Q-Network (DQN) trained Agent
- Navigate to the `DQN/` directory.
- Run `DQN_test.py`. 

#### Note: Modify dataset paths in the code if necessary. The default dataset is LOS_highspeed2.

## Example Saved Models
- Pre-trained models for SAC and DQN agents are available in the ./models folder for reference.

## IAS Implementation

### 1. IAS Implementation Live on SAC Agent  
- Navigate to the `SAC/` directory.
- Run the Notebook `1.SAC_Agent_action_steering.ipynb`. 

## Repository Structure

```
A2/
│
├── custom_mimo_env.py                 # Root file for the MIMO environment simulator
├── Datasets/                          # Contains training and testing datasets
│   ├── LOS_lowspeed_64_7.hdf5
│   ├── LOS_highspeed1_64_7.hdf5
│   ├── LOS_highspeed2_64_7.hdf5
│   ├── NLOS_lowspeed_64_7.hdf5
│   ├── NLOS_highspeed1_64_7.hdf5
│   ├── NLOS_highspeed2_64_7.hdf5
│   ├── LOS_test_64_7.hdf5             # Test datasets
│   └── NLOS_test_64_7.hdf5
│
├── DQN/                               # DQN Agent implementation
│   ├── DQN_main.py
│   └── DQN_Agent.py
│
├── SAC/                               # SAC Agent implementation [1]
│   ├── SAC_main.py
│   ├── action_space.py
│   ├── model.py
│   ├── replay_memory.py
│   ├── sac.py
│   ├── SACArgs.py
│   ├── Action_Steering/               # Utility files for IAS Algorithm
│   ├── 1. SAC_Agent_action_steering.ipynb   # Notebook to run the IAS Algorithm
│   └── 2. paper_plots.ipynb                 # Notebook for the plots used in the paper
│
└── README.md
```


## TESTED ENVIRONMENT
- Linux: Ubuntu 20.04 / 22.04 LTS
- GPU: NVIDIA A100, 40 GB
- Python: 3.10.14

### Contact
For queries or issues, please reach out to RESILIENT AI NETWORK GROUP






