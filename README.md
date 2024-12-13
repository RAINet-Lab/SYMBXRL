# SYMBXRL: Symbolic Explainable Deep Reinforcement Learning for Mobile Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and resources for the research paper:

> **SYMBXRL: Symbolic Explainable Deep Reinforcement Learning for Mobile Networks**
>
> *Anonymous Authors*

**Abstract:**

This paper introduces SYMBXRL, a novel technique for Explainable Reinforcement Learning (XRL) that synthesizes human-interpretable explanations for Deep Reinforcement Learning (DRL) agents operating in mobile network environments. SYMBXRL leverages symbolic AI, specifically First-Order Logic (FOL), to generate explanations that describe key concepts, relationships, and the decision-making process of DRL agents through intuitive symbols and rules. This approach offers more comprehensible descriptions of agent behavior compared to existing XRL methods.

**Key Contributions:**

-   Proposes SYMBXRL, a new XRL framework for DRL agents, employing symbolic representations and FOL for generating human-interpretable explanations.
-   Validates SYMBXRL in two diverse use cases:
    -   **A1-NetworkSlicingSchedulingPolicy:** DRL agent controlling Radio Access Network (RAN) slicing and scheduling.
    -   **A2-MIMOResourceScheduler:** DRL agent for resource scheduling in Massive MIMO.
-   Demonstrates that SYMBXRL's symbolic representation enables Intent-based Action Steering (IAS), improving cumulative rewards and enforcing operational constraints.
-   Shows that IAS outperforms existing XRL methods like METIS in terms of reward improvement.

Refer [Manuscript-details](./Manuscript-details) for the complete paper and further details.

**Repository Structure:**
```
SYMBXRL/
├── A1-NetworkSlicingSchedulingPolicy/ # Code for the A1 agent (RAN slicing and scheduling)
│ ├── data/ # Data files (raw and processed)
│ ├── results/ # Generated results (graphs, heatmaps, etc.)
│ ├── script/ # Python scripts for data processing, symbolic representation, analysis
│ ├── 1_data_preprocess.ipynb # Jupyter Notebook: Loads and preprocesses data for A1 agent
│ ├── 2_probabilistic_analysis.ipynb # Jupyter Notebook: Probabilistic analysis (distributions, heatmaps)
│ ├── 3_graph_visualization.ipynb # Jupyter Notebook: Graph-based analysis and visualization
│ ├── 4_Plots_for_Paper.ipynb # Jupyter Notebook: Generates plots used in the paper
│ ├── clean_exps.py # Python script: Handles and cleans experiment data
│ ├── constants.py # Python script: Defines project constants
│ ├── experiments_constants.py # Python script: Defines agent-specific constants and configurations
│ ├── load_data.py # Python script: Loads and preprocesses data
│ ├── p_square_quantile_approximator.py# Python script: Implements P² quantile approximation
│ ├── probability_comparison.py # Python script: Functions for probability distribution comparison
│ └── symbolic_representation.py # Python script: Creates symbolic representations from numerical data
│ └── utils.py # Python script: Utility functions
│ └── README.md # README for the A1 agent
├── A2-MIMOResourceScheduler/ # Code for the A2 agent (Massive MIMO resource scheduling)
│ ├── Datasets/ # Training and testing datasets
│ │ ├── LOS_highspeed1_64_7.hdf5 # HDF5 files for different scenarios (LOS, NLOS, varying speeds)
│ │ ├── LOS_highspeed2_64_7.hdf5
│ │ ├── LOS_lowspeed_64_7.hdf5
│ │ ├── LOS_test_64_7.hdf5
│ │ ├── NLOS_highspeed1_64_7.hdf5
│ │ ├── NLOS_highspeed2_64_7.hdf5
│ │ ├── NLOS_lowspeed_64_7.hdf5
│ │ └── NLOS_test_64_7.hdf5
│ ├── DQN/ # Implementation of the DQN agent
│ │ ├── DQNAgent.py # DQN agent class
│ │ ├── DQN_main.py # Main script for training the DQN agent
│ │ └── DQN_test.py # Script for testing the trained DQN agent
│ ├── SAC/ # Implementation of the SAC agent
│ │ ├── action_space.py # Defines the action space for the agent
│ │ ├── model.py # Neural network models for SAC
│ │ ├── replay_memory.py # Replay buffer implementation
│ │ ├── sac.py # SAC algorithm implementation
│ │ ├── SAC_main.py # Main script for training the SAC agent
│ │ ├── SAC_test.py # Script for testing the trained SAC agent
│ │ ├── SACArgs.py # Arguments for SAC
│ │ ├── smartfunc.py # Utility functions for SAC
│ │ ├── Action_Steering/ # Code related to Action Steering
│ │ │ ├── action_steering_utils.py # Utility functions for action steering
│ │ │ ├── decision_graph.py # Creates a decision graph based on agent's behavior
│ │ │ ├── experiment_constants.py # Constants for experiments
│ │ │ ├── p_square_quantile_approximator.py # P-square algorithm for quantile approximation
│ │ │ ├── symbolic_representation.py # Functions to generate symbolic representations
│ │ │ ├── 1. SAC_Agent_action_steering.ipynb # Jupyter Notebook: Implements and tests action steering
│ │ │ └── 2. paper_plots.ipynb # Jupyter Notebook: Generates plots for the paper
│ │ └── Agents_Numeric_Symbolic_Raw_Data/ # Data for plotting and evaluation (CSV files)
│ │ └── processed_csvs/ # Processed data
│ │ └── processed_action_steering_results.csv
│ ├── custom_mimo_env.py # Custom Gym environment for MIMO
│ ├── models/ # Trained model parameters
│ │ ├── DQN_956.59_300_dtLOS_HS2_final.pth
│ │ ├── SACG_884.53_551_dtLOS_HS2_checkpointed.pth
│ │ └── SACG_1989.88_205_dtNLOS_HS2_final.pth
│ ├── .gitignore # Git ignore file
│ ├── init.py # Init file
│ └── README.md # README for the A2 agent
├── paper-results/ # Directory containing the results presented in the paper.
├── .gitignore # Git ignore file
├── init.py
├── conda-environment.yml # Conda environment file
├── constants.py # Project-wide constants
├── pip-requirements.txt # Pip requirements file
└── README.md # Main README file (this file)
```

**Getting Started:**

1. **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd SYMBXRL
    ```

2. **Environment Setup:**

    -   It is recommended to create a virtual environment (e.g., using `conda` or `venv`) to manage dependencies.
    -   Each agent's directory may contain its specific environment file:
        - For the A1 agent, refer to the instructions in `A1-NetworkSlicingSchedulingPolicy/README.md`.
        - For the A2 agent, refer to the instructions in `A2-MIMOResourceScheduler/README.md`.

        To run the code, you can create an environment using conda:

        ```bash
        conda env create -f conda-environment.yml
        conda activate symbxrl-env
        ```

        or using pip:

        ```bash
        pip install -r pip-requirements.txt
        ```

        then update the root path of the project in the `constants.py` file.
        ```
        #### Root Address of the project
        PROJ_ADDR = '<path_to_project>'
        ```

3. **Explore the Code:**

    -   Navigate to the `A1-NetworkSlicingSchedulingPolicy/` or `A2-MIMOResourceScheduler/` directories to explore the code for each agent.
    -   Each agent's directory contains Jupyter Notebooks and Python scripts for data processing, analysis, and visualization.
    -   Refer to the agent-specific README files (e.g., `A1-NetworkSlicingSchedulingPolicy/README.md`, `A2-MIMOResourceScheduler/README.md`) for detailed instructions on how to run the code.

**Data:**

-   **A1 Agent:** The `A1-NetworkSlicingSchedulingPolicy/data/` directory is intended to store the raw and processed data for the A1 agent. You will need to place the raw data files in this directory. The raw data is not included in the repository. Refer to `A1-NetworkSlicingSchedulingPolicy/README.md` for more information about how to obtain and place it in the correct directory.
-   **A2 Agent:** The `A2-MIMOResourceScheduler/Datasets/` directory contains the datasets (HDF5 files) for the A2 agent. The datasets are already included in the repository.
-   The `paper-results/` folder contains the data used for the results presented in the paper.

**Results:**

-   The `results/` directory within each agent's folder will store the generated results (plots, heatmaps, graphs) after running the analysis scripts or notebooks.

**Contributing:**

-   If you wish to contribute to this project, please fork the repository and submit a pull request.

**License:**

-   This project is licensed under the MIT License - see the `LICENSE` file for details.

**Contact:**

-   For any questions or issues, please open an issue on the GitHub repository.

---