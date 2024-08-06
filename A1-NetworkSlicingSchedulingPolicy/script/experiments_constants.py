import os
from constants import proj_address


# The association of folder number with agents
AGENT_EXPERIMENT_INFO = {
    "embb-trf1": {
        "name": "embb-trf1",
        "experiment_directories": [1,2,3,4,5,6,7,8],
        "num_of_users": {
            1:6,
            2:5,
            3:4,
            4:3,
            5:2,
            6:1,
            7:1,
            8:1
        }
    },
    "embb-trf2":{
        "name": "embb-trf2",
        "experiment_directories": [9,10,11,12,13,14,15,16],
        "num_of_users": {
            9:6,
            10:5,
            11:4,
            12:3,
            13:2,
            14:1,
            15:1,
            16:1
        }
    },
    "urllc-trf1":{
        "name": "urllc-trf1",
        "experiment_directories": [27,28,29,30,31,32,33,34],
        "num_of_users": {
            27:6,
            28:5,
            29:4,
            30:3,
            31:2,
            32:1,
            33:1,
            34:1
        }
    },
    "urllc-trf2":{
        "name": "urllc-trf2",
        "experiment_directories": [35,36,37,38,39,40,41,42],
        "num_of_users": {
            35:6,
            36:5,
            37:4,
            38:3,
            39:2,
            40:1,
            41:1,
            42:1
        }
    }  
}


# List of KPIs provided by the environment
ENV_KPI_NAME_LIST = ['tx_brate', 'tx_pckts', 'dl_buffer']

# Change Threshold for KPIS and PRB
KPI_CHANGE_THRESHOLD_PERCENT = 5

PRB_CATEGORY_LIST = {
    "C1": (0, 10),  # Values from 0 to 10 (inclusive) belong to category C1
    "C2": (11, 20),  # Values from 11 to 20 belong to category C2
    "C3": (21, 30),
    "C4": (31, 40),
    "C5": (41, 50)
}


# # The address for loading experiment data in the main file
EXPERIMENT_DATA_DIR_ADRESS = f"{proj_address}/A1-NetworkSlicingSchedulingPolicy/data/raw-exps/" 
EXPERIMENT_DATA_LOG_FILE_NAME = "/xapp_drl_sched_slicing_ric_26_agent.log"

# # The address for loading experiment data with reward
# EXPERIMENT_DATA_DIR_ADRESS_ACTION_STEERING = "../../../data/action-steering/" 
# EXPERIMENT_DATA_LOG_FILE_NAME_REWARD = "/xapp_drl_sched_slicing_ric_26_agent.log"

# # New constants for directory and names of the csv files
STORAGE_DIRECTORY = f"{proj_address}/A1-NetworkSlicingSchedulingPolicy/data/symbxrl/"
CLEANED_EXPERIMENT_DATA_FILE_SUFFIX = "_cleaned_experiment_data.csv"
CLEANED_SCHEDULING_POLICY_DATA_FILE_SUFFIX = "_cleaned_scheduling_policy_data.csv"
SYMBOLIC_DATA_FILE_SUFFIX = "_symbolic_data.csv"
QUANTILE_DATA_FILE_SUFFIX = "_quantile_data.csv"


# AGENT_EXPERIMENT_INFO_ACTION_STEERING = {
#     "embb-trf1": {
#         "name": "embb-trf1",
#         "experiment_directories": [9, 13, 15, 21, 25, 27, 29, 31]
#     },
#     "embb-trf2": {
#         "name": "embb-trf2",
#         "experiment_directories": [11, 17, 19, 23, 27, 29, 31, 33]
#     },
#     "urllc-trf1": {
#         "name": "urllc-trf1",
#         "experiment_directories": [10, 14, 16, 22, 26, 28, 30, 32]
#     },
#     "urllc-trf2": {
#         "name": "urllc-trf2",
#         "experiment_directories": [12, 18, 20, 24, 28, 30, 32, 34]
#     }
# }

AGENT_WITH_REWARD_FOLDER = {
    "winter-2023": [27, 28, 29, 30, 31, 32, 33, 34],
    "spring-2023": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
}

# # Action steering experimetns with their mode
EXP_BASELINE = {"embb-trf1": 27,"embb-trf2": 29,"urllc-trf1": 28,"urllc-trf2": 30}
# EXP_BASELINE_OBS_20 = {"embb-trf1": 21,"embb-trf2": 23,"urllc-trf1": 22,"urllc-trf2": 24}

# EXP_MAX_REWARD = {"embb-trf1": 31,"embb-trf2": 33,"urllc-trf1": 32,"urllc-trf2": 34}
# EXP_MAX_REWARD_OBS_20 = {"embb-trf1": 9,"embb-trf2": 11,"urllc-trf1": 10,"urllc-trf2": 12}

# # exp_min_reward = {"embb-trf1": 13,"embb-trf2": 17,"urllc-trf1": 14,"urllc-trf2": 18}
# # exp_min_reward_obs_20 = {"embb-trf1": 15,"embb-trf2": 19,"urllc-trf1": 16,"urllc-trf2": 20}
# EXP_IMP_TX_BRATE= {"embb-trf1": 25,"embb-trf2": 27,"urllc-trf1": 26,"urllc-trf2": 28}
# EXP_IMP_TX_BRATE_OBS_20 = {"embb-trf1": 29,"embb-trf2": 31,"urllc-trf1": 30,"urllc-trf2": 32}

# # Agent modes
# AGENT_MODES_LIST = {
#     'baseline': EXP_BASELINE,
#     'baseline_lw': EXP_BASELINE_OBS_20,
#     'improve_brate': EXP_IMP_TX_BRATE,
#     'improve_brate_lw': EXP_BASELINE_OBS_20
# }
