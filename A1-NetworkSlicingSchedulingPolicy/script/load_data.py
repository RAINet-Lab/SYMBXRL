from script.experiments_constants import EXPERIMENT_DATA_DIR_ADRESS
from script.experiments_constants import EXPERIMENT_DATA_LOG_FILE_NAME
from script.experiments_constants import STORAGE_DIRECTORY
from script.experiments_constants import CLEANED_EXPERIMENT_DATA_FILE_SUFFIX
from script.experiments_constants import CLEANED_SCHEDULING_POLICY_DATA_FILE_SUFFIX
from script.utils import get_list_of_experiments_number_for_number_of_users
import pandas as pd
import re
import os

def handle_data(agent_info, user_number = 6):
    """
    This function helps us to reduce the execution time of the program by order of magnitude for after first run. 
    It checks if the program has been ran before, and csv files are stored, just retrieve data from csv files.
    """ 

    # create list of directories according to number of users
    directories = get_list_of_experiments_number_for_number_of_users(agent_info['num_of_users'], user_number)
    str_helper = f"{user_number}-users_exp-{'-'.join(str(x) for x in directories)}"

    # Set address of the csv files
    agent_data_folder_address = f"{STORAGE_DIRECTORY}/{agent_info['name']}/{str_helper}/cleaned-data"
    agent_kpi_data_csv_address = f"{agent_data_folder_address}/{agent_info['name']}_{str_helper}{CLEANED_EXPERIMENT_DATA_FILE_SUFFIX}"
    agent_decision_data_csv_address = f"{agent_data_folder_address}/{agent_info['name']}_{str_helper}{CLEANED_SCHEDULING_POLICY_DATA_FILE_SUFFIX}"
    
    if not os.path.exists(agent_data_folder_address):
        os.makedirs(agent_data_folder_address)

    if os.path.exists(agent_kpi_data_csv_address) and os.path.exists(agent_decision_data_csv_address):
        # print(f"Cleaned data File for agent {agent_info['name']} exists, we load data from: {agent_kpi_data_csv_address}")
        
        kpi_data = pd.read_csv(agent_kpi_data_csv_address)
        decision_data = pd.read_csv(agent_decision_data_csv_address)
        # Todo: We may need to convert decision columns to tuple/list later 
        
    else:
        # print(f"Cleaned data File for agent {agent_info['name']} does not exists, we load data from log file for {str_helper}")
        raw_experiment_data, raw_prb_decision, raw_scheduling_policy = load_agent_data(directories, user_number)

        
        kpi_data, decision_data = clean_process_experiment_data(raw_experiment_data, raw_prb_decision, raw_scheduling_policy)
        
        kpi_data.to_csv(agent_kpi_data_csv_address, index=False)
        decision_data.to_csv(agent_decision_data_csv_address, index=False)

    return kpi_data, decision_data

################### ---------------------- Load Data From Files ----------------------- ############################
def load_agent_data(agent_data_directories, user_number):
    agent_experiment_data = []
    agent_prb_decision_data = []
    agent_scheduling_policy_data = []

    timestep = 0

    for dir in agent_data_directories:
        
        # Load data of a log file
        received_data_df, prb_decision_df, sending_to_du_df = process_log_file(EXPERIMENT_DATA_DIR_ADRESS + f"exp{str(dir)}" + EXPERIMENT_DATA_LOG_FILE_NAME)
        
        # filter for having each a new data frame that at each time stamp the data 
        for i in range(0, len(received_data_df)):
            timestep += 1

            for data in received_data_df.iloc[i].data:

                extracted_info = data.split(",")

                if check_received_data_elemets_by_length(extracted_info):
                    agent_experiment_data.append({
                        "timestep": int(timestep),
                        "slice_id": int(extracted_info[0]),
                        "tx_brate": float(extracted_info[2]),
                        "tx_pckts": float(extracted_info[5]),
                        "dl_buffer": float(extracted_info[1]),
                        "slice_prb": int(extracted_info[4]),
                    })
            
            agent_prb_decision_data.append({
                "timestep": timestep,
                "prb_decision": prb_decision_df.iloc[i].data
            })
             
            agent_scheduling_policy_data.append({
                "timestep": timestep,
                "scheduling_policy": sending_to_du_df.iloc[i].data
            })
    
    
    return pd.DataFrame(agent_experiment_data), pd.DataFrame(agent_prb_decision_data), pd.DataFrame(agent_scheduling_policy_data)

def check_received_data_elemets_by_length(array, length = 6):
    return len(array) == length

def process_log_file(file_path):
    """
    Processes the log file and returns dataframes for received data and data sent to DU.

    Args:
    file_path (str): Path to the log file.

    Returns:
    tuple: Two pandas DataFrames, one for received data and another for data sent to DU.
    """
    log_received_data = []
    log_prb_decision_data = []
    log_send_to_du_data = []
    try:
        with open(file_path, 'r') as file:
            # valid_entry falg will determine whether the received data is updated or not, 
            # if the received data is not updated it means the state of the agent is not updated and
            # as a result the decision of the agent hasn't changed and there is no need to record the decision.
            valid_entry = False
            for line in file:
                if "Received data:" in line:
                    valid_entry = True
                    received_data = parse_received_data(line)
                    log_received_data.append({'data': received_data})
                elif "Using previous socket data" in line:
                    valid_entry = False
                elif "Action means slice_prb" in line and valid_entry:
                    prb_decision = parse_prb_data(line)
                    log_prb_decision_data.append({'data': prb_decision})
                elif "Sending to DU:" in line and valid_entry:
                    sending_data = parse_sending_data(line)
                    log_send_to_du_data.append({'data': sending_data})

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        # Handle the exception or re-raise

    return pd.DataFrame(log_received_data), pd.DataFrame(log_prb_decision_data), pd.DataFrame(log_send_to_du_data)

def parse_prb_data(line):
    """
    Extract prb decisions for the next timestep from log file
    """
    match = re.search(r"slice_prb \[([0-9, ]+)\]", line)
    if match:
        numbers_str = match.group(1)
        numbers = numbers_str.split(", ")
        return numbers
    else:
        print("problem")

def parse_received_data(line):
    """
    Extracts and processes received data from a line.
    """

    return line.split("Received data:")[1][2:-2].split('\\n')

def parse_sending_data(line):
    """
    Extracts and processes sending data from a line.
    """
    return line.split("Sending to DU:")[1][2:7].strip().split(",")

################### ---------------------- Clean and Process Data ----------------------- ############################
def clean_process_experiment_data(extracted_data, prb_decisions, scheduling_policy):
    """
    Receives the raw read log data and perform cleaning and arranging on them.
    Output is two dataframe that the timestep is sync between them.
    At each time step the experiment data shows the KPI at that timestep and 
    the decision dataframe shows the decisions made according to that KPI which will be effective on the next timestep.
    """
    cleaned_experiment_data = []
    cleaned_decision = []
    
    timestep_counter = 0
    
    for i in range(1, extracted_data['timestep'].max()+1):
        data_at_timestep = extracted_data.loc[extracted_data.timestep == i]
        
        if is_complete_data(data_at_timestep):
            timestep_counter += 1
            
            timestep_data = group_and_average_data(data_at_timestep)
            cleaned_experiment_data.extend(
                build_experiment_data(timestep_counter, timestep_data)
            )

            policy_state = scheduling_policy.loc[scheduling_policy.timestep == i]
            prb_state = prb_decisions.loc[prb_decisions.timestep == i]
            
            decision = make_decision(prb_state, policy_state, timestep_counter)
            
            cleaned_decision.append(decision)

    
    return drop_non_frequent_slice_timesteps(pd.DataFrame(cleaned_experiment_data), pd.DataFrame(cleaned_decision))

def drop_non_frequent_slice_timesteps(kpi_data, decision_data):
    """
    Filters out timesteps from kpi_data and decision_data DataFrames that do not contain
    the most frequently occurring set of slice_ids. Rows with timesteps not containing the most
    frequent set are removed directly from both dataframes.
    """
    # Group by 'timestep' and collect unique 'slice_id' for each group in kpi_data
    slice_sets_per_timestep = kpi_data.groupby('timestep')['slice_id'].apply(lambda x: set(x))
    
    # Identify the most frequent set of 'slice_id'
    set_frequency = slice_sets_per_timestep.value_counts()
    most_frequent_set = set_frequency.idxmax()
    
    # Identify timesteps that do not contain the most frequent set
    timesteps_not_with_most_frequent_set = slice_sets_per_timestep[slice_sets_per_timestep != most_frequent_set].index
    
    # Filter out these timesteps from both kpi_data and decision_data
    filtered_kpi_data = kpi_data[~kpi_data['timestep'].isin(timesteps_not_with_most_frequent_set)]
    filtered_decision_data = decision_data[~decision_data['timestep'].isin(timesteps_not_with_most_frequent_set)]
    
    # Adjust the 'timestep' in filtered_kpi_data and filtered_decision_data to be continuous starting from 1
    filtered_kpi_data['timestep'], _ = pd.factorize(filtered_kpi_data['timestep'], sort=True)
    filtered_kpi_data['timestep'] += 1

    filtered_decision_data['timestep'], _ = pd.factorize(filtered_decision_data['timestep'], sort=True)
    filtered_decision_data['timestep'] += 1


    # Reset indexes
    filtered_kpi_data.reset_index(drop=True, inplace=True)
    filtered_decision_data.reset_index(drop=True, inplace=True)

    return filtered_kpi_data, filtered_decision_data

def is_complete_data(data):
    return len(data['slice_id'].unique()) >= 1

def group_and_average_data(data):
    return data.groupby(['slice_id']).mean().reset_index()

def build_experiment_data(timestep, grouped_data):
    return [
        {
            "timestep": timestep,
            "slice_id": int(item['slice_id']),
            "tx_brate": item['tx_brate'],
            "tx_pckts": item['tx_pckts'],
            "dl_buffer": item['dl_buffer']
        }
        for key, item in grouped_data.iterrows()
    ]

def make_decision(prb_data, policy_state, timestep):
    prb_state = define_prb_state(prb_data)
    scheduling_policy_state = define_scheduling_policy_state(policy_state)
    
    return {
        "timestep": timestep,
        "decision": tuple(prb_state + scheduling_policy_state),
        "prb_decision": prb_state,
        "policy_decision": scheduling_policy_state
    }

def define_prb_state(prb_data):    
    return [int(i) for i in prb_data['prb_decision'].iloc[0]]

def define_scheduling_policy_state(df_timestep_scheduling_policy_data):
    return [int(i) for i in df_timestep_scheduling_policy_data['scheduling_policy'].iloc[0]]