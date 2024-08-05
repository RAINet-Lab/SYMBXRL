import os
from script.experiments_constants import STORAGE_DIRECTORY, KPI_CHANGE_THRESHOLD_PERCENT, AGENT_WITH_REWARD_FOLDER

def get_exp_folder(exp_number):
    """
    this function will receive the number of the experiment and returns the folder in which the experiment is stored:
    winter-2023 or spring-2023
    """
    for key, value in AGENT_WITH_REWARD_FOLDER.items():
        if exp_number in value:
            return key

def get_list_of_experiments_number_for_number_of_users(dictionary, value):
    return [key for key, val in dictionary.items() if val == value]

def get_kpi_change_threshold_percent():
    return KPI_CHANGE_THRESHOLD_PERCENT

def create_plot_dir_for_analysis(analysis_name):
    """Create the directory path for storing data."""
    str_helper = f"resulting_plots/{analysis_name}"
    path = os.path.join(STORAGE_DIRECTORY, str_helper)
    ensure_directory_exists(path)
    return path
    
def ensure_directory_exists(directory_path):
    """Ensure that the directory exists."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)