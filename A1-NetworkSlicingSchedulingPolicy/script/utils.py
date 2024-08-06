import os
import pandas as pd
import numpy as np
from graphviz import Digraph
from script.experiments_constants import STORAGE_DIRECTORY, KPI_CHANGE_THRESHOLD_PERCENT, AGENT_WITH_REWARD_FOLDER
from script.experiments_constants import ENV_KPI_NAME_LIST, PRB_CATEGORY_LIST


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


# Plotting utils
def create_effects_list(kpis=ENV_KPI_NAME_LIST, changes=['dec', 'const', 'inc']):
    return {
        kpi: [f'{change}({kpi}, Q{quartile})' for quartile in range(1, 5) for change in changes] for kpi in kpis
    }

def create_decisions_list(predicates=['dec', 'const', 'inc'], 
                          quartiles=['C1', 'C2', 'C3', 'C4', 'C5'], 
                          sched_policies=['const', 'toRR', 'toWF', 'toPF']):
    decisions = []
    for predicate in predicates:
        for quartile in quartiles:
            for policy in sched_policies:
                decision = f"{predicate}(prb, {quartile}) - {policy}(sched)"
                decisions.append(decision)
    return decisions

def create_prb_decisions_list(states = list(PRB_CATEGORY_LIST.keys())):
    # Define the possible decision types
    decision_types = ['inc', 'dec', 'const']
    
    # Initialize an empty list to store all possible decisions
    all_decisions = []
    # Generate all possible decisions
    for decision in decision_types:
        for i, start_state in enumerate(states):
            if decision == 'const':  # 'const' decisions have a single state
                all_decisions.append(f"{decision}(prb, {start_state})")
            elif decision == 'inc':
                for end_state in states[i:]:
                    if start_state != end_state:
                        all_decisions.append(f"{decision}(prb, {start_state}, {end_state})")
            elif decision == 'dec':
                for end_state in states[:i]:
                    if start_state != end_state:
                        all_decisions.append(f"{decision}(prb, {start_state}, {end_state})")
    
    return all_decisions

