import numpy as np
import pandas as pd
import os
import ast
from script.experiments_constants import STORAGE_DIRECTORY, SYMBOLIC_DATA_FILE_SUFFIX, QUANTILE_DATA_FILE_SUFFIX, ENV_KPI_NAME_LIST 
from script.utils import get_list_of_experiments_number_for_number_of_users, get_kpi_change_threshold_percent
from script.experiments_constants import PRB_CATEGORY_LIST
from script.p_square_quantile_approximator import PSquareQuantileApproximator

class KPIQuantileManager:
    def __init__(self):
        self.quantile_approximators = {
            "tx_brate": PSquareQuantileApproximator(p=50),
            "tx_pckts": PSquareQuantileApproximator(p=50),
            "dl_buffer": PSquareQuantileApproximator(p=50),
        }

    def fit(self):
        for approximator in self.quantile_approximators.values():
            approximator.fit([])

    def partial_fit(self, kpi_name, value):
        if kpi_name in self.quantile_approximators:
            self.quantile_approximators[kpi_name].partial_fit(value)

    def get_markers(self, kpi_name):
        if kpi_name in self.quantile_approximators:
            return self.quantile_approximators[kpi_name].get_markers()
        else:
            return []

    def reset(self):
        for kpi in self.quantile_approximators:
            self.quantile_approximators[kpi].fit([])

class SymbolicDataCreator:
    def __init__(self, quantile_manager, kpi_name_list):
        self.quantile_manager = quantile_manager
        self.kpi_name_list = kpi_name_list
        self.marker_data = []

        
    def create_symbolic_data(self, df_kpi, df_decision):
        """
        Create symbolic representation of data.
        """
        
        effects_symbolic_representation = []
        max_timestep = df_kpi['timestep'].max()
        self._add_timestep_kpi_data_to_approximator(df_kpi[df_kpi['timestep'] == 1], timestep=1)
        
        for i in range(2, max_timestep - 1):
            resulting_kpi = df_kpi[df_kpi['timestep'] == i + 1]
            effective_decision = df_decision[df_decision['timestep'] == i]['decision'].iloc[0]
            previous_kpi = df_kpi[df_kpi['timestep'] == i]
            previous_decision = df_decision[df_decision['timestep'] == i - 1]['decision'].iloc[0]
            
            self._add_timestep_kpi_data_to_approximator(previous_kpi, timestep=i)
            
            slices = resulting_kpi['slice_id'].unique()
            
            effect_symbolic_decision = self.calculate_decision_symbolic_state(effective_decision, previous_decision, slices)
            
            for slice_id in slices:
                symbolic_effect_for_slice = self._calculate_kpi_symbolic_state(
                    resulting_kpi[resulting_kpi['slice_id'] == slice_id][['tx_brate', 'tx_pckts', 'dl_buffer']], 
                    previous_kpi[previous_kpi['slice_id'] == slice_id][['tx_brate', 'tx_pckts', 'dl_buffer']]
                )

                effects_symbolic_representation.append({
                    "timestep": i,
                    "slice_id": slice_id,
                    "prb_decision": effect_symbolic_decision.loc[effect_symbolic_decision['slice_id'] == slice_id, 'prb'].iloc[0],
                    "sched_decision": effect_symbolic_decision.loc[effect_symbolic_decision['slice_id'] == slice_id, 'sched'].iloc[0],
                    **symbolic_effect_for_slice
                })
        
        return pd.DataFrame(effects_symbolic_representation), pd.DataFrame(self.marker_data)
    
    def _add_timestep_kpi_data_to_approximator(self, timestep_data, timestep):
        """Adds KPI data of one timestep to the quantile approximators."""
        
        for kpi_name in self.kpi_name_list:
            self.quantile_manager.partial_fit(kpi_name, timestep_data[kpi_name].to_numpy())
            markers = self.quantile_manager.get_markers(kpi_name)
            if len(markers) == 5:
                self.marker_data.append({
                    "timestep": timestep,
                    "kpi": kpi_name,
                    "q0": markers[0],
                    "q1": markers[1],
                    "q2": markers[2],
                    "q3": markers[3],
                    "q4": markers[4],
                })

    def calculate_decision_symbolic_state(self, current_decision, previous_decision, slices):
        """Calculate the symbolic state for a decision based on current and previous decision values."""
        
        symbolic_decision = []

        current_decision = ast.literal_eval(current_decision) if not isinstance(current_decision, tuple) else current_decision
        previous_decision = ast.literal_eval(previous_decision) if not isinstance(previous_decision, tuple) else previous_decision

        for slice_id in slices:
            symbolic_decision.append({
                "slice_id": slice_id,
                "prb": f"{self._define_prb_change_symbolic_representation(current_decision[slice_id], previous_decision[slice_id], 'prb')}",
                "sched": f"{self._define_scheduling_policy_change_symbol(current_decision[slice_id + 3], previous_decision[slice_id + 3])}(sched)"
            })

        return pd.DataFrame(symbolic_decision)

    def _define_prb_change_symbolic_representation(self, curr_value, prev_value, variable_name):
        """
        Convert the changes in the KPIs to symbolic representation.
        """
        curr_cat = self._get_prb_category(curr_value)
        prev_cat = self._get_prb_category(prev_value)
        
        change_percentage = self._find_change_percentage(curr_value, prev_value)
        predicate = self._get_predicate(change_percentage)
        
        if curr_cat == prev_cat:
            return f"const({variable_name}, {curr_cat})"
        else:
            # return f"{predicate}({variable_name}, {prev_cat}, {curr_cat})"
            return f"{predicate}({variable_name}, {curr_cat})"
        
    # def _define_prb_change_symbolic_representation(self, curr_value, prev_value, variable_name):
    #     """
    #     Convert the changes in the KPIs to symbolic representation.
    #     """
    #     change_percentage = self._find_change_percentage(curr_value, prev_value)
    #     predicate = self._get_predicate(change_percentage)
        
    #     if predicate == "const":
    #         return f"{predicate}({variable_name})"
    #     else:
    #         return f"{predicate}({variable_name}, {abs(curr_value - prev_value)}, {curr_value})"

    def _get_prb_category(self, value, category_map=PRB_CATEGORY_LIST):
        """ 
        This function will return the category in which the prb value sits in which category
        Maps a numerical value to its corresponding category based on the provided category map.
        Args:
            value: The numerical value to be categorized.
            category_map: A dictionary where keys are category names and values are tuples representing the range (inclusive) for each category.
        Returns:
            The category name to which the value belongs, or None if no matching category is found.
        """
        for category, (lower_bound, upper_bound) in category_map.items():
            if lower_bound <= value <= upper_bound:
                return category
        return None
        
    def _define_scheduling_policy_change_symbol(self, curr_sch_pl, prev_sch_pl):
        """Convert the changes in the scheduling policy to symbolic representation."""
        scheduling_policy_string_helper = {0: "RR", 1: "WF", 2: "PF"}
        return "const" if curr_sch_pl == prev_sch_pl else f"to{scheduling_policy_string_helper[curr_sch_pl]}"

    def _calculate_kpi_symbolic_state(self, curr_kpi, prev_kpi):
        """Calculate the symbolic state for a KPI slice based on current and previous KPI values."""    
        return {
                f"{kpi}": f"{self._define_change_symbolic_representation(curr_kpi[kpi].iloc[0], prev_kpi[kpi].iloc[0], kpi)}" for kpi in curr_kpi.columns
            }

    def _define_change_symbolic_representation(self, curr_value, prev_value, variable_name):
        """Define symbolic representation of changes for KPIs."""
        """Convert the changes in the KPIs to symbolic representation."""
        change_percentage = self._find_change_percentage(curr_value, prev_value)
        predicate = self._get_predicate(change_percentage)
        
        # if predicate == "const":
        #     return f"{predicate}({variable_name})"
        # else:        
        return f"{predicate}({variable_name}, {self._get_kpi_quantile(variable_name, curr_value)})"

    def _find_change_percentage(self, curr_value, prev_value):
        """ This function will calculate the change percentage of the given parameter """
        if prev_value == 0:
            if curr_value == 0:
                return 0
            else:
                return 'inf'
        else:
            return int(100 * (curr_value - prev_value) / prev_value)

    def _get_predicate(self, change_percentage):
        """ This function will return the correct predicate according to the change percentage """
        if change_percentage == 'inf':
            return "inc"
        elif change_percentage > get_kpi_change_threshold_percent():
            return "inc"
        elif change_percentage < -get_kpi_change_threshold_percent():
            return "dec"
        else:
            return "const"

    def _get_kpi_quantile(self, kpi_name, kpi_value):
        """
        This function will return the quarter or if the value is min/max of the observed data
        """
        markers = self.quantile_manager.get_markers(kpi_name)
        
        if len(markers) < 5:
            return "NaN"
        # Check for values at or below the minimum marker or at or above the maximum marker
        if kpi_value <= markers[1]:
            return "Q1"
        elif kpi_value <= markers[2]:
            return "Q2"
        elif kpi_value <= markers[3]:
            return "Q3"
        else:
            return "Q4"
    
def create_directory_path(agent_info, user_number, storage_directory):
    """
    Create the directory path for storing data based on agent information and user number.
    """
    
    directories = get_list_of_experiments_number_for_number_of_users(agent_info['num_of_users'], user_number)
    str_helper = f"{user_number}-users_exp-{'-'.join(str(x) for x in directories)}"
    
    return os.path.join(storage_directory, agent_info['name'], str_helper)

def ensure_directory_exists(directory_path):
    """Ensure that the directory exists."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def get_symbolic_data_csv_path(directory_path, agent_info, user_number, symbolic_data_file_suffix):
    """
    Construct the full path to the symbolic data CSV file based on directory path, agent information, and user number.
    """
    
    directories = get_list_of_experiments_number_for_number_of_users(agent_info['num_of_users'], user_number)
    str_helper = f"{user_number}-users_exp-{'-'.join(str(x) for x in directories)}"
    
    return os.path.join(directory_path, f"{agent_info['name']}_{str_helper}{symbolic_data_file_suffix}")

def create_symbolic_state_decision_matrix(df_kpi, df_decision, agent_info, user_number, force_overwrite=False):
    """
    Main function to create or read symbolic state decision matrix.
    
    Args:
        df_kpi (DataFrame): DataFrame containing KPI data.
        df_decision (DataFrame): DataFrame containing decision data.
        agent_info (dict): Dictionary containing agent information.
        user_number (int): Number of users.
        force_overwrite (bool): If True, force overwrite the existing CSV file.
    """

    directory_path = create_directory_path(agent_info, user_number, STORAGE_DIRECTORY)
    ensure_directory_exists(directory_path)

    symbolic_data_csv_path = get_symbolic_data_csv_path(directory_path, agent_info, user_number, SYMBOLIC_DATA_FILE_SUFFIX)
    qunatile_data_csv_path = get_symbolic_data_csv_path(directory_path, agent_info, user_number, QUANTILE_DATA_FILE_SUFFIX)
    
    
    # Check if the CSV exists and force_overwrite is False
    if not force_overwrite and os.path.exists(symbolic_data_csv_path):
        # print("Reading existing symbolic data CSV: ", symbolic_data_csv_path, "and", qunatile_data_csv_path)
        return pd.read_csv(symbolic_data_csv_path), pd.read_csv(qunatile_data_csv_path)
    else:
        print("Creating new symbolic data CSV...")
        quantile_manager = KPIQuantileManager()
        symbolic_creator = SymbolicDataCreator(quantile_manager, ENV_KPI_NAME_LIST)
        
        quantile_manager.reset()
        
        df_symbolic_representation, df_marker_data = symbolic_creator.create_symbolic_data(df_kpi, df_decision)
        df_symbolic_representation.to_csv(symbolic_data_csv_path, index=False)
        
        # print(qunatile_data_csv_path)
        df_marker_data.to_csv(qunatile_data_csv_path, index=False)

        return pd.DataFrame(df_symbolic_representation), pd.DataFrame(df_marker_data)
