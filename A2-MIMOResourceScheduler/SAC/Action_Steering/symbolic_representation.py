'''
The following code is part of "SymbXRL: Symbolic Explainable Deep Reinforcement Learning for Mobile Networks" 
Copyright - RESILIENT AI NETWORK LAB, IMDEA NETWORKS

DISCLAIMER: THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
'''


## Imports

from Action_Steering.p_square_quantile_approximator import PSquareQuantileApproximator
import pandas as pd
import numpy as np
import ast

KPI_CHANGE_THRESHOLD_PERCENT = 5

# Symbolizer that receives one or 2 timestep of data and returns the symbolic representation
class QuantileManager:
    '''
    A class to manage quantile approximations for multiple KPIs using the PSquareQuantileApproximator.
    '''
    def __init__(self, kpi_list, p=50):        
        self.quantile_approximators = {kpi: PSquareQuantileApproximator(p) for kpi in kpi_list}
    
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
            self.quantile_approximators[kpi].reset() # TODO: the reset, set's markers to 1, 2, 3, 4, 5 Maybe we should do something about this
    
    def represent_markers(self):
        """
        This function will return a dictionary that keys are q0 to q4
        """
        markers_data = []
        for kpi in self.quantile_approximators:
            markers = self.get_markers(kpi)
            if len(markers) == 5:
                markers_data.append({
                    "kpi": kpi,
                    "q0": markers[0],
                    "q1": markers[1],
                    "q2": markers[2],
                    "q3": markers[3],
                    "q4": markers[4],
                })
        
        return pd.DataFrame(markers_data)
    
class Symbolizer:
    def __init__(self, quantile_manager: QuantileManager, kpi_list, users):
        self.quantile_manager = quantile_manager
        self.kpis_name_list = kpi_list
        self.users = users
        
        self.prev_state_df = {}
        self.prev_decision_df = {}
        
        self.prev_state_candid_df = {}
        self.prev_decision_candid_df = {}
    
    def create_symbolic_form(self, state_t_df, decision_t_df):
        """
            This function will receive the state and decision of an agent in a timestep and returns the symbolic representation of it
        """
        effects_symbolic_representation = []
        
        ## Scrape groups and each groups memebers from 
        groups = self._get_list_of_existing_groups_in_timestep(state_t_df)
        agent_complete_decision = self._get_actions_full_represetntation(decision_t_df['action'].iloc[0])
        
        for group, group_members in groups.items():
            ## For Each group check if we have a previous record for that group, if existst then compare with it and create symbolic format if not store it and wait for the next one
            members_refined = None
            kpi_columns = list(np.concatenate([self._get_list_of_kpi_column_for_users(kpi_name, group_members) for kpi_name in self.kpis_name_list]))
            
            if group in self.prev_state_df:
                group_symbolic_effect = self._calculate_kpi_symbolic_state(state_t_df, self.prev_state_df[group], group_members)
                
                group_symbolic_decision = self._calculate_decision_symbolic_state(decision_t_df, self.prev_decision_df[group], group, group_members)
                
                members_refined = self._clean_member_state_according_to_scheduling(group_members, decision_t_df['action'].iloc[0])
                
                effects_symbolic_representation.append({
                    "timestep": state_t_df['timestep'].iloc[0],
                    "group": group,
                    "group_members": str(group_members),
                    **group_symbolic_effect,
                    "sched_members": str(members_refined),
                    "sched_members_complete": str(agent_complete_decision),
                    "decision": group_symbolic_decision
                })
            else:
                members_refined = self._clean_member_state_according_to_scheduling(group_members, decision_t_df['action'].iloc[0])
            
            decision_to_be_rememeberd = decision_t_df.copy()
            decision_to_be_rememeberd.at[decision_to_be_rememeberd.index[0], 'action'] = members_refined[0]
            
            self.prev_state_candid_df[group] = state_t_df[kpi_columns]
            self.prev_decision_candid_df[group] = decision_to_be_rememeberd
            self.quantile_manager.partial_fit('scheduled_user', [len(members_refined[0])])
            
        self._add_timestep_kpi_data_to_approximator(state_t_df)
        return pd.DataFrame(effects_symbolic_representation)
    
    def step(self):
        """
        This function will step to the next timestep and update the previous state and decision
        """
        self.prev_state_df = self.prev_state_candid_df.copy()
        self.prev_decision_df = self.prev_decision_candid_df.copy()
    
    def _clean_member_state_according_to_scheduling(self, members_list, decision):
        """
        This function receives a list of members and returns a list of 2 lists, the first element contains the shceduled users and the second element contains the list of unscheduled users
        """
        decision = set(ast.literal_eval(decision) if not isinstance(decision, tuple) else decision)
        scheduled_members = [member for member in members_list if member in decision]
        unscheduled_members = [member for member in members_list if member not in decision]
        return [scheduled_members, unscheduled_members]
    
    def _calculate_decision_symbolic_state(self, current_decision_df, previous_decision, group_num, group_users):
        """Calculate the symbolic state for a decision based on current and previous decision values."""
        # inc(G0, Quartile, Percent)
        
        current_decision = self._clean_member_state_according_to_scheduling(group_users, current_decision_df['action'].iloc[0])
        previous_decision = ast.literal_eval(previous_decision['action'].iloc[0]) if not isinstance(previous_decision['action'].iloc[0], list) else previous_decision['action'].iloc[0]
        
        scheduled_users_count = len(current_decision[0])
        total_users_count = len(current_decision[0]) + len(current_decision[1])
        
        ## Set predicate
        predicate = "const"
        ## Compare the number of schedueld user with previous one and set direction
        if scheduled_users_count > len(previous_decision):
            predicate = "inc"
        elif scheduled_users_count < len(previous_decision):
            predicate = 'dec'

        ## Set Group Name
        group_name = f"G{group_num}"
        
        ## Set Quartile of Scheduled User number
        quartile = self._get_kpi_quantile("scheduled_user", scheduled_users_count)
        
        ## Set the percentage of scheduled users from the group
        # scheduled_percentage = round((scheduled_users_count / total_users_count) * 100 / 10) * 10
        scheduled_percentage = round((scheduled_users_count / total_users_count) * 100 / 25) * 25
        
        return f"{predicate}({group_name}, {quartile}, {scheduled_percentage})" 
    
    def _calculate_kpi_symbolic_state(self, curr_state_df:pd.DataFrame, prev_state_df:pd.DataFrame, members:list):
        """
        Calculate the symbolic state for a KPI slice based on current and previous KPI values.
        """           
        kpi_symbolic_representatino = {}
        
        for kpi_group in self.kpis_name_list:
            # calculate the symbolic form of mean of the kpis
            curr_mean = round(curr_state_df[self._get_list_of_kpi_column_for_users(kpi_group, members)].iloc[0].mean(), 4)
            prev_mean = round(prev_state_df.filter(regex=f"^{kpi_group}").iloc[0].mean(), 4)
            
            kpi_symbolic_representatino[f'{kpi_group}'] = self._define_MSE_or_DTU_symbolic_state(curr_mean, prev_mean, f'{kpi_group}', kpi_group)
            # kpi_symbolic_representatino[f'{kpi_group}_mean'] = self._define_MSE_or_DTU_symbolic_state(curr_mean, prev_mean, f'{kpi_group}_mean', kpi_group)
            # display(kpi_symbolic_representatino)
            # detail = []
            # for member in members:
            #     detail.append(self._define_MSE_or_DTU_symbolic_state(curr_kpi[f'{kpi_group}{member}'].iloc[0], prev_kpi[f'{kpi_group}{member}'].iloc[0], f'{kpi_group}{member}', kpi_group))
            # kpi_symbolic_representatino[f'{kpi_group}_detail'] = detail
        return kpi_symbolic_representatino
    
    def _define_MSE_or_DTU_symbolic_state(self, curr_value, prev_value, kpi_column, kpi_name):
        """
            This function will calculate and returns the symbolic representation of the MSE column for different users
        """
        change_percentage = self._find_change_percentage(curr_value, prev_value)
        predicate = self._get_predicate(change_percentage)
        return f'{predicate}({kpi_column}, {self._get_kpi_quantile(kpi_name, curr_value)})'
    
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
        elif change_percentage > KPI_CHANGE_THRESHOLD_PERCENT:
            return "inc"
        elif change_percentage < -KPI_CHANGE_THRESHOLD_PERCENT:
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
        elif kpi_value <= 0.999*markers[4]:
            return "Q4"
        else:
            return "MAX"
    
    def _add_timestep_kpi_data_to_approximator(self, timestep_df):
        """Adds KPI data of one timestep to the quantile approximators."""
        for kpi_name in self.kpis_name_list:
            # Create list of df columns according to the kpi_name
            kpi_columns = self._get_list_of_kpi_column_for_users(kpi_name, self.users)
            
            # send list of new values to the quartile manager to handle
            self.quantile_manager.partial_fit(kpi_name, timestep_df[kpi_columns].iloc[0].to_numpy())
    
    def _get_list_of_kpi_column_for_users(self, kpi_name, user_list):
        """
        This function will combine the kpi name and user list given. The output is used to fetch columns from the receied state dataframe
        """
        return [f'{kpi_name}{user}' for user in user_list]
    
    def _get_list_of_existing_groups_in_timestep(self, data):
        """"
        This function will receive a timestep of data and returns a dictionary that the keys are group number and the elements are the list of users in that group
        """
        # Find group numbers
        groups = {}
        # Iterate over each 'UGUr' column to determine the group for each user
        for i in self.users:  # Assuming there are 7 UGUr columns (0 to 6)
            group_number = int(data[f'UGUr{i}'].iloc[0])
            
            # Add the user number (i) to the corresponding group number key in the dictionary
            if group_number in groups:
                groups[group_number].append(i)
            else:
                groups[group_number] = [i]
        return groups
    
    def _get_actions_full_represetntation(self, agent_action_tuple):
        """
        This function will receive the action and return the full representation of the action
        """
        # Convert the input tuple to a list
        members = list(agent_action_tuple)
        
        # Create a full set of numbers from 0 to 6
        full_set = set(range(7))
        
        # Convert the input tuple to a set
        input_set = set(agent_action_tuple)
        
        # Find the missing numbers by subtracting the input set from the full set
        missing_numbers = list(full_set - input_set)
        
        # Return the result as a list of two lists
        return [members, missing_numbers]