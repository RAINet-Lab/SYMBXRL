'''
The following code is part of "SymbXRL: Symbolic Explainable Deep Reinforcement Learning for Mobile Networks" 
Copyright - RESILIENT AI NETWORK LAB, IMDEA NETWORKS

DISCLAIMER: THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
'''

import ast
import random
import numpy as np
import pandas as pd

def extract_decision_from_suggested(suggested_decision):
    """
    Extracts and sorts decisions from a list of suggested decisions.
    This function takes a list of suggested decisions, where each decision can be either a string representation
    of a list or a list itself. It converts any string representations to lists, extracts the first element from
    each list, and then sorts and returns these elements as a tuple.
    Args:
        suggested_decision (list): A list of suggested decisions, where each decision is either a string 
                                   representation of a list or a list itself.
    Returns:
        tuple: A sorted tuple of the extracted decisions.
    """
    extracted_decision = []
    for item in suggested_decision:
        converted_decision = ast.literal_eval(item) if type(item) == str else item            
        extracted_decision.extend(converted_decision[0])
    
    return tuple(sorted(extracted_decision))

def do_action_steering_this_timestep(curr_state_df, history_df, rt_decision_graph):
    """
    Perform action steering for the current timestep based on the current state, history, and real-time decision graph.
    Parameters:
    curr_state_df (pd.DataFrame): DataFrame containing the current state information.
    history_df (pd.DataFrame): DataFrame containing the historical state information.
    rt_decision_graph (dict): Dictionary containing the real-time decision graphs for each group.
    Returns:
    tuple: A tuple containing:
        - pd.Series: The scheduled members for the action-steered timestep.
        - float: The reward for the action-steered timestep.
        If no common timesteps are found, returns (False, None).
    """
    prev_state_df = history_df[history_df['timestep'] == history_df['timestep'].tail(1).iloc[0]] 
    groups = curr_state_df['group'].unique()
    common_timesteps = set(history_df['timestep'])
    
    for group in groups:
        if not prev_state_df[prev_state_df['group'] == group].empty:
            # print("group to filter from previous timestep: ", group)
            G = rt_decision_graph[group].get_graph(mode="networkX")
            node_id = prev_state_df[prev_state_df['group'] == group]['decision'].iloc[0]
            neighbors = list(G.neighbors(node_id))
            
            # Collect neighbors with their mean rewards
            neighbors_with_mean_rewards = []
            for neighbor in neighbors:
                edge_data = G.get_edge_data(node_id, neighbor)
                mean_reward = edge_data.get('mean_reward', 0)
                neighbors_with_mean_rewards.append((neighbor, mean_reward))
            
            # Sort neighbors by mean reward in descending order
            neighbors_with_mean_rewards.sort(key=lambda x: x[1], reverse=True)
            
            # Get the top 3 neighbors (or fewer if less than 3)
            top_neighbors = neighbors_with_mean_rewards[:3]
            
            # Construct the actions with top 3 most probable actions and the current action
            actions = [x[0] for x in top_neighbors] + [curr_state_df[curr_state_df['group'] == group]['decision'].iloc[0]]
            
            # Filter all previous timesteps that have same State for the group and agent made the same decision
            conditioned_timesteps = set(history_df[(
                (history_df['timestep'].isin(common_timesteps)) &
                (history_df['group'] == group) & 
                (history_df['group_members'] == curr_state_df[curr_state_df['group'] == group]['group_members'].iloc[0]) & 
                (history_df['MSEUr'] == curr_state_df[curr_state_df['group'] == group]['MSEUr'].iloc[0]) & 
                (history_df['DTUr'] == curr_state_df[curr_state_df['group'] == group]['DTUr'].iloc[0]) & 
                (history_df['decision'].isin(actions))
            )]['timestep'])
            
            common_timesteps &= conditioned_timesteps
        else:
            # Filter timesteps in common_timesteps that have the group
            group_timesteps = set(history_df[(
                (history_df['timestep'].isin(common_timesteps)) &
                (history_df['group'] == group) & 
                (history_df['group_members'] == curr_state_df[curr_state_df['group'] == group]['group_members'].iloc[0]) & 
                (history_df['MSEUr'] == curr_state_df[curr_state_df['group'] == group]['MSEUr'].iloc[0]) & 
                (history_df['DTUr'] == curr_state_df[curr_state_df['group'] == group]['DTUr'].iloc[0])
            )]['timestep'])
            common_timesteps &= group_timesteps
        
        if len(common_timesteps) == 0:
            return False, None
    
    action_steered_timestep = history_df[history_df['timestep'].isin(common_timesteps)].groupby('timestep').first().reset_index().nlargest(1, 'reward')['timestep'].iloc[0]
    return history_df[history_df['timestep'] == action_steered_timestep]['sched_members'], history_df[history_df['timestep'] == action_steered_timestep]['reward'].iloc[0]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def do_action_steering_this_timestep_randomized(curr_state_df, history_df, rt_decision_graph, agent_expected_reward):
    """
    Perform action steering for the current timestep based on historical data and a randomized approach.
    Parameters:
    curr_state_df (pd.DataFrame): The current state dataframe containing information about the current timestep.
    history_df (pd.DataFrame): The historical dataframe containing information about previous timesteps.
    rt_decision_graph (dict): A dictionary of decision graphs for each group, where each graph is represented in networkX format.
    agent_expected_reward (float): The expected reward for the agent.
    Returns:
    tuple: A tuple containing:
        - sched_members (pd.Series): The scheduled members for the steered action timestep.
        - reward (float): The reward associated with the steered action timestep.
    If no suitable timestep is found, returns (False, None).
    """
    prev_state_df = history_df[history_df['timestep'] == history_df['timestep'].tail(1).iloc[0]]
    
    groups = curr_state_df['group'].unique()
    common_timesteps = set(history_df['timestep'])
    
    for group in groups:
        if not prev_state_df[prev_state_df['group'] == group].empty:
            G = rt_decision_graph[group].get_graph(mode="networkX")
            node_id = prev_state_df[prev_state_df['group'] == group]['decision'].iloc[0]
            neighbors = list(G.neighbors(node_id))
            
            # Collect neighbors with their mean rewards
            neighbors_with_mean_rewards = []
            for neighbor in neighbors:
                edge_data = G.get_edge_data(node_id, neighbor)
                mean_reward = edge_data.get('mean_reward', 0)
                neighbors_with_mean_rewards.append((neighbor, mean_reward))
            
            # Sort neighbors by mean reward in descending order
            neighbors_with_mean_rewards.sort(key=lambda x: x[1], reverse=True)
            
            # Get the top 3 neighbors (or fewer if less than 3)
            top_neighbors = neighbors_with_mean_rewards[:3]
            
            # Construct the actions with top 3 most probable actions and the current action
            actions = [x[0] for x in top_neighbors] + [curr_state_df[curr_state_df['group'] == group]['decision'].iloc[0]]
            
            # Filter all previous timesteps that have same State for the group and agent made the same decision
            conditioned_timesteps = set(history_df[(
                (history_df['timestep'].isin(common_timesteps)) &
                (history_df['group'] == group) & 
                (history_df['group_members'] == curr_state_df[curr_state_df['group'] == group]['group_members'].iloc[0]) & 
                (history_df['MSEUr'] == curr_state_df[curr_state_df['group'] == group]['MSEUr'].iloc[0]) & 
                (history_df['DTUr'] == curr_state_df[curr_state_df['group'] == group]['DTUr'].iloc[0]) & 
                (history_df['decision'].isin(actions))
            )]['timestep'])
            
            common_timesteps &= conditioned_timesteps
        else:
            # Filter timesteps in common_timesteps that have the group
            group_timesteps = set(history_df[(
                (history_df['timestep'].isin(common_timesteps)) &
                (history_df['group'] == group) & 
                (history_df['group_members'] == curr_state_df[curr_state_df['group'] == group]['group_members'].iloc[0]) & 
                (history_df['MSEUr'] == curr_state_df[curr_state_df['group'] == group]['MSEUr'].iloc[0]) & 
                (history_df['DTUr'] == curr_state_df[curr_state_df['group'] == group]['DTUr'].iloc[0])
            )]['timestep'])
            common_timesteps &= group_timesteps
        
        if len(common_timesteps) == 0:
            return False, None
    # Filter timesteps with better or equal reward than agent's expected reward
    better_timesteps = history_df[(history_df['timestep'].isin(common_timesteps)) & (history_df['reward'] >= agent_expected_reward)]
    
    if better_timesteps.empty:
        return False, None

    # Collect the actions and their rewards
    action_rewards = better_timesteps.groupby('decision', as_index=False)['reward'].mean()
    
    # Apply softmax to the rewards to calculate weights
    action_rewards['weight'] = softmax(action_rewards['reward'])
    
    # Randomly choose an action based on the weights
    chosen_action = random.choices(
        population=action_rewards['decision'].tolist(),
        weights=action_rewards['weight'].tolist(),
        k=1
    )[0]
       
    action_steered_timestep = better_timesteps[better_timesteps['decision'] == chosen_action].sort_values(by='reward', ascending=False).iloc[0]['timestep']   
    
    return history_df[history_df['timestep'] == action_steered_timestep]['sched_members'], history_df[history_df['timestep'] == action_steered_timestep]['reward'].iloc[0]


def transform_action(action, high=1, low=-1, tot_act=127):
    k = (high - low) / (tot_act - 1)
    return round((action - low) / k)


def process_buffer(buff, transform_action, sel_ue,  mode, timestep = 0, agent_type = 'SAC'):
    """
    Processes a buffer of transitions and returns two DataFrames: one for states and one for actions and rewards.
    Args:
        buff (list): A list of transitions, where each transition is a tuple (state, action).
        transform_action (function): A function to transform the action if the agent type is 'SAC'.
        sel_ue (function): A function to select the user equipment (UE) from the action.
        mode (str): The mode of processing, either 'buffer' or another mode.
        timestep (int, optional): The timestep to use if mode is not 'buffer'. Defaults to 0.
        agent_type (str, optional): The type of agent, defaults to 'SAC'.
    Returns:
        tuple: A tuple containing two DataFrames:
            - states_df (pd.DataFrame): DataFrame containing the processed states.
            - actions_rewards_df (pd.DataFrame): DataFrame containing the processed actions and rewards.
    """

    buff_state_columns = ["MSEUr0", "MSEUr1", "MSEUr2", "MSEUr3", "MSEUr4", "MSEUr5", "MSEUr6",
                 "DTUr0", "DTUr1", "DTUr2", "DTUr3", "DTUr4", "DTUr5", "DTUr6",
                 "UGUr0", "UGUr1", "UGUr2", "UGUr3", "UGUr4", "UGUr5", "UGUr6"]
    buff_states = []
    buff_actions_rewards = []

    for transition in buff:
        state, action = transition

        state_1d = state.flatten()
        buff_states.append(state_1d)

        action_reward = [action[0]]
        buff_actions_rewards.append(action_reward)

    states_df = pd.DataFrame(buff_states, columns=buff_state_columns)
 
    if mode == 'buffer':
        states_df['timestep'] = states_df.index + 1
    else:
        states_df['timestep'] = timestep
    cols = ['timestep'] + [col for col in states_df.columns if col != 'timestep']
    states_df = states_df[cols]
    actions_rewards_df = pd.DataFrame(buff_actions_rewards, columns=["action"])
    if agent_type == 'SAC':
        actions_rewards_df["action"] = actions_rewards_df["action"].apply(transform_action)
    actions_rewards_df["action"] = actions_rewards_df["action"].apply(lambda x: sel_ue(x)[0])
    if mode == 'buffer':
        actions_rewards_df['timestep'] = actions_rewards_df.index + 1
    else:
        actions_rewards_df['timestep'] = timestep
    cols = ['timestep'] + [col for col in actions_rewards_df.columns if col != 'timestep']
    actions_rewards_df = actions_rewards_df[cols]

    return states_df, actions_rewards_df