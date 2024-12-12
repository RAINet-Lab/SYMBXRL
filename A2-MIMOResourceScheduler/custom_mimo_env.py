'''
The following code is part of "SymbXRL: Symbolic Explainable Deep Reinforcement Learning for Mobile Networks" 
Copyright - RESILIENT AI NETWORK LAB, IMDEA NETWORKS

DISCLAIMER: THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
'''

########################### IMPORTS ######################################################
import numpy as np
import gymnasium as gym
import random
from collections import Counter
import numpy.matlib 
from itertools import combinations

########################### OPEN AI GYM CLASS DEFINATIONS ######################################################

class MimoEnv(gym.Env):
    def __init__(self, H, se_max):
        super(MimoEnv, self).__init__()
        """
        Initialize the 7 User MIMO environment.
        Args:
            H (numpy.ndarray): Channel matrix(CSI).
            se_max (numpy.ndarray): Maximum achievable spectral efficiency of 7 users.
        """
        self.H = H
        self.se_max = se_max
        self.num_ue = H.shape[2]
        self.current_step = 0
        self.total_steps = H.shape[0]
        ue_history = np.zeros((H.shape[2], ))
        self.ue_history = ue_history
        self.obs_state = []
        self.usrgrp_cntr = []
        action_space_size = 127  # 0 to 126, inclusive
        self.action_space = gym.spaces.Discrete(action_space_size)

        low = np.array([-np.inf, 0, 0] * 7)  # Min values for state variables
        high = np.array([np.inf, np.inf, 6] * 7)  # Max values for state variables
        self.observation_space = gym.spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float64)

        self.total_reward = None
        self.history = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        """
        Reset the environment to the initial state.
        Returns:
            numpy.ndarray: Initial observation/state.
            dict: Information about the environment.
        """
        self.current_step = 0
        self.total_reward = 0
        self.history = {}
        self.jfi = 0
        self.sys_se = 0
        group_idx = usr_group(np.squeeze(self.H[self.current_step,:,:]))
        self.usrgrp_cntr.append(group_idx)
        self.ue_history = np.zeros((7,))
        initial_state = np.concatenate((np.reshape(self.se_max[self.current_step,:],(1,self.num_ue)),np.reshape(self.ue_history,(1,self.num_ue)),np.reshape(group_idx,(1,-1))),axis = 1)
        # initial_state = initial_state.flatten()
        self.obs_state.append(initial_state)
        info = self.getinfo()     
        return initial_state, info
    
    def step(self, action):
        """
        Takes a step in the environment.
        Args:
            action: Action taken by the agent.
        Returns:
            numpy.ndarray: Next observation/state.
            float: Reward received for the action.
            bool: Whether the episode is finished.
            dict: Information about the environment.
        """
        ue_select, idx = sel_ue(action)
        mod_select = np.ones((idx,)) * 4
        ur_se_total, ur_min_snr, ur_se = data_process(np.reshape(self.H[self.current_step,:,ue_select],(64,-1)),idx,mod_select)
        reward, self.ue_history, jfi, sys_se = self.calculate_reward(ur_se_total, ur_min_snr, ur_se, ue_select, idx, self.usrgrp_cntr[self.current_step], self.se_max[self.current_step])
        self.jfi = jfi
        self.sys_se = sys_se
        self.total_reward += reward
        self.current_step += 1 
        done_pm = self.total_steps - 1
        done = self.current_step >= done_pm
        truncated = False
        group_idx = usr_group(np.squeeze(self.H[(self.current_step),:,:]))
        self.usrgrp_cntr.append(group_idx)
        next_state = np.concatenate((np.reshape(self.se_max[(self.current_step),:],(1,self.num_ue)),np.reshape(self.ue_history,(1,self.num_ue)),np.reshape(group_idx,(1,-1))),axis = 1)
        # next_state = next_state.flatten()
        self.obs_state.append(next_state)       
        info = self.getinfo()
        history = self.update_history(info)   
        return next_state, reward, done, truncated, info
    
    def get_reward(self, action):
        """
        Calculate the reward for the given action without advancing the environment.    
        Args:
            action: Action taken by the agent.
        Returns:
            float: Reward for the action.
        """
        # Saves the current state
        current_step = self.current_step
        ue_history = self.ue_history.copy()
        
        # Calculates reward
        ue_select, idx = sel_ue(action)
        mod_select = np.ones((idx,)) * 4
        ur_se_total, ur_min_snr, ur_se = data_process(np.reshape(self.H[current_step, :, ue_select], (64, -1)), idx, mod_select)
        reward, _, _, _ = self.calculate_reward(ur_se_total, ur_min_snr, ur_se, ue_select, idx, self.usrgrp_cntr[current_step], self.se_max[current_step], se_noise=True)
        
        # Restores the state
        self.current_step = current_step
        self.ue_history = ue_history
        
        return reward

    def set_state(self, state):
        """
        Set the environment to a specific state.
        Args:
            state (numpy.ndarray): State to set the environment to.
        """
        # Extracting the spectral efficiencies from the state
        spectral_efficiencies = state[:self.num_ue]        
        # Finding the row in se_max that matches the spectral efficiencies with some tolerance defined
        tolerance = 1
        # Finding the row index in se_max_ur that matches next_obs2 up to 2 decimal places
        row_index = -1
        for idx, row in enumerate(self.se_max):
            if np.all(np.isclose(row, spectral_efficiencies, atol=tolerance)):
                row_index = idx
                break
        if row_index == -1:
            raise ValueError("The provided state does not match any row in se_max.")
    
        self.current_step = row_index
        # Setting the ue_history based on the provided state
        self.ue_history = state[self.num_ue:self.num_ue*2]
        # Updating the group index history
        group_idx = state[self.num_ue*2:]
        if len(self.usrgrp_cntr) > self.current_step:
            self.usrgrp_cntr[self.current_step] = group_idx
        else:
            self.usrgrp_cntr.append(group_idx)
        
        # Creating the initial state
        initial_state = np.concatenate(
            (np.reshape(self.se_max[self.current_step, :], (1, self.num_ue)),
             np.reshape(self.ue_history, (1, self.num_ue)),
             np.reshape(group_idx, (1, -1))),
            axis=1
        )
        self.obs_state.append(initial_state)
        info = self.getinfo()
    
    def getinfo(self):
        """
        Get information about the environment.
        Returns:
            dict: Information about the environment.
        """
        return dict(current_step = self.current_step, NSSE = self.sys_se, JFI = self.jfi)
    
    def update_history(self, info):
        """
        Update the history of the environment.
        Args:
            info (dict): Information about the environment.
        """
        if not self.history:
            self.history = {key: [] for key in info.keys()}
        for key, value in info.items():
            self.history[key].append(value)
    
    def calculate_reward(self, ur_se_total, ur_min_snr, ur_se, ue_select, idx, usrgrp, semax, se_noise = False):
        """
        Calculate the reward based on the received spectral efficiency.
        Args:
            ur_se_total (float): Total spectral efficiency.
            ur_min_snr (float): Minimum signal-to-noise ratio.
            ur_se (numpy.ndarray): Spectral efficiency for each user.
            ue_select (int): Selected user index.
            idx (int): Number of selected users.
            usrgrp (int): User group index.
            semax (numpy.ndarray): Maximum achievable spectral efficiency.
        Returns:
            float: Calculated reward.
            numpy.ndarray: Updated user history.
        """
        beta = 0.5 # reward weight 
        bin_act = transform_input_to_output(ue_select, 7) # Converting Action to Binary Encoding
        usrgrp2 = usrgrp + 1
        sel = usrgrp2 * bin_act
        non_zero_elements = sel[sel != 0]
        ue_select = np.array(ue_select)
        sum_semax = np.sum(semax)
        Norm_Const = 1.15
        if se_noise:
            ur_se, ur_se_total = adjust_se_interfernce(non_zero_elements, ur_se, ur_se_total, usrgrp, ue_select)
        #Reward calculation
        ur_se_total = ur_se_total / (sum_semax*Norm_Const) # Normalizing due to Randomization
        for i in range(0,idx):
            self.ue_history[ue_select[i]] += ur_se[i]
        jfi = np.square((np.sum(self.ue_history))) / (7 * np.sum(np.square(self.ue_history)))
        reward  = round((beta*ur_se_total) + ((1-beta)*jfi), 3)
        return reward, self.ue_history, jfi, ur_se_total

    def render(self, mode = 'human'):
        pass

    def __call__(self):
        # Implement the __call__ method to make the class callable
        return self
        
    def close(self):
        pass
  

########################### IMPORTED FUNCTION BLOCKS ######################################################
'''
[reference] Use and modified code from https://github.com/qinganrice/SMART
[reference] Use and modified code from https://github.com/renew-wireless/RENEWLab
[reference] Qing An, Chris Dick, Santiago Segarra, Ashutosh Sabharwal, Rahman Doost-Mohammady, ``A Deep Reinforcement Learning-Based Resource Scheduler for Massive MIMO Networks'', arXiv:2303.00958, 2023
'''

def transform_input_to_output(input_sequence, total_variables):
    """
    Transform input action to binary coded output action.
    Args:
        input_sequence (list): Input sequence [1,3,4].
        total_variables (int): Total number of users 7.
    Returns:
        list: Output sequence  [0 1 0 1 1 0 0].
    """
    output_sequence = [0] * total_variables  # Initializing the output sequence with all zeros
    for index in input_sequence:
        if index < total_variables:
            output_sequence[index] = 1  # Setting the corresponding position to 1 if it's in the input sequence
    return output_sequence

def transform_array(arr):
    """
    Transform the array based on maximum occurrences.
    Args:
        arr (list): Input array.
    Returns:
        list: Transformed array.
    """
    counts = {}
    result = []
    for num in arr:
        if num != 0:
            counts[num] = counts.get(num, 0) + 1
    max_count = max(counts.values()) if counts else 0
    max_occurrence_numbers = [num for num, count in counts.items() if count == max_count]
    chosen_number = random.choice(max_occurrence_numbers) if max_occurrence_numbers else 0
    for num in arr:
        if num == chosen_number and num != 0:
            result.append(1)
        else:
            result.append(0)       
    return result

def get_selected_indices_and_values(arr):
    """
    Get selected indices and their values from the array.
    Args:
        arr (list): Input array.
    Returns:
        tuple: Number of selected indices and their values.
    """
    selected_indices = [i for i, num in enumerate(arr) if num != 0]
    return len(selected_indices), selected_indices


def count_occurrences(arr):
    """
    This function counts the maximum occurrence of a variable in an array.
    Args:
        arr: A list of integers.
    Returns:
        A tuple containing the variable with the maximum occurrence and its count.
    """
    counts = Counter(arr)
    max_value = max(counts.values())
    max_variable = [var for var, count in counts.items() if count == max_value]
    max_indexes = [i for i, x in enumerate(arr) if x in max_variable]
    return max_variable[0], max_value, max_indexes

def adjust_se_interfernce(non_zero_elements, ur_se, ur_se_total, usrgrp, ue_select):
    """
    Adjust the spectral efficiency based on the interference.
    Args:
        non_zero_elements (list): Non-zero elements.
        ur_se (numpy.ndarray): Spectral efficiency for each user.
        ur_se_total (float): Total spectral efficiency.
        usrgrp (int): User group index.
        ue_select (int): Selected user index.
    Returns:
        numpy.ndarray: Adjusted spectral efficiency.
        float: Adjusted total spectral efficiency.
    """
    intf_penalty = 0.5
    bonus_reward = [1.1, 1.2, 1.25]
    if np.any(non_zero_elements != non_zero_elements[0]):
        ur_se = ur_se * intf_penalty
        ur_se_total = ur_se_total * intf_penalty
    else:
        _,case,max_ind = count_occurrences(usrgrp)
        all_ind = np.arange(0,7)
        min_ind = np.setdiff1d(all_ind,max_ind)
        if case == 7:
            ur_se = ur_se
            ur_se_total = ur_se_total
        elif case == 6:
            if np.all(ue_select) in max_ind:
                ur_se[np.arange(0, len(ue_select))] = ur_se[np.arange(0, len(ue_select))]
            else:
                ur_se[np.arange(0, len(ue_select))] = ur_se[np.arange(0, len(ue_select))] * bonus_reward[2]
            ur_se_total = np.sum(ur_se)
        elif case == 5:
            if np.all(ue_select) in max_ind:
                ur_se[np.arange(0, len(ue_select))] = ur_se[np.arange(0, len(ue_select))] * bonus_reward[0]
            else:
                ur_se[np.arange(0, len(ue_select))] = ur_se[np.arange(0, len(ue_select))] * bonus_reward[2]
            ur_se_total = np.sum(ur_se)
        elif case == 4:
            if np.all(ue_select) in max_ind:
                ur_se[np.arange(0, len(ue_select))] = ur_se[np.arange(0, len(ue_select))] * bonus_reward[1]
            else:
                ur_se[np.arange(0, len(ue_select))] = ur_se[np.arange(0, len(ue_select))] * bonus_reward[1]
            ur_se_total = np.sum(ur_se)
        else:
            ur_se[np.arange(0, len(ue_select))] = ur_se[np.arange(0, len(ue_select))] * bonus_reward[2]
            ur_se_total = np.sum(ur_se)
    return ur_se, ur_se_total


def usr_group(H):
    """
    This function groups users based on the correlation of their channel vectors(CSI).
    Parameters:
    H (numpy.ndarray): A matrix of channel vectors where each column corresponds to a user and each row corresponds to a base station antenna.
    Returns:
    numpy.ndarray: An array where each element represents the group index of the corresponding user.
    """    
    N_UE = 7  # Number of user equipment (UE)
    num_bs = 64  # Number of base station antennas
    ur_group = [[] for i in range(N_UE)]  # Initialize a list of lists to store groups of users
    group_idx = np.zeros(N_UE)  # Initialize an array to store the group index of each user
    ur_group[0].append(0)  # Assign the first user to the first group
    N_group = 1  # Start with one group
    corr_h = 0.5  # Correlation threshold for grouping users
    meet_all = 0  # Flag to indicate if the user meets the correlation criteria for all users in a group
    assigned = 0  # Flag to indicate if the user has been assigned to a group
    # Loop over all users starting from the second user
    for i in range(1, N_UE):
        # Loop over all existing groups
        for j in range(N_group):
            # Loop over all users in the current group
            for k in ur_group[j]:
                # Compute the correlation between the current user's channel vector and the channel vector of the k-th user in the current group
                g_i = np.matrix(np.reshape(H[:, i], (num_bs, 1))).getH()  # Hermitian transpose of the current user's channel vector
                corr = abs(np.dot(g_i, np.reshape(H[:, k], (num_bs, 1)))) / (np.linalg.norm(np.reshape(H[:, i], (num_bs, 1))) * np.linalg.norm(np.reshape(H[:, k], (num_bs, 1))))
                if corr > corr_h:
                    # If the correlation is above the threshold, break and try the next group
                    break
                else:
                    if k == ur_group[j][-1]:
                        # If the current user meets the correlation criteria for all users in the group, set meet_all flag
                        meet_all = 1
                    continue
            if meet_all == 1:
                # If the user meets the correlation criteria for all users in the group, add the user to the group
                ur_group[j].append(i)
                meet_all = 0  # Reset the meet_all flag
                assigned = 1  # Set the assigned flag
                break
            else:
                continue
        if assigned == 0:
            # If the user was not assigned to any existing group, create a new group for the user
            ur_group[N_group].append(i)
            N_group += 1  # Increment the number of groups
        else:
            assigned = 0  # Reset the assigned flag
    # Assign group indices to each user
    for i in range(N_group):
        for j in ur_group[i]:
            group_idx[j] = i
            
    return group_idx



def data_process (H, N_UE, MOD_ORDER):
    """
    This function converts channel vectors to Spectral efficiency per user and SINR per user.
    Parameters:
    H (numpy.ndarray): A matrix of channel vectors where each column corresponds to a user and each row corresponds to a base station antenna.
    N_UE: No of Users
    MOD_ORDER: Modulation Order
    Returns:
    System Spectral Efficiency, SINR (all users), spectral effecincy (all users)
    """    
    # Waveform params
    N_OFDM_SYMS             = 24  # Number of OFDM symbols
    # MOD_ORDER               = 4  # Modulation order (2/4/16/64 = BSPK/QPSK/16-QAM/64-QAM)
    TX_SCALE                = 1.0 # Scale for Tdata waveform ([0:1])

    # OFDM params
    SC_IND_PILOTS           = np.array([7, 21, 43, 57])  # Pilot subcarrier indices
    SC_IND_DATA             = np.r_[1:7,8:21,22:27,38:43,44:57,58:64]  # Data subcarrier indices
    N_SC                    = 64           # Number of subcarriers
    # CP_LEN                  = 16          # Cyclic prefidata length
    N_DATA_SYMS             = N_OFDM_SYMS * len(SC_IND_DATA)     # Number of data symbols (one per data-bearing subcarrier per OFDM symbol)
    SAMP_FREQ               = 20e6

    # Massive-MIMO params
    # N_UE                    = 7
    N_BS_ANT                = 64  # N_BS_ANT >> N_UE
    # N_UPLINK_SYMBOLS        = N_OFDM_SYMS
    N_0                     = 1e-2
    H_var                   = 0.1

    # LTS for CFO and channel estimation
    lts_f = np.array([0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1])
    pilot_in_mat = np.zeros((N_UE, N_SC, N_UE))
    for i in range (0, N_UE):
        pilot_in_mat [i, :, i] = lts_f

    lts_f_mat = np.zeros((N_BS_ANT, N_SC, N_UE))
    for i in range (0, N_UE):
        lts_f_mat[:, :, i] = numpy.matlib.repmat(lts_f, N_BS_ANT, 1)

    ## Uplink
    # Generate a payload of random integers
    tx_ul_data = np.zeros((N_UE, N_DATA_SYMS),dtype='int')
    for n_ue in range (0,N_UE):
        tx_ul_data[n_ue,:] = np.random.randint(low = 0, high = MOD_ORDER[n_ue], size=(1, N_DATA_SYMS))
    # Map the data values on to complex symbols
    tx_ul_syms = np.zeros((N_UE, N_DATA_SYMS),dtype='complex')
    vec_mod = np.vectorize(modulation)
    for n_ue in range (0,N_UE):
        tx_ul_syms[n_ue,:] = vec_mod(MOD_ORDER[n_ue], tx_ul_data[n_ue,:])
    # Reshape the symbol vector to a matrix with one column per OFDM symbol
    tx_ul_syms_mat = np.reshape(tx_ul_syms, (N_UE, len(SC_IND_DATA), N_OFDM_SYMS))
    # Define the pilot tone values as BPSK symbols
    pt_pilots = np.transpose(np.array([[1, 1, -1, 1]]))
    # Repeat the pilots across all OFDM symbols
    pt_pilots_mat = np.zeros((N_UE, 4, N_OFDM_SYMS),dtype= 'complex')
    for i in range (0,N_UE):
        pt_pilots_mat[i,:,:] = numpy.matlib.repmat(pt_pilots, 1, N_OFDM_SYMS)
    ## IFFT
    # Construct the IFFT input matrix
    data_in_mat = np.zeros((N_UE, N_SC, N_OFDM_SYMS),dtype='complex')
    # Insert the data and pilot values; other subcarriers will remain at 0
    data_in_mat[:, SC_IND_DATA, :] = tx_ul_syms_mat
    data_in_mat[:, SC_IND_PILOTS, :] = pt_pilots_mat
    tx_mat_f = np.concatenate((pilot_in_mat, data_in_mat),axis=2)
    # Reshape to a vector
    tx_payload_vec = np.reshape(tx_mat_f, (N_UE, -1))
    # UL noise matrix
    Z_mat = np.sqrt(N_0/2) * ( np.random.random((N_BS_ANT,tx_payload_vec.shape[1])) + 1j*np.random.random((N_BS_ANT,tx_payload_vec.shape[1])))
    # H = np.sqrt(H_var/2) * ( np.random.random((N_BS_ANT, N_UE)) + 1j*np.random.random((N_BS_ANT, N_UE)))
    rx_payload_vec = np.matmul(H, tx_payload_vec) + Z_mat
    rx_mat_f = np.reshape(rx_payload_vec, (N_BS_ANT, N_SC, N_UE + N_OFDM_SYMS))
    
    csi_mat = np.multiply(rx_mat_f[:, :, 0:N_UE], lts_f_mat)
    fft_out_mat = rx_mat_f[:, :, N_UE:]
    # precoding_mat = np.zeros((N_BS_ANT, N_SC, N_UE),dtype='complex')
    demult_mat = np.zeros((N_UE, N_SC, N_OFDM_SYMS),dtype='complex')
    sc_csi_mat = np.zeros((N_BS_ANT, N_UE),dtype='complex')

    for j in range (0,N_SC):
        sc_csi_mat = csi_mat[:, j, :]
        zf_mat = np.linalg.pinv(sc_csi_mat)   # ZF
        demult_mat[:, j, :] = np.matmul(zf_mat, np.squeeze(fft_out_mat[:, j, :]))

    payload_syms_mat = demult_mat[:, SC_IND_DATA, :]
    payload_syms_mat = np.reshape(payload_syms_mat, (N_UE, -1))

    tx_ul_syms_vecs = np.reshape(tx_ul_syms_mat, (N_UE, -1))
    ul_evm_mat = np.mean(np.square(np.abs(payload_syms_mat - tx_ul_syms_vecs)),1) / np.mean(np.square(np.abs(tx_ul_syms_vecs)),1)
    ul_sinrs = 1 / ul_evm_mat

    ## Spectrual Efficiency
    ul_se = np.zeros(N_UE)
    for n_ue in range (0,N_UE):
        ul_se[n_ue] = np.log2(1+ul_sinrs[n_ue])
    ul_se_total = np.sum(ul_se)

    return ul_se_total, ul_sinrs, ul_se


def modulation (mod_order,data):
    '''
    Sub Functions of Previous Main Function - Data Process
    '''
    modvec_bpsk   =  (1/np.sqrt(2))  * np.array([-1, 1]) # and QPSK
    modvec_16qam  =  (1/np.sqrt(10)) * np.array([-3, -1, +3, +1])
    modvec_64qam  =  (1/np.sqrt(43)) * np.array([-7, -5, -1, -3, +7, +5, +1, +3])
    
    if (mod_order == 2): #BPSK
        return complex(modvec_bpsk[data],0) # data = 0/1
    elif (mod_order == 4): #QPSK
        return complex(modvec_bpsk[data>>1],modvec_bpsk[np.mod(data,2)])
    elif (mod_order == 16): #16-QAM
        return complex(modvec_16qam[data>>2],modvec_16qam[np.mod(data,4)])
    elif (mod_order == 64): #64-QAM
        return complex(modvec_64qam[data>>3],modvec_64qam[np.mod(data,8)])

def demodulation (mod_order, data):
    if (mod_order == 2): #BPSK
        return float(np.real(data)>0) # data = 0/1
    elif (mod_order == 4): #QPSK
        return float(2*(np.real(data)>0) + 1*(np.imag(data)>0))
    elif (mod_order == 16): #16-QAM
        return float((8*(np.real(data)>0)) + (4*(abs(np.real(data))<0.6325)) + (2*(np.imag(data)>0)) + (1*(abs(np.imag(data))<0.6325)))
    elif (mod_order == 64): #64-QAM
        return float((32*(np.real(data)>0)) + (16*(abs(np.real(data))<0.6172)) + (8*((abs(np.real(data))<(0.9258))and((abs(np.real(data))>(0.3086))))) + (4*(np.imag(data)>0)) + (2*(abs(np.imag(data))<0.6172)) + (1*((abs(np.imag(data))<(0.9258))and((abs(np.imag(data))>(0.3086))))))


def sel_ue(action):
    '''
    Converting Action into User Indexed Action
    '''
    user_set = [0,1,2,3,4,5,6]
    sum_before = 0
    # ue_select = []
    # idx = 0
    for i in range (1,8):
        sum_before += len(list(combinations(user_set, i)))
        if ((action+1)>sum_before):
            continue
        else:
            idx = i
            sum_before -= len(list(combinations(user_set, i)))
            ue_select = list(combinations(user_set, i))[action-sum_before]
            break
    return ue_select,idx

def reverse_sel_ue(ue_select):
    '''
    Reversing User Indexed action to system action
    '''
    user_set = [0,1,2,3,4,5,6]
    idx = len(ue_select)
    action = 0
    for i in range(1, idx):
        action += len(list(combinations(user_set, i)))
    comb_list = list(combinations(user_set, idx))
    position = comb_list.index(tuple(ue_select))
    action += position
    return action