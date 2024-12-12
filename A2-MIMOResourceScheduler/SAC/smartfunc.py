'''
The following code is part of "SymbXRL: Symbolic Explainable Deep Reinforcement Learning for Mobile Networks" 
[reference] Use and modified code from https://github.com/jimkon/Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces
[reference] Use and modified code from https://github.com/qinganrice/SMART
[reference] Use and modified code from https://github.com/renew-wireless/RENEWLab
[reference] Qing An, Chris Dick, Santiago Segarra, Ashutosh Sabharwal, Rahman Doost-Mohammady, ``A Deep Reinforcement Learning-Based Resource Scheduler for Massive MIMO Networks'', arXiv:2303.00958, 2023


DISCLAIMER: THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
'''

import numpy as np
import numpy.matlib 
from itertools import combinations
import itertools
import torch
import math

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

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

