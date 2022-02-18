import numpy as np

import torch as th
from torch import nn
import gym
import minerl
from tqdm.notebook import tqdm
from colabgymrender.recorder import Recorder
from pyvirtualdisplay import Display
import logging
import cv2
import pickle
import torch
import random

from torch import nn
import gym
import minerl
from tqdm.notebook import tqdm
from colabgymrender.recorder import Recorder
from pyvirtualdisplay import Display
import logging
import cv2
import pickle
import torch
import time
import math
import torchvision
from collections import Counter
import os
from hummingbird.ml import convert
import matplotlib.pyplot as plt
import copy
from torch.autograd import Variable
import sys
from sklearn.cluster import KMeans

abc =['observation$inventory$coal',
'observation$inventory$cobblestone',
'observation$inventory$crafting_table',
'observation$inventory$dirt',
'observation$inventory$furnace',
'observation$inventory$iron_axe',
'observation$inventory$iron_ingot',
'observation$inventory$iron_ore',
'observation$inventory$iron_pickaxe',
'observation$inventory$log',
'observation$inventory$planks',
'observation$inventory$stick',
'observation$inventory$stone',
'observation$inventory$stone_axe',
'observation$inventory$stone_pickaxe',
'observation$inventory$torch',
'observation$inventory$wooden_axe',
'observation$inventory$wooden_pickaxe',
'observation$equipped_items.mainhand.damage',
 'observation$equipped_items.mainhand.maxDamage',
 'observation$equipped_items.mainhand.type']
abcd = ['action$forward', 'action$left', 'action$back','action$right',
            'action$jump', 'action$sneak','action$sprint','action$attack',
            'action$camera', 'action$place', 'action$equip', 'action$craft',
            'action$nearbyCraft', 'action$nearbySmelt']

key_as = ['action$place', 'action$equip', 'action$craft',
            'action$nearbyCraft', 'action$nearbySmelt']
key_as_v = [[ 'none','cobblestone', 'crafting_table', 'dirt', 'furnace', 'stone', 'torch'],
            ['none', 'iron_axe', 'iron_pickaxe', 'stone_axe', 'stone_pickaxe', 'wooden_axe',
             'wooden_pickaxe'],
            ['none', 'crafting_table', 'planks', 'stick', 'torch'],
            ['none','furnace', 'iron_axe', 'iron_pickaxe' , 'stone_axe' ,'stone_pickaxe',
             'wooden_axe' ,'wooden_pickaxe'],
            ['none', 'coal', 'iron_ingot']]
tmp = copy.deepcopy(key_as_v)
counter = 0
key_as_v = {}
for k in key_as:
    key_as_v[k] = tmp[counter]
    counter +=1

DATA_DIR = "data"
EPOCHS = 2
LEARNING_RATE = 0.0001
BATCH_SIZE = 128
DATA_SAMPLES = 1000000

max_last_action = 15
max_his = 20
action_shape_his = 18
max_action_top = 80
index_1 = np.flip(np.arange(0, int(max_his / 2), 1))
index_2 = index_1 * 15
main_all_i = np.concatenate((index_1, index_2), axis=0)
with open('others.pkl', 'rb') as f:
    others = pickle.load(f)
class NatureCNN(nn.Module):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param input_shape: A three-item tuple telling image dimensions in (C, H, W)
    :param output_dim: Dimensionality of the output vector
    """

    def __init__(self, input_shape, output_dim, hidden_size=64):
        super().__init__()
        n_input_channels = input_shape[0]
        self.hidden_size = hidden_size
        self.layer_dim = 2
        self.hidden_dim = 100
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.zeros(1, *input_shape)).shape[1]

        self.linear_stack = nn.Sequential(
            nn.Linear((max_action_top + 16) * max_last_action, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
        )
        self.rnn_s = nn.LSTM(n_flatten + 512, self.hidden_dim, self.layer_dim, batch_first=True)


        self.rnn = nn.LSTM(n_flatten + 512, self.hidden_dim, self.layer_dim, batch_first=True)

        # self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim + self.hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, observations: th.Tensor, data: th.Tensor) -> th.Tensor:
        batch_size, timesteps, C, H, W = observations.size()
        c_in = observations.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        data_out = self.linear_stack(data)
        r_in_cnn = c_out.view(batch_size, timesteps, -1)
        r_in_data = data_out.view(batch_size, timesteps, -1)

        r_in = torch.cat((r_in_cnn[:, int(max_his / 2):], r_in_data[:, int(max_his / 2):]), dim=-1)
        h0 = torch.zeros(self.layer_dim, observations.size(0), self.hidden_dim).requires_grad_().cuda()
        c0 = torch.zeros(self.layer_dim, observations.size(0), self.hidden_dim).requires_grad_().cuda()
        out_s, (hn, cn) = self.rnn_s(r_in, (h0.detach(), c0.detach()))


        r_in = torch.cat((r_in_cnn[:, :int(max_his / 2)], r_in_data[:,:int(max_his / 2)]), dim=-1)
        h0 = torch.zeros(self.layer_dim, observations.size(0), self.hidden_dim).requires_grad_().cuda()
        c0 = torch.zeros(self.layer_dim, observations.size(0), self.hidden_dim).requires_grad_().cuda()
        out, (hn, cn) = self.rnn(r_in, (h0.detach(), c0.detach()))

        # out = self.fc(out[:,-1,:])


        out_in = torch.cat((out[:, -1, :], out_s[:, -1, :]), dim=-1)
        out = self.fc(out_in)

        return out

    def initHidden(self, BATCH):
        return (Variable(torch.zeros(self.layer_dim, BATCH, self.hidden_dim)).cuda(),
                Variable(torch.zeros(self.layer_dim, BATCH, self.hidden_dim)).cuda())

with open('action_connect_to_vector.pkl', 'rb') as f:
    action_vector_all = pickle.load(f)
action_vector_all.keys()
data = minerl.data.make("MineRLObtainDiamondVectorObf-v0", data_dir='data', num_workers=4)
trajectory_names = data.get_trajectory_names()
random.shuffle(trajectory_names)

print("Loading data")

all_a = []

vector_obs = None
action_obs = None
R_S = None
key_as_vetor = {}
for k in abcd:
    key_as_vetor[k] = None
for trajectory_name in trajectory_names:
    obs_vector = np.load(os.path.join(DATA_DIR, 'MineRLObtainDiamondVectorObf-v0', trajectory_name, 'rendered.npz'))
    obs_non = np.load(os.path.join(DATA_DIR, 'MineRLObtainDiamond-v0', trajectory_name, 'rendered.npz'))
    a_v = np.array(obs_vector['action$vector'])
    r = np.array(obs_vector['reward'])

    os_v = np.array(obs_vector['observation$vector'])
    if action_obs is None:
        action_obs = np.array(a_v)
        R_S = np.array(r)
        vector_obs = np.array(os_v)
        for k in abcd:
            key_as_vetor[k] = obs_non[k]

    else:

        action_obs = np.concatenate((action_obs,
                                     np.array(a_v)), axis=0)
        vector_obs = np.concatenate((vector_obs,
                                     np.array(obs_vector['observation$vector'])), axis=0)
        R_S = np.concatenate((R_S,r), axis=0)
        for k in abcd:
            key_as_vetor[k] = np.concatenate((key_as_vetor[k],
                                     np.array(obs_non[k])), axis=0)

key_as_vetor['c1'] = key_as_vetor['action$camera'][:,0]
key_as_vetor['c2'] = key_as_vetor['action$camera'][:,1]
del key_as_vetor['action$camera']
counter = 0
for k in key_as:
    value = key_as_v[k]
    counter_2 = 0
    for k_2 in value:
        index = np.where(key_as_vetor[k] == k_2)[0]
        key_as_vetor[k][index] = counter_2
        counter_2 += 1

    key_as_vetor[k] = np.array(key_as_vetor[k], dtype=np.uint8)
    counter += 1
# a = key_as_vetor['action$craft']
# np.unique(a)
# for k in abcd:
#     print(len(key_as_vetor[k]))

index = np.where(((R_S != 0) &
                  (R_S != 1) &
                  (R_S != 1024) &
                  (R_S != 64)) &
                  (R_S != 16))[0]
action_key = action_obs[index, :]

action_key = np.unique(action_key, axis=0)

counter_a= []
for a in action_key:
    counter_a.append(np.sum(np.all(action_obs == a, axis=1)))

action_key = action_key[np.flip(np.argsort(counter_a))]
counter_a = np.flip(np.sort(counter_a))
action_0_0 = action_key[1]

index_2 = None
len(action_key)
for i in range(len(action_key) - 2):
    a = action_key[i + 2]
    if index_2 is None:
        index_2 = np.all(action_obs == a, axis=1)
    else:
        index_2 = index_2 | np.all(action_obs == a, axis=1)
assert np.sum(index_2) == np.sum(counter_a[2:])
action_obs = action_obs[~index_2]

for k in key_as_vetor.keys():
    key_as_vetor[k] = key_as_vetor[k][~index_2]
full_a = []

d_1 = np.sum((action_0_0 - action_obs) ** 2, axis=1)
index_tmp = np.where((0.095 > d_1) & (d_1 > 0.059))[0]
len(index_tmp)
len(index_tmp)/len(d_1)
full_2 = np.where(0.095 <= d_1)[0]
full_a.append([action_0_0,  np.where( 0.059 >= d_1)[0]])

c_t = copy.deepcopy(key_as_vetor)
for k in c_t.keys():
    c_t[k] = c_t[k][index_tmp]
# a = c_t['action$nearbySmelt']
# np.unique(a)
index_2 = None
for k in c_t.keys():
    if k != 'action$camera':
        defl = 0
        if k in ['action$forward', 'action$attack']:
            defl = 0
        if index_2 is None:
            index_2 = c_t[k] == defl
        else:
            index_2 = index_2 & (c_t[k] == defl)
np.sum(index_2)/len(index_tmp)

sum_all = None
for k in c_t.keys():
    if k not in ['c1', 'c2']:
        if sum_all is None:
            sum_all = c_t[k][:,None]
        else:
            sum_all = np.concatenate((sum_all, c_t[k][:,None]),axis=1)
sum_x = np.sum(sum_all, axis=1)
np.max(sum_x)

full_1 = action_obs[index_tmp]
index_main = copy.deepcopy(index_tmp)
stack_zero_action = []
for i in range(9):
    unipue_a = np.unique(full_1, axis=0, return_counts=True, return_index=True)
    counter_a = np.flip(np.sort(unipue_a[2]))
    print(counter_a[0])
    index_sort = np.flip(np.argsort(unipue_a[2]))
    main_index = unipue_a[1][index_sort]
    unipue_a = unipue_a[0][np.flip(np.argsort(unipue_a[2]))]
    main_action = unipue_a[0]
    stack_zero_action.append(main_action)
    index = main_index[0]
    key_fail = None
    for k in c_t.keys():
        if k not in ['action$camera','c1','c2']:
            if c_t[k][index] != 0:
                print(k)
                key_fail = k
                break
    d_1 = np.sum((main_action - full_1) ** 2, axis=1)
    main_condition = d_1 > 0.04
    index_2 = None
    for k in c_t.keys():
        if k not in ['action$camera', 'c1', 'c2']:
            defl = 0
            if k == key_fail:
                defl = 1
            if index_2 is None:
                index_2 = c_t[k][~main_condition] == defl
            else:
                index_2 = index_2 & (c_t[k][~main_condition] == defl)
    print(np.sum(index_2) / len(d_1))

    a = c_t['action$place'][~main_condition] == 0
    np.unique(a)
    full_a.append([action_0_0, full_1[np.where(~main_condition)[0]]])

    index_tmp2 = np.where(main_condition)[0]
    full_1 = full_1[index_tmp2]
    index_main = index_main[index_tmp2]
    print(full_1.shape)
    for k in c_t.keys():
        c_t[k] = c_t[k][index_tmp2]

len(index_main)
index_tmp2 = np.sort(np.concatenate((index_main, full_2)))
index_tmp2 = np.sort(np.concatenate((copy_i2, i_non)))

full_1 = action_obs[index_tmp2]
c_t = copy.deepcopy(key_as_vetor)
for k in c_t.keys():
    c_t[k] = c_t[k][index_tmp2]
full_1.shape

#
# index_tmp2 = np.sort(np.concatenate((copy_i2, i_non)))
#
# len(index_tmp2)
# full_1 = full_1[index_tmp2]
# for k in c_t.keys():
#     c_t[k] = c_t[k][index_tmp2]

len(index_tmp2)
non_take = []
for main_action in stack_zero_action:
    select_index = 1
    main_action = stack_zero_action[select_index]
    d_1 = np.sum((main_action - full_1) ** 2, axis=1)
    main_condition = (d_1 > 0.04) & (d_1 < 0.12)
    i_non = index_tmp2[d_1 >= 0.12]

    # main_condition = (d_1 > 0.12) & (d_1 < 0.14)
    # i_non = index_tmp2[d_1 >= 0.16]

    copy_i2 = index_tmp2[main_condition]

    index_tmp2 = np.where(main_condition)[0]


    full_1 = full_1[index_tmp2]
    print(full_1.shape)
    for k in c_t.keys():
        c_t[k] = c_t[k][index_tmp2]
    for _ in range(7):
        unipue_a = np.unique(full_1, axis=0, return_counts=True, return_index=True)
        counter_a = np.flip(np.sort(unipue_a[2]))
        print(counter_a[0])
        index_sort = np.flip(np.argsort(unipue_a[2]))
        main_index = unipue_a[1][index_sort]
        unipue_a = unipue_a[0][np.flip(np.argsort(unipue_a[2]))]
        main_action = unipue_a[0]
        stack_zero_action.append(main_action)
        index = main_index[0]
        counter__ = 0
        for i in range(len(stack_zero_action)):
            if i != select_index:
                i = stack_zero_action[i]
                print(np.sum((i - main_action) ** 2))

        key_fail = {}
        for k in c_t.keys():
            if k not in ['action$camera', 'c1', 'c2']:
                if c_t[k][index] != 0:
                    key_fail[k] = c_t[k][index]

                    if k in key_as:
                        print(key_as_v[k][c_t[k][index]])
                        # assert False
                    # print(key_as_v[k][c_t[k][index]])
        print(key_fail)
        d_1 = np.sum((main_action - full_1) ** 2, axis=1)
        main_condition = d_1 > 0.04
        print(np.sum(~main_condition) / len(d_1))
        index_2 = None
        for k in c_t.keys():
            if k not in ['action$camera', 'c1', 'c2']:
                defl = 0
                if k in key_fail:
                    defl = key_fail[k]
                if index_2 is None:
                    index_2 = c_t[k] == defl
                else:
                    index_2 = index_2 & (c_t[k] == defl)
        print(np.sum(index_2) / len(d_1))
        index_tmp2 = np.where(main_condition)[0]
        full_1 = full_1[index_tmp2]
        copy_i2 = copy_i2[index_tmp2]
        # index_main_2 = index_main_2[index_tmp2]
        print(full_1.shape)
        for k in c_t.keys():
            c_t[k] = c_t[k][index_tmp2]
    # for k in c_t.keys():
    #     if k not in ['action$camera','c1','c2']:
    #         if c_t[k][index] != 0:
    #             print(k)
    #             key_fail = k

    sum_all = None
    for k in c_t.keys():
        if k not in ['c1', 'c2']:
            if sum_all is None:
                sum_all = c_t[k][:, None]
            else:
                sum_all = np.concatenate((sum_all, c_t[k][:, None]), axis=1)
    sum_x = np.sum(sum_all, axis=1)
    print(np.max(sum_x))
len(stack_zero_action)
len(stack_new)
# stack_zero_action = stack_zero_action[:9]
stack_new = []
stack_new_dict = []
key_sima = {}
for i in range(80):
    unipue_a = np.unique(full_1, axis=0, return_counts=True, return_index=True)
    counter_a = np.flip(np.sort(unipue_a[2]))
    # print(counter_a[0])
    index_sort = np.flip(np.argsort(unipue_a[2]))
    main_index = unipue_a[1][index_sort]
    unipue_a = unipue_a[0][np.flip(np.argsort(unipue_a[2]))]
    main_action = unipue_a[0]
    x = np.sum((main_action - stack_zero_action) ** 2, axis=1)
    # print(np.max(x), np.min(x))
    index = main_index[0]

    if np.min(x) >= 0.10:
        check = False
        key_fail = {}
        for k in c_t.keys():
            if k not in ['action$camera','c1','c2']:
                if c_t[k][index] != 0:
                    key_fail[k] = c_t[k][index]

                    if k in key_as:
                        # print(key_as_v[k][c_t[k][index]])
                        check = True
                        # assert False
                    # print(key_as_v[k][c_t[k][index]])
        if len(stack_new) != 0:
            x2 = np.sum((main_action - stack_new) ** 2, axis=1)
            if np.min(x2) > 0.075:
                if check:
                    print(key_fail, stack_new_dict[np.argmin(x2)], np.min(x2))
                stack_new.append(main_action)
                stack_new_dict.append(key_fail)
            else:
                if np.argmin(x2) not in key_sima:
                    key_sima[np.argmin(x2)] = []
                key_sima[np.argmin(x2)].append(main_action)
                if check:
                    print('NONE ',key_fail, stack_new_dict[np.argmin(x2)], np.min(x2))
        else:
            stack_new.append(main_action)
            stack_new_dict.append(key_fail)
    d_1 = np.sum((main_action - full_1) ** 2, axis=1)
    main_condition = d_1 > 0.04
    # print(np.sum(~main_condition)/ len(d_1))
    index_2 = None
    for k in c_t.keys():
        if k not in ['action$camera', 'c1', 'c2']:
            defl = 0
            if k in key_fail:
                defl = key_fail[k]
            if index_2 is None:
                index_2 = c_t[k] == defl
            else:
                index_2 = index_2 & (c_t[k] == defl)
    # print(np.sum(index_2) / len(d_1))

    a = c_t['action$place'][~main_condition] == 0
    np.unique(a)
    index_tmp2 = np.where(main_condition)[0]
    full_1 = full_1[index_tmp2]
    # index_main_2 = index_main_2[index_tmp2]
    for k in c_t.keys():
        c_t[k] = c_t[k][index_tmp2]

    # print('---DONE---')

    len(main_action)

# len(key_sima)

for i in range(len(action_key) - 2):
    stack_new.append(action_key[i + 2])

len(stack_new)