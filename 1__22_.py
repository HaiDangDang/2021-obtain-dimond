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


index = np.where(((R_S != 0) &
                  (R_S != 1) &
                  (R_S != 1024) &
                  (R_S != 64)) &
                  (R_S != 16))[0]


action_key = action_obs[index, :]

action_key = np.unique(action_key, axis=0)

counter_a = []
for a in action_key:
    counter_a.append(np.sum(np.all(action_obs == a, axis=1)))

action_key = action_key[np.flip(np.argsort(counter_a))]
counter_a = np.flip(np.sort(counter_a))
action_0_0 = action_key[1]

index_2 = None
for i in range(len(action_key) - 2):
    a = action_key[i + 2]
    if index_2 is None:
        index_2 = np.all(action_obs == a, axis=1)
    else:
        index_2 = index_2 | np.all(action_obs == a, axis=1)
assert np.sum(index_2) == np.sum(counter_a[2:])

action_obs = action_obs[~index_2]
big_index = np.arange(0, len(action_obs))

for k in key_as_vetor.keys():
    key_as_vetor[k] = key_as_vetor[k][~index_2]
full_a = []

d_1 = np.sum((action_0_0 - action_obs) ** 2, axis=1)
index_tmp = np.where((0.095 > d_1) & (d_1 > 0.059))[0]
full_2 = np.where(0.095 <= d_1)[0]

full_a.append([action_0_0,  np.where((0.059 >= d_1) & (0.0 < d_1))[0]])
print(len(np.where((0.059 >= d_1) & (0.0 < d_1))[0]))

c_t = copy.deepcopy(key_as_vetor)
for k in c_t.keys():
    c_t[k] = c_t[k][index_tmp]
np.sum(key_as_vetor['action$jump'])


full_1 = action_obs[index_tmp]
big_index = np.arange(0, len(action_obs))[index_tmp]

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
    full_a.append([main_action, big_index[np.where(~main_condition)[0]]])

    index_tmp2 = np.where(main_condition)[0]
    full_1 = full_1[index_tmp2]
    big_index = big_index[index_tmp2]

    index_main = index_main[index_tmp2]
    print(full_1.shape)
    for k in c_t.keys():
        c_t[k] = c_t[k][index_tmp2]

tmp = copy.deepcopy(full_a[7])
full_a[7] = full_a[3]
full_a[3] = tmp
# len(full_a[3][1])

index_tmp2 = np.sort(np.concatenate((index_main, full_2)))

full_1 = action_obs[index_tmp2]
big_index = np.arange(0, len(action_obs))[index_tmp2]

c_t = copy.deepcopy(key_as_vetor)
for k in c_t.keys():
    c_t[k] = c_t[k][index_tmp2]
i_non = None
copy_i2 = None
for select_index in [0,1]:
    if i_non is not None:
        index_tmp2 = np.sort(np.concatenate((copy_i2, i_non)))
        full_1 = action_obs[index_tmp2]
        big_index = np.arange(0, len(action_obs))[index_tmp2]

        c_t = copy.deepcopy(key_as_vetor)
        for k in c_t.keys():
            c_t[k] = c_t[k][index_tmp2]

    main_action = stack_zero_action[select_index]
    d_1 = np.sum((main_action - full_1) ** 2, axis=1)
    main_condition = (d_1 > 0.04) & (d_1 < 0.12)
    i_non = index_tmp2[d_1 >= 0.12]

    copy_i2 = index_tmp2[main_condition]
    index_tmp2 = np.where(main_condition)[0]

    full_1 = full_1[index_tmp2]
    big_index = big_index[index_tmp2]

    for k in c_t.keys():
        c_t[k] = c_t[k][index_tmp2]

    for _ in range(7):
        unipue_a = np.unique(full_1, axis=0, return_counts=True, return_index=True)
        counter_a = np.flip(np.sort(unipue_a[2]))

        index_sort = np.flip(np.argsort(unipue_a[2]))
        main_index = unipue_a[1][index_sort]
        unipue_a = unipue_a[0][np.flip(np.argsort(unipue_a[2]))]
        main_action = unipue_a[0]

        delta = np.sum((stack_zero_action[:9] - main_action) ** 2,axis=1)
        delta[select_index] = 10
        loc = np.argmin(delta)
        print(loc + 1)
        stack_zero_action.append(main_action)
        index = main_index[0]
        d_1 = np.sum((main_action - full_1) ** 2, axis=1)
        main_condition = d_1 > 0.04

        # full_a.append([main_action, big_index[np.where(~main_condition)[0]]])
        full_a[loc+1][1] = np.concatenate((full_a[loc+1][1], big_index[np.where(~main_condition)[0]]))
        index_tmp2 = np.where(main_condition)[0]
        full_1 = full_1[index_tmp2]
        big_index = big_index[index_tmp2]

        copy_i2 = copy_i2[index_tmp2]
        for k in c_t.keys():
            c_t[k] = c_t[k][index_tmp2]

len(full_a)
x = None
for a in full_a:
    if x is None:
        x = a[1]
    else:
        x = np.concatenate((x,a[1]))
x.shape
x = np.unique(x)
x.shape
len(action_obs) - x.shape[0]
# len(full_a)
index_tmp2 = np.sort(np.concatenate((copy_i2, i_non)))
full_1 = action_obs[index_tmp2]
big_index = np.arange(0, len(action_obs))[index_tmp2]

c_t = copy.deepcopy(key_as_vetor)
for k in c_t.keys():
    c_t[k] = c_t[k][index_tmp2]


stack_new = []
stack_new_dict = []
key_sima = {}
len(stack_zero_action)
for i in range(120):
    unipue_a = np.unique(full_1, axis=0, return_counts=True, return_index=True)
    counter_a = np.flip(np.sort(unipue_a[2]))
    print(counter_a[0])
    index_sort = np.flip(np.argsort(unipue_a[2]))
    main_index = unipue_a[1][index_sort]
    unipue_a = unipue_a[0][np.flip(np.argsort(unipue_a[2]))]
    main_action = unipue_a[0]
    x = np.sum((main_action - stack_zero_action) ** 2, axis=1)
    # print(np.max(x), np.min(x))
    index = main_index[0]
    key_fail = {}

    d_1 = np.sum((main_action - full_1) ** 2, axis=1)
    main_condition = d_1 > 0.04

    if np.min(x) >= 0.10:
        check = False
        for k in c_t.keys():
            if k not in ['action$camera', 'c1', 'c2']:
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
    else:
        next_actions = []
        for i2 in range(7):
            next_actions.append(full_a[i2][0])
        delta = np.sum((main_action - next_actions) ** 2, axis=1)
        loc = np.argmin(delta)
        full_a[loc][1] = np.concatenate((full_a[loc][1], big_index[np.where(~main_condition)[0]]))

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
    print(full_1.shape)
    # index_main_2 = index_main_2[index_tmp2]
    for k in c_t.keys():
        c_t[k] = c_t[k][index_tmp2]

    # print('---DONE---')

    len(main_action)

len(stack_new)
for k in key_sima:
    stack_new[k] = [ stack_new[k]]

    for x in key_sima[k]:
        stack_new[k].append(x)
for i in range(len(action_key) - 2):
    stack_new.append(action_key[i + 2])

len(full_a)

stack_all_final = []
n_cluster = 14
counter = 0
for i in full_a:
    # counter = 7
    #
    # i = full_a[counter]
    if counter != 9:
        as_ = action_obs[i[1]]
        print(len(as_))
        if counter == 1:
            n_cluster = 16
        elif counter == 3:
            n_cluster = 14
        elif counter in [8,7]:
            n_cluster = 4
        elif counter in [4,5,6]:
            n_cluster = 8

        kmeans = KMeans(n_clusters=n_cluster)
        kmeans.fit(as_)
        action_centroids = kmeans.cluster_centers_
        c_l = []
        for a in action_centroids:
            print("---1---")
            d_1 = np.sum((a - as_) ** 2, axis=1)
            index = np.argmin(d_1)
            c_l.append(as_[index])
            for k in c_t.keys():
                if key_as_vetor[k][i[1]][index] != 0:
                    print(k, key_as_vetor[k][i[1]][index])
            print("---2---")

        print("END")
        stack_all_final.append(c_l)
    else:
        stack_all_final.append(i[0])

    counter += 1


# with open(f'stack_all_knn.pkl', 'wb') as fout:
#     pickle.dump(stack_all_final, fout)
#
# with open(f'stack_new.pkl', 'wb') as fout:
#     pickle.dump(stack_new, fout)

with open('stack_all_knn.pkl', 'rb') as f:
    stack_all_final = pickle.load(f)

with open('stack_new.pkl', 'rb') as f:
    stack_new = pickle.load(f)

max_action_top = len(stack_new)
all_a = []
for x in stack_all_final:
    if len(x) != 64:
        max_action_top += len(x)
        for a in x:
            all_a.append(a)
    else:
        max_action_top += 1
        all_a.append(x)

other_a = []
his = []
for x in stack_new:
    if len(x) != 64:
        all_a.append(x[0])
        for a in x:
            other_a.append(a)
            his.append(len(all_a) -1)
    else:
        all_a.append(x)
other_a = np.array(other_a)
all_a = np.array(all_a)
# all_a.shape

simalar = []
counter = 0
for a in all_a:
    d_1 = np.sum((a - all_a) ** 2, axis=1)
    print(np.round(np.sort(d_1)[1], 4))
    for j in range(len(d_1)):
        if j != counter:
            if d_1[j] <= 0.0001:
                simalar.append(counter)
    counter+=1

simalar = np.unique(simalar)
all_a = np.delete(all_a, simalar, axis=0)


main_all_i = np.concatenate((index_1, index_2), axis=0)
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
            nn.Linear((max_action_top ) * max_last_action, 1024),
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



all_actions = []
all_pov_obs = np.zeros((1600000, 64, 64,3), dtype=np.uint8)
all_last_action = np.zeros((1600000, (max_action_top ) * max_last_action), dtype=np.uint8)
c_history = []
counter = 0

for trajectory_name in trajectory_names:
    trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)
    last_action = None
    stack_logs = []
    stack_all_a_ = [np.zeros((1, max_action_top)) for _ in range(max_last_action)]
    for obs, action, r, _, _ in trajectory:main_all_i
        av = action["vector"]

        if np.any(av != others[4]):
            all_pov_obs[len(all_actions)] = obs["pov"]
            all_last_action[len(all_actions)] = np.ndarray.flatten(np.array(stack_all_a_))
            d_1 = np.sum((av - other_a) ** 2, axis=1)
            tmp = np.zeros((1, max_action_top))
            if np.min(d_1) == 0:
                a = his[np.argmin(d_1)]
                all_actions.append(a)
                tmp[0][a] = 1
            else:
                d_1 = np.sum((av - all_a) ** 2, axis=1)
                a = np.argmin(d_1)
                all_actions.append(a)
                tmp[0][a] = 1
            c_history.append(counter)
            stack_all_a_.append(tmp)
            del stack_all_a_[0]
    counter += 1
c_history.append(100000)

all_actions = np.array(all_actions)
np.bincount(all_actions)
# all_last_action = np.array(all_last_action)
c_history = np.array(c_history)
# c_history = np.concatenate((c_history, [100000]),axis=0)

network = NatureCNN((3, 64, 64), max_action_top).cuda()
optimizer = th.optim.Adam(network.parameters(), lr=LEARNING_RATE)
loss_function = nn.CrossEntropyLoss()

len(c_history)

num_samples = all_actions.shape[0]

update_count = 0
losses = []
len(all_actions)
import time
c_time =time.time()

zeros_index = len(c_history) -1
for index__ in range(10):
    print("New EPS: ", index__)
    if index__ != 0:
        th.save(network.state_dict(), 'a1.pth')

    epoch_indices = np.arange(num_samples)
    np.random.shuffle(epoch_indices)
    for batch_i in range(0, num_samples, BATCH_SIZE):

        batch_indices = epoch_indices[batch_i:batch_i + BATCH_SIZE]
        # batch_indices = all_index_new[batch_indices]

        new_b = np.expand_dims(batch_indices, -1)
        new_b = new_b - main_all_i
        i_error = np.where(new_b < 0)
        new_b[i_error[0], i_error[1]] = zeros_index
        c_h = c_history[batch_indices]
        o_h = c_history[new_b] - np.expand_dims(c_h, -1)
        i_error = np.where(o_h != 0)
        new_b[i_error[0], i_error[1]] = zeros_index

        obs = all_pov_obs[new_b]
        l_a = all_last_action[new_b]
        # l_a.shape

        obs = obs.transpose(0, 1, 4, 2, 3)
        obs = th.from_numpy(obs).float().cuda()
        # l_a[0][-1]
        obs /= 255.0


        inven_0 = l_a
        inven_0 = th.from_numpy(inven_0).float().cuda()

        logits = network(obs, inven_0)
        actions = all_actions[batch_indices]
        loss = loss_function(logits, th.from_numpy(actions).long().cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        update_count += 1
        losses.append([loss.item()])
        if (update_count % 1000) == 0:
            mean_loss = np.mean(losses, axis=0)
            tqdm.write("Iteration {}. Loss {:<10.3f}  {:<10.3f}".format(
                update_count, mean_loss[0], time.time() - c_time))
            c_time = time.time()
            losses.clear()



network = NatureCNN((3, 64, 64), max_action_top).cuda()
network.load_state_dict(th.load('a1.pth'))




env = gym.make('MineRLObtainDiamondVectorObf-v0')

rewards = []
for episode in range(50):
    obs = env.reset()

    total_reward = 0
    steps = 0
    stack_all = []
    stack_all_last = [np.zeros((1, max_last_action * (max_action_top )))]
    stack_all_a_ = [np.zeros((1, (max_action_top ))) for _ in range(max_last_action)]
    # cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    # cv2.resizeWindow('image', 1000, 1000)
    # cv2.moveWindow('image', -30, 30)
    for i in range(18000):

        final = obs['pov']
        final = np.expand_dims(final, axis=0)
        stack_all.append(final)

        final = np.zeros((max_his, 64, 64, 3), dtype=np.float32)
        l_a = np.zeros((max_his, (max_action_top ) * max_last_action), dtype=np.float32)
        counter_i = 0
        index_c = len(stack_all) - 1
        for b in main_all_i:
            n_index = index_c - b
            if n_index >= 0:
                final[counter_i] = stack_all[n_index]
                l_a[counter_i] = stack_all_last[n_index]
            counter_i += 1

        final = final.transpose(0, 3, 1, 2).astype(np.float32)
        final = th.from_numpy(final[None]).float().cuda()
        final /= 255.0


        inven = l_a
        inven = th.from_numpy(inven[None]).float().cuda()
        p = network(final, inven)
        probabilities = th.softmax(p, dim=1)[0]
        probabilities = probabilities.detach().cpu().numpy()
        a_1 = np.random.choice(np.arange(max_action_top), p=probabilities)


        action = env.action_space.noop()
        action['vector'] = all_a[a_1]
        tmp = np.zeros((1, max_action_top))
        tmp[0][a_1] = 1

        obs, reward, done, info = env.step(action)

        stack_all_a_.append(tmp)

        if len(stack_all_a_) > max_last_action:
            del stack_all_a_[0]

        stack_all_last.append(np.ndarray.flatten(np.array(stack_all_a_))[None])


        total_reward += reward
        steps += 1
        if done:
            break
        # cv2.imshow('image', cv2.cvtColor(obs['pov'], cv2.COLOR_BGR2RGB))
        # cv2.resizeWindow('image', 950, 950)
        # if cv2.waitKey(10) & 0xFF == ord('o'):
        #     break
        # time.sleep(0.1)

    rewards.append(total_reward)

    print(f'Episode #{episode + 1} reward: {total_reward}\t\t episode length: {steps}\n')

    cv2.destroyAllWindows()

np.mean(rewards)


counter = 0

a = all_a[counter]
print("---1---")
d_1 = np.sum((a - action_obs) ** 2, axis=1)
index = np.argmin(d_1)
for k in key_as_vetor.keys():
    if key_as_vetor[k][index] != 0:
        v = key_as_vetor[k][index]
        if k in key_as:
            v = key_as_v[k][v]
        print(k,v )
print("---2---")
counter += 1
main_index = None
for k in key_as_vetor.keys():
    data_v = 0
    if k not in ['c1', 'c2']:

        index = key_as_vetor[k] == data_v
    if main_index is None:
        main_index = index
    else:
        main_index = index & main_index

c1 = key_as_vetor['c1'][main_index]
c2 = key_as_vetor['c2'][main_index]
x = action_obs[main_index]
d_1 = np.sum((others[4] - x) ** 2, axis=1)

len(x)

len(x)
np.min(c2)
index = (np.abs(c1) <= 0) & (np.abs(c1) == 0) &\
        (np.abs(c2) <= 20) & (np.abs(c2) > 10)
index = d_1 > 0.04
print(c1[index][:5])
print(c2[index][:5])
print(d_1[index][:5])
np.max(d_1[index])
np.min(d_1[index])

d_3 = c1 + c2
d_1 = np.sum((others[4] - x) ** 2, axis=1)
d_1 = np.sum((others[4] - full_a[0][0]) ** 2)

print(np.sum((others[4] - x[index]) ** 2, axis=1)[:5])

a = np.flip(np.argsort(d_1))
a = np.argsort(d_1)
x = a[d_1[a] > 0.059]
len(x)
np.round(d_1[x][:5], 3)
np.round(c1[x][:5], 3)
np.round(c2[x][:5], 3)


np.sum((others[4] - full_a[7][0]) ** 2)
np.min(d_1)

simalar = []
counter = 0
for a in all_a:
    d_1 = np.sum((a - all_a) ** 2, axis=1)
    print(np.round(np.sort(d_1)[1], 4))
    for j in range(len(d_1)):
        if j != counter:
            if d_1[j] <= 0.0001:
                simalar.append(counter)
    counter+=1

simalar = np.unique(simalar)
all_a = np.delete(all_a, simalar, axis=0)

counter = 0

for i in range(len(all_a)):
    a = all_a[i]
    print("---1---")
    d_1 = np.sum((a - action_obs) ** 2, axis=1)
    index = np.argmin(d_1)
    for k in key_as_vetor.keys():
        if key_as_vetor[k][index] != 0:
            v = key_as_vetor[k][index]
            if k in key_as:
                v = key_as_v[k][v]
            print(k,v )
    print("---2---")

len(all_a)