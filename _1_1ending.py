
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

action_keys_s = np.array(['attack','back','craft','equip','forward',
                'jump','left','nearbyCraft','nearbySmelt','place','right','sneak','sprint','c1','c2'])

obs_inve_keys_s = np.array(['coal', 'cobblestone','crafting_table', 'dirt','furnace',
                   'iron_axe','iron_ingot','iron_ore','iron_pickaxe','log','planks','stick','stone','stone_axe',
                   'stone_pickaxe','torch','wooden_axe','wooden_pickaxe'])
with open('action_key.pkl', 'rb') as f:
    action_key = pickle.load(f)
# action_key[2] =['0', 'f', 'b', 'l', 'r', 'j_f', 'j_r', 'j_l']
len(action_key[3])

DATA_DIR = "data"  # path to MineRL dataset (should contain "MineRLObtainIronPickaxeVectorObf-v0" directory).
EPOCHS = 2  # how many times we train over dataset.
LEARNING_RATE = 0.0001  # learning rate for the neural network.
BATCH_SIZE = 128
DATA_SAMPLES = 1000000

max_last_action = 15
max_his = 20
action_shape_his = 18
max_action_top = 80
index_1 = np.flip(np.arange(0, int(max_his / 2), 1))
index_2 = index_1 * 15

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
list_key = ['action$forward', 'action$left', 'action$back','action$right',
            'action$jump', 'action$sneak','action$sprint','action$attack',
            'action$camera', 'action$place', 'action$equip', 'action$craft',
            'action$nearbyCraft', 'action$nearbySmelt']

non_others = {}
for k in list_key:
    non_others[k] = []
#
print("Loading data")
vector_obs = None
action_obs = None
R_S = None
all = None
counter = 0
all_a = []

len(all_a)
x = np.unique(all_a, axis=0)
len(x)
c1 = None
c2 = None

for trajectory_name in trajectory_names:
    obs_vector = np.load(os.path.join(DATA_DIR, 'MineRLObtainDiamondVectorObf-v0', trajectory_name, 'rendered.npz'))
    obs_non = np.load(os.path.join(DATA_DIR, 'MineRLObtainDiamond-v0', trajectory_name, 'rendered.npz'))
    r = np.array(obs_vector['reward'])
    # if np.sum(r == 256) != 0:
    #     index = np.where(obs_non['action$equip'] == 'iron_pickaxe')[0]
    #     if np.sum(obs_non['action$equip'] == 'iron_pickaxe') != 0:
    #         counter += 1
    #         for aa in index:
    #             all_a.append(obs_vector['action$vector'][aa])

    # if np.sum(r == 256) != 0:
    #     a_v = obs_vector['action$vector']
    #     a_v = np.unique(a_v, axis=0)
    #     a = action_vector_all['equip_iron_pickaxe'][0]
    #     x = np.all(a[None] == a_v, axis=1)
    #     if np.sum(x) != 0:
    #         counter += 1
    #
    #     if action_obs is None:
    #         action_obs = a_v
    #     else:
    #         index = []
    #         counter = 0
    #         for a in action_obs:
    #             x= np.all(a[None] == a_v, axis=1)
    #             if np.sum(x) != 0:
    #                 index.append(counter)
    #             counter += 1
    #         # tmp = np.in1d(action_obs, a_v)
    #         # tmp = tmp.reshape(len(action_obs), 64)
    #         # tmp = np.all(tmp, axis=1)
    #         action_obs = action_obs[index]
    #     print(len(action_obs))
    os_v = np.array(obs_vector['observation$vector'])
    delta = os_v[1:-1] - np.roll(os_v, -1, axis=0)[1:-1]
    delta = np.sum(delta,axis=1)
    a_v = np.array(obs_vector['action$vector'])
    # delta = np.sum(a_v - np.roll(a_v, -1, axis=0), axis=1)[:-1]
    # a_v = delta
    if np.array(obs_vector['observation$vector']).shape[0] != np.array(obs_vector['action$vector']).shape[0]:
        print(np.array(obs_vector['observation$vector']).shape[0]- np.array(obs_vector['action$vector']).shape[0])
    if vector_obs is None:
        action_obs = np.array(a_v)
        c1 = np.array(obs_non['action$camera'][:-1,0])
        c2 = np.array(obs_non['action$camera'][:-1,1])

        R_S = np.array(r)
        vector_obs = np.array(os_v)

        # all = delta
    else:

        action_obs = np.concatenate((action_obs,
                                     np.array(a_v)), axis=0)
        vector_obs = np.concatenate((vector_obs,
                                     np.array(obs_vector['observation$vector'])), axis=0)
        R_S = np.concatenate((R_S,r), axis=0)
        # all = np.concatenate((all,delta), axis=0)

        c1 = np.concatenate((c1,
                                     np.array(obs_non['action$camera'][:-1,0])), axis=0)
        c2 = np.concatenate((c2,
                                     np.array(obs_non['action$camera'][:-1,1])), axis=0)


c1_ = c1[np.flip(np.argsort(action_obs))]
c2_ = c2[np.flip(np.argsort(action_obs))]

np.max(c1_)
vector_obs.shape
list_none = ['action$place', 'action$equip', 'action$craft',
            'action$nearbyCraft', 'action$nearbySmelt']
list_none = ['action$equip']
counter = 0
all_a = None
for trajectory_name in trajectory_names:
    obs_vector = np.load(os.path.join(DATA_DIR, 'MineRLObtainDiamondVectorObf-v0', trajectory_name, 'rendered.npz'))
    obs_non = np.load(os.path.join(DATA_DIR, 'MineRLObtainDiamond-v0', trajectory_name, 'rendered.npz'))
    index = None
    for k in list_none:
        tmp = np.where(obs_non[k] != 'none')[0]
        if index is None:
            index = tmp
        else:
            index = np.concatenate((index, tmp))

    index_2 = np.where(obs_non['action$place'] == 'crafting_table')[0]
    # index_2 = index
    os_v = np.array(obs_vector['observation$vector'])
    a_v = np.array(obs_vector['action$vector'])
    vector_obs_shift = np.roll(os_v, -1, axis=0)[:-1]
    vector_obs_none = os_v[:-1]

    action_obs_shift = np.roll(a_v, -1, axis=0)
    action_obs_shift_u = np.roll(a_v, 1, axis=0)
    action_obs_none = a_v
    #
    # index_2 = np.where(((np.all(vector_obs_shift == vector_obs_none, axis=1))
    #                 & (np.all(action_obs_shift_u != action_obs_none, axis=1))
    #                   & (np.all(action_obs_shift != action_obs_none, axis=1))
    #                     & (np.all(action_obs_shift == action_obs_shift_u, axis=1))))
    # len(os_v)
    if all_a is None:
        all_a = a_v[index_2]
    else:

        all_a = np.concatenate((all_a,
                                     a_v[index_2]), axis=0)

    # index_2 = np.where((np.all(vector_obs_shift != vector_obs_none, axis=1)))

    # vector_obs_shift[263]
    # vector_obs_none[263]
    if ~np.all(np.in1d(index,index_2)):
        counter += 1

len(action_obs)
len(all_a)
a= np.unique(all_a,axis=0)
a.shape
len(a)
all_rest_a = np.unique(action_obs,return_counts=True, axis=0)
all_rest_a[1] = np.sum(all_rest_a[1], axis=1)

tmp = all_rest_a[0][np.flip(np.argsort(all_rest_a[1]))]
len(tmp)
a = np.flip(np.sort(all_rest_a[1]))
a = a[:120]

action_obs.shape
action_obs = np.unique(action_obs, axis=0)
len(action_obs)
# counter = 0
# for x in action_obs:
#     for k,v in action_vector_all.items():
#         if np.all(x == v[0]) :
#             print(k, counter)
#     counter+=1

index = np.where((r == 16))[0]
a = 2350
x = np.array(obs_vector['observation$vector'][a]) - np.array(obs_vector['observation$vector'][a + 1])
np.sum(x)
delta = os_v[1:-1] - np.roll(os_v, -1, axis=0)[1:-1]
delta.shape
delta = np.sum(delta, axis=1)
x = delta[np.where(delta != 0)[0]]
len(x)
b = np.unique(action_obs, axis=0)

len(b)
counter = []
for x in a:
    counter.append(np.sum((c - x)** 2))

index = np.where(((R_S != 0) &
                  (R_S != 1) &
                  (R_S != 1024) &
                  (R_S != 64)) &
                  (R_S != 16))[0]

action_key = action_obs[index, :]
a = np.unique(action_key, axis=0)
len(b)
a = np.unique(action_obs, axis=0)
len(a)
kmeans = KMeans(n_clusters=120)
kmeans.fit(action_obs)
action_centroids = kmeans.cluster_centers_
action_centroids.shape

kmeans_2 = KMeans(n_clusters=90)
kmeans_2.fit(action_centroids)
action_centroids_2 = kmeans_2.cluster_centers_

distances = np.sum((all_a - action_centroids[:, None]) ** 2, axis=2)
# Get the index of the closest centroid to each action.
# This is an array of (batch_size,)
actions = np.argmin(distances, axis=0)
np.unique(actions)
action_vector_all.keys()
a = action_vector_all['_f_0_0'][0]
b = action_vector_all['_j_f_0_0'][0]
c = action_vector_all['attack_f_-1_-1'][0]

np.sum((b-a)** 2)
np.sum((c-a)** 2)

# for _ in ['equip_iron_pickaxe', 'equip_stone_pickaxe', 'equip_wooden_pickaxe',
#           'place_crafting_table', 'place_furnace']:
#     a = np.concatenate((a,action_vector_all[_][0][None]))
# a.shape
#
#
# a = np.unique(a, axis=0)
# len(a)
#
# for x in a:
#     tmp_2 = np.sum(action_obs - x, axis=1)
#     print(len( np.where(tmp_2 == 0)[0]))
# x = np.where(all != 0)[0]
# len(x)
# b= np.unique(all)
# len(b)
#
# with open('others.pkl', 'rb') as f:
#     others = pickle.load(f)
# founded_index = None
# unknow = np.arange(0, len(action_obs))
# for x in others:
#     tmp_2 = np.sum(action_obs - x, axis=1)
#     # tmp_2 = np.sum(action_obs[unknow,:] - x, axis=1)
#     if founded_index is None:
#         founded_index = np.where(tmp_2 == 0)[0]
#     else:
#         founded_index = np.concatenate((founded_index,  np.where(tmp_2 == 0)[0]))

# len(founded_index) + len(unknow)
# founded_index = np.unique(founded_index)
# len(unknow)
# len(action_obs)
# c = np.in1d(unknow,founded_index)
# # len(c)
# unknow = action_obs[~c]
# len(unknow) + len(founded_index)
# all_rest_a = np.unique(unknow,return_counts=True, axis=0)
# len(all_rest_a)
# a = np.flip(np.argsort(all_rest_a[1]))
# action_top = all_rest_a[0][a]
# a = np.flip(np.sort(all_rest_a[1]))
#
# for x in action_top[:50]:
#     tmp_2 = np.sum(action_obs - x, axis=1)
#     # tmp_2 = np.sum(action_obs[unknow,:] - x, axis=1)
#     if founded_index is None:
#         founded_index = np.where(tmp_2 == 0)[0]
# for a in action_top[:50]:
#     x= np.all(unknow == a[None],axis=1)
#     print(np.sum(x))


# with open(f'action_top.pkl', 'wb') as fout:
#     pickle.dump(action_top[:400], fout)

with open('action_centroids.pkl', 'rb') as f:
    action_centroids = pickle.load(f)
action_centroids = action_centroids[:80]
len(action_centroids)
#
# kmeans = KMeans(n_clusters=100)
# kmeans.fit(unknow)
# action_centroids = kmeans.cluster_centers_


# data = minerl.data.make("MineRLObtainDiamondVectorObf-v0", data_dir='data', num_workers=4)
# trajectory_names = data.get_trajectory_names()
# with open(f'action_centroids.pkl', 'wb') as fout:
#     pickle.dump(action_centroids, fout)
# with open('action_centroids.pkl', 'rb') as f:
#     action_centroids = pickle.load(f)

# with open(f'others.pkl', 'wb') as fout:
#     pickle.dump(a, fout)
with open('others.pkl', 'rb') as f:
    others = pickle.load(f)

# for x in others:
#     tmp_2 = np.sum(action_obs - x, axis=1)
#     print(len( np.where(tmp_2 == 0)[0]))

counter = 0
for x in action_obs:
    for k,v in action_vector_all.items():
        if np.all(x == v[0]) :
            print(k, counter)
    counter+=1

all_actions = []
all_pov_obs = np.zeros((1600000, 64, 64,3), dtype=np.uint8)
all_last_action = np.zeros((1600000, (max_action_top + 16) * max_last_action), dtype=np.uint8)
c_history = []
counter = 0

for trajectory_name in trajectory_names:
    trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)

    # stack_all_a_ = [np.zeros((1, max_action_top)) for _ in range(max_last_action)]

    last_action = None
    last_inven_raw = None
    stack_logs = []
    stack_all_a_ = [np.zeros((1, max_action_top +16)) for _ in range(max_last_action)]
    for obs, action, r, _, _ in trajectory:
        av = action["vector"]

        if np.sum(av - others[4]) != 0:
            all_pov_obs[len(all_actions)] = obs["pov"]
            all_last_action[len(all_actions)]= np.ndarray.flatten(np.array(stack_all_a_))

            distances = np.sum((av - action_centroids) ** 2, axis=1)
            a_1 = np.argmin(distances)
            a_2 = 4
            counter_2 = 0
            for x in others:
                if np.all(x == av):
                    a_2 = counter_2
                    break
                counter_2 += 1
            all_actions.append([a_1, a_2])
            c_history.append(counter)
            tmp = np.zeros((1, max_action_top + 16))
            tmp[0][a_1] = 1
            tmp[0][a_2 + max_action_top] = 1
            stack_all_a_.append(tmp)
            del stack_all_a_[0]
    counter += 1



#
# np.histogram(all_actions[:, 1])
# np.histogram(all_actions[:, 1])
# all_data_obs[0].append(np.zeros(len(all_data_obs[0][0])))
# all_last_action.append(np.zeros(len(all_last_action[0])))
#

all_actions = np.array(all_actions)
np.bincount(all_actions[:,1])
# all_actions[0].shape

# all_last_action = np.array(all_last_action)
c_history = np.array(c_history)
# c_history = np.concatenate((c_history, [100000]),axis=0)

network = NatureCNN((3, 64, 64), max_action_top).cuda()
optimizer = th.optim.Adam(network.parameters(), lr=LEARNING_RATE)
loss_function = nn.CrossEntropyLoss()

network_1 = NatureCNN((3, 64, 64), 16).cuda()
optimizer_1 = th.optim.Adam(network_1.parameters(), lr=LEARNING_RATE)
loss_function_1 = nn.CrossEntropyLoss()

num_samples = all_actions.shape[0]

update_count = 0
losses = []
len(all_actions)
import time
c_time =time.time()
# print(np.bincount(all_actions[:,0]))

# tmp = all_actions[:, 0]
# index_0 = np.where(tmp == a3_0_Index)[0]
# index_1 = np.where(tmp == 1)[0]
# index_2 = np.where(((tmp != a3_0_Index) & (tmp != 1)))[0]
# np.random.shuffle(index_0)
# np.random.shuffle(index_1)
# index_0 = index_0[:len(index_2) * 10]
# index_1 = index_1[:len(index_2) * 10]
# all_index_new = np.concatenate((index_0, index_1, index_2))
# all_index_new = np.sort(all_index_new)


#
# tmp = all_actions[:, 2]
#
# index_2 = np.where(~((tmp >= a2_0_Index - 1) & (tmp <= a2_0_Index + 1)))[0]
# print(len(index_2))
#
# index_0 = np.where(tmp == a2_0_Index)[0]
# np.random.shuffle(index_0)
# index_0 = index_0[:int(len(index_2)/3)]
#
# print(len(index_0))
# index_1 = np.where((tmp == a2_0_Index - 1))[0]
# np.random.shuffle(index_1)
# index_1 = index_1[:int(len(index_2)/3)]
# print(len(index_1))
# index_1_1 = np.where((tmp == a2_0_Index + 1))[0]
# np.random.shuffle(index_1_1)
# index_1_1 = index_1_1[:int(len(index_2)/3)]
#
# print(len(index_1_1))
#
# all_index_new = np.concatenate((index_0,index_1,index_1_1, index_2))
# all_index_new = np.sort(all_index_new)
# # a = np.unique(all_index_new)
# # len(index_0)
# # len(a)
# num_samples = all_index_new.shape[0]
# np.bincount(all_actions[all_index_new,2])
zeros_index = len(c_history) -1
for index__ in range(10):
    print("New EPS: ", index__)
    if index__ != 0:
        th.save(network.state_dict(), 'a1.pth')
        th.save(network_1.state_dict(), 'a2.pth')


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
        #
        # inven_1 = np.concatenate((inven_1, l_a), axis=-1)
        # inven_1 = th.from_numpy(inven_1).float().cuda()
        #
        # inven_2 = np.concatenate((inven_2, l_a), axis=-1)
        # inven_2 = th.from_numpy(inven_2).float().cuda()
        #
        # inven_3 = np.concatenate((inven_3, l_a), axis=-1)
        # inven_3 = th.from_numpy(inven_3).float().cuda()

        inven_0 = l_a
        inven_0 = th.from_numpy(inven_0).float().cuda()

        logits = network(obs, inven_0)
        actions = all_actions[batch_indices, 0]
        loss = loss_function(logits, th.from_numpy(actions).long().cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logits = network_1(obs, inven_0)
        actions = all_actions[batch_indices, 1]
        loss_1 = loss_function_1(logits, th.from_numpy(actions).long().cuda())
        optimizer_1.zero_grad()
        loss_1.backward()
        optimizer_1.step()


        update_count += 1
        losses.append([loss.item(), loss_1.item()])
        if (update_count % 1000) == 0:
            mean_loss = np.mean(losses, axis=0)
            tqdm.write("Iteration {}. Loss {:<10.3f} {:<10.3f}  {:<10.3f}".format(
                update_count, mean_loss[0], mean_loss[1], time.time() - c_time))
            c_time = time.time()
            losses.clear()

# TRAIN_MODEL_NAME = 'another_potato_1.pth'  # name to use when saving the trained agent.


th.save(network.state_dict(), 'a_1_2.pth')
th.save(network_1.state_dict(), 'a_2_2.pth')
th.save(network_2.state_dict(), 'a_3_2.pth')
th.save(network_3.state_dict(), 'a_4_2.pth')

th.save(network.state_dict(), 'a_1_3.pth')
th.save(network_1.state_dict(), 'a_2_3.pth')
th.save(network_2.state_dict(), 'a_3_3.pth')
th.save(network_3.state_dict(), 'a_4_3.pth')

del data


network = NatureCNN((3, 64, 64), max_action_top).cuda()
network.load_state_dict(th.load('a1.pth'))

network_1 = NatureCNN((3, 64, 64), 16).cuda()
network_1.load_state_dict(th.load('a2.pth'))


env = gym.make('MineRLObtainDiamondVectorObf-v0')

rewards = []
for episode in range(50):
    obs = env.reset()

    total_reward = 0
    steps = 0
    stack_all = []
    stack_all_last = [np.zeros((1, max_last_action * (max_action_top + 16)))]
    stack_all_a_ = [np.zeros((1, (max_action_top + 16))) for _ in range(max_last_action)]
    # cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    # cv2.resizeWindow('image', 1000, 1000)
    # cv2.moveWindow('image', -30, 30)
    # len(stack_all_last)
    for i in range(18000):

        final = obs['pov']
        final = np.expand_dims(final, axis=0)
        stack_all.append(final)

        final = np.zeros((max_his, 64, 64, 3), dtype=np.float32)
        l_a = np.zeros((max_his, (max_action_top + 16) * max_last_action), dtype=np.float32)
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

        p = network_1(final, inven)
        probabilities = th.softmax(p, dim=1)[0]
        probabilities = probabilities.detach().cpu().numpy()
        a_2 = np.random.choice(np.arange(16), p=probabilities)

        action = env.action_space.noop()
        action['vector'] = action_centroids[a_1]
        tmp = np.zeros((1, max_action_top + 16))
        tmp[0][a_1] = 1
        if a_2 != 4:
            tmp[0][a_1] = 0
            tmp[0][a_2 + max_action_top] = 1
            action['vector'] = others[a_2]

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


a = plt.imshow(obs['pov'])
plt.show()

# stack_all_last[-1].sh