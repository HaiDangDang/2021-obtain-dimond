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
from torch.nn.utils.rnn import pad_sequence
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

index_1 = np.flip(np.arange(1, 20, 2)) + 10
index_2 =[10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
main_all_i = np.concatenate((index_1, index_2), axis=0)

class NatureCNN(nn.Module):
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

    def forward(self, observations: th.Tensor, data: th.Tensor, slice_index, len_batch) -> th.Tensor:
        c_out = self.cnn(observations)
        data_out = self.linear_stack(data)

        c_out = torch.split(c_out, slice_index)
        data_out = torch.split(data_out, slice_index)
        final_out = None
        for j in range(len_batch):
            c_1 = c_out[j][None]
            d_1 = data_out[j][None]
            r_in = torch.cat((c_1, d_1), dim=-1)
            h0 = torch.zeros(self.layer_dim, 1, self.hidden_dim).requires_grad_().cuda()
            c0 = torch.zeros(self.layer_dim, 1, self.hidden_dim).requires_grad_().cuda()
            out_s, (hn, cn) = self.rnn_s(r_in, (h0.detach(), c0.detach()))

            c_1 = c_out[j + len_batch][None]
            d_1 = data_out[j + len_batch][None]
            r_in = torch.cat((c_1, d_1), dim=-1)

            out, (hn, cn) = self.rnn(r_in, (hn, cn))
            out_in = torch.cat((out[:, -1, :], out_s[:, -1, :]), dim=-1)

            out = self.fc(out_in)
            if final_out is None:
                final_out = out
            else:
                final_out = torch.cat((final_out, out), dim=0)
        return final_out



with open('others.pkl', 'rb') as f:
    others = pickle.load(f)

with open('action_connect_to_vector.pkl', 'rb') as f:
    action_vector_all = pickle.load(f)

action_vector_all.keys()
data = minerl.data.make("MineRLObtainDiamondVectorObf-v0", data_dir='data', num_workers=4)
trajectory_names = data.get_trajectory_names()
random.shuffle(trajectory_names)

print("Loading data")


with open('stack_all_knn.pkl', 'rb') as f:
    stack_all_final = pickle.load(f)

with open('stack_new.pkl', 'rb') as f:
    stack_new = pickle.load(f)

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
    for j in range(len(d_1)):
        if j != counter:
            if d_1[j] <= 0.0001:
                simalar.append(counter)
    counter+=1

simalar = np.unique(simalar)
all_a = np.delete(all_a, simalar, axis=0)
max_action_top = len(all_a)


all_actions = []
all_pov_obs = np.zeros((1600000, 64, 64,3), dtype=np.uint8)
all_last_action = np.zeros((1600000, (max_action_top ) * max_last_action), dtype=np.uint8)
c_history = []
c_h_1 = []

c_h_2 = []
counter = 0
main_index = []
back_ward = 500

for trajectory_name in trajectory_names:
    trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)
    last_action = None
    stack_logs = []
    stack_all_a_ = [np.zeros((1, max_action_top)) for _ in range(max_last_action)]
    stack_r = []
    eps_i = len(all_actions) - 1
    for obs, action, r, _, _ in trajectory:
        av = action["vector"]
        if np.any(av != others[4]):
            all_pov_obs[len(all_actions)] = obs["pov"]

            all_last_action[len(all_actions)] = np.ndarray.flatten(np.array(stack_all_a_))
            tmp = np.zeros((1, max_action_top))

            d_1 = np.sum((av - all_a) ** 2, axis=1)
            a = np.argmin(d_1)
            all_actions.append(a)

            current_h = len(all_actions) - 1
            tmp_st = copy.deepcopy(stack_r)
            tmp_st.append(current_h)
            c_h_2.append(tmp_st)

            c_11 = []
            for j in reversed(main_all_i):
                if current_h - j <= eps_i:
                    break
                c_11.insert(0, current_h - j)
            c_h_1.append(c_11)

            tmp[0][a] = 1
            c_history.append(counter)
            stack_all_a_.append(tmp)
            del stack_all_a_[0]
            if r != 0:
                stack_r.append(len(all_actions) - 1)
    super_break = False
    first_stop = eps_i
    for i in range(len(stack_r)):
        main_i = stack_r[i]
        if i == len(stack_r) - 1:
            end_stop = len(all_actions) - 1
        else:
            end_stop = stack_r[i + 1]
        for j in range(back_ward):
            if main_i - j < 0 or main_i - j <= first_stop:
                break
            if main_i - j in main_index:
                super_break = True
                break
            main_index.append(main_i - j)
        for j in range(1,back_ward):
            if main_i + j >= end_stop:
                break
            if main_i + j in main_index:
                super_break = True
                break
            main_index.append(main_i + j)
        first_stop = np.max(main_index)
    if super_break:
        break
    counter += 1

print(len(main_index))
main_index = np.unique(main_index)
len(main_index)


all_actions = np.array(all_actions)
np.bincount(all_actions)
c_history = np.array(c_history)

network = NatureCNN((3, 64, 64), max_action_top).cuda()
optimizer = th.optim.Adam(network.parameters(), lr=LEARNING_RATE)
loss_function = nn.CrossEntropyLoss()

len(c_history)
c_h_1 =np.array(c_h_1)
c_h_2 =np.array(c_h_2)

import time
c_time =time.time()

num_samples = len(main_index) + 200000
x_none = np.arange(0, len(all_actions))
x_none = x_none[~np.in1d(x_none, main_index)]

update_count = 0
losses = []
for index__ in range(10):
    print("New EPS: ", index__)
    if index__ != 0:
        th.save(network.state_dict(), 'a1.pth')

    np.random.shuffle(x_none)
    x_batch = np.copy(x_none)
    x_batch = x_batch[:200000]
    x_batch = np.concatenate((x_batch, main_index))
    np.random.shuffle(x_batch)

    epoch_indices = np.arange(num_samples)
    np.random.shuffle(epoch_indices)
    for batch_i in range(0, num_samples, BATCH_SIZE):

        batch_indices = epoch_indices[batch_i:batch_i + BATCH_SIZE]
        batch_indices = x_batch[batch_indices]
        slices = []
        c_1 = c_h_1[batch_indices]
        c_2 = c_h_2[batch_indices]
        index_c = []
        for i in range(len(batch_indices)):
            slices.append(c_2[i])
            index_c.append(len(c_2[i]))

        for i in range(len(batch_indices)):
            slices.append(c_1[i])
            index_c.append(len(c_1[i]))


        slices = np.concatenate(slices)
        obs = all_pov_obs[slices]
        l_a = all_last_action[slices]

        obs = obs.transpose(0, 3, 1, 2)
        obs = th.from_numpy(obs).float().cuda()
        # l_a[0][-1]
        obs /= 255.0
        inven_0 = th.from_numpy(l_a).float().cuda()
        # len(obs)
        # a =network.cnn(obs)
        # a = torch.split(a, index_c)
        # a[0].size()
        # x = pad_sequence(a, batch_first=True)
        # x[0][0]

        logits = network(obs, inven_0, index_c, len(batch_indices))
        # x = torch.nn.utils.rnn.pack_padded_sequence()
        actions = all_actions[batch_indices]
        loss = loss_function(logits, th.from_numpy(actions).long().cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        update_count += 1
        losses.append([loss.item()])
        if (update_count % 100) == 0:
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
    stack_all_last = [np.zeros(( max_last_action * (max_action_top )))]
    stack_all_a_ = [np.zeros(( max_action_top)) for _ in range(max_last_action)]

    # cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    # cv2.resizeWindow('image', 1000, 1000)
    # cv2.moveWindow('image', -30, 30)
    stack_hist = []
    for i in range(18000):

        final = obs['pov']
        stack_all.append(final)


        current_h = len(stack_all) - 1
        c_11 = []
        for j in reversed(main_all_i):
            if current_h - j < 0:
                break
            c_11.insert(0, current_h - j)

        slices = []
        index_c = []



        tmp_C2 = copy.deepcopy(stack_hist)
        tmp_C2.append(len(stack_all) - 1)

        slices.append(tmp_C2)
        index_c.append(len(tmp_C2))

        slices.append(c_11)
        index_c.append(len(c_11))

        slices = np.concatenate(slices)
        x = np.array(stack_all)

        final = np.array(stack_all)[slices]
        l_a = np.array(stack_all_last)[slices]

        final = final.transpose(0, 3, 1, 2).astype(np.float32)
        final = th.from_numpy(final).float().cuda()
        final /= 255.0


        inven = l_a
        inven = th.from_numpy(inven).float().cuda()

        p = network(final, inven, index_c, 1)
        probabilities = th.softmax(p, dim=1)[0]
        probabilities = probabilities.detach().cpu().numpy()
        a_1 = np.random.choice(np.arange(max_action_top), p=probabilities)


        action = env.action_space.noop()
        action['vector'] = all_a[a_1]
        tmp = np.zeros(( max_action_top))
        tmp[a_1] = 1

        obs, reward, done, info = env.step(action)
        if reward != 0:
            stack_hist.append(len(stack_all[0]) - 1)
            print(reward)
        stack_all_a_.append(tmp)

        if len(stack_all_a_) > max_last_action:
            del stack_all_a_[0]

        stack_all_last.append(np.ndarray.flatten(np.array(stack_all_a_)))


        total_reward += reward
        steps += 1
        if done:
            break
        # cv2.imshow('image', cv2.cvtColor(obs['pov'], cv2.COLOR_BGR2RGB))
        # cv2.resizeWindow('image', 950, 950)
        # if cv2.waitKey(10) & 0xFF == ord('o'):
        #     break

    del stack_all
    rewards.append(total_reward)

    print(f'Episode #{episode + 1} reward: {total_reward}\t\t episode length: {steps}\n')

    cv2.destroyAllWindows()

np.mean(rewards)
cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow('image', 1000, 1000)
cv2.moveWindow('image', -30, 30)
for x in main_index:

    cv2.imshow('image', cv2.cvtColor(all_pov_obs[x], cv2.COLOR_BGR2RGB))
    cv2.resizeWindow('image', 950, 950)
    if cv2.waitKey(10) & 0xFF == ord('o'):
        break
    time.sleep(0.01)
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