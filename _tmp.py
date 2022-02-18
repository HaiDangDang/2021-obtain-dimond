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

from hummingbird.ml import convert
import matplotlib.pyplot as plt
import copy
from torch.autograd import Variable

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

max_last_action = 10
max_his = 20

class NatureCNN(nn.Module):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param input_shape: A three-item tuple telling image dimensions in (C, H, W)
    :param output_dim: Dimensionality of the output vector
    """

    def __init__(self, input_shape, output_dim, data_size=16, hidden_size=64):
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
            nn.Linear(data_size + 8 * max_last_action, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        self.rnn_s = nn.LSTM(n_flatten + 512, self.hidden_dim, self.layer_dim, batch_first=True)


        self.rnn = nn.LSTM(n_flatten + 512 +  self.hidden_dim, self.hidden_dim, self.layer_dim, batch_first=True)

        self.fc = nn.Linear(self.hidden_dim, output_dim)
        # self.fc = nn.Sequential(
        #     nn.Linear(self.hidden_dim + self.hidden_dim, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, output_dim),
        # )

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


        r_in = torch.cat((r_in_cnn[:, :int(max_his / 2)], r_in_data[:,:int(max_his / 2)], out_s), dim=-1)
        h0 = torch.zeros(self.layer_dim, observations.size(0), self.hidden_dim).requires_grad_().cuda()
        c0 = torch.zeros(self.layer_dim, observations.size(0), self.hidden_dim).requires_grad_().cuda()
        out, (hn, cn) = self.rnn(r_in, (h0.detach(), c0.detach()))
        out = self.fc(out[:,-1,:])


        # out_in = torch.cat((out[:, -1, :], out_s[:, -1, :]), dim=-1)
        # out = self.fc(out_in)

        return out

    def initHidden(self, BATCH):
        return (Variable(torch.zeros(self.layer_dim, BATCH, self.hidden_dim)).cuda(),
                Variable(torch.zeros(self.layer_dim, BATCH, self.hidden_dim)).cuda())


def reset_craft_action():
    return {"craft_crafting_table": False, "nearbyCraft_furnace": False, "place_crafting_table": False, "craft_planks": False,
          "craft_stick": False, "nearbyCraft_wooden_pickaxe": False, "nearbyCraft_stone_pickaxe": False, "place_furnace": False,
          "nearbySmelt_iron_ingot": False, "nearbyCraft_iron_pickaxe": False}


item_by_attack = ['coal', 'cobblestone', 'dirt', 'iron_ore', 'log', 'stone', 'crafting_table', 'furnace']
item_by_attack_index = [0, 1, 3, 7, 9, 12, 2, 4]

max_shape = 3


print("Prepare_Data")

range_angle = np.array([-30, -15, -10, -5, 0, 5, 10, 15, 30]) #8

range_angle_2 = np.array([-45, -30, -15, -10, -5, 0, 5, 10, 15, 30, 45])#8
list_key = [ np.array(action_key[3]), action_key[2], range_angle_2, range_angle ]

to_action_0 = [[{}, {'equip': 'carrot'}, {'equip': 'fence'},
               {'equip': 'fence_gate'}, {'equip': 'snowball'}, {'equip': 'wheat'},
               {'equip': 'wheat_seeds'}, {'attack': 1}, {'use': 1}],
                [{}, {'forward': 1}, {'back': 1}, {'left': 1}, {'right': 1}, {'forward': 1, 'jump': 1}, {'right': 1, 'jump': 1}, {'left': 1, 'jump': 1}]
            , range_angle_2,  range_angle]

list_key = np.array(list_key)
a1_0_Index = np.argmin(np.abs(range_angle - 0))
a2_0_Index = np.argmin(np.abs(range_angle_2 - 0))
range_angle_delta = np.max(range_angle) - np.min(range_angle)
range_angle_2_delta = np.max(range_angle_2) - np.min(range_angle_2)

dict_action = {}

a3_0_Index = np.where(list_key[0] == '0')[0][0]


index_1 = np.flip(np.arange(0, int(max_his / 2), 1))
index_2 = index_1 * 100
main_all_i = np.concatenate((index_1, index_2), axis=0)
main_all_i[-1] = 50
# main_all_i = np.flip(np.sort(main_all_i))

def dataset_action_batch_to_actions(actions):
    action_with_key = copy.deepcopy(actions)
    dict_action = {}

    a1 = action_with_key['camera'][0]
    a2 = action_with_key['camera'][1]

    ks = [f'0', f'0', 0 ,0]

    a1 = np.abs(range_angle - a1)
    ks[3] = np.argmin(a1)

    a2 = np.abs(range_angle_2 - a2)
    ks[2] = np.argmin(a2)

    if action_with_key['jump'] == 1:
        if action_with_key['forward'] == 1:
            ks[1] = f'j_f'
        elif action_with_key['right'] == 1:
            ks[1] = f'j_r'
        elif action_with_key['left'] == 1:
            ks[1] = f'j_l'
        else:
            ks[1] = f'j_f'

    elif action_with_key['forward'] == 1:
        dict_action['forward'] = 1
        ks[1] = f'f'
    elif action_with_key['back'] == 1:
        dict_action['back'] = 1
        ks[1] = f'b'
    elif action_with_key['left'] == 1:
        dict_action['left'] = 1
        ks[1] = f'l'
    elif action_with_key['right'] == 1:
        dict_action['right'] = 1
        ks[1] = f'r'

    ks[1] = np.where(np.array(list_key[1]) == ks[1])[0][0]

    if action_with_key['craft'] != 'none':
        craft = action_with_key['craft']
        ks[0] = f'craft_{craft}'
        dict_action['craft'] = craft
    elif action_with_key['equip'] != 'none':

        equip = action_with_key['equip']
        if equip == 'iron_axe': equip = 'iron_pickaxe'
        if equip == 'stone_axe': equip = 'stone_pickaxe'
        if equip == 'wooden_axe': equip = 'wooden_pickaxe'
        ks[0] = f'equip_{equip}'
        dict_action['equip'] = equip

    elif action_with_key['nearbyCraft'] != 'none':
        nearbyCraft = action_with_key['nearbyCraft']
        if nearbyCraft == 'iron_axe': nearbyCraft = 'iron_pickaxe'
        if nearbyCraft == 'stone_axe': nearbyCraft = 'stone_pickaxe'
        if nearbyCraft == 'wooden_axe': nearbyCraft = 'wooden_pickaxe'
        ks[0] = f'nearbyCraft_{nearbyCraft}'
        dict_action['nearbyCraft'] = nearbyCraft

    elif action_with_key['nearbySmelt'] != 'none':
        nearbySmelt = action_with_key['nearbySmelt']
        ks[0] = f'nearbySmelt_{nearbySmelt}'
        dict_action['nearbySmelt'] = nearbySmelt
    elif action_with_key['place'] != 'none':
        place = action_with_key['place']
        # if place not in ['cobblestone', 'dirt', 'stone']:
        ks[0] = f'place_{place}'
        dict_action['place'] = place
    elif action_with_key['attack'] == 1:
        dict_action['attack'] = 1
        ks[0] = 'attack'
    ks[0] = np.where(np.array(list_key[0]) == ks[0])[0][0]

    return ks


def process_inventory(obs, angle_1, angle_2, test=False):

    # rs = [1, 2, 4, 8, 16, 32, 32, 64, 128, 256, 1024]
    data = np.zeros(20)
    if test:
        if obs['equipped_items']['mainhand']['type'] == 'iron_pickaxe':
            data[0] = 1
        if obs['equipped_items']['mainhand']['type'] == 'stone_pickaxe':
            data[1] = 1
        if obs['equipped_items']['mainhand']['type'] == 'wooden_pickaxe':
            data[2] = 1
        if obs['equipped_items']['mainhand']['type'] in ['other', 'none', 'air']:
            data[3] = 1

    else:
        if obs['equipped_items.mainhand.type'] in ['iron_pickaxe', 'iron_iron_axe']:
            data[0] = 1
        if obs['equipped_items.mainhand.type'] in ['stone_pickaxe', 'stone_axe']:
            data[1] = 1
        if obs['equipped_items.mainhand.type'] in ['wooden_pickaxe', 'wooden_axe']:
            data[2] = 1
        if obs['equipped_items.mainhand.type'] in ['other', 'none', 'air']:
            data[3] = 1

    data[4] = np.clip(obs['inventory']['furnace'] / 2, 0, 1)
    data[5] = np.clip(obs['inventory']['crafting_table'] / 2, 0, 1)

    data[6] = np.clip(obs['inventory']['planks'] / 40, 0, 1)
    data[7] = np.clip(obs['inventory']['stick'] / 40, 0, 1)

    data[8] = np.clip((obs['inventory']['stone_pickaxe'] + obs['inventory']['stone_axe']) / 2, 0, 1)
    data[9] = np.clip((obs['inventory']['wooden_pickaxe'] + obs['inventory']['wooden_axe']) / 2, 0, 1)
    data[10] = np.clip((obs['inventory']['iron_pickaxe'] + obs['inventory']['iron_axe']) / 2, 0, 1)

    data[11] = np.clip(obs['inventory']['log'] / 40, 0, 1)
    data[12] = np.clip((obs['inventory']['cobblestone']) / 40, 0, 1)
    # data[13] = np.clip((obs['inventory']['stone']) / 20, 0, 1)

    data[13] = np.clip(obs['inventory']['iron_ore'] / 20, 0, 1)
    data[14] = np.clip(obs['inventory']['iron_ingot'] / 20, 0, 1)
    data[15] = np.clip(obs['inventory']['coal'] / 20, 0, 1)
    # data[17] = np.clip(obs['inventory']['dirt'] / 50, 0, 1)
    data[16] = np.clip(obs['inventory']['torch'] / 20, 0, 1)

    data[17] = (angle_1 + 90) / 180

    data[19] = 0
    if angle_2 > 180:
        angle_2 = 360 - angle_2
        data[19] = 1
    data[18] = np.clip(np.abs(angle_2) / 180, 0, 1)

    return data
# np.sum(all_data_obs[0][:,14] > 0.)
# len(all_last_action[0])


data = minerl.data.make("MineRLObtainDiamond-v0", data_dir='data', num_workers=4)
trajectory_names = data.get_trajectory_names()
# np.bincount(all_actions[:,0])

random.shuffle(trajectory_names)

all_actions = []
all_pov_obs = []
all_last_action = []
c_history = []
all_index_x = []

print("Loading data")

counter = 0

all_data_obs = [[], [], [], []]
counter_fail = 0
for trajectory_name in trajectory_names:
    trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)

    angle_1 = 0
    angle_2 = 0

    stack_all_a_ = [np.zeros((1, 8)) for _ in range(max_last_action)]

    for obs, action, r, _, _ in trajectory:

        main_data = []
        final = obs['pov']
        key = dataset_action_batch_to_actions(action)
        after_proces = process_inventory(obs, angle_1, angle_2)
        main_data.append(after_proces)

        # x = np.array([action['attack'], action['craft'], action['nearbyCraft'], action['nearbySmelt']])
        # after_proces = np.concatenate((after_proces, x), axis=0)
        # main_data.append(after_proces)
        #
        # x = np.array([action['jump'], action['forward'], action['back']
        #                  , action['right'], action['left']])
        # after_proces = np.concatenate((after_proces, x), axis=0)
        # main_data.append(after_proces)
        #
        # after_proces = np.concatenate(
        #     (after_proces, [(range_angle_2[key[2]] + np.max(range_angle_2)) / range_angle_2_delta]), axis=0)
        # main_data.append(after_proces)

        angle_2 += action['camera'][1]
        if angle_2 > 360:
            angle_2 = angle_2 - 360
        elif angle_2 < 0:
            angle_2 = 360 + angle_2

        angle_1 -= action['camera'][0]
        angle_1 = np.clip(angle_1, -90, 90)

        if not (key[3] == a1_0_Index and key[2] == a2_0_Index and key[0] == a3_0_Index and key[1] == 0):
            # all_index_x.append(len(all_pov_obs))
            all_pov_obs.append(final)
            all_actions.append(key)
            all_last_action.append(np.ndarray.flatten(np.array(stack_all_a_)))
            c_history.append(counter)
            all_data_obs[0].append(main_data[0])
            # all_data_obs[1].append(main_data[1])
            # all_data_obs[2].append(main_data[2])
            # all_data_obs[3].append(main_data[3])
        else:
            counter_fail += 1
        action['c2'] = (range_angle_2[key[2]] + np.min(range_angle_2) * -1) / range_angle_2_delta
        action['c1'] = (range_angle[key[3]] + np.min(range_angle) * -1) / range_angle_delta
        del action['camera']
        del action['equip']
        del action['sprint']
        del action['sneak']

        del action['craft']
        del action['nearbyCraft']
        del action['nearbySmelt']
        del action['place']

        stack_all_a_.append(np.expand_dims(np.array(list(action.values())), axis=0))
        if len(stack_all_a_) > max_last_action:
            del stack_all_a_[0]
        del main_data
    print(len(all_data_obs[0]), counter_fail)
    counter += 1
    del stack_all_a_


#
# np.histogram(all_actions[:, 1])
# np.histogram(all_actions[:, 1])
all_actions = np.array(all_actions)
for i in range(len(all_data_obs)):
    all_data_obs[i] = np.array(all_data_obs[i])
all_last_action = np.array(all_last_action)
c_history = np.array(c_history)

network = NatureCNN((max_shape, 64, 64), len(list_key[0]), all_data_obs[0].shape[1]).cuda()
optimizer = th.optim.Adam(network.parameters(), lr=LEARNING_RATE)
loss_function = nn.CrossEntropyLoss()

network_1 = NatureCNN((max_shape, 64, 64), len(list_key[1]), all_data_obs[0].shape[1]).cuda()
optimizer_1 = th.optim.Adam(network_1.parameters(), lr=LEARNING_RATE)
loss_function_1 = nn.CrossEntropyLoss()

network_2 = NatureCNN((max_shape, 64, 64), len(list_key[2]), all_data_obs[0].shape[1]).cuda()
optimizer_2 = th.optim.Adam(network_2.parameters(), lr=LEARNING_RATE)
loss_function_2 = nn.CrossEntropyLoss()

network_3 = NatureCNN((max_shape, 64, 64), len(list_key[3]), all_data_obs[0].shape[1]).cuda()
optimizer_3 = th.optim.Adam(network_3.parameters(), lr=LEARNING_RATE)
loss_function_3 = nn.CrossEntropyLoss()

all_index_x = np.array(all_index_x)
num_samples = all_actions.shape[0]

update_count = 0
losses = []
len(all_actions)
import time
c_time =time.time()
# print(np.bincount(all_actions[:,0]))
for index__ in range(10):
    epoch_indices = np.arange(num_samples)
    np.random.shuffle(epoch_indices)
    for batch_i in range(0, num_samples, BATCH_SIZE):
        batch_indices = epoch_indices[batch_i:batch_i + BATCH_SIZE]
        # batch_indices = all_index_x[batch_indices]
        obs = np.zeros((len(batch_indices), max_his, 64, 64, max_shape), dtype=np.float32)
        l_a = np.zeros((len(batch_indices), max_his, all_last_action.shape[1]), dtype=np.float32)
        inven_0 = np.zeros((len(batch_indices), max_his, all_data_obs[0].shape[1]), dtype=np.float32)
        #
        # inven_1 = np.zeros((len(batch_indices), max_his, all_data_obs[1].shape[1]), dtype=np.float32)
        # inven_2 = np.zeros((len(batch_indices), max_his, all_data_obs[2].shape[1]), dtype=np.float32)
        # inven_3 = np.zeros((len(batch_indices), max_his, all_data_obs[3].shape[1]), dtype=np.float32)
        for j in range(len(batch_indices)):
            index = batch_indices[j]
            c_h = c_history[index]
            counter_i = 0
            for b in main_all_i:
                n_index = index - b
                if n_index >= 0:
                    o_h = c_history[n_index]
                    if o_h == c_h:
                        obs[j, counter_i] = all_pov_obs[n_index]
                        l_a[j, counter_i] = all_last_action[n_index]
                        inven_0[j, counter_i] = all_data_obs[0][n_index]
                        #
                        # inven_1[j, counter_i] = all_data_obs[1][n_index]
                        # inven_2[j, counter_i] = all_data_obs[2][n_index]
                        # inven_3[j, counter_i] = all_data_obs[3][n_index]

                counter_i += 1
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

        inven_0 = np.concatenate((inven_0, l_a), axis=-1)
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

        logits = network_2(obs, inven_0)
        actions = all_actions[batch_indices, 2]
        loss_2 = loss_function_2(logits, th.from_numpy(actions).long().cuda())
        optimizer_2.zero_grad()
        loss_2.backward()
        optimizer_2.step()

        logits = network_3(obs, inven_0)
        actions = all_actions[batch_indices, 3]
        loss_3 = loss_function_3(logits, th.from_numpy(actions).long().cuda())
        optimizer_3.zero_grad()
        loss_3.backward()
        optimizer_3.step()


        update_count += 1
        losses.append([loss.item(), loss_1.item(), loss_2.item(), loss_3.item()])
        if (update_count % 1000) == 0:
            mean_loss = np.mean(losses, axis=0)
            tqdm.write("Iteration {}. Loss {:<10.3f} {:<10.3f} {:<10.3f}  {:<10.3f} {:<10.3f}".format(
                update_count, mean_loss[0], mean_loss[1], mean_loss[2], mean_loss[3], time.time() - c_time))
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


network = NatureCNN((max_shape, 64, 64), len(list_key[0]), 20).cuda()
network.load_state_dict(th.load('a_1_3.pth'))

network_1 = NatureCNN((max_shape, 64, 64), len(list_key[1]), 20).cuda()
network_1.load_state_dict(th.load('a_2_3.pth'))

network_2 = NatureCNN((max_shape, 64, 64), len(list_key[2]), 20).cuda()
network_2.load_state_dict(th.load('a_3_3.pth'))

network_3 = NatureCNN((max_shape, 64, 64), len(list_key[3]), 20).cuda()
network_3.load_state_dict(th.load('a_4_3.pth'))

env = gym.make('MineRLObtainDiamond-v0')

rewards = []
for episode in range(50):
    obs = env.reset()
    total_reward = 0
    steps = 0

    done = False
    angle_2 = 0
    angle_1 = 0

    stack_all = []
    stack_all_data = []
    stack_all_data_1 = []
    stack_all_data_2 = []
    stack_all_data_3 = []
    stack_all_last = [np.zeros((1, max_last_action * 8))]
    stack_all_a_ = [np.zeros((1, 8)) for _ in range(max_last_action)]
    # cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    # cv2.resizeWindow('image', 1000, 1000)
    # cv2.moveWindow('image', -30, 30)
    for i in range(17900):
        new_a = env.action_space.noop()

        final = obs['pov']
        final = np.expand_dims(final, axis=0)
        stack_all.append(final)

        final = np.zeros((max_his, 64, 64, max_shape), dtype=np.float32)
        l_a = np.zeros((max_his, 8 * max_last_action), dtype=np.float32)
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

        # -----#
        inven_m = process_inventory(obs, angle_1, angle_2,True)

        # -----#
        # inven = np.expand_dims(inven_m, axis=0)
        stack_all_data.append(inven_m)

        inven = np.zeros((max_his, inven_m.shape[0]), dtype=np.float32)
        counter_i = 0
        index_c = len(stack_all_data) - 1
        for b in main_all_i:
            n_index = index_c - b
            if n_index >= 0:
                inven[counter_i] = stack_all_data[n_index]
            counter_i += 1
        inven = np.concatenate((inven, l_a), axis=-1)
        # inven[0,-1]
        inven = th.from_numpy(inven[None]).float().cuda()
        p = network(final, inven)
        probabilities = th.softmax(p, dim=1)[0]
        probabilities = probabilities.detach().cpu().numpy()
        action = np.random.choice(np.arange(len(list_key[0])), p=probabilities)

        key_3 = action_key[3][action]
        if key_3 != '0':
            if key_3 == 'attack':
                new_a['attack'] = 1
            else:
                slipt = key_3.split('_')
                if len(slipt) == 2:
                    new_a[slipt[0]] = slipt[1]
                else:
                    new_a[slipt[0]] = slipt[1] + '_' + slipt[2]
        if key_3 not in ['0', 'attack']:
            print(key_3)

        # x = np.array([new_a['place'], new_a['attack'], new_a['craft'], new_a['nearbyCraft'], new_a['nearbySmelt']])
        # inven_m = np.concatenate((inven_m, x), axis=0)
        #
        # stack_all_data_1.append(inven_m)
        #
        # inven = np.zeros((max_his, inven_m.shape[0]), dtype=np.float32)
        # counter_i = 0
        # index_c = len(stack_all_data_1) - 1
        # for b in main_all_i:
        #     n_index = index_c - b
        #     if n_index >= 0:
        #         inven[counter_i] = stack_all_data_1[n_index]
        #     counter_i += 1
        # inven = np.concatenate((inven, l_a), axis=-1)
        #
        # inven = th.from_numpy(inven[None]).float().cuda()
        p = network_1(final, inven)
        probabilities = th.softmax(p, dim=1)[0]
        probabilities = probabilities.detach().cpu().numpy()
        action = np.random.choice(np.arange(len(to_action_0[1])), p=probabilities)
        for k, v in to_action_0[1][action].items():
            new_a[k] = v
        # -----#

        # -----#

        # x = np.array([new_a['jump'], new_a['forward'], new_a['back']
        #                  , new_a['right'], new_a['left']])
        #
        # inven_m = np.concatenate((inven_m, x), axis=0)
        #
        # stack_all_data_2.append(inven_m)
        # inven = np.zeros((max_his, inven_m.shape[0]), dtype=np.float32)
        # counter_i = 0
        # index_c = len(stack_all_data_2) - 1
        # for b in main_all_i:
        #     n_index = index_c - b
        #     if n_index >= 0:
        #         inven[counter_i] = stack_all_data_2[n_index]
        #     counter_i += 1
        # inven = np.concatenate((inven, l_a), axis=-1)
        #
        # inven = th.from_numpy(inven[None]).float().cuda()
        p = network_2(final, inven)
        probabilities = th.softmax(p, dim=1)[0]
        probabilities = probabilities.detach().cpu().numpy()
        action = np.random.choice(np.arange(len(to_action_0[2])), p=probabilities)
        new_a['camera'][1] = to_action_0[2][action]

        # -----#

        # -----#
        # inven_m = np.concatenate((inven_m, [(new_a['camera'][1] + np.max(range_angle_2))
        #                                     / range_angle_2_delta]), axis=0)
        #
        # stack_all_data_3.append(inven_m)
        # inven = np.zeros((max_his, inven_m.shape[0]), dtype=np.float32)
        # counter_i = 0
        # index_c = len(stack_all_data_3) - 1
        # for b in main_all_i:
        #     n_index = index_c - b
        #     if n_index >= 0:
        #         inven[counter_i] = stack_all_data_3[n_index]
        #     counter_i += 1
        # inven = np.concatenate((inven, l_a), axis=-1)
        #
        # inven = th.from_numpy(inven[None]).float().cuda()
        p = network_3(final, inven)
        probabilities = th.softmax(p, dim=1)[0]
        probabilities = probabilities.detach().cpu().numpy()
        action = np.random.choice(np.arange(len(to_action_0[3])), p=probabilities)
        a1 = to_action_0[3][action]
        new_a['camera'][0] = a1

        # -----#
        angle_1 -= new_a['camera'][0]
        angle_1 = np.clip(angle_1, -90, 90)

        angle_2 += new_a['camera'][1]
        if angle_2 > 360:
            angle_2 = angle_2 - 360
        elif angle_2 < 0:
            angle_2 = 360 + angle_2

        obs, reward, done, info = env.step(new_a)
        # if new_a['use'] == 1:
        #     print(obs['equipped_items']['mainhand']['type'])
        new_a['c2'] = (new_a['camera'][1] + np.min(range_angle_2) * -1) / range_angle_2_delta

        new_a['c1'] = (new_a['camera'][0] + np.min(range_angle) * -1) / range_angle_delta
        del new_a['camera']
        del new_a['equip']
        del new_a['sprint']
        del new_a['sneak']
        del new_a['craft']
        del new_a['nearbyCraft']
        del new_a['nearbySmelt']
        del new_a['place']

        tmp = np.expand_dims(np.array(list(new_a.values())), axis=0)
        stack_all_a_.append(tmp)
        if len(stack_all_a_) > max_last_action:
            del stack_all_a_[0]

        stack_all_last.append(np.ndarray.flatten(np.array(stack_all_a_))[None])


        total_reward += reward
        steps += 1
        # cv2.imshow('image', cv2.cvtColor(obs['pov'], cv2.COLOR_BGR2RGB))
        # cv2.resizeWindow('image', 950, 950)
        if done:
            break
        # if cv2.waitKey(10) & 0xFF == ord('o'):
        #     break
        # time.sleep(0.01)

    rewards.append(total_reward)

    print(f'Episode #{episode + 1} reward: {total_reward}\t\t episode length: {steps}\n')

    cv2.destroyAllWindows()
    print(obs['equipped_items'])

np.mean(rewards)



data = minerl.data.make("MineRLObtainDiamond-v0", data_dir='data', num_workers=4)
trajectory_names = data.get_trajectory_names()
random.shuffle(trajectory_names)
# np.bincount(all_actions[:,0])
cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow('image', 1000, 1000)
cv2.moveWindow('image', -30, 30)

cv2.namedWindow('grey', cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow('grey', 1000, 1000)
# Add trajectories to the data until we reach the required DATA_SAMPLES.
action_at = []
import time
np.min(action_at)


stack_all_pov = None
stack_all_action = None
stack_all_hand = None
stack_all_inventory = None
total = 0
for trajectory_name in trajectory_names:
    print("aa")
    stack_all_inventory = np.load(f'data/MineRLObtainDiamond-v0/{trajectory_name}/inventory_saver.npy')
    stack_all_action = np.load(f'data/MineRLObtainDiamond-v0/{trajectory_name}/action_saver.npy')
    stack_all_hand = np.load(f'data/MineRLObtainDiamond-v0/{trajectory_name}/hand_types_saver.npy')
    index = np.where((stack_all_hand[index[0]] == 'stone_pickaxe'))
    index_keys = np.where(obs_inve_keys_s == 'log')[0]
    total += np.max(stack_all_inventory[:,index_keys])

    if stack_all_action is None:
        # stack_all_pov = np.load(f'data/MineRLObtainDiamond-v0/{trajectory_name}/obs_pov_saver.npy')
        stack_all_action = np.load(f'data/MineRLObtainDiamond-v0/{trajectory_name}/action_saver.npy')
        stack_all_hand = np.load(f'data/MineRLObtainDiamond-v0/{trajectory_name}/hand_types_saver.npy')
        stack_all_inventory = np.load(f'data/MineRLObtainDiamond-v0/{trajectory_name}/inventory_saver.npy')
    # else:
    #     # stack_all_pov = np.concatenate((stack_all_pov,  np.load(f'data/MineRLObtainDiamond-v0/{trajectory_name}/obs_pov_saver.npy')), axis=0)
    #     stack_all_action = np.concatenate((stack_all_action,  np.load(f'data/MineRLObtainDiamond-v0/{trajectory_name}/action_saver.npy')), axis=0)
    #     stack_all_hand = np.concatenate((stack_all_hand,  np.load(f'data/MineRLObtainDiamond-v0/{trajectory_name}/hand_types_saver.npy')), axis=0)
    #     stack_all_inventory = np.concatenate((stack_all_inventory,  np.load(f'data/MineRLObtainDiamond-v0/{trajectory_name}/inventory_saver.npy')), axis=0)
    index_keys = np.where(obs_inve_keys_s == 'log')[0]
    total += np.max(stack_all_inventory[:,index_keys])

    print(obs['inventory']["cobblestone"])
    # log when make planks
    index_keys = np.where(action_keys_s == 'nearbyCraft')[0]
    index = np.where((stack_all_action[:,index_keys] == 'stone_axe'))
    print(len(index[0]))

    index = np.where((stack_all_hand[index[0]] == 'stone_axe'))
    print(len(index[0]))

    index_keys = np.where(obs_inve_keys_s == 'cobblestone')[0]
    logs_mean = stack_all_inventory[index, index_keys]
    np.mean(logs_mean)
    shift_feture = np.roll(stack_all_inventory, 1, axis=0)
    delta = shift_feture - stack_all_inventory
    delta = np.sum(delta, axis=1)
    index = np.where(delta > 1)
    len(delta)
    index_keys = np.where(obs_inve_keys_s == 'coal')[0][0]
    np.sum(delta[:, index_keys] > 0)


    index_keys = np.where(action_keys_s == 'jump')[0]
    index = np.where((stack_all_action[:,index_keys] == '1'))
    print(len(index[0]))
    index_keys = np.where(action_keys_s == 'forward')[0]
    index = np.where((stack_all_action[index[0],index_keys] == '1'))
    print(len(index[0]))

    index_keys = np.where(action_keys_s == 'jump')[0]
    index = np.where((stack_all_action[:,index_keys] == '1'))
    print(len(index[0]))
    index_keys = np.where(action_keys_s == 'back')[0]
    index = np.where((stack_all_action[index[0], index_keys] == '1'))
    print(len(index[0]))

    stack_all_action[:, index_keys[0]].shape
    index_keys = np.where(action_keys_s == 'c1')[0]
    c1 = stack_all_action[:, index_keys[0]].astype(np.float32)
    index_keys = np.where(action_keys_s == 'c2')[0]
    c2 = stack_all_action[:, index_keys[0]].astype(np.float32)


    index = np.where((c1 != 0))
    print(len(index[0])/2)
    index = np.where((c2 != 0))
    print(len(index[0])/2)

    angle_1 = 0
    angle_2 = 0
    angle_2_f = 0
    stack_grey ={}

    obs_pov_saver = []
    inventory_saver = []
    hand_types_saver = []
    action_saver = []
    for obs, action, r, _, _ in trajectory:

        # actions_eps = np.array(actions_eps)
        # for i in range(2):
        #     for j in range(4):
        #         np.save(f'data/MineRLObtainDiamond-v0/{trajectory_name}/{i}_{j}.npy', actions_eps[:, i, j], )
        a, _ = dataset_action_batch_to_actions(action)
        grey = np.dot(obs['pov'][..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        angle_2_fix = (np.round(int(angle_2) / 15) * 15)
        stack_grey[angle_2_fix] = grey

        all_angle_had = np.array(list(stack_grey.keys()))
        index_current = np.where(all_angle_had == angle_2_fix)[0][0]
        a_img = []
        for location in list_obs:
            a2 = angle_2_fix + location
            if a2 > 360:
                a2 = a2 - 360
            elif a2 < 0:
                a2 = 360 + a2
            a2 = np.abs(all_angle_had - a2)
            a2[index_current] += 1000
            a_min = np.min(a2)
            if a_min > 20:
                a_img.append(np.zeros((64, 64, 1)))
            else:
                a_img.append(stack_grey[all_angle_had[np.argmin(a2)]])

        angle_1 -= action['camera'][0]
        angle_1 = np.clip(angle_1, -90, 90)
        if a[3] not in [0,1]:
            print(action_key[3][a[3]])

        angle_2 += action['camera'][1]
        if angle_2 > 360:
            angle_2 = angle_2 - 360
        elif angle_2 < 0:
            angle_2 = 360 + angle_2
        cv2.imshow('image', cv2.cvtColor(obs['pov'], cv2.COLOR_BGR2RGB))
        cv2.resizeWindow('image', 950, 950)

        cv2.imshow('grey', a_img[0])
        cv2.resizeWindow('grey', 950, 950)

        if cv2.waitKey(10) & 0xFF == ord('o'):
            print(angle_1)
            break
        time.sleep(0.1)


cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow('image', 1000, 1000)
cv2.moveWindow('image', -30, 30)
total = []
for trajectory_name in trajectory_names:
    trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)
    angle_1 = 0
    angle_2 = 0

    for obs, action, r, _, _ in trajectory:
        angle_1 -= action['camera'][0]
        angle_1 = np.clip(angle_1, -90, 90)

        angle_2 += action['camera'][1]
        if angle_2 > 360:
            angle_2 = angle_2 - 360
        elif angle_2 < 0:
            angle_2 = 360 + angle_2

    cv2.imshow('image', cv2.cvtColor(obs['pov'], cv2.COLOR_BGR2RGB))
    cv2.resizeWindow('image', 1000, 1000)

    # cv2.imshow('grey', a_img[0])
    # cv2.resizeWindow('grey', 950, 950)


    if cv2.waitKey(10) & 0xFF == ord('o'):
        break
    time.sleep(5)

for trajectory_name in trajectory_names:
    trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)

    angle_1 = 0
    angle_2 = 0
    angle_2_f = 0
    stack_grey = {}

    obs_pov_saver = []
    inventory_saver = []
    hand_types_saver = []
    action_saver = []
    for obs, action, r, _, _ in trajectory:
        obs_pov_saver.append(obs['pov'])
        hand_types_saver.append(obs['equipped_items.mainhand.type'])
        inventory_saver.append(np.array(list(obs['inventory'].values())))
        action['c1'] = action['camera'][0]
        action['c2'] = action['camera'][1]
        del action['camera']
        action_saver.append(np.array(list(action.values())))
    # obs_pov_saver = np.array(obs_pov_saver)
    # np.save(f'data/MineRLObtainDiamond-v0/{trajectory_name}/obs_pov_saver.npy', obs_pov_saver)
    #
    # inventory_saver = np.array(inventory_saver)
    # np.save(f'data/MineRLObtainDiamond-v0/{trajectory_name}/inventory_saver.npy', inventory_saver)
    # hand_types_saver = np.array(hand_types_saver)
    # np.save(f'data/MineRLObtainDiamond-v0/{trajectory_name}/hand_types_saver.npy', hand_types_saver)
    action_saver = np.array(action_saver)
    np.save(f'data/MineRLObtainDiamond-v0/{trajectory_name}/action_saver.npy', action_saver)

cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow('image', 1000, 1000)
cv2.moveWindow('image', -30, 30)
for i in range(len(all_visibale_obs)):
    image = all_visibale_obs[len(all_visibale_obs) - i -1]
    cv2.imshow('image', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cv2.resizeWindow('image', 1000, 1000)
    if cv2.waitKey(10) & 0xFF == ord('o'):
        break

cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow('image', 1000, 1000)
cv2.moveWindow('image', -30, 30)
for i in range(62805,len(all_visibale_obs)):
    image = all_visibale_obs[i]
    cv2.imshow('image', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cv2.resizeWindow('image', 1000, 1000)
    if cv2.waitKey(1) & 0xFF == ord('o'):
        break