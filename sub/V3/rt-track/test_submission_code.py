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
from torch.autograd import Variable

from hummingbird.ml import convert
import matplotlib.pyplot as plt
import copy

class EpisodeDone(Exception):
    pass

MAX_TEST_EPISODE_LEN = 18000  # 18k is the default for MineRLObtainDiamond.

class Episode(gym.Env):
    """A class for a single episode."""
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._done = False

    def reset(self):
        if not self._done:
            return self.env.reset()

    def step(self, action):
        s, r, d, i = self.env.step(action)
        if d:
            self._done = True
            raise EpisodeDone()
        else:
            return s, r, d, i

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


# def process_a(action, action_new):
#     key = list_key_index[action]
#     for k, v in action_key[key].items():
#         action_new[k] = v
#     # action_new['attack'] = 1
#     return

# def process_inventory(obs, attack, t, full_previous, angle_1, angle_2, inven_process):
#     """
#     Turn a batch of actions from dataset (`batch_iter`) to a numpy
#     array that corresponds to batch of actions of ActionShaping wrapper (_actions).
#
#     Camera margin sets the threshold what is considered "moving camera".
#
#     Note: Hardcoded to work for actions in ActionShaping._actions, with "intuitive"
#         ordering of actions.
#         If you change ActionShaping._actions, remember to change this!
#
#     Array elements are integers corresponding to actions, or "-1"
#     for actions that did not have any corresponding discrete match.
#     """
#     data = np.zeros(22)
#
#     if inven_process['mainhand'] == 0: #iron
#         data[0] = 1
#     if inven_process['mainhand'] == 1:# stone
#         data[1] = 1
#     if inven_process['mainhand'] == 2:# wodden
#         data[2] = 1
#     if inven_process['mainhand'] == 3:# other
#         data[3] = 1
#
#     data[4] = np.clip(inven_process['furnace'] / 3, 0, 1)
#     data[5] = np.clip(inven_process['crafting_table'] / 3, 0, 1)
#
#     data[6] = np.clip(inven_process['planks'] / 30, 0, 1)
#     data[7] = np.clip(inven_process['stick'] / 20, 0, 1)
#
#     data[8] = np.clip(inven_process['stone_pickaxe'] / 3, 0, 1)
#     data[9] = np.clip(inven_process['wooden_pickaxe'] / 3, 0, 1)
#     data[10] = np.clip(inven_process['iron_pickaxe'] / 3, 0, 1)
#
#     data[11] = np.clip(inven_process['log'] / 20, 0, 1)
#     data[12] = np.clip((inven_process['cobblestone']) / 20, 0, 1)
#     data[13] = np.clip((inven_process['stone']) / 20, 0, 1)
#
#     data[14] = np.clip(inven_process['iron_ore'] / 10, 0, 1)
#     data[15] = np.clip(inven_process['iron_ingot'] / 10, 0, 1)
#     data[16] = np.clip(inven_process['coal'] / 10, 0, 1)
#     data[17] = np.clip(inven_process['dirt'] / 50, 0, 1)
#
#     # data[18] = t/18000
#     data[18] = (angle_1 + 90) / 180
#
#     angle_2 = int(angle_2)
#     data[19] = np.clip(((angle_2 + 180) % 360) / 360, 0, 1)
#
#     data[20] = np.clip(attack / 300, 0, 1)
#     data[21] = np.clip((inven_process['dirt'] + inven_process['stone']
#                         + inven_process['cobblestone']) / 300, 0, 1)
#
#     # data[15] = attack
#     # data = np.concatenate((data, full_previous), axis=0)
#
#     # a2 = int(a2)
#     # data[15] = np.clip((a1 + 180)/ 180, 0, 1)
#     # data[16] = np.clip(((a2 + 180)% 360)/360, 0, 1)
#     # data[15] = np.clip((a1 + 180)/ 180, 0, 1)
#     # data[16] = np.clip(((a2 + 180)% 360)/360, 0, 1)
#     return data

def process_inventory(obs, attack, t, full_previous, angle_1, angle_2, inven_process):
    """
    Turn a batch of actions from dataset (`batch_iter`) to a numpy
    array that corresponds to batch of actions of ActionShaping wrapper (_actions).

    Camera margin sets the threshold what is considered "moving camera".

    Note: Hardcoded to work for actions in ActionShaping._actions, with "intuitive"
        ordering of actions.
        If you change ActionShaping._actions, remember to change this!

    Array elements are integers corresponding to actions, or "-1"
    for actions that did not have any corresponding discrete match.
    """

    data = np.zeros(23)

    if inven_process['mainhand'] == 0: #iron
        data[0] = 1
    if inven_process['mainhand'] == 1:# stone
        data[1] = 1
    if inven_process['mainhand'] == 2:# wodden
        data[2] = 1
    if inven_process['mainhand'] == 3:# other
        data[3] = 1

    data[4] = np.clip(inven_process['furnace'] / 3, 0, 1)
    data[5] = np.clip(inven_process['crafting_table'] / 3, 0, 1)

    data[6] = np.clip(inven_process['planks'] / 30, 0, 1)
    data[7] = np.clip(inven_process['stick'] / 20, 0, 1)

    data[8] = np.clip(inven_process['stone_pickaxe'] / 3, 0, 1)
    data[9] = np.clip(inven_process['wooden_pickaxe'] / 3, 0, 1)
    data[10] = np.clip(inven_process['iron_pickaxe'] / 3, 0, 1)

    data[11] = np.clip(inven_process['log'] / 20, 0, 1)
    data[12] = np.clip((inven_process['cobblestone']) / 20, 0, 1)
    data[13] = np.clip((inven_process['stone']) / 20, 0, 1)

    data[14] = np.clip(inven_process['iron_ore'] / 10, 0, 1)
    data[15] = np.clip(inven_process['iron_ingot'] / 10, 0, 1)
    data[16] = np.clip(inven_process['coal'] / 10, 0, 1)
    data[17] = np.clip(inven_process['dirt'] / 50, 0, 1)

    # data[18] = t/18000
    data[18] = (angle_1 + 90) / 180

    angle_2 = int(angle_2)
    data[19] = np.clip(((angle_2 + 180) % 360) / 360, 0, 1)

    data[20] = np.clip(attack / 300, 0, 1)
    data[21] = np.clip((inven_process['dirt'] + inven_process['stone']
                        + inven_process['cobblestone']) / 300, 0, 1)

    data[22] = np.clip(inven_process['torch'] / 20, 0, 1)


    # data[15] = attack
    # data = np.concatenate((data, full_previous), axis=0)

    # a2 = int(a2)
    # data[15] = np.clip((a1 + 180)/ 180, 0, 1)
    # data[16] = np.clip(((a2 + 180)% 360)/360, 0, 1)
    # data[15] = np.clip((a1 + 180)/ 180, 0, 1)
    # data[16] = np.clip(((a2 + 180)% 360)/360, 0, 1)
    return data


def check_successed_craft(craft_type, items):
    if craft_type == 'craft_stick':
        items['planks'] -= 2
        items['stick'] += 4
        if items['planks'] < 0: items['planks'] = 0

    elif craft_type == 'craft_planks':
        items['log'] -= 1
        items['planks'] += 4
        if items['log'] < 0: items['log'] = 0

    elif craft_type == 'craft_crafting_table':
        items['planks'] -= 4
        items['crafting_table'] += 1
        if items['planks'] < 0: items['planks'] = 0

    elif craft_type == 'craft_torch':
        items['stick'] -= 1
        items['coal'] -= 1
        items['torch'] += 4
        if items['stick'] < 0: items['stick'] = 0
        if items['coal'] < 0: items['coal'] = 0

    elif craft_type == 'nearbyCraft_furnace':
        items['cobblestone'] -= 8
        items['furnace'] += 1
        if items['cobblestone'] < 0: items['cobblestone'] = 0

    elif craft_type == 'nearbyCraft_stone_pickaxe':
        items['cobblestone'] -= 3
        items['stick'] -= 2
        items['stone_pickaxe'] += 1
        if items['cobblestone'] < 0: items['cobblestone'] = 0
        if items['stick'] < 0: items['stick'] = 0

    elif craft_type == 'nearbyCraft_wooden_pickaxe':
        items['planks'] -= 3
        items['stick'] -= 2
        items['wooden_pickaxe'] += 1
        if items['planks'] < 0: items['planks'] = 0
        if items['stick'] < 0: items['stick'] = 0

    elif craft_type == 'nearbyCraft_iron_pickaxe':
        items['iron_ingot'] -= 3
        items['stick'] -= 2
        items['iron_pickaxe'] += 1
        if items['iron_ingot'] < 0: items['iron_ingot'] = 0
        if items['stick'] < 0: items['stick'] = 0

    elif craft_type == 'nearbySmelt_iron_ingot':
        items['iron_ingot'] += 1
        items['iron_ore'] -= 1
        if items['iron_ore'] < 0: items['iron_ore'] = 0
        if items['planks'] != 0:
            items['planks'] -= 1
        elif items['coal'] != 0:
            items['coal'] -= 1
    elif craft_type == 'nearbySmelt_coal':
        items['coal'] += 1
        items['log'] -= 1
        if items['log'] < 0: items['log'] = 0

    elif craft_type == 'place_crafting_table':
        items['crafting_table'] -= 1
        if items['crafting_table'] < 0: items['crafting_table'] = 0
    elif craft_type == 'place_furnace':
        items['furnace'] -= 1
        if items['furnace'] < 0: items['furnace'] = 0
    return items


def check_can_craft(craft_type, items):
    if craft_type == 'craft_stick':
        if items['planks'] >= 1:
            return True

    elif craft_type == 'craft_planks':
        if items['log'] >= 1:
            return True

    elif craft_type == 'craft_crafting_table':
        if items['planks'] >= 4:
            return True
    elif craft_type == 'craft_torch':
        if items['stick'] >= 1 and items['coal'] >= 1:
            return True

    elif craft_type == 'nearbyCraft_furnace':
        if items['cobblestone'] >= 8:
            return True

    elif craft_type == 'nearbyCraft_stone_pickaxe':
        if items['stick'] >= 2 and items['cobblestone'] >= 3:
            return True

    elif craft_type == 'nearbyCraft_wooden_pickaxe':
        if items['stick'] >= 2 and items['planks'] >= 3:
            return True
    elif craft_type == 'nearbyCraft_iron_pickaxe':
        if items['stick'] >= 2 and items['iron_ingot'] >= 3:
            return True
    elif craft_type == 'nearbySmelt_iron_ingot':
        if (items['coal'] >= 1 or items['planks'] >= 1) and items['iron_ore'] >= 1:
            return True
    elif craft_type == 'nearbySmelt_coal':
        if items['log'] >= 1:
            return True
    elif craft_type == 'place_crafting_table':
        if items['crafting_table'] >= 1:
            return True
    elif craft_type == 'place_furnace':
        if items['furnace'] >= 1:
            return True
    elif craft_type in ['equip_iron_pickaxe',
       'equip_stone_pickaxe', 'equip_wooden_pickaxe']:
        return True
    return False

class MineRLAgent():
    """
    To compete in the competition, you are required to implement the two
    functions in this class:
        - load_agent: a function that loads e.g. network models
        - run_agent_on_episode: a function that plays one game of MineRL

    By default this agent behaves like a random agent: pick random action on
    each step.

    NOTE:
        This class enables the evaluator to run your agent in parallel in Threads,
        which means anything loaded in load_agent will be shared among parallel
        agents. Take care when tracking e.g. hidden state (this should go to run_agent_on_episode).
    """

    def load_agent(self):
        """
        This method is called at the beginning of the evaluation.
        You should load your model and do any preprocessing here.
        THIS METHOD IS ONLY CALLED ONCE AT THE BEGINNING OF THE EVALUATION.
        DO NOT LOAD YOUR MODEL ANYWHERE ELSE.
        """
        # This is a random agent so no need to do anything
        # YOUR CODE GOES HERE

        # np.save("all_key_actions.npy", action_key)
        # with open('action_key_1.pkl', 'wb') as f:
        #     pickle.dump(action_key, f)
        with open('./train/action_key_1.pkl', 'rb') as f:
            self.action_key = pickle.load(f)
        self.list_key_index = np.array(list(self.action_key.keys()))

        self.network = NatureCNN((7, 64, 64),  len(self.list_key_index)).cuda()
        self.network.load_state_dict(th.load("./train/another_potato_4.pth"))



        # inventory_list = ['coal', 'cobblestone', 'crafting_table', 'dirt', 'furnace', 'iron_ore', 'iron_ingot',
        #                   'iron_pickaxe',
        #                   'log', 'planks', 'stick', 'stone', 'stone_pickaxe', 'torch', 'wooden_pickaxe', 'mainhand']
        #


        with open('./train/action_connect_to_vector.pkl', 'rb') as f:
            self.action_vector_all = pickle.load(f)

        next_items_list = ['coal', 'cobblestone', 'crafting_table', 'dirt', 'furnace', 'iron_ore', 'log', 'stone']
        KNN = np.load("./train/KNN.npy")
        self.KNN = KNN[:, 64:] - KNN[:, :64]
        self.y_mask = np.load('./train/y_mask.npy')
        pass

    def run_agent_on_episode(self, single_episode_env: Episode):
        """This method runs your agent on a SINGLE episode.

        You should just implement the standard environment interaction loop here:
            obs  = env.reset()
            while not done:
                env.step(self.agent.act(obs))
                ...

        NOTE:
            This method will be called in PARALLEL during evaluation.
            So, only store state in LOCAL variables.
            For example, if using an LSTM, don't store the hidden state in the class
            but as a local variable to the method.

        Args:
            env (gym.Env): The env your agent should interact with.
        """
        # An implementation of a random agent
        # YOUR CODE GOES HERE
        env = single_episode_env
        none_move_action = np.array(['attack', 'craft_crafting_table',
                                     'craft_planks', 'craft_stick', 'equip_iron_pickaxe',
                                     'equip_stone_pickaxe', 'equip_wooden_pickaxe',
                                     'nearbyCraft_furnace', 'nearbyCraft_iron_pickaxe',
                                     'nearbyCraft_stone_pickaxe', 'nearbyCraft_wooden_pickaxe',
                                     'nearbySmelt_iron_ingot', 'place_crafting_table', 'place_furnace',
                                     'nearbySmelt_coal', 'craft_torch', 'place_torch'])

        obs = env.reset()
        last_inventory = obs['vector']

        list_obs = [[-15, 0], [15, 0], [0, -15], [0, 15]]
        next_equip = False
        next_items_list = ['coal', 'cobblestone', 'crafting_table', 'dirt', 'furnace', 'iron_ore', 'log', 'stone']
        counter_craft = []
        last_craft = []


        inventory_list = ['coal', 'cobblestone', 'crafting_table', 'dirt', 'furnace', 'iron_ore', 'iron_ingot',
                          'iron_pickaxe', 'torch',
                          'log', 'planks', 'stick', 'stone', 'stone_pickaxe', 'torch', 'wooden_pickaxe', 'mainhand']

        items_counter = {}
        for k_a in inventory_list:
            items_counter[k_a] = 0
        items_counter['mainhand'] = 3

        done = False
        total_reward = 0
        steps = 0

        action_list = np.arange(len(self.list_key_index))

        angle_1 = 0
        angle_2 = 0

        time_add = 0

        time_attack_no_new = 0
        current_item = 0

        previous = None
        check_attack = False

        stack_grey = {}
        first_add = 0
        for i in range(MAX_TEST_EPISODE_LEN):

            hsv = obs['pov']
            grey = np.dot(obs['pov'][..., :3], [0.2989, 0.5870, 0.1140])

            angle_2_fix = ((int(angle_2) + 180) % 360)
            stack_grey[tuple((angle_1, angle_2_fix))] = grey

            for location in list_obs:
                a1 = angle_1 + location[0]
                a2 = ((int(angle_2 + location[1]) + 180) % 360)
                new_tuple = tuple((a1, a2))
                if new_tuple in stack_grey:
                    hsv = np.concatenate((hsv, stack_grey[new_tuple][:, :, None]), axis=-1)
                else:
                    hsv = np.concatenate((hsv, np.zeros((64, 64, 1))), axis=-1)
            # hsv = np.concatenate((hsv, first_obs, second_obs), axis=-1)
            hsv = hsv.transpose(2, 0, 1).astype(np.float32)
            hsv = th.from_numpy(hsv[None]).float().cuda()
            hsv /= 255.0


            inven = process_inventory(obs, time_attack_no_new, time_add, [], angle_1, angle_2, items_counter)
            inven = th.from_numpy(inven[None]).float().cuda()

            probabilities = th.softmax(self.network(hsv, inven), dim=1)[0]
            probabilities = probabilities.detach().cpu().numpy()
            action = np.random.choice(action_list, p=probabilities)

            key = self.list_key_index[action]
            if key == 'attack_0_0' and (check_attack or time_attack_no_new >= 300):
                action = np.random.choice([0, 1, 2, 3, 5, 6, 7, 8], p=np.repeat(0.125, 8))
                key = self.list_key_index[action]
                time_attack_no_new = 0

            if next_equip or time_add % 1000 == 0 or key in \
                    ['equip_stone_pickaxe', 'iron_pickaxe',
                     'equip_wooden_pickaxe']:
                if items_counter['iron_pickaxe'] != 0:
                    key = 'equip_iron_pickaxe'
                    items_counter['mainhand'] = 0
                elif items_counter['stone_pickaxe'] != 0:
                    key = 'equip_stone_pickaxe'
                    items_counter['mainhand'] = 1
                elif items_counter['wooden_pickaxe'] != 0:
                    key = 'equip_wooden_pickaxe'
                    items_counter['mainhand'] = 2
                else:
                    items_counter['mainhand'] = 3
                next_equip = False

            if key in none_move_action:
                # if check_can_craft(key, items_counter):
                last_craft.append(key)
                counter_craft.append(time_add)

            slipt = key.split('_')
            if 'attack' in slipt:
                if time_attack_no_new == 0:
                    current_item = copy.deepcopy(last_inventory)

                    time_attack_no_new += 1

                else:
                    if np.sum(current_item - last_inventory) != 0.0:
                        time_attack_no_new = 0
                    else:
                        time_attack_no_new += 1
            else:
                time_attack_no_new = 0


            if len(slipt) >= 2:
                if slipt[-1] == '-1':
                    angle_2 -= 15
                elif slipt[-1] == '1':
                    angle_2 += 15

                if slipt[-2] == '-1':
                    angle_1 -= 15
                elif slipt[-2] == '1':
                    angle_1 += 15

                angle_1 = np.clip(angle_1, -90, 90)


            action = env.action_space.noop()
            action['vector'] = self.action_vector_all[key][0]
            obs, reward, done, info = env.step(action)
            time_add += 1

            inventory_change = np.sum(last_inventory - obs['vector']) != 0
            craft_success = False
            if len(last_craft) != 0:
                first_craft = last_craft[0]
                first_time = counter_craft[0]
                if first_time + 3 == time_add:
                    if inventory_change:
                        craft_success = True
                        items_counter = check_successed_craft(first_craft, items_counter)
                        if first_craft in ['nearbyCraft_iron_pickaxe','nearbyCraft_stone_pickaxe',
                                           'nearbyCraft_wooden_pickaxe']:
                            next_equip = True
                    del last_craft[0]
                    del counter_craft[0]

            if inventory_change and not craft_success:
                # pov_items = np.concatenate((previous, obs['pov']), axis=-1)
                # pov_items = pov_items.transpose(2, 0, 1).astype(np.float32)
                # pov_items = th.from_numpy(pov_items[None]).float().cuda()
                # p_2 = th.softmax(self.network_3(pov_items), dim=1)[0]
                # p_2 = p_2.detach().cpu().numpy()
                # if next_items_list[np.argmax(p_2)] not in ['crafting_table', 'furnace']:
                #     items_counter[next_items_list[np.argmax(p_2)]] += 1
                # elif next_items_list[np.argmax(p_2)] == 'crafting_table':
                #     items_counter['log'] += 1
                if first_add != 0:
                    error_distance = np.sum(np.sqrt((self.KNN - (obs['vector'] - last_inventory)) ** 2), axis=1)
                    items_counter[next_items_list[self.y_mask[np.argmin(error_distance)]]] += 1
                first_add += 1

            last_inventory = obs['vector']

            if previous is not None:
                delta = previous - obs['pov']
                delta = np.sum(delta)
                if delta == 0:
                    check_attack = True
                else:
                    check_attack = False
            previous = obs['pov']

            total_reward += reward
            steps += 1
            if done:
                break

