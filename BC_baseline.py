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

with open('action_key_2.pkl', 'rb') as f:
    action_key = pickle.load(f)
list_key_index = np.array(list(action_key.keys()))
print(len(list_key_index))

DATA_DIR = "data"  # path to MineRL dataset (should contain "MineRLObtainIronPickaxeVectorObf-v0" directory).
EPOCHS = 2  # how many times we train over dataset.
LEARNING_RATE = 0.0001  # learning rate for the neural network.
BATCH_SIZE = 64
DATA_SAMPLES = 1000000

class NatureCNN(nn.Module):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param input_shape: A three-item tuple telling image dimensions in (C, H, W)
    :param output_dim: Dimensionality of the output vector
    """

    def __init__(self, input_shape, output_dim):
        super().__init__()
        n_input_channels = input_shape[0]
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
            nn.Linear(23, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        # self.flat_cnn = nn.Sequential(
        #     nn.Linear(n_flatten, 512),
        #     nn.ReLU(),
        # )
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + 512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, observations: th.Tensor, data: th.Tensor) -> th.Tensor:
        x1 = self.cnn(observations)
        x2 = self.linear_stack(data)
        x = torch.cat((x1, x2), dim=1)

        return self.linear(x)



def process_a(action, action_new):
    key = list_key_index[action]
    action_new = copy.deepcopy(action_new)
    tmp = copy.deepcopy(action_key)
    for k, v in tmp[key].items():
        action_new[k] = v
        if k == 'camera':
            action_new[k][0] =action_new[k][0]*2
            action_new[k][1] = action_new[k][1]*2

    return action_new

def dataset_action_batch_to_actions(actions):
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
    action_with_key = actions
    dict_action = {}
    if action_with_key['craft'] != 'none':
        craft = action_with_key['craft']
        k = f'craft_{craft}'
        dict_action['craft'] = craft

    elif action_with_key['equip'] != 'none':

        equip = action_with_key['equip']
        if equip == 'iron_axe': equip = 'iron_pickaxe'
        if equip == 'stone_axe': equip = 'stone_pickaxe'
        if equip == 'wooden_axe': equip = 'wooden_pickaxe'
        k = f'equip_{equip}'
        dict_action['equip'] = equip

    elif action_with_key['nearbyCraft'] != 'none':
        nearbyCraft = action_with_key['nearbyCraft']
        if nearbyCraft == 'iron_axe': nearbyCraft = 'iron_pickaxe'
        if nearbyCraft == 'stone_axe': nearbyCraft = 'stone_pickaxe'
        if nearbyCraft == 'wooden_axe': nearbyCraft = 'wooden_pickaxe'
        k = f'nearbyCraft_{nearbyCraft}'
        dict_action['nearbyCraft'] = nearbyCraft

    elif action_with_key['nearbySmelt'] != 'none':
        nearbySmelt = action_with_key['nearbySmelt']
        k = f'nearbySmelt_{nearbySmelt}'
        dict_action['nearbySmelt'] = nearbySmelt

    elif action_with_key['place'] != 'none':
        place = action_with_key['place']
        k = f'place_{place}'
        dict_action['place'] = place
    elif action_with_key['attack'] == 1:
        dict_action['attack'] = 1
        k, dict_action = find_key('jump', action_with_key, 'attack', dict_action)
    else:
        k, dict_action = find_key('jump', action_with_key, '', dict_action)
    return k, dict_action

def find_key(byWhat, action_with_key, key, dict_action):
    first_angle = action_with_key['camera'][0]
    second_angle = action_with_key['camera'][1]
    if byWhat == 'camera':
        camera = []
        if np.abs(first_angle) > 3:
            if first_angle < 0:
                key += '_-1'
                camera.append(-5)
            else:
                key += '_1'
                camera.append(5)
        else:
            key += '_0'
            camera.append(0)

        if np.abs(second_angle) > 3:
            if second_angle < 0:
                key += '_-1'
                camera.append(-5)
            else:
                key += '_1'
                camera.append(5)
        else:
            key += '_0'
            camera.append(0)
        dict_action['camera'] = camera
        return key, dict_action
    elif byWhat == 'jump':
        if action_with_key['jump'] == 1:
            key += '_j'
            dict_action['jump'] = 1
            key, dict_action = find_key('move', action_with_key, key, dict_action)

        elif action_with_key['back'] == 1:
            key += '_b'
            dict_action['back'] = 1
        elif action_with_key['right'] == 1:
            key += '_r'
            dict_action['right'] = 1
        elif action_with_key['left'] == 1:
            key += '_l'
            dict_action['left'] = 1
        else:
            key, dict_action = find_key('move', action_with_key, key, dict_action)
    elif byWhat == 'move':
        if action_with_key['forward'] == 1:
            key += '_f'
            dict_action['forward'] = 1
        key, dict_action = find_key('camera', action_with_key, key, dict_action)
    return key, dict_action


def process_inventory(obs, attack, angle_1, angle_2):
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
    # rs = [1, 2, 4, 8, 16, 32, 32, 64, 128, 256, 1024]
    data = np.zeros(23)

    if obs['equipped_items.mainhand.type'] in ['iron_pickaxe', 'iron_iron_axe']:
        data[0] = 1
    if obs['equipped_items.mainhand.type'] in ['stone_pickaxe', 'stone_axe']:
        data[1] = 1
    if obs['equipped_items.mainhand.type'] in ['wooden_pickaxe', 'wooden_axe']:
        data[2] = 1
    if obs['equipped_items.mainhand.type'] in ['other', 'none', 'air']:
        data[3] = 1

    data[4] = np.clip(obs['inventory']['furnace'] / 3, 0, 1)
    data[5] = np.clip(obs['inventory']['crafting_table'] / 3, 0, 1)

    data[6] = np.clip(obs['inventory']['planks'] / 30, 0, 1)
    data[7] = np.clip(obs['inventory']['stick'] / 20, 0, 1)

    data[8] = np.clip(obs['inventory']['stone_pickaxe'] / 3, 0, 1)
    data[9] = np.clip(obs['inventory']['wooden_pickaxe'] / 3, 0, 1)
    data[10] = np.clip(obs['inventory']['iron_pickaxe'] / 3, 0, 1)

    data[11] = np.clip(obs['inventory']['log'] / 20, 0, 1)
    data[12] = np.clip((obs['inventory']['cobblestone']) / 20, 0, 1)
    data[13] = np.clip((obs['inventory']['stone']) / 20, 0, 1)

    data[14] = np.clip(obs['inventory']['iron_ore'] / 10, 0, 1)
    data[15] = np.clip(obs['inventory']['iron_ingot'] / 10, 0, 1)
    data[16] = np.clip(obs['inventory']['coal'] / 10, 0, 1)
    data[17] = np.clip(obs['inventory']['dirt'] / 50, 0, 1)

    # data[18] = t/18000
    data[18] = (angle_1 + 90) / 180

    angle_2 = int(angle_2)
    data[19] = np.clip(((angle_2 + 180) % 360) / 360, 0, 1)

    data[20] = np.clip(attack / 300, 0, 1)
    data[21] = np.clip((obs['inventory']['dirt'] + obs['inventory']['stone']
                        + obs['inventory']['cobblestone']) / 300, 0, 1)

    data[22] = np.clip(obs['inventory']['torch'] / 20, 0, 1)
    # for r in rewards
    # data[23] = np.clip(obs['inventory']['torch'] / 50, 0, 1)

    # data[15] = attack
    # data = np.concatenate((data, full_previous), axis=0)

    # a2 = int(a2)
    # data[15] = np.clip((a1 + 180)/ 180, 0, 1)
    # data[16] = np.clip(((a2 + 180)% 360)/360, 0, 1)
    # data[15] = np.clip((a1 + 180)/ 180, 0, 1)
    # data[16] = np.clip(((a2 + 180)% 360)/360, 0, 1)
    return data

def process_inventory_test(obs, attack, angle_1, angle_2):
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
    # rs = [1, 2, 4, 8, 16, 32, 32, 64, 128, 256, 1024]
    data = np.zeros(23)

    if obs['equipped_items']['mainhand']['type'] == 'iron_pickaxe':
        data[0] = 1
    if obs['equipped_items']['mainhand']['type'] == 'stone_pickaxe':
        data[1] = 1
    if obs['equipped_items']['mainhand']['type'] == 'wooden_pickaxe':
        data[2] = 1
    if obs['equipped_items']['mainhand']['type'] in ['other', 'none', 'air']:
        data[3] = 1

    data[4] = np.clip(obs['inventory']['furnace'] / 3, 0, 1)
    data[5] = np.clip(obs['inventory']['crafting_table'] / 3, 0, 1)

    data[6] = np.clip(obs['inventory']['planks'] / 30, 0, 1)
    data[7] = np.clip(obs['inventory']['stick'] / 20, 0, 1)

    data[8] = np.clip(obs['inventory']['stone_pickaxe'] / 3, 0, 1)
    data[9] = np.clip(obs['inventory']['wooden_pickaxe'] / 3, 0, 1)
    data[10] = np.clip(obs['inventory']['iron_pickaxe'] / 3, 0, 1)

    data[11] = np.clip(obs['inventory']['log'] / 20, 0, 1)
    data[12] = np.clip((obs['inventory']['cobblestone']) / 20, 0, 1)
    data[13] = np.clip((obs['inventory']['stone']) / 20, 0, 1)

    data[14] = np.clip(obs['inventory']['iron_ore'] / 10, 0, 1)
    data[15] = np.clip(obs['inventory']['iron_ingot'] / 10, 0, 1)
    data[16] = np.clip(obs['inventory']['coal'] / 10, 0, 1)
    data[17] = np.clip(obs['inventory']['dirt'] / 50, 0, 1)

    # data[18] = t/18000
    data[18] = (angle_1 + 90) / 180

    angle_2 = int(angle_2)
    data[19] = np.clip(((angle_2 + 180) % 360) / 360, 0, 1)

    data[20] = np.clip(attack / 300, 0, 1)
    data[21] = np.clip((obs['inventory']['dirt'] + obs['inventory']['stone']
                        + obs['inventory']['cobblestone']) / 300, 0, 1)

    data[22] = np.clip(obs['inventory']['torch'] / 20, 0, 1)
    # for r in rewards
    # data[23] = np.clip(obs['inventory']['torch'] / 50, 0, 1)

    # data[15] = attack
    # data = np.concatenate((data, full_previous), axis=0)

    # a2 = int(a2)
    # data[15] = np.clip((a1 + 180)/ 180, 0, 1)
    # data[16] = np.clip(((a2 + 180)% 360)/360, 0, 1)
    # data[15] = np.clip((a1 + 180)/ 180, 0, 1)
    # data[16] = np.clip(((a2 + 180)% 360)/360, 0, 1)
    return data

def checking_craft_place(obs, action):
    inventory = obs['inventory']
    if action["craft"] == 'planks':
        if inventory['log'] == 0:
            action["craft"] = 'none'

    if action["craft"] == 'stick':
        if inventory['planks'] <= 1:
            action["craft"] = 'none'
    if action["craft"] == 'crafting_table':
        if inventory['planks'] < 4:
            action["craft"] = 'none'
    if action["craft"] == 'torch':
        if inventory['coal'] < 1 or inventory['stick'] < 1:
            action["craft"] = 'none'

    if action["nearbyCraft"] == 'wooden_pickaxe':
        if inventory['stick'] < 2 or inventory['planks'] < 3:
            action["nearbyCraft"] = 'none'
    if action["nearbyCraft"] == 'stone_pickaxe':
        if inventory['stick'] < 2 or inventory['cobblestone'] < 3:
            action["nearbyCraft"] = 'none'
    if action["nearbyCraft"] == 'furnace':
        if inventory['cobblestone'] < 8 and inventory['stone'] < 8:
            action["nearbyCraft"] = 'none'
    if action["nearbyCraft"] == 'iron_pickaxe':
        if inventory['stick'] < 2 or inventory['iron_ingot'] < 3:
            action["nearbyCraft"] = 'none'

    if action["place"] == 'furnace':
        if inventory['furnace'] == 0:
            action["place"] = 'none'
    if action["place"] == 'crafting_table':
        if inventory['crafting_table'] == 0:
            action["place"] = 'none'
        # elif inventory['stick'] < 2 and:
        #     if not (inventory['furnace'] == 0 and inventory['cobblestone'] >= 8):
        #         action["place"] = 'none'
        # else:
        #     if not (inventory['furnace'] == 0 and inventory['cobblestone'] >= 8):
        #         if (inventory['planks'] < 3 and
        #                 inventory['cobblestone'] < 3 and
        #                 (inventory['iron_ingot'] +
        #                  inventory['iron_ore'] < 3)):
        #             action["place"] = 'none'
        #         elif (inventory['cobblestone'] < 3 and
        #               (inventory['iron_ingot'] +
        #                inventory['iron_ore'] < 3)):
        #             action["place"] = 'none'
        #         elif (inventory['iron_ingot'] +
        #               inventory['iron_ore'] < 3):
        #             action["place"] = 'none'

    return action

item_by_attack = ['coal', 'cobblestone', 'dirt', 'iron_ore', 'log', 'stone']
item_by_attack_index = [0, 1, 3, 7, 9, 12]
none_move_action = np.array(['attack', 'craft_crafting_table',
                             'craft_planks', 'craft_stick', 'equip_iron_pickaxe',
                             'equip_stone_pickaxe', 'equip_wooden_pickaxe',
                             'nearbyCraft_furnace', 'nearbyCraft_iron_pickaxe',
                             'nearbyCraft_stone_pickaxe', 'nearbyCraft_wooden_pickaxe',
                             'nearbySmelt_iron_ingot', 'place_crafting_table', 'place_furnace',
                             'nearbySmelt_coal', 'craft_torch', 'place_torch', 'place_cobblestone',
                             'place_stone', 'place_dirt'])
list_obs = [[-10, 0], [10, 0], [0, -10], [0, 10]]

def train():
    print("Prepare_Data")


    data = minerl.data.make("MineRLObtainDiamond-v0",  data_dir='data', num_workers=4)

    all_actions = []
    all_pov_obs = []
    all_data_obs = []

    keyss = []
    # action_key = {}

    print("Loading data")
    trajectory_names = data.get_trajectory_names()
    random.shuffle(trajectory_names)


    change_attack = {}
    for it_ in item_by_attack:
        change_attack[it_] = 0

    # Add trajectories to the data until we reach the required DATA_SAMPLES.
    for trajectory_name in trajectory_names:
        trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)

        last_inventory = None
        time = 0
        angle_1 = 0
        angle_2 = 0
        time_attack_no_new = 0
        current_item = 0
        stack_grey = {}
        stack_index = []
        stack_index_max = []
        for obs, action, r, _, _ in trajectory:
            b_inventory = np.array(list(obs['inventory'].values()))
            b_inventory = b_inventory[item_by_attack_index]
            final = obs['pov']
            grey = np.dot(obs['pov'][..., :3], [0.2989, 0.5870, 0.1140])

            action = checking_craft_place(obs, action)

            if action['equip'] != 'none':
                if ((action['equip'] == 'iron_pickaxe' and
                     (obs['equipped_items.mainhand.type'] == 'iron_pickaxe' or
                      obs['inventory']['iron_pickaxe'] == 0))
                        or (action['equip'] == 'stone_pickaxe' and
                            (obs['equipped_items.mainhand.type'] == 'stone_pickaxe' or
                             obs['inventory']['stone_pickaxe'] == 0))
                        or (action['equip'] == 'wooden_pickaxe' and
                            (obs['equipped_items.mainhand.type'] == 'wooden_pickaxe' or
                             obs['inventory']['wooden_pickaxe'] == 0))):
                    action['equip'] = 'none'

            key, dict_action = dataset_action_batch_to_actions(action)
            keyss.append(key)

            angle_2_fix = (np.round(int(angle_2)/10)*10)
            angle_1_fix = (np.round(int(angle_1)/10)*10)
            stack_grey[tuple((angle_1_fix, angle_2_fix))] = [grey, time]

            for location in list_obs:
                a1 = angle_1_fix + location[0]
                a2 = angle_2_fix + location[1]
                # a1 = np.clip(a1, -90, 90)
                if a2 > 360:
                    a2 = a2 - 360
                elif a2 < 0:
                    a2 = 360 + a2
                new_tuple = tuple((a1, a2))
                if new_tuple in stack_grey:
                    grey_1 = stack_grey[new_tuple]
                    if grey_1[1] >= time - 1000:
                        final = np.concatenate((final, grey_1[0][:, :, None]), axis=-1)
                    else:
                        final = np.concatenate((final, np.zeros((64, 64, 1))), axis=-1)

                else:
                    final = np.concatenate((final, np.zeros((64, 64, 1))), axis=-1)
            final = final.astype(np.uint8)
            # action_key[key] = dict_action
            # if key not in action_key:
            #     action_key[key] = dict_action
            # if len(action_key.keys()) ==  97:
            #     print("aa")
            a = np.where(list_key_index == key)[0][0]

            after_proces = process_inventory(obs, time_attack_no_new, angle_1, angle_2)

            if action['attack']:
                if time_attack_no_new == 0:
                    current_item = np.sum(np.array(list(obs['inventory'].values())))
                    time_attack_no_new += 1

                else:
                    check_new = np.sum(np.array(list(obs['inventory'].values())))
                    if check_new != current_item:
                        time_attack_no_new = 0
                    else:
                        time_attack_no_new += 1
            else:
                time_attack_no_new = 0

            angle_1 += action['camera'][0]
            angle_1 = np.clip(angle_1, -90, 90)
            angle_2 += action['camera'][1]
            if angle_2 > 360:
                angle_2 = angle_2 - 360
            elif angle_2 < 0:
                angle_2 = 360 + angle_2

            time += 1
            if key != '_0_0':
                if obs['equipped_items.mainhand.type'] in ['stone_pickaxe', 'iron_pickaxe']:
                    stack_index.append([final, a , after_proces])
                    add = False
                    if key in none_move_action:
                        stack_index_max.append(stack_index)
                        stack_index = []
                        add = True
                    elif last_inventory is not None:
                        deal_t_e = b_inventory - last_inventory
                        index_e = np.where(deal_t_e > 0)[0]
                        if len(index_e) != 0:
                            if item_by_attack[index_e[0]] not in ['cobblestone', 'stone', 'dirt']:
                                stack_index_max.append(stack_index)
                                stack_index = []
                                change_attack[item_by_attack[index_e[0]]] += 1
                                add = True
                            else:
                                # stack_index_max.append(stack_index)
                                stack_index = []
                                if len(stack_index_max) > 2:
                                    del stack_index_max[0]
                                # trash_counter += 1
                    if add:
                        for v1 in stack_index_max:
                            for v2 in v1:
                                all_pov_obs.append(v2[0])
                                all_actions.append(v2[1])
                                all_data_obs.append(v2[2])
                        stack_index_max = []
                        stack_index = []
                else:
                    if key in none_move_action:
                        keyss.append(key)
                    if last_inventory is not None:
                        deal_t_e = b_inventory - last_inventory
                        index_e = np.where(deal_t_e > 0)[0]
                        if len(index_e) != 0:
                            change_attack[item_by_attack[index_e[0]]] += 1
                    # index_selected.append(index_counter)
                    all_pov_obs.append(final)
                    all_actions.append(a)
                    all_data_obs.append(after_proces)
                # len(index_selected) / 655850

            last_inventory = b_inventory

            # all_pov_obs.append(final)
            # all_actions.append(a)
            # all_data_obs.append(after_proces)


        if len(all_actions) >= DATA_SAMPLES:
            break


    a = Counter(keyss)
    a['_0_0']

    # key = list(action_key.keys())
    # key.sort()
    # new_dict = {}
    # for j in key:
    #     # if j not in list_key_index:
    #     #     print(j)
    #     new_dict[j] = action_key[j]
    # action_key = new_dict
    # with open('action_key_2.pkl', 'wb') as f:
    #     pickle.dump(action_key, f)



    all_actions = np.array(all_actions)
    # all_pov_obs = np.array(all_pov_obs)
    all_data_obs = np.array(all_data_obs)
    # all_actions_r = np.array(all_actions_r)
    # np.bincount(all_actions)
    # print(len(all_actions)/ 1916597)
    # np.sum(all_actions == 80)

    network = NatureCNN((7, 64, 64), len(list_key_index)).cuda()
    optimizer = th.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    num_samples = all_actions.shape[0]
    update_count = 0
    losses = []

    print("Training")
    for _ in range(15):
        # Randomize the order in which we go over the samples
        epoch_indices = np.arange(num_samples)
        np.random.shuffle(epoch_indices)
        for batch_i in range(0, num_samples, BATCH_SIZE):
            # break
            # NOTE: this will cut off incomplete batches from end of the random indices
            batch_indices = epoch_indices[batch_i:batch_i + BATCH_SIZE]

            obs = np.zeros((len(batch_indices), 64, 64, 7), dtype=np.float32)
            for j in range(len(batch_indices)):
                obs[j] = all_pov_obs[batch_indices[j]]
            # Load the inputs and preprocess
            # obs = all_pov_obs[batch_indices].astype(np.float32)
            # Transpose observations to be channel-first (BCHW instead of BHWC)
            obs = obs.transpose(0, 3, 1, 2)

            obs = th.from_numpy(obs).float().cuda()

            # Normalize observations. Do this here to avoid using too much memory (images are uint8 by default)
            obs /= 255.0

            # Map actions to their closest centroids
            # actions = all_actions[batch_indices]

            actions = all_actions[batch_indices]
            # distances = np.sum((action_vectors - action_centroids[:, None]) ** 2, axis=2)
            # actions = np.argmin(distances, axis=0)
            # Obtain logits of each action
            inven = th.from_numpy(all_data_obs[batch_indices]).float().cuda()
            logits = network(obs, inven)

            # Minimize cross-entropy with target labels.
            # We could also compute the probability of demonstration actions and
            # maximize them.
            loss = loss_function(logits, th.from_numpy(actions).long().cuda())

            # Standard PyTorch update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_count += 1
            losses.append(loss.item())
            if (update_count % 1000) == 0:
                mean_loss = sum(losses) / len(losses)
                tqdm.write("Iteration {}. Loss {:<10.3f}".format(update_count, mean_loss))
                losses.clear()



    TRAIN_MODEL_NAME = 'another_potato_1.pth'  # name to use when saving the trained agent.

    th.save(network.state_dict(), TRAIN_MODEL_NAME)
    del data


network = NatureCNN((7, 64, 64), len(list_key_index)).cuda()
network.load_state_dict(th.load('another_potato_1.pth'))

env = gym.make('MineRLObtainDiamond-v0')


rewards = []

for episode in range(20):
    obs = env.reset()
    total_reward = 0

    done = False
    steps = 0
    # BC part to get some logs:
    action_list = np.arange(len(list_key_index))

    angle_1 = 0
    angle_2 = 0

    last_attack = 0
    time_add = 0
    a2 = 0
    all_obs = []
    action_before = []
    last_action = 0

    data_obs = []
    time_attack_no_new = 0
    current_item = 0
    counter_0_0 = 0
    check_fix = False
    start_fix = 0

    previous = None
    check_attack = False
    time = 0
    # cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    # cv2.resizeWindow('image', 1000, 1000)
    # cv2.moveWindow('image', -30, 30)
    stack_grey = {}
    time_wait_equi = 0
    for i in range(18000):

        hsv = obs['pov']
        grey = np.dot(obs['pov'][..., :3], [0.2989, 0.5870, 0.1140])

        angle_2_fix = (np.round(int(angle_2) / 10) * 10)
        angle_1_fix = (np.round(int(angle_1) / 10) * 10)
        stack_grey[tuple((angle_1_fix, angle_2_fix))] = [grey, time_add]

        for location in list_obs:
            a1 = angle_1_fix + location[0]
            a2 = angle_2_fix + location[1]
            a1 = np.clip(a1, -90, 90)
            if a2 > 360:
                a2 = a2 - 360
            elif a2 < 0:
                a2 = 360 + a2
            new_tuple = tuple((a1, a2))
            if new_tuple in stack_grey:
                grey_1 = stack_grey[new_tuple]
                if grey_1[1] >= time_add - 1000:
                    hsv = np.concatenate((hsv, grey_1[0][:, :, None]), axis=-1)
                else:
                    hsv = np.concatenate((hsv, np.zeros((64, 64, 1))), axis=-1)

            else:
                hsv = np.concatenate((hsv, np.zeros((64, 64, 1))), axis=-1)
        all_obs.append(copy.deepcopy(obs['pov']))

        hsv = hsv.transpose(2, 0, 1).astype(np.float32)
        hsv = th.from_numpy(hsv[None]).float().cuda()
        hsv /= 255.0

        inven = process_inventory_test(obs, time_attack_no_new, angle_1, angle_2)
        inven = th.from_numpy(inven[None]).float().cuda()

        probabilities = th.softmax(network(hsv, inven), dim=1)[0]
        # Into numpy
        probabilities = probabilities.detach().cpu().numpy()
        # Sample action according to the probabilities
        action = np.random.choice(action_list, p=probabilities)

        key = list_key_index[action]
        if key == 'attack_0_0' and (check_attack or time_attack_no_new >= 300):
            action = np.random.choice([0, 1, 2, 3, 5, 6, 7, 8], p=np.repeat(0.125, 8))
            key = list_key_index[action]
            time_attack_no_new = 0
            # print("aa")

        if key in none_move_action:
            a = 3
            # print(key)

        action_before.append(action)
        data_obs.append(inven)
        last_action = action

        slipt = key.split('_')
        if 'attack' in slipt:
            last_attack = 1
            if time_attack_no_new == 0:
                current_item = 0
                for items_k in ['planks', 'log', 'cobblestone', 'stone', 'iron_ore', 'coal', 'dirt',
                                'crafting_table', 'furnace']:
                    current_item += obs['inventory'][items_k]
                time_attack_no_new += 1

            else:
                check_new = 0
                for items_k in ['planks', 'log', 'cobblestone', 'stone', 'iron_ore', 'coal', 'dirt', 'crafting_table',
                                'furnace']:
                    check_new += obs['inventory'][items_k]
                if check_new != current_item:
                    time_attack_no_new = 0
                else:
                    time_attack_no_new += 1
        else:
            last_attack = 0
            time_attack_no_new = 0

        time_add += 1

        # for k,v in action_key.items():
        #     print(k,v)
        # action_key['_f_0_0']
        # _f_0_0

        if len(slipt) >= 2:
            if slipt[-1] == '-1':
                angle_2 -= 10
            elif slipt[-1] == '1':
                angle_2 += 10

            if slipt[-2] == '-1':
                angle_1 -= 10
            elif slipt[-2] == '1':
                angle_1 += 10
            angle_1 = np.clip(angle_1, -90, 90)
            if angle_2 > 360:
                angle_2 = angle_2 - 360
            elif angle_2 < 0:
                angle_2 = 360 + angle_2

        action = process_a(action, env.action_space.noop())
        # print(key)
        obs, reward, done, info = env.step(action)
        # if key in ['place_furnace', 'place_crafting_table']:
        #     for __i in range(2):
        #         action = process_a(4, env.action_space.noop())
        #         obs, reward, done, info = env.step(env.action_space.noop())
        #         total_reward += reward
        #         time_add += 1

        if previous is not None:
            delta = previous - obs['pov']
            delta = np.sum(delta)
            if delta == 0:
                check_attack = True
            else:
                check_attack = False
        previous = obs['pov']
        #
        # cv2.imshow('image', cv2.cvtColor(obs['pov'], cv2.COLOR_BGR2RGB))
        # cv2.resizeWindow('image', 950, 950)
        total_reward += reward
        steps += 1
        if done:
            break
        if cv2.waitKey(10) & 0xFF == ord('o'):
            break
        # time.sleep(0.01)
    cv2.destroyAllWindows()
    print(obs['equipped_items'])

    rewards.append(total_reward)

    print(f'Episode #{episode + 1} reward: {total_reward}\t\t episode length: {steps}\n')

np.mean(rewards)
