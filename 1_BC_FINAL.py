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
            nn.Linear(22, 512),
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




def process_inventory(obs, time_attack_no_new, angle_1, angle_2, angle_2_f, last_action, success_action, his, test=False):

    # rs = [1, 2, 4, 8, 16, 32, 32, 64, 128, 256, 1024]
    data = np.zeros(22)
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

    # data[18] = t/18000
    data[17] = (angle_1 + 90) / 180

    data[19] = 0
    if angle_2 > 180:
        angle_2 = 360 - angle_2
        data[19] = 1
    data[18] = np.clip(np.abs(angle_2) / 180, 0, 1)


    attack = 0
    jumping = 0
    if last_action is not None:
        attack = last_action['attack']
        jumping = last_action['jump']

    data[20] = attack
    data[21] = jumping
    # data[20] = np.clip(time_attack_no_new / 200, 0, 1)

    # success_action = np.int64(list(success_action.values()))
    # data = np.concatenate((data, success_action, np.ndarray.flatten(his[0]), np.ndarray.flatten(his[1])), axis=0)

    data_f = copy.deepcopy(data)
    data_f[19] = 0
    if angle_2_f > 180:
        angle_2_f = 360 - angle_2_f
        data_f[19] = 1
    data_f[18] = np.clip(np.abs(angle_2_f) / 180, 0, 1)

    return data, data_f


range_angle = np.array([-30, -15, -10, -7, -5, -3, -1, -0.5, 0, 0.5, 1, 3, 5, 7, 10, 15, 30]) #8
# range_angle = np.array([0, -30, -15, -10, -7 ,-3,  3, 7, 10, 15, 30])

range_angle_2 = np.array([-45, -30, -15, -8, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 8, 15, 30, 45])#8

# range_angle_2 = np.array([0, -45, -30, -15, -10, -5, 5, 10, 15, 30, 45])

# action_key = [[], [], ['0'], ['0', 'attack']]
# with open('action_key.pkl', 'wb') as f:
#     pickle.dump(action_key, f)


def dataset_action_batch_to_actions(actions):
    action_with_key = copy.deepcopy(actions)
    dict_action = {}

    a1 = action_with_key['camera'][0]
    a2 = action_with_key['camera'][1]

    ks = [0, 0, f'0', f'0']

    a1 = np.abs(range_angle - a1)
    ks[0] = np.argmin(a1)

    a2 = np.abs(range_angle_2 - a2)
    ks[1] = np.argmin(a2)

    if action_with_key['jump'] == 1:
        if action_with_key['forward'] == 1:
            ks[2] = f'j_f'
        elif action_with_key['right'] == 1:
            ks[2] = f'j_r'
        elif action_with_key['left'] == 1:
            ks[2] = f'j_l'
        else:
            ks[2] = f'j_f'

    elif action_with_key['forward'] == 1:
        dict_action['forward'] = 1
        ks[2] = f'f'
    elif action_with_key['back'] == 1:
        dict_action['back'] = 1
        ks[2] = f'b'
    elif action_with_key['left'] == 1:
        dict_action['left'] = 1
        ks[2] = f'l'
    elif action_with_key['right'] == 1:
        dict_action['right'] = 1
        ks[2] = f'r'

    # if ks[2] not in action_key[2]:
    #     action_key[2].append(ks[2])
    #     ks[2] = len(action_key[2]) - 1
    # else:
    ks[2] = np.where(np.array(action_key[2]) == ks[2])[0][0]

    if action_with_key['craft'] != 'none':
        craft = action_with_key['craft']
        ks[3] = f'craft_{craft}'
        dict_action['craft'] = craft
    elif action_with_key['equip'] != 'none':

        equip = action_with_key['equip']
        if equip == 'iron_axe': equip = 'iron_pickaxe'
        if equip == 'stone_axe': equip = 'stone_pickaxe'
        if equip == 'wooden_axe': equip = 'wooden_pickaxe'
        ks[3] = f'equip_{equip}'
        dict_action['equip'] = equip

    elif action_with_key['nearbyCraft'] != 'none':
        nearbyCraft = action_with_key['nearbyCraft']
        if nearbyCraft == 'iron_axe': nearbyCraft = 'iron_pickaxe'
        if nearbyCraft == 'stone_axe': nearbyCraft = 'stone_pickaxe'
        if nearbyCraft == 'wooden_axe': nearbyCraft = 'wooden_pickaxe'
        ks[3] = f'nearbyCraft_{nearbyCraft}'
        dict_action['nearbyCraft'] = nearbyCraft

    elif action_with_key['nearbySmelt'] != 'none':
        nearbySmelt = action_with_key['nearbySmelt']
        ks[3] = f'nearbySmelt_{nearbySmelt}'
        dict_action['nearbySmelt'] = nearbySmelt
    elif action_with_key['place'] != 'none':
        place = action_with_key['place']
        # if place not in ['cobblestone', 'dirt', 'stone']:
        ks[3] = f'place_{place}'
        dict_action['place'] = place
    elif action_with_key['attack'] == 1:
        dict_action['attack'] = 1
        ks[3] = 'attack'

    ks[3] = np.where(np.array(action_key[3]) == ks[3])[0][0]

    ks_f = copy.deepcopy(ks)
    a2 = action_with_key['camera'][1] * -1
    a2 = np.abs(range_angle_2 - a2)
    ks_f[1] = np.argmin(a2)

    return ks, ks_f


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

def reset_craft_action():
    return {"craft_crafting_table": False, "nearbyCraft_furnace": False, "place_crafting_table": False, "craft_planks": False,
          "craft_stick": False, "nearbyCraft_wooden_pickaxe": False, "nearbyCraft_stone_pickaxe": False, "place_furnace": False,
          "nearbySmelt_iron_ingot": False, "nearbyCraft_iron_pickaxe": False}


item_by_attack = ['coal', 'cobblestone', 'dirt', 'iron_ore', 'log', 'stone', 'crafting_table', 'furnace']
item_by_attack_index = [0, 1, 3, 7, 9, 12, 2, 4]
none_move_action = np.array(['attack', 'craft_crafting_table',
                             'craft_planks', 'craft_stick', 'equip_iron_pickaxe',
                             'equip_stone_pickaxe', 'equip_wooden_pickaxe',
                             'nearbyCraft_furnace', 'nearbyCraft_iron_pickaxe',
                             'nearbyCraft_stone_pickaxe', 'nearbyCraft_wooden_pickaxe',
                             'nearbySmelt_iron_ingot', 'place_crafting_table', 'place_furnace',
                             'nearbySmelt_coal', 'craft_torch', 'place_torch', 'place_cobblestone',
                             'place_stone', 'place_dirt'])
list_obs = [-30, -15, 15, 30]
max_shape = 3
new_items = {'cobblestone': [0, 0, 0, 0, 0, 1], 'iron_ore': [0, 0, 0, 0, 1, 0],
             'log': [0, 0, 0, 1, 0, 0], 'crafting_table': [0, 0, 1, 0, 0, 0],
             'furnace': [0, 1, 0, 0, 0, 0], 'coal': [1, 0, 0, 0, 0, 0]}
old_items = {'iron_ore': [0, 0, 0, 1], 'crafting_table': [0, 0, 1, 0],
             'furnace': [0, 1, 0, 0], 'log':  [1, 0, 0, 0]}

print("Prepare_Data")

data = minerl.data.make("MineRLObtainDiamond-v0", data_dir='data', num_workers=4)


trajectory_names = data.get_trajectory_names()
# np.bincount(all_actions[:,0])

all_actions = []
all_pov_obs = []
all_data_obs = []

print("Loading data")

a1_a = []
a2_a = []
#
# a1_a = np.array(a1_a)
# a2_a = np.array(a2_a)
#
# x = a1_a[np.where(((a1_a != 0) ))]
# print(len(x))
# x = x[np.where(((x >= 0.5) ))]
# print(len(x))
# x = x[np.where(((x >= 3) ))]
# print(len(x))
# x = x[np.where(((x >= 5) ))]
# print(len(x))
#
# x = x[np.where(((x >= 45) ))]
# print(len(x))
#
# x = x[np.where(((x >= 45) ))]
# print(len(x))
#
# len(x)
# np.sum(x)
# np.histogram(np.abs(x))
change_attack = {}
for it_ in item_by_attack:
    change_attack[it_] = 0


counter_added = 0
current_added_log = 0
all_visibale_obs = []
total_useless = 0
for trajectory_name in trajectory_names:
    trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)
    last_inventory = None
    time = 0
    angle_1 = 0
    angle_2 = 0
    angle_2_f = 0

    stack_grey = {}
    stack_grey_f = {}

    current_item = 0

    stack_index_max = []
    stack_index = []
    actions_eps = []
    last_action = None
    key_a_list = None
    last_inven_raw = None

    time_attack_no_new = 0

    success_action = reset_craft_action()
    stack_new_items = np.zeros((15,6))
    stack_old_itmes = np.zeros((5,4))
    for obs, action, r, _, _ in trajectory:
        if last_action is not None:
            key_3 = action_key[3][key_a_list[3]]
            if key_3 in success_action:
                if not success_action[key_3]:
                    if key_3 == ['place_crafting_table']:
                        if last_inven_raw['crafting_table'] > obs['inventory']['crafting_table']:
                            success_action[key_3] = True
                    elif key_3 == ['place_furnace']:
                        if last_inven_raw['furnace'] > obs['inventory']['furnace']:
                            success_action[key_3] = True
                    else:
                        split = key_3.split('_')
                        item_name = split[-1]
                        if len(split) == 3:
                            item_name = split[-2] + "_" + split[-1]
                        if last_inven_raw[item_name] < obs['inventory'][item_name]:
                            success_action[key_3] = True
            if success_action['place_crafting_table']:
                if last_inven_raw['crafting_table'] < obs['inventory']['crafting_table']:
                    success_action['place_crafting_table'] = False
            if success_action['place_furnace']:
                if last_inven_raw['furnace'] < obs['inventory']['furnace']:
                    success_action['place_furnace'] = False
            new_i = np.array(list(obs['inventory'].values()))
            old_i = np.array(list(last_inven_raw.values()))
            delta = new_i - old_i
            if np.sum(delta) > 0:
                index = np.where(delta > 0)[0]
                if len(index) != 0:
                    new_k_t = obs_inve_keys_s[index[0]]
                    if new_k_t in new_items:
                        stack_new_items = np.concatenate((stack_new_items, np.array(new_items[new_k_t])[None]), axis=0)
                        stack_new_items = stack_new_items[1:]

                    if new_k_t == 'log':
                        p_1 = 0.5
                        p_0 = 0.5
                        d_le = 400
                        if current_added_log < counter_added - 300:
                            p_1 = 1
                            p_0 = 0
                            d_le = 500
                        elif current_added_log < counter_added - 200:
                            p_1 = 0.7
                            p_0 = 0.3
                        elif current_added_log < counter_added - 100:
                            d_le = 200
                        elif current_added_log < counter_added - 50:
                            p_1 = 0.3
                            p_0 = 0.7
                            d_le = 200
                        choice = np.random.choice([0, 1], p=[p_0, p_1])

                        for backward in range(d_le):
                            new_index_b = counter_added - backward - 20
                            if new_index_b <= current_added_log or new_index_b < 0\
                                    or choice == 0:
                                break
                            # for _m in range(2):
                            new_c = 1
                            if all_actions[new_index_b][3] == 1:
                                new_c = np.random.choice([0, 1], p=[0.5, 0.5])
                            if new_c == 1:
                                all_pov_obs.append(all_pov_obs[new_index_b])
                                all_actions.append(all_actions[new_index_b])
                                all_data_obs.append(all_data_obs[new_index_b])
                            # if backward%2 == 0:
                            #     all_visibale_obs.append(all_pov_obs[new_index_b])
                        counter_added = len(all_pov_obs)
                    current_added_log = len(all_pov_obs) - 1

            elif np.sum(delta) < 0:
                index = np.where(delta < 0)[0]
                if len(index) != 0:
                    if obs_inve_keys_s[index[0]] in old_items:
                        stack_old_itmes = np.concatenate((stack_old_itmes, np.array(old_items[obs_inve_keys_s[index[0]]])[None]), axis=0)
                        stack_old_itmes = stack_old_itmes[1:]

        last_inven_raw = obs['inventory']
        a1_a.append(action['camera'][0])
        a2_a.append(action['camera'][1])

        b_inventory = np.array(list(obs['inventory'].values()))
        b_inventory = b_inventory[item_by_attack_index]

        key_a_list, key_a_list_f = dataset_action_batch_to_actions(action)

        final = obs['pov']
        final_f = np.fliplr(obs['pov'])

        # grey = np.dot(obs['pov'][..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        # angle_2_fix = (np.round(int(angle_2) / 15) * 15)
        # stack_grey[angle_2_fix] = grey
        #
        # all_angle_had = np.array(list(stack_grey.keys()))
        # index_current = np.where(all_angle_had == angle_2_fix)[0][0]
        #
        # for location in list_obs:
        #     a2 = angle_2_fix + location
        #     if a2 > 360:
        #         a2 = a2 - 360
        #     elif a2 < 0:
        #         a2 = 360 + a2
        #     a2 = np.abs(all_angle_had - a2)
        #     a2[index_current] += 1000
        #     a_min = np.min(a2)
        #     if a_min > 20:
        #         final = np.concatenate((final, np.zeros((64, 64, 1))), axis=-1)
        #     else:
        #         final = np.concatenate((final, stack_grey[all_angle_had[np.argmin(a2)]][:, :, None]), axis=-1)
        #
        # final = final.astype(np.uint8)

        # grey_f = np.dot(final_f[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        # angle_2_fix_f = (np.round(int(angle_2_f) / 15) * 15)
        # stack_grey_f[angle_2_fix_f] = grey_f
        #
        # all_angle_had = np.array(list(stack_grey_f.keys()))
        # index_current = np.where(all_angle_had == angle_2_fix_f)[0][0]
        #
        # for location in list_obs:
        #     a2 = angle_2_fix_f + location
        #     if a2 > 360:
        #         a2 = a2 - 360
        #     elif a2 < 0:
        #         a2 = 360 + a2
        #     a2 = np.abs(all_angle_had - a2)
        #     a2[index_current] += 1000
        #     a_min = np.min(a2)
        #     if a_min > 15:
        #         final_f = np.concatenate((final_f, np.zeros((64, 64, 1))), axis=-1)
        #     else:
        #         final_f = np.concatenate((final_f, stack_grey_f[all_angle_had[np.argmin(a2)]][:, :, None]), axis=-1)
        # final_f = final_f.astype(np.uint8)

        # key_a_list, key_a_list_f = dataset_action_batch_to_actions(action)
        after_proces, after_proces_f = process_inventory(obs, time_attack_no_new, angle_1, angle_2, angle_2_f,
                                                         last_action, success_action, [stack_new_items, stack_old_itmes], False)
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

        last_action = action
        angle_2 += action['camera'][1]
        if angle_2 > 360:
            angle_2 = angle_2 - 360
        elif angle_2 < 0:
            angle_2 = 360 + angle_2

        action['camera'][1] *= -1
        angle_2_f += action['camera'][1]
        if angle_2_f > 360:
            angle_2_f = angle_2_f - 360
        elif angle_2_f < 0:
            angle_2_f = 360 + angle_2_f

        angle_1 -= action['camera'][0]
        angle_1 = np.clip(angle_1, -90, 90)

        time += 1
        if not (key_a_list[0] == 8 and key_a_list[1] == 8 and key_a_list[2] == 0 and key_a_list[3] == 0):
            all_pov_obs.append(final)
            all_actions.append(key_a_list)
            all_data_obs.append(after_proces)

            all_pov_obs.append(final_f)
            all_actions.append(key_a_list_f)
            all_data_obs.append(after_proces_f)
            counter_added += 2
            # if obs['equipped_items.mainhand.type'] in ['stone_pickaxe', 'iron_pickaxe', 'stone_axe', 'iron_axe']:
            #     stack_index.append([final, key_a_list, after_proces, final_f, key_a_list_f, after_proces_f])
            #     # stack_index.append([final, key_a_list, after_proces])
            #
            #     add = False
            #     if key_a_list[3] not in [0, 1]:
            #         stack_index_max.append(stack_index)
            #         stack_index = []
            #         add = True
            #     elif last_inventory is not None:
            #         deal_t_e = b_inventory - last_inventory
            #         index_e = np.where(deal_t_e > 0)[0]
            #         if len(index_e) != 0:
            #             if item_by_attack[index_e[0]] not in ['cobblestone', 'stone', 'dirt']:
            #                 stack_index_max.append(stack_index)
            #                 stack_index = []
            #                 change_attack[item_by_attack[index_e[0]]] += 1
            #                 add = True
            #             else:
            #                 stack_index_max.append(stack_index)
            #                 stack_index = []
            #                 if len(stack_index_max) > 1:
            #                     del stack_index_max[0]
            #                 # trash_counter += 1
            #     if add:
            #         for v1 in stack_index_max:
            #             for v2 in v1:
            #                 all_pov_obs.append(v2[0])
            #                 all_actions.append(v2[1])
            #                 all_data_obs.append(v2[2])
            #                 if np.random.choice([0, 1], p=[0.3, 0.7]) == 1:
            #                     all_pov_obs.append(v2[3])
            #                     all_actions.append(v2[4])
            #                     all_data_obs.append(v2[5])
            #         stack_index_max = []
            #         stack_index = []
            # else:
            #a
            #     if last_inventory is not None:
            #         deal_t_e = b_inventory - last_inventory
            #         index_e = np.where(deal_t_e > 0)[0]
            #         if len(index_e) != 0:
            #             change_attack[item_by_attack[index_e[0]]] += 1
            #     all_pov_obs.append(final)
            #     all_actions.append(key_a_list)
            #     all_data_obs.append(after_proces)
            #     if np.random.choice([0, 1], p=[0.2, 0.8]) == 1:
            #         all_pov_obs.append(final_f)
            #         all_actions.append(key_a_list_f)
            #         all_data_obs.append(after_proces_f)

        else:
            current_item += 1


        last_inventory = b_inventory
    total_useless += current_item
    print(len(all_actions), current_item, total_useless)
    del trajectory
    # del stack_grey_f
    # del stack_grey
    del stack_index
    del stack_index_max


#
# np.histogram(all_actions[:, 1])
# np.histogram(all_actions[:, 1])

all_actions = np.array(all_actions)
# all_pov_obs = np.array(all_pov_obs)
all_data_obs = np.array(all_data_obs)

network = NatureCNN((max_shape, 64, 64), len(range_angle)).cuda()
optimizer = th.optim.Adam(network.parameters(), lr=LEARNING_RATE)
loss_function = nn.CrossEntropyLoss()

network_1 = NatureCNN((max_shape, 64, 64), len(range_angle_2)).cuda()
optimizer_1 = th.optim.Adam(network_1.parameters(), lr=LEARNING_RATE)
loss_function_1 = nn.CrossEntropyLoss()

network_2 = NatureCNN((max_shape, 64, 64), len(action_key[2])).cuda()
optimizer_2 = th.optim.Adam(network_2.parameters(), lr=LEARNING_RATE)
loss_function_2 = nn.CrossEntropyLoss()

network_3 = NatureCNN((max_shape, 64, 64), len(action_key[3])).cuda()
optimizer_3 = th.optim.Adam(network_3.parameters(), lr=LEARNING_RATE)
loss_function_3 = nn.CrossEntropyLoss()

num_samples = all_actions.shape[0]
update_count = 0
losses = []
np.sum(all_actions == 7)
# th.save(network.state_dict(), 'a_1_2.pth')
# th.save(network_1.state_dict(), 'a_2_2.pth')
# th.save(network_2.state_dict(), 'a_3_2.pth')
# th.save(network_3.state_dict(), 'a_4_2.pth')
BATCH_SIZE = 128
print("Training")
for _ in range(15):
    # Randomize the order in which we go over the samples
    epoch_indices = np.arange(num_samples)
    np.random.shuffle(epoch_indices)
    np.random.shuffle(epoch_indices)
    np.random.shuffle(epoch_indices)

    for batch_i in range(0, num_samples, BATCH_SIZE):
        # break
        # NOTE: this will cut off incomplete batches from end of the random indices
        batch_indices = epoch_indices[batch_i:batch_i + BATCH_SIZE]

        obs = np.zeros((len(batch_indices), 64, 64, max_shape), dtype=np.float32)
        for j in range(len(batch_indices)):
            obs[j] = all_pov_obs[batch_indices[j]]
        # Load the inputs and preprocess
        # obs = all_pov_obs[batch_indices].astype(np.float32)
        # Transpose observations to be channel-first (BCHW instead of BHWC)
        obs = obs.transpose(0, 3, 1, 2)

        obs = th.from_numpy(obs).float().cuda()

        # Normalize observations. Do this here to avoid using too much memory (images are uint8 by default)
        obs /= 255.0

        inven = th.from_numpy(all_data_obs[batch_indices]).float().cuda()
        logits = network(obs, inven)

        actions = all_actions[batch_indices, 0]
        loss = loss_function(logits, th.from_numpy(actions).long().cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logits = network_1(obs, inven)
        actions = all_actions[batch_indices, 1]
        loss_1 = loss_function_1(logits, th.from_numpy(actions).long().cuda())
        optimizer_1.zero_grad()
        loss_1.backward()
        optimizer_1.step()

        logits = network_2(obs, inven)
        actions = all_actions[batch_indices, 2]
        loss_2 = loss_function_2(logits, th.from_numpy(actions).long().cuda())
        optimizer_2.zero_grad()
        loss_2.backward()
        optimizer_2.step()

        logits = network_3(obs, inven)
        actions = all_actions[batch_indices, 3]
        loss_3 = loss_function_3(logits, th.from_numpy(actions).long().cuda())
        optimizer_3.zero_grad()
        loss_3.backward()
        optimizer_3.step()

        update_count += 1
        losses.append([loss.item(), loss_1.item(), loss_2.item(), loss_3.item()])
        if (update_count % 1000) == 0:
            mean_loss = np.mean(losses, axis=0)
            tqdm.write("Iteration {}. Loss {:<10.3f} {:<10.3f} {:<10.3f}  {:<10.3f}".format(
                update_count, mean_loss[0], mean_loss[1], mean_loss[2], mean_loss[3]))
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


network = NatureCNN((max_shape, 64, 64), len(range_angle)).cuda()
network.load_state_dict(th.load('a_1_3.pth'))

network_1 = NatureCNN((max_shape, 64, 64), len(range_angle_2)).cuda()
network_1.load_state_dict(th.load('a_2_3.pth'))

network_2 = NatureCNN((max_shape, 64, 64), len(action_key[2])).cuda()
network_2.load_state_dict(th.load('a_3_3.pth'))

network_3 = NatureCNN((max_shape, 64, 64), len(action_key[3])).cuda()
network_3.load_state_dict(th.load('a_4_3.pth'))

env = gym.make('MineRLObtainDiamond-v0')

rewards = []
aa = [{}, {'forward': 1}, {'back': 1}, {'left': 1}, {'right': 1}, {'forward': 1, 'jump': 1}, {'right': 1, 'jump': 1}, {'left': 1, 'jump': 1}]
all_visibale_obs = []
for episode in range(50):
    obs = env.reset()
    total_reward = 0

    done = False
    steps = 0
    # BC part to get some logs:

    angle_1 = 0
    angle_2 = 0

    last_attack = 0
    time_add = 0
    a2 = 0
    all_obs = []
    action_before = []
    last_action = None

    data_obs = []
    time_attack_no_new = 0
    current_item = 0
    counter_0_0 = 0
    check_fix = False
    start_fix = 0

    previous = None
    check_attack = False
    time = 0
    stack_grey = {}
    time_wait_equi = 0
    last_inven_raw = None
    success_action = reset_craft_action()

    # cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    # cv2.resizeWindow('image', 1000, 1000)
    # cv2.moveWindow('image', -30, 30)
    stack_new_items = np.zeros((15,6))
    stack_old_itmes = np.zeros((5,4))
    for i in range(18000):
        all_visibale_obs.append(obs['pov'])
        if last_action is not None:
            key_3 = action_key[3][x[3]]
            if key_3 in success_action:
                if not success_action[key_3]:
                    if key_3 == ['place_crafting_table']:
                        if last_inven_raw['crafting_table'] > obs['inventory']['crafting_table']:
                            success_action[key_3] = True
                    elif key_3 == ['place_furnace']:
                        if last_inven_raw['furnace'] > obs['inventory']['furnace']:
                            success_action[key_3] = True
                    else:
                        split = key_3.split('_')
                        item_name = split[-1]
                        if len(split) == 3:
                            item_name = split[-2] + "_" + split[-1]
                        if last_inven_raw[item_name] < obs['inventory'][item_name]:
                            success_action[key_3] = True
            if success_action['place_crafting_table']:
                if last_inven_raw['crafting_table'] < obs['inventory']['crafting_table']:
                    success_action['place_crafting_table'] = False
            if success_action['place_furnace']:
                if last_inven_raw['furnace'] < obs['inventory']['furnace']:
                    success_action['place_furnace'] = False
            new_i = np.array(list(obs['inventory'].values()))
            old_i = np.array(list(last_inven_raw.values()))
            delta = new_i - old_i
            if np.sum(delta) > 0:
                index = np.where(delta > 0)[0]
                if len(index) != 0:
                    new_k_t = obs_inve_keys_s[index[0]]
                    if new_k_t in new_items:
                        stack_new_items = np.concatenate((stack_new_items, np.array(new_items[new_k_t])[None]), axis=0)
                        stack_new_items = stack_new_items[1:]

            elif np.sum(delta) < 0:
                index = np.where(delta < 0)[0]
                if len(index) != 0:
                    if obs_inve_keys_s[index[0]] in old_items:
                        stack_old_itmes = np.concatenate(
                            (stack_old_itmes, np.array(old_items[obs_inve_keys_s[index[0]]])[None]), axis=0)
                        stack_old_itmes = stack_old_itmes[1:]
        last_inven_raw = obs['inventory']
        final = obs['pov']
        # grey = np.dot(obs['pov'][..., :3], [0.2989, 0.5870, 0.1140])
        # angle_2_fix = (np.round(int(angle_2) / 15) * 15)
        # stack_grey[angle_2_fix] = grey
        #
        # all_angle_had = np.array(list(stack_grey.keys()))
        # index_current = np.where(all_angle_had == angle_2_fix)[0][0]
        # for location in list_obs:
        #     a2 = angle_2_fix + location
        #     if a2 > 360:
        #         a2 = a2 - 360
        #     elif a2 < 0:
        #         a2 = 360 + a2
        #     a2 = np.abs(all_angle_had - a2)
        #     a2[index_current] += 1000
        #     a_min = np.min(a2)
        #     if a_min > 15:
        #         final = np.concatenate((final, np.zeros((64, 64, 1))), axis=-1)
        #     else:
        #         final = np.concatenate((final, stack_grey[all_angle_had[np.argmin(a2)]][:, :, None]), axis=-1)
        # final = final.astype(np.uint8)

        final = final.transpose(2, 0, 1).astype(np.float32)
        final = th.from_numpy(final[None]).float().cuda()
        final /= 255.0

        inven,_ = process_inventory(obs, time_attack_no_new, angle_1, angle_2, 0, last_action,success_action, [stack_new_items, stack_old_itmes], True)
        inven = th.from_numpy(inven[None]).float().cuda()

        new_a = env.action_space.noop()
        x = []
        probabilities = th.softmax(network(final, inven), dim=1)[0]
        probabilities = probabilities.detach().cpu().numpy()
        action = np.random.choice(np.arange(len(range_angle)), p=probabilities)
        # print(range_angle[action])
        x.append(action)

        probabilities = th.softmax(network_1(final, inven), dim=1)[0]
        probabilities = probabilities.detach().cpu().numpy()
        action = np.random.choice(np.arange(len(range_angle_2)), p=probabilities)
        # print(range_angle_2[action])
        x.append(action)

        probabilities = th.softmax(network_2(final, inven), dim=1)[0]
        probabilities = probabilities.detach().cpu().numpy()
        action = np.random.choice(np.arange(len(action_key[2])), p=probabilities)
        # print(action_key[2][action])

        x.append(action)

        probabilities = th.softmax(network_3(final, inven), dim=1)[0]
        probabilities = probabilities.detach().cpu().numpy()
        action = np.random.choice(np.arange(len(action_key[3])), p=probabilities)
        # print(action_key[3][action])

        x.append(action)

        # dict_action = all_k_combo[tuple((x[0], x[1], x[2], x[3]))]
        new_a['camera'] = [range_angle[x[0]], range_angle_2[x[1]]]
        if action_key[2][x[2]] != '0':
            for k, v in aa[x[2]].items():
                new_a[k] = v
        key_3 = action_key[3][x[3]]
        if key_3 != '0':
            if key_3 == 'attack':
                new_a['attack'] = 1
            else:
                slipt = key_3.split('_')
                if len(slipt) == 2:
                    new_a[slipt[0]] = slipt[1]
                else:
                    new_a[slipt[0]] = slipt[1] + '_' + slipt[2]


        action_before.append(action)
        data_obs.append(inven)

        if new_a['attack'] == 1:
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
            last_attack = 0
            time_attack_no_new = 0

        time_add += 1

        # for k,v in action_key.items():
        #     print(k,v)
        # action_key['_f_0_0']
        # _f_0_0

        angle_1 -= new_a['camera'][0]
        angle_1 = np.clip(angle_1, -90, 90)

        angle_2 += new_a['camera'][1]
        if angle_2 > 360:
            angle_2 = angle_2 - 360
        elif angle_2 < 0:
            angle_2 = 360 + angle_2

        # print(key)
        # new_a['craft'] = 'crafting_table'
        obs, reward, done, info = env.step(new_a)
        if x[3] not in [0, 1]:
            for _ in range(3):
                if done:
                    break
                obs, reward, done, info = env.step(env.action_space.noop())
                total_reward += reward

        # print(new_a['camera'][1])
        last_action = new_a

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
    print(obs['inventory']['log'])

    rewards.append(total_reward)

    print(f'Episode #{episode + 1} reward: {total_reward}\t\t episode length: {steps}\n')

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