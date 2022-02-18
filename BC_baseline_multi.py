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
import time
import math
import torchvision
from collections import Counter

from hummingbird.ml import convert
import matplotlib.pyplot as plt
import copy

DATA_DIR = "data"  # path to MineRL dataset (should contain "MineRLObtainIronPickaxeVectorObf-v0" directory).
EPOCHS = 4  # how many times we train over dataset.
LEARNING_RATE = 0.0001  # learning rate for the neural network.
BATCH_SIZE = 64
NUM_ACTION_CENTROIDS = 100  # number of KMeans centroids used to cluster the data.
Inventory_MAX = {'furnace': 3, 'crafting_table': 3, 'planks': 30, 'stick': 20,
                 'stone_pickaxe': 2, 'wooden_pickaxe': 3, 'iron_pickaxe': 3, 'log': 20, 'cobblestone': 20,
                 'stone': 20, 'iron_ore': 10, 'iron_ingot': 10, 'coal': 10, 'dirt': 50, 'torch': 20}
with open('Inventory_MAX.pkl', 'wb') as f:
    pickle.dump(Inventory_MAX, f)
# np.save("all_key_actions.npy", action_key)
# with open('action_key_1.pkl', 'wb') as f:
#     pickle.dump(action_key, f)
with open('action_key_1.pkl', 'rb') as f:
    action_key = pickle.load(f)
list_key_index = np.array(list(action_key.keys()))
print(len(list_key_index))
none_move_action = np.array(['attack', 'craft_crafting_table',
                             'craft_planks', 'craft_stick', 'equip_iron_pickaxe',
                             'equip_stone_pickaxe', 'equip_wooden_pickaxe',
                             'nearbyCraft_furnace', 'nearbyCraft_iron_pickaxe',
                             'nearbyCraft_stone_pickaxe', 'nearbyCraft_wooden_pickaxe',
                             'nearbySmelt_iron_ingot', 'place_crafting_table', 'place_furnace',
                             'nearbySmelt_coal', 'craft_torch', 'place_torch'])


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
    for k, v in action_key[key].items():
        action_new[k] = v
    # action_new['attack'] = 1
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
        k = f'equip_{equip}'
        dict_action['equip'] = equip

    elif action_with_key['nearbyCraft'] != 'none':
        nearbyCraft = action_with_key['nearbyCraft']
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
        if np.abs(first_angle) > 3 and np.abs(first_angle) <= 18:
            if first_angle < 0:
                key += '_-1'
                camera.append(-15)
            else:
                key += '_1'
                camera.append(15)
        else:
            key += '_0'
            camera.append(0)

        if np.abs(second_angle) > 3 and np.abs(second_angle) <= 18:
            if second_angle < 0:
                key += '_-1'
                camera.append(-15)
            else:
                key += '_1'
                camera.append(15)
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


def rgb_to_hsv(image: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    # The first or last occurrence is not guaranteed before 1.6.0
    # https://github.com/pytorch/pytorch/issues/20414
    maxc, _ = image.max(-3)
    maxc_mask = image == maxc.unsqueeze(-3)
    _, max_indices = ((maxc_mask.cumsum(-3) == 1) & maxc_mask).max(-3)
    minc: torch.Tensor = image.min(-3)[0]

    v: torch.Tensor = maxc  # brightness

    deltac: torch.Tensor = maxc - minc
    s: torch.Tensor = deltac / (v + eps)

    # avoid division by zero
    deltac = torch.where(deltac == 0, torch.ones_like(deltac, device=deltac.device, dtype=deltac.dtype), deltac)

    maxc_tmp = maxc.unsqueeze(-3) - image
    rc: torch.Tensor = maxc_tmp[..., 0, :, :]
    gc: torch.Tensor = maxc_tmp[..., 1, :, :]
    bc: torch.Tensor = maxc_tmp[..., 2, :, :]

    h = torch.stack([bc - gc, 2.0 * deltac + rc - bc, 4.0 * deltac + gc - rc], dim=-3)

    h = torch.gather(h, dim=-3, index=max_indices[..., None, :, :])
    h = h.squeeze(-3)
    h = h / deltac

    h = (h / 6.0) % 1.0

    h = 2 * 180 * h

    return torch.stack([h, s * 255, v], dim=-3)


def process_inventory(obs, attack, t, full_previous, angle_1, angle_2):
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
    hand = obs['equipped_items']['mainhand']['type']
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


list_obs = [[-15, 0], [15, 0], [0, -15], [0, 15]]

item_by_attack = ['coal', 'cobblestone', 'dirt', 'iron_ore', 'log', 'stone']
item_by_attack_index = [0, 1, 3, 7, 9, 12]


#
# item_by_attack = ['coal', 'crafting_table', 'furnace', 'iron_ingot', 'iron_ore', 'iron_pickaxe']
# item_by_attack_index = [0, 2, 4, 6, 7, 8 ]
# Counter({'craft_planks': 1693,
#          'craft_stick': 909,
#          'craft_crafting_table': 291,
#          'place_crafting_table': 823,
#          'nearbyCraft_wooden_pickaxe': 336,

#          'equip_wooden_pickaxe': 270,

#          'nearbyCraft_stone_pickaxe': 262,
#          'equip_stone_pickaxe': 234,

#          'nearbyCraft_furnace': 300,
#          'place_furnace': 240,
#          'nearbySmelt_iron_ingot': 1103,
#          'nearbyCraft_iron_pickaxe': 313,
#          'equip_iron_pickaxe': 139,
#          'place_torch': 480,
#          'nearbySmelt_coal': 156,
#          'craft_torch': 109})
def train():
    print("Prepare_Data")

    all_pov_obs = []
    all_actions = []
    all_data_obs = []
    all_rs = []

    time_attack_no_new_all = []
    keyss = []

    change_attack = {}
    for it_ in item_by_attack:
        change_attack[it_] = 0

    total_data = 0
    index_counter = -1
    # index_selected = []
    # with open('index_selected.pkl', 'rb') as f:
    #     index_selected = pickle.load(f)
    a_counter = 0
    esp_all = []
    esp = None
    len(esp_all)
    esp_all[0]
    for i in range(73):
        with open(f'history_{i}.pkl', 'rb') as f:
            data = pickle.load(f)
        total_data += len(data['obs'])
        print(len(data['action']), i)
        if esp is not  None:
            esp_all.append(esp)
        esp = {'obs': [], 'infos': None, 'acts': [], 'rews': []}
        all_obs = []
        action_before = []
        data_obs = []
        time = 0

        angle_1 = 0
        angle_2 = 0
        time_attack_no_new = 0
        current_item = 0
        seri_of_craft = []
        stack_grey = {}

        last_inventory = None

        stack_index = []
        stack_index_max = []
        for j in range(len(data['rewards'])):
            action = data['action'][j]
            obs = data['obs'][j]

            reward = data['rewards'][j]

            b_inventory = np.array(list(obs['inventory'].values()))
            b_inventory = b_inventory[item_by_attack_index]

            action = checking_craft_place(obs, action)

            if action['equip'] != 'none':
                if ((action['equip'] == 'iron_pickaxe' and
                     (obs['equipped_items']['mainhand']['type'] == 'iron_pickaxe' or
                      obs['inventory']['iron_pickaxe'] == 0))
                        or (action['equip'] == 'stone_pickaxe' and
                            (obs['equipped_items']['mainhand']['type'] == 'stone_pickaxe' or
                             obs['inventory']['stone_pickaxe'] == 0))
                        or (action['equip'] == 'wooden_pickaxe' and
                            (obs['equipped_items']['mainhand']['type'] == 'wooden_pickaxe' or
                             obs['inventory']['wooden_pickaxe'] == 0))):
                    action['equip'] = 'none'

            index_counter += 1
            if 'angle' in obs:
                if obs['angle'][-1] == 0:
                    last_inventory = None
                    stack_index = []
                    stack_index_max = []
                    if len(esp['acts']) != 0:
                        esp_all.append(esp)
                    esp = {'obs': [], 'infos': None, 'acts': [], 'rews': []}


            # key, dict_action = dataset_action_batch_to_actions(action)
            # keyss.append(key)
            #
            # if key != '_0_0':
            #     if obs['equipped_items']['mainhand']['type'] in ['stone_pickaxe', 'iron_pickaxe']:
            #         # stack_index.append([final, a , after_proces])
            #         stack_index.append(0)
            #         add = False
            #         if key in none_move_action:
            #             stack_index_max.append(stack_index)
            #             stack_index = []
            #             add = True
            #
            #         elif last_inventory is not None:
            #             deal_t_e = b_inventory - last_inventory
            #             index_e = np.where(deal_t_e > 0)[0]
            #             if len(index_e) != 0:
            #                 if item_by_attack[index_e[0]] not in ['cobblestone', 'stone', 'dirt']:
            #                     stack_index_max.append(stack_index)
            #                     stack_index = []
            #                     change_attack[item_by_attack[index_e[0]]] += 1
            #                     add = True
            #                 else:
            #                     stack_index_max.append(stack_index)
            #                     stack_index = []
            #                     if len(stack_index_max) > 2:
            #                         del stack_index_max[0]
            #                     # trash_counter += 1
            #         if add:
            #             for v1 in stack_index_max:
            #                 for v2 in v1:
            #                     a_counter += 1
            #             # print(len(stack_index_max))
            #             stack_index_max = []
            #             stack_index = []
            #         # last_inventory = copy.deepcopy(b_inventory)
            #     else:
            #         if last_inventory is not None:
            #             deal_t_e = b_inventory - last_inventory
            #             index_e = np.where(deal_t_e > 0)[0]
            #             if len(index_e) != 0:
            #                 change_attack[item_by_attack[index_e[0]]] += 1
            #         a_counter += 1
            #     # len(index_selected) a_counter / 655850
            # last_inventory = copy.deepcopy(b_inventory)


            # if action['place'] == 'crafting_table':
            #     if j + 5 < len(data['obs']):
            #         if (data['obs'][j + 5]['inventory']['crafting_table'] >=
            #                 obs['inventory']['crafting_table']):
            #             action['place'] = 'none'
            # elif action['place'] == 'furnace':
            #     if j + 5 < len(data['obs']):
            #         if (data['obs'][j + 5]['inventory']['furnace'] >=
            #                 obs['inventory']['furnace']):
            #             action['place'] = 'none'
            # if action['place'] == 'crafting_table':
            #     print("aaasw")
            #     break

            if 'angle' in obs:
                if obs['angle'][-1] == 0:
                    print('Reset', len(all_obs))
                    all_obs = []
                    action_before = []
                    data_obs = []
                    time = 0
                    time_attack_no_new = 0
                    current_item = 0
                    seri_of_craft = []
                    stack_grey = {}
                    last_inventory = None

                angle_1 = obs['angle'][0]
                angle_2 = obs['angle'][1]


            grey = np.dot(obs['pov'][..., :3], [0.2989, 0.5870, 0.1140])

            angle_2_fix = ((int(angle_2) + 180) % 360)
            stack_grey[tuple((angle_1, angle_2_fix))] = [grey, time]

            key, dict_action = dataset_action_batch_to_actions(action)
            keyss.append(key)
            # if key == 'equip_wooden_pickaxe':
            #     break
            # if key not in action_key:
            #     action_key[key] = dict_action

            a = np.where(list_key_index == key)[0][0]

            all_obs.append(obs['pov'])

            final = obs['pov']
            for location in list_obs:
                a1 = angle_1 + location[0]
                a2 = ((int(angle_2 + location[1]) + 180) % 360)
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
            time_attack_no_new_all.append(time_attack_no_new)

            after_proces = process_inventory(obs, time_attack_no_new, time, [], angle_1, angle_2)

            if key in none_move_action:
                seri_of_craft.append(np.where(none_move_action == key)[0][0] /
                                     len(none_move_action))


            if key != '_0_0':
                if obs['equipped_items']['mainhand']['type'] in ['stone_pickaxe', 'iron_pickaxe']:
                    stack_index.append([final, a , after_proces])
                    stack_index = []
                    # add = False
                    # if key in none_move_action:
                    #     stack_index_max.append(stack_index)
                    #     stack_index = []
                    #     add = True
                    # elif last_inventory is not None:
                    #     deal_t_e = b_inventory - last_inventory
                    #     index_e = np.where(deal_t_e > 0)[0]
                    #     if len(index_e) != 0:
                    #         if item_by_attack[index_e[0]] not in ['cobblestone', 'stone', 'dirt']:
                    #             stack_index_max.append(stack_index)
                    #             stack_index = []
                    #             change_attack[item_by_attack[index_e[0]]] += 1
                    #             add = True
                    #         else:
                    #             # stack_index_max.append(stack_index)
                    #             stack_index = []
                    #             if len(stack_index_max) > 2:
                    #                 del stack_index_max[0]
                    #             # trash_counter += 1
                    # if add:
                    #     for v1 in stack_index_max:
                    #         for v2 in v1:
                    #             all_pov_obs.append(v2[0])
                    #             all_actions.append(v2[1])
                    #             all_data_obs.append(v2[2])
                    #     stack_index_max = []
                    #     stack_index = []
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
                    esp['rews'].append(reward)
                    esp['obs'].append({'pov': final, 'inventory': after_proces})
                    esp['acts'].append(a)

                # len(index_selected) / 655850
            last_inventory = copy.deepcopy(b_inventory)

            # if index_counter in index_selected:
            # all_pov_obs.append(final)
            # all_actions.append(a)
            # all_data_obs.append(after_proces)

            if action['attack']:
                if time_attack_no_new == 0:
                    current_item = 0
                    for items_k in ['planks', 'log', 'cobblestone', 'stone', 'iron_ore',
                                    'coal', 'dirt', 'crafting_table', 'furnace']:
                        current_item += obs['inventory'][items_k]
                    time_attack_no_new += 1

                else:
                    check_new = 0
                    for items_k in ['planks', 'log', 'cobblestone', 'stone', 'iron_ore', 'coal', 'dirt',
                                    'crafting_table', 'furnace']:
                        check_new += obs['inventory'][items_k]
                    if check_new != current_item:
                        time_attack_no_new = 0
                    else:
                        time_attack_no_new += 1

            else:
                time_attack_no_new = 0

            if action['camera'][0] < 0:
                angle_1 -= 15
            elif action['camera'][0] > 0:
                angle_1 += 15
            angle_1 = np.clip(angle_1, -90, 90)
            if action['camera'][1] < 0:
                angle_2 -= 15
            elif action['camera'][1] > 0:
                angle_2 += 15
            if angle_2 > 360:
                angle_2 = angle_2 - 360
            elif angle_2 < 0:
                angle_2 = 360 + angle_2
            action_before.append(a)
            data_obs.append(after_proces)
            time += 1

    Counter(keyss)
    # key = list(action_key.keys())
    # key.sort()
    # new_dict = {}
    # for j in key:
    #     # if j not in list_key_index:
    #     #     print(j)
    #     new_dict[j] = action_key[j]
    # action_key = new_dict
    # with open('action_key_1.pkl', 'wb') as f:
    #     pickle.dump(action_key, f)

    all_actions = np.array(all_actions)
    # all_pov_obs = np.array(all_pov_obs)
    all_data_obs = np.array(all_data_obs)
    # all_actions_r = np.array(all_actions_r)
    # np.bincount(all_actions)
    # print(len(all_actions)/ 655850)
    # np.sum(all_actions == 80)
    network = NatureCNN((7, 64, 64), len(list_key_index)).cuda()
    optimizer = th.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    num_samples = all_actions.shape[0]
    update_count = 0
    losses = []

    print("Training")
    for _ in range(12):
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

    TRAIN_MODEL_NAME = 'another_potato_7.pth'  # name to use when saving the trained agent.

    th.save(network.state_dict(), TRAIN_MODEL_NAME)
    del data


# plt.imshow(final[:,:,4])
# plt.show()



network = NatureCNN((7, 64, 64), len(list_key_index)).cuda()
network.load_state_dict(th.load('another_potato_7.pth'))
a = th.load('another_potato_7.pth')
a
env = gym.make('MineRLObtainDiamond-v0')
# env1 = Recorder(env, './video', fps=60)  # saving environment before action shaping to use with scripted part
# env = ActionShaping(env, always_attack=True)
# env.seed(0)

#
# num_actions = env.action_space.n
# action_list = np.arange(num_actions)

# action_sequence = get_action_sequence()

rewards = []

# rewards= [15,19,9,16,0]tranf = torchvision.transforms.Grayscale(num_output_channels=1)
# list_obs = [[-15, 0], [15, 0], [0, -15], [0, 15]]

for episode in range(50):
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

    # cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    # cv2.resizeWindow('image', 1000, 1000)
    # cv2.moveWindow('image', -30, 30)
    stack_grey = {}
    time_wait_equi = 0
    for i in range(18000):

        hsv = obs['pov']
        grey = np.dot(obs['pov'][..., :3], [0.2989, 0.5870, 0.1140])

        angle_2_fix = ((int(angle_2) + 180) % 360)
        stack_grey[tuple((angle_1, angle_2_fix))] = [grey, time_add]

        all_obs.append(copy.deepcopy(obs['pov']))
        for location in list_obs:
            a1 = angle_1 + location[0]
            a2 = ((int(angle_2 + location[1]) + 180) % 360)
            new_tuple = tuple((a1, a2))
            if new_tuple in stack_grey:
                grey_1 = stack_grey[new_tuple]
                if grey_1[1] >= time_add - 1000:
                    hsv = np.concatenate((hsv, grey_1[0][:, :, None]), axis=-1)
                else:
                    hsv = np.concatenate((hsv, np.zeros((64, 64, 1))), axis=-1)
            else:
                hsv = np.concatenate((hsv, np.zeros((64, 64, 1))), axis=-1)
        # hsv = np.concatenate((hsv, first_obs, second_obs), axis=-1)
        hsv = hsv.transpose(2, 0, 1).astype(np.float32)
        hsv = th.from_numpy(hsv[None]).float().cuda()
        hsv /= 255.0

        inven = process_inventory(obs, time_attack_no_new, time_add, [], angle_1, angle_2)
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
                angle_2 -= 15
            elif slipt[-1] == '1':
                angle_2 += 15

            if slipt[-2] == '-1':
                angle_1 -= 15
            elif slipt[-2] == '1':
                angle_1 += 15
            if slipt[-1] == '0' and slipt[-2] == '0':
                counter_0_0 += 1
            else:
                counter_0_0 = 0

            if counter_0_0 >= 300 and time_attack_no_new >= 200:
                check_fix = True
            angle_1 = np.clip(angle_1, -90, 90)

        action = process_a(action, env.action_space.noop())
        obs, reward, done, info = env.step(action)
        if key in ['place_furnace', 'place_crafting_table']:
            for __i in range(2):
                action = process_a(4, env.action_space.noop())
                obs, reward, done, info = env.step(env.action_space.noop())
                total_reward += reward
                time_add += 1

        if previous is not None:
            delta = previous - obs['pov']
            delta = np.sum(delta)
            if delta == 0:
                check_attack = True
            else:
                check_attack = False
        previous = obs['pov']

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

# rewards= [30,21,25,11,23, 22,30,32,26,0, 23,23,28,12,10, 20,27,26,29,14]
# np.mean(rewards)

# while(True):
#     cv2.imshow('image', np.array(tree_obs.cpu().numpy(), dtype=(np.uint8)))
#     cv2.imshow('image', np.array(grey.cpu().numpy(), dtype=(np.uint8)))
#
#     cv2.resizeWindow('image', 950, 950)
#     if cv2.waitKey(10) & 0xFF == ord('o'):
#         break

"""
<Item reward="1" type="log" />
<Item reward="2" type="planks" />
<Item reward="4" type="stick" />
<Item reward="4" type="crafting_table" />
<Item reward="8" type="wooden_pickaxe" />
<Item reward="16" type="cobblestone" />
<Item reward="32" type="furnace" />
<Item reward="32" type="stone_pickaxe" />
<Item reward="64" type="iron_ore" />
<Item reward="128" type="iron_ingot" />
<Item reward="256" type="iron_pickaxe" />
<Item reward="1024" type="diamond" />
"""
