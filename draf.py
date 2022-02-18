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
BATCH_SIZE = 32
NUM_ACTION_CENTROIDS = 100  # number of KMeans centroids used to cluster the data.
Inventory_MAX = {'furnace': 3, 'crafting_table': 3, 'planks': 30, 'stick': 20,
                 'stone_pickaxe': 2, 'wooden_pickaxe': 3, 'iron_pickaxe': 3, 'log': 20, 'cobblestone': 20,
                 'stone': 20, 'iron_ore': 10,'iron_ingot':10, 'coal': 10, 'dirt': 50, 'torch': 20}
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
                             'nearbySmelt_iron_ingot', 'place_crafting_table', 'place_furnace'])


class PovOnlyObservation(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        # 64 x 64 + 64 x64 + 18 [cpa;, cobblestone]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(8210,), dtype=np.float64)
        self.observation_space = self.env.observation_space['pov']
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4114,), dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 2), dtype=np.uint8)

        self.names_stack = []
        self.max_name = 120
        self.max_item = 50

        dict_data = {}
        with open('data_1.pkl', 'rb') as f:
            dict_data = pickle.load(f)

        h_range = []
        s_range = []
        v_range = []

        names = []
        kind = []
        counter = 0
        tree = 0
        for k, v in dict_data.items():
            for i in range(len(v['H'])):
                h_range.append(v['H'][i])
                s_range.append(v['S'][i])
                v_range.append(v['V'][i])

                names.append(v['name'][i])
                if k == "tree_chunk":
                    kind.append(255)
                else:
                    kind.append(0)

            # if k == tre
            # counter += 1

        h_range = np.expand_dims(h_range, axis=0)
        h_range = np.repeat(h_range, 64, axis=0)
        h_range = np.expand_dims(h_range, axis=0)
        h_range = np.repeat(h_range, 64, axis=0)
        h_range = np.array(h_range, dtype=np.float64)
        self.h_range = torch.Tensor(h_range)

        s_range = np.expand_dims(s_range, axis=0)
        s_range = np.repeat(s_range, 64, axis=0)
        s_range = np.expand_dims(s_range, axis=0)
        s_range = np.repeat(s_range, 64, axis=0)
        s_range = np.array(s_range, dtype=np.float64)
        self.s_range = torch.Tensor(s_range)

        v_range = np.expand_dims(v_range, axis=0)
        v_range = np.repeat(v_range, 64, axis=0)
        v_range = np.expand_dims(v_range, axis=0)
        v_range = np.repeat(v_range, 64, axis=0)
        v_range = np.array(v_range, dtype=np.float64)
        self.v_range = torch.Tensor(v_range)

        self.kind = np.array(kind, dtype=np.uint8)

    def observation(self, observation):
        image = observation['pov']
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv_64 = np.array(hsv, dtype=np.float64)
        hsv_64 = torch.Tensor(hsv_64)

        h = torch.unsqueeze(hsv_64[:, :, 0], -1)
        s = torch.unsqueeze(hsv_64[:, :, 1], -1)
        v = torch.unsqueeze(hsv_64[:, :, 2], -1)

        h_error = torch.abs(h - self.h_range)
        index = torch.where(h_error >= 89.5)
        h_error[index[0], index[1], index[2]] = 179 - h_error[index[0], index[1], index[2]]

        d_error = torch.sqrt(torch.pow(h_error, 2) + torch.pow((s / 4 - self.s_range / 4), 2)
                             + torch.pow((v / 6 - self.v_range / 6), 2))

        d_error = torch.argmin(d_error, axis=2)
        tree_obs = self.kind[d_error]
        tree_obs = np.array(tree_obs, dtype=np.uint8)

        hsv[:, :, 0] = 100
        hsv[:, :, 1] = 2

        obs = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        obs = np.expand_dims(obs, axis=-1)

        tree_obs = np.expand_dims(tree_obs, axis=-1)
        obs = np.concatenate((obs, tree_obs), axis=-1)

        return obs


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


class ActionShaping(gym.ActionWrapper):
    """
    The default MineRL action space is the following dict:

    Dict(attack:Discrete(2),
         back:Discrete(2),
         camera:Box(low=-180.0, high=180.0, shape=(2,)),
         craft:Enum(crafting_table,none,planks,stick,torch),
         equip:Enum(air,iron_axe,iron_pickaxe,none,stone_axe,stone_pickaxe,wooden_axe,wooden_pickaxe),
         forward:Discrete(2),
         jump:Discrete(2),
         left:Discrete(2),
         nearbyCraft:Enum(furnace,iron_axe,iron_pickaxe,none,stone_axe,stone_pickaxe,wooden_axe,wooden_pickaxe),
         nearbySmelt:Enum(coal,iron_ingot,none),
         place:Enum(cobblestone,crafting_table,dirt,furnace,none,stone,torch),
         right:Discrete(2),
         sneak:Discrete(2),
         sprint:Discrete(2))

    It can be viewed as:
         - buttons, like attack, back, forward, sprint that are either pressed or not.
         - mouse, i.e. the continuous camera action in degrees. The two values are pitch (up/down), where up is
           negative, down is positive, and yaw (left/right), where left is negative, right is positive.
         - craft/equip/place actions for items specified above.
    So an example action could be sprint + forward + jump + attack + turn camera, all in one action.

    This wrapper makes the action space much smaller by selecting a few common actions and making the camera actions
    discrete. You can change these actions by changing self._actions below. That should just work with the RL agent,
    but would require some further tinkering below with the BC one.
    """

    def __init__(self, env, camera_angle=15, always_attack=False):
        super().__init__(env)

        self.camera_angle = camera_angle
        self.always_attack = always_attack

        self._actions = [
            [('craft', 'planks')],  # 0
            [('craft', 'stick')],  # 1
            [('craft', 'crafting_table')],  # 2

            [('forward', 1), ('jump', 1)],  # 20
            [('camera', [-self.camera_angle, 0])],  # 21
            [('camera', [self.camera_angle, 0])],  # 22
            [('camera', [0, self.camera_angle])],
            [('camera', [0, -self.camera_angle])],
        ]
        self.actions = []
        for actions in self._actions:
            act = self.env.action_space.noop()
            for a, v in actions:
                act[a] = v
            if self.always_attack:
                act['attack'] = 1
            self.actions.append(act)

        # self.action_space = gym.spaces.Discrete(len(self.actions))
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(len(self.actions),), dtype=np.uint8)

    def action(self, action):

        action_new = self.env.action_space.noop()
        key = list_key_index[action]
        for k, v in action_key[key].items():
            action_new[k] = v
        # action_new['attack'] = 1
        return action_new


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

    return data


def checking_craft_place(obs, action):
    inventory = obs['inventory']
    hand = obs['equipped_items']['mainhand']['type']
    if action["craft"] == 'planks':
        if inventory['log'] == 0:
            action["craft"] = 'none'

    if action["craft"] == 'stick':
        if inventory['planks'] == 0:
            action["craft"] = 'none'
    if action["craft"] == 'crafting_table':
        if inventory['planks'] < 4:
            action["craft"] = 'none'

    if action["nearbyCraft"] == 'wooden_pickaxe':
        if inventory['stick'] < 2 or inventory['planks'] < 3:
            action["nearbyCraft"] = 'none'
    if action["nearbyCraft"] == 'stone_pickaxe':
        if inventory['stick'] < 2 or inventory['cobblestone'] < 3:
            action["nearbyCraft"] = 'none'
    if action["nearbyCraft"] == 'furnace':
        if inventory['cobblestone'] < 8:
            action["nearbyCraft"] = 'none'
    if action["nearbyCraft"] == 'iron_pickaxe':
        if inventory['stick'] < 2 or inventory['iron_ingot'] < 3:
            action["nearbyCraft"] = 'none'

    if action["place"] == 'furnace':
        if inventory['iron_ore'] == 0 or inventory['furnace'] == 0:
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
inventory_list = ['coal', 'cobblestone', 'dirt', 'iron_ore', 'log', 'stone']
inventory_list_index = [0, 1, 3, 7, 9, 12]


def train():
    print("Prepare_Data")

    change = {}
    for i in inventory_list:
        change[i] = 0

    iter_count = 0
    losses = []
    all_pov_obs = []
    all_actions = []
    all_data_obs = []
    # action_key = {}

    time_attack_no_new_all = []
    # np.mean(time_attack_no_new_all)
    # np.histogram(time_attack_no_new_all)
    keyss = []
    # imag = plt.hist(time_attack_no_new_all)
    # plt.show()
    for i in range(73):
        with open(f'history_{i}.pkl', 'rb') as f:
            data = pickle.load(f)
        print(len(data['action']), i)

        all_obs = []
        action_before = []
        data_obs = []
        time = 0
        last_action = 0
        angle_1 = 0
        angle_2 = 0
        time_attack_no_new = 0
        current_item = 0
        seri_of_craft = []
        stack_grey = {}
        last_items = None
        for j in range(len(data['obs'])):
            action = data['action'][j]
            obs = data['obs'][j]
            before_items = np.array(list(obs['inventory'].values()))
            before_items = before_items[inventory_list_index]
            action = checking_craft_place(obs, action)
            if action['place'] == 'crafting_table':
                if j + 5 < len(data['obs']):
                    if (data['obs'][j + 5]['inventory']['crafting_table'] >=
                            obs['inventory']['crafting_table']):
                        action['place'] = 'none'
            elif action['place'] == 'furnace':
                if j + 5 < len(data['obs']):
                    if (data['obs'][j + 5]['inventory']['furnace'] >=
                            obs['inventory']['furnace']):
                        action['place'] = 'none'
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
                    last_action = 0
                    time_attack_no_new = 0
                    current_item = 0
                    seri_of_craft = []
                    stack_grey = {}
                    last_items = None

                angle_1 = obs['angle'][0]
                angle_2 = obs['angle'][1]
                if obs['equipped_items']['mainhand']['type'] == 'iron_pickaxe':
                    continue
            else:
                if obs['equipped_items']['mainhand']['type'] == 'iron_pickaxe':
                    break
            if last_items is not None:
                delta = before_items - last_items
                index = np.where(delta > 0)[0]
                if len(index) > 0:
                    change[inventory_list[index[0]]] += 1
            last_items = before_items

            grey = np.dot(obs['pov'][..., :3], [0.2989, 0.5870, 0.1140])

            angle_2_fix = ((int(angle_2) + 180) % 360)
            stack_grey[tuple((angle_1, angle_2_fix))] = [grey, time]
            if action['equip'] != 'none':
                if ((action['equip'] == 'iron_pickaxe' and
                     (obs['equipped_items']['mainhand']['type'] == 'iron_pickaxe' or
                     obs['inventory']['iron_pickaxe'] == 0))
                        or (action['equip'] == 'stone_pickaxe' and
                           ( obs['equipped_items']['mainhand']['type'] == 'stone_pickaxe'or
                            obs['inventory']['stone_pickaxe'] == 0))
                        or (action['equip'] == 'wooden_pickaxe' and
                            (obs['equipped_items']['mainhand']['type'] == 'wooden_pickaxe'or
                     obs['inventory']['wooden_pickaxe'] == 0))):
                    action['equip'] = 'none'
            key, dict_action = dataset_action_batch_to_actions(action)
            keyss.append(key)

            # if key == 'equip_wooden_pickaxe':
            #     break
            # if key not in action_key:
            #     action_key[key] = dict_action

            a = np.where(list_key_index == key)[0][0]

            # data_obs.append()
            all_obs.append(obs['pov'])
            # first_obs = copy.deepcopy(obs['pov'])
            # if len(all_obs) > 3:
            #     first_obs = all_obs[len(all_obs) - 3]
            #
            # second_obs = copy.deepcopy(first_obs)
            # if len(all_obs) > 6:
            #     second_obs = all_obs[len(all_obs) - 6]
            # final = np.concatenate((obs['pov'], first_obs, second_obs), axis=-1)

            if len(seri_of_craft) >= 20:
                add_a = seri_of_craft[len(seri_of_craft) - 20:]
            else:
                add_a = np.zeros(20)
                add_a[20 - len(seri_of_craft):] = seri_of_craft

            final = obs['pov']
            for location in list_obs:
                a1 = angle_1 + location[0]
                a2 = ((int(angle_2 + location[1]) + 180) % 360)
                new_tuple = tuple((a1, a2))
                if new_tuple in stack_grey:
                    grey_1 = stack_grey[new_tuple]
                    if grey_1[1] >= time - 1000:
                        final = np.concatenate((final, grey_1[0][:,:,None]), axis=-1)
                    else:
                        final = np.concatenate((final, np.zeros((64, 64, 1))), axis=-1)

                else:
                    final = np.concatenate((final, np.zeros((64, 64, 1))), axis=-1)
            final = final.astype(np.uint8)
            # add_a = np.array(add_a)/len(list_key_index)
            #
            # stack = np.zeros(18*3)
            # if len(data_obs) >= 2:
            #     stack[:18] = data_obs[len(data_obs) - 1]
            # if len(data_obs) >= 4:
            #     stack[18:36] = data_obs[len(data_obs) - 3]
            # if len(data_obs) >= 6:
            #     stack[36:] = data_obs[len(data_obs) - 5]
            time_attack_no_new_all.append(time_attack_no_new)
            after_proces = process_inventory(obs, time_attack_no_new, time, add_a, angle_1, angle_2)
            # stack = np.concatenate((after_proces, stack))

            if key in none_move_action:
                seri_of_craft.append(np.where(none_move_action == key)[0][0] /
                                     len(none_move_action))
            if key != '_0_0':
                all_pov_obs.append(final)
                all_actions.append(a)
                all_data_obs.append(after_proces)

            if action['attack']:
                last_attack = 1
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
                last_attack = 0
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

            # if action['camera'][0] < 0:
            #     angle_1 += action['camera'][0]
            # elif action['camera'][0] > 0:
            #     angle_1 += action['camera'][0]
            # angle_1 = np.clip(angle_1, -90, 90)
            #
            # if action['camera'][1] < 0:
            #     angle_2 += action['camera'][1]
            # elif action['camera'][1] > 0:
            #     angle_2 += action['camera'][1]

            last_action = a
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
    #all_pov_obs = np.array(all_pov_obs)
    all_data_obs = np.array(all_data_obs)

    # We know ActionShaping has seven discrete actions, so we create
    # a network to map images to seven values (logits), which represent
    # likelihoods of selecting those actions
    network = NatureCNN((7, 64, 64), len(list_key_index)).cuda()
    optimizer = th.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    num_samples = all_actions.shape[0]
    update_count = 0
    losses = []
    # We have the data loaded up already in all_actions and all_pov_obs arrays.
    # Let's do a manual training loop
    print("Training")
    for _ in range(8):
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

    TRAIN_MODEL_NAME = 'another_potato_4.pth'  # name to use when saving the trained agent.

    th.save(network.state_dict(), TRAIN_MODEL_NAME)

    del data

# plt.imshow(final[:,:,4])
# plt.show()

def str_to_act(env, actions):
    """
    Simplifies specifying actions for the scripted part of the agent.
    Some examples for a string with a single action:
        'craft:planks'
        'camera:[10,0]'
        'attack'
        'jump'
        ''
    There should be no spaces in single actions, as we use spaces to separate actions with multiple "buttons" pressed:
        'attack sprint forward'
        'forward camera:[0,10]'

    :param env: base MineRL environment.
    :param actions: string of actions.
    :return: dict action, compatible with the base MineRL environment.
    """
    act = env.action_space.noop()
    for action in actions.split():
        if ":" in action:
            k, v = action.split(':')
            if k == 'camera':
                act[k] = eval(v)
            else:
                act[k] = v
        else:
            act[action] = 1
    return act


network = NatureCNN((7, 64, 64), len(list_key_index)).cuda()
network.load_state_dict(th.load('another_potato_4.pth'))

env = gym.make('MineRLObtainDiamond-v0')
# env1 = Recorder(env, './video', fps=60)  # saving environment before action shaping to use with scripted part
# env = ActionShaping(env, always_attack=True)
env.seed(0)

#
# num_actions = env.action_space.n
# action_list = np.arange(num_actions)

# action_sequence = get_action_sequence()

rewards = []
tranf = torchvision.transforms.Grayscale(num_output_channels=1)
list_obs = [[-15, 0], [15, 0], [0, -15], [0, 15]]

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
    seri_of_craft = []

    previous = None
    check_attack = False

    cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('image', 1000, 1000)
    cv2.moveWindow('image', -30, 30)
    stack_grey = {}
    time_wait_equi = 0
    for i in range(18000):
        # Process the action:
        #   - Add/remove batch dimensions
        #   - Transpose image (needs to be channels-last)
        #   - Normalize image

        hsv = obs['pov']
        grey = np.dot(obs['pov'][..., :3], [0.2989, 0.5870, 0.1140])

        angle_2_fix = ((int(angle_2) + 180) % 360)
        stack_grey[tuple((angle_1, angle_2_fix))] = [grey, time_add]

        all_obs.append(copy.deepcopy(obs['pov']))

        first_obs = copy.deepcopy(obs['pov'])
        if len(all_obs) > 3:
            first_obs = all_obs[len(all_obs) - 3]

        second_obs = copy.deepcopy(first_obs)
        if len(all_obs) > 6:
            second_obs = all_obs[len(all_obs) - 6]

        hsv = obs['pov']
        for location in list_obs:
            a1 = angle_1 + location[0]
            a2 = ((int(angle_2 + location[1]) + 180) % 360)
            new_tuple = tuple((a1, a2))
            if new_tuple in stack_grey:
                grey_1 = stack_grey[new_tuple]
                if grey_1[1] >= time_add - 1000:
                    hsv = np.concatenate((hsv, grey_1[0][:,:,None]), axis=-1)
                else:
                    hsv = np.concatenate((hsv, np.zeros((64, 64, 1))), axis=-1)
            else:
                hsv = np.concatenate((hsv, np.zeros((64, 64, 1))), axis=-1)
        # hsv = np.concatenate((hsv, first_obs, second_obs), axis=-1)
        hsv = hsv.transpose(2, 0, 1).astype(np.float32)
        hsv = th.from_numpy(hsv[None]).float().cuda()
        hsv /= 255.0

        # if len(action_before) >= 100:
        #     add_a = action_before[len(action_before) - 100:]
        # else:
        #     add_a = np.repeat(last_action, 100)
        #     add_a[100 - len(action_before):] = action_before
        #
        # add_a = np.array(add_a) / len(list_key_index)
        if len(seri_of_craft) >= 20:
            add_a = seri_of_craft[len(seri_of_craft) - 20:]
        else:
            add_a = np.zeros(20)
            add_a[20 - len(seri_of_craft):] = seri_of_craft

        # stack = np.zeros(18 * 3)
        # if len(data_obs) >= 2:
        #     stack[:18] = data_obs[len(data_obs) - 1]
        # if len(data_obs) >= 4:
        #     stack[18:36] = data_obs[len(data_obs) - 3]
        # if len(data_obs) >= 6:
        #     stack[36:] = data_obs[len(data_obs) - 5]

        inven = process_inventory(obs, time_attack_no_new, time_add, add_a, angle_1, angle_2)

        # stack = np.concatenate((inven, stack))
        # stack = th.from_numpy(stack[None]).float().cuda()

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
            seri_of_craft.append(np.where(none_move_action == key)[0][0] /
                                 len(none_move_action))
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
        # print(total_reward)
        # if check_fix:
        #     if start_fix < 4:
        #         action = 6
        #         start_fix += 1
        #         print("check_fix")
        #     else:
        #         check_fix = False
        #         counter_0_0 = 0
        action = process_a(action, env.action_space.noop())
        obs, reward, done, info = env.step(action)
        if key in ['place_furnace', 'place_crafting_table']:
            for i in range(2):
                action = process_a(4, env.action_space.noop())
                obs, reward, done, info = env.step(action)
                total_reward += reward

        # x = obs['inventory']
        # y = obs['equipped_items']
        # print('p: ', x['planks'],
        #       'log: ', x['log'], 'stick: ', x['stick'],
        #       'ct: ', x['crafting_table'],
        #       'furnace: ', x['furnace'],
        #       'c_ston: ', x['cobblestone'],
        #       'stone: ', x['stone'],
        #       'iron_ingot: ', x['iron_ingot'],
        #       'wooden_pickaxe: ', x['wooden_pickaxe'],
        #       'stone_pickaxe: ', x['stone_pickaxe'],
        #       'iron_pickaxe: ', x['iron_pickaxe'],
        #       'hand: ', y['mainhand']['type'], end='\r')
        if previous is not None:
            delta = previous - obs['pov']
            delta = np.sum(delta)
            if delta == 0:
                check_attack = True
            else:
                check_attack = False
        previous = obs['pov']

        cv2.imshow('image', cv2.cvtColor(obs['pov'], cv2.COLOR_BGR2RGB))
        cv2.resizeWindow('image', 950, 950)
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
# rewards= [15,19,9,16,0]
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
