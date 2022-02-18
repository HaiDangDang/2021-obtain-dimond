import random
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
import torch as th
from torch import nn
import gym
import minerl
import cv2
import matplotlib.pyplot as plt
import pickle
import torch
from queue import Queue
from utils import actions_from_file
import os


# Parameters:
fail_data = ['v3_absolute_grape_changeling-17_418-2303',
     'v3_equal_olive_chimera-9_25021-26077',
     'v3_agonizing_kale_tree_nymph-20_58203-59745',
     'v3_kindly_lemon_mummy-11_2752-6180',
     'v3_juvenile_apple_angel-26_862-2906']
DATA_DIR = "data"  # path to MineRL dataset (should contain "MineRLObtainIronPickaxeVectorObf-v0" directory).
NUM_ACTION_CENTROIDS = 100  # number of KMeans centroids used to cluster the data.
data = minerl.data.make("MineRLObtainDiamondVectorObf-v0", data_dir=DATA_DIR, num_workers=1)
data_IT = 'MineRLObtainDiamond-v0'
data_RT = 'MineRLObtainDiamondVectorObf-v0'


print("KMeans done")
def find_key(byWhat, action_with_key, key):
    first_angle = action_with_key['camera'][0]
    second_angle = action_with_key['camera'][1]
    keys = list(action_with_key.keys())

    if byWhat == 'camera':
        # if first_angle == 0 :
        #     key += '_0'
        # elif first_angle > 0 :
        #     key += '_+'
        # elif first_angle < 0 :
        #     key += '_-'
        # if second_angle == 0 :
        #     key += '_0'
        # elif second_angle > 0 :
        #     key += '_+'
        # elif second_angle < 0 :
        #     key += '_-'

        if np.abs(first_angle) > 3 and np.abs(first_angle) < 15 :
            if first_angle < 0:
                key += '_-1'
            else:
                key += '_1'
        else:
            key += '_0'

        if np.abs(second_angle) > 3 and np.abs(second_angle) < 15  :
            if second_angle < 0:
                key += '_-1'
            else:
                key += '_1'
        else:
            key += '_0'

        return key
    elif byWhat == 'jump':
        if 'jump' in keys:
            key += '_j'
        return find_key('move', action_with_key, key)
    elif byWhat == 'move':
        if 'forward' in keys:
            key += '_f'
        elif 'back' in keys or 'left' in keys or 'right' in keys:
            key += '_a'
        # elif 'back' in keys:
        #     key += '_b'
        #
        # if 'left' in keys:
        #     key += '_l'
        # elif 'right' in keys:
        #     key += '_r'

        return find_key('camera', action_with_key, key)
    return key

output_file = f'MineRLObtainDiamond-v0.pkl'
with open(output_file, 'rb') as file:
    data_map = pickle.load(file)

list_actions = list(data_map.keys())

list_craft = ['craft: planks', 'craft: crafting_table', 'craft: stick',
              'craft: torch', 'nearbyCraft: iron_axe']
counter = 0
list_place = ['place: cobblestone', 'place: crafting_table', 'place: dirt',
              'place: furnace', 'place: stone', 'place: torch']
string_int = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','-', '.']
dict_actions = []
for i in range(len(list_actions)):
    grade = 0
    dicts = {}
    for a in list_actions[i]:
        string_split = a.split()
        if string_split[0] == 'camera:':
            first_angle = ''
            second_angle = ''
            found = False
            next = 1
            for j in a:
                if j in string_int:
                    if (first_angle == '' or found) and next == 1:
                        first_angle += j
                        found = True
                    if (second_angle == '' or found) and next == 2:
                        second_angle += j
                        found = True

                else:
                    found = False
                    if first_angle != '':
                        next = 2
            if first_angle[-1] == '.':
                first_angle += '0'
            if second_angle[-1] == '.':
                second_angle += '0'

            first_angle = float(first_angle)
            second_angle = float(second_angle)
            if first_angle == 0 and second_angle == 0:
                grade += 1
            dicts['camera'] = [first_angle, second_angle]
        elif string_split[0] == 'craft:':
            dicts['craft'] = string_split[1]
        elif string_split[0] == 'nearbyCraft:':
            dicts['nearbyCraft'] = string_split[1]
        elif string_split[0] == 'nearbySmelt:':
            dicts['nearbySmelt'] = string_split[1]
        elif string_split[0] == 'place:':
            dicts['place'] = string_split[1]
        elif string_split[0] == 'equip:':
            dicts['equip'] = string_split[1]
        else:
            dicts[a] = 1
    dicts['action'] = data_map[list_actions[i]]
    dict_actions.append(dicts)

len(dict_actions)
len(list_actions)
data_map[list_actions[102744]]
dict_actions_keys = { 'a_non_c'}


knn_action = []
dict_keys = {}
counter = 0
f_a =[]
s_a = []
for i in range(len(dict_actions)):
    keys = list(dict_actions[i].keys())
    action_with_key = dict_actions[i]
    first_angle = action_with_key['camera'][0]
    f_a.append(first_angle)
    second_angle = action_with_key['camera'][1]
    s_a.append(second_angle)
    if 'craft' in keys:
        craft = action_with_key['craft']
        k = f'craft_{craft}'
        if k not in dict_keys:
            dict_keys[k] = [action_with_key['action']]
        else:
            # print(k)
            assert False
    elif 'equip' in keys:
        equip = action_with_key['equip']
        k = f'equip_{equip}'
        if k not in dict_keys:
            dict_keys[k] = [action_with_key['action']]
        else:
            dict_keys[k].append(action_with_key['action'])
    elif 'nearbyCraft' in keys:
        nearbyCraft = action_with_key['nearbyCraft']
        k = f'nearbyCraft_{nearbyCraft}'
        if k not in dict_keys:
            dict_keys[k] = [action_with_key['action']]
        else:
            print(k)
            assert False
    elif 'nearbySmelt' in keys:
        nearbySmelt = action_with_key['nearbySmelt']
        k = f'nearbySmelt_{nearbySmelt}'
        if k not in dict_keys:
            dict_keys[k] = [action_with_key['action']]
        else:
            assert False
    elif 'place' in keys:
        place = action_with_key['place']
        k = f'place_{place}'
        if k not in dict_keys:
            dict_keys[k] = [action_with_key['action']]
            #knn_action.append(action_with_key['action'])
        else:
            dict_keys[k].append(action_with_key['action'])
    elif 'attack' in keys:
        if len(keys) == 2:
            if np.abs(first_angle) == 0 and np.abs((second_angle)) == 0:
                key = 'attack_0_0'
            else:
                key = find_key('camera',action_with_key,'attack')
        else:
            key = find_key('jump',action_with_key,'attack')

        if key not in dict_keys:
            dict_keys[key] = [action_with_key['action']]
            #knn_action.append(action_with_key['action'])
        else:
            dict_keys[key].append(action_with_key['action'])
    else:
        key = find_key('jump', action_with_key, '')
        if key not in dict_keys:
            dict_keys[key] = [action_with_key['action']]
            #knn_action.append(action_with_key['action'])
        else:
            dict_keys[key].append(action_with_key['action'])

np.max(s_a)
len(dict_keys)
for k,v in dict_keys.items():
    print(k, len(v))



knn_number = {
    'attack_0_0': 2, # attack_0_0 49473
    'attack_0_1': 8, # attack_0_1 15386
    'attack_1_0': 8, # attack_1_0 19504
    'attack_1_1': 10, # attack_1_1 6697

    'attack_f_0_0': 2, # attack_f_0_0 15446
    'attack_f_1_0': 8, # attack_f_0_1 3916
    'attack_f_0_1': 8, # attack_f_1_0 4654
    'attack_f_1_1': 10, # attack_f_1_1 1669

    'attack_a_0_0': 1, # attack_a_0_0 4810
    'attack_a_1_1': 1, # attack_a_1_1 576
    'attack_a_0_1': 1, # attack_a_0_1 1728
    'attack_a_1_0': 1, # attack_a_1_0 889

    'attack_j_f_0_0': 2,  # attack_j_f_0_0 980
    'attack_j_f_0_1': 6,  # attack_j_f_0_1 124
    'attack_j_f_1_0': 6,  # attack_j_f_1_0 375
    'attack_j_f_1_1': 12,  # attack_j_f_1_1 93

    'attack_j_0_0': 2,  # attack_j_0_0 86
    'attack_j_1_1': 4,  # attack_j_1_1 9
    'attack_j_1_0': 4,  # attack_j_1_0 15
    'attack_j_0_1': 4,  # attack_j_0_1 0

    'attack_j_a_0_0': 2,  # attack_j_a_0_0 90
    'attack_j_a_1_0': 4,  # attack_j_a_1_0 17
    'attack_j_a_0_1': 4,  # attack_j_a_0_1 28
    'attack_j_a_1_1': 'attack_j_a_0_0',  # attack_j_a_1_1 0

    '_0_0': '_1_0',  # _0_0 45926
    '_1_0': 6,  # _1_0 17717
    '_0_1': 6,  # _0_1 23795
    '_1_1': 6,  # _1_1 13601

    '_f_0_0': 2,  # _f_0_0 47697
    '_f_1_0': 8,  # _f_1_0 7760
    '_f_0_1': 8,  # _f_0_1 25061
    '_f_1_1': 12,  # _f_1_1 6934

    '_j_f_0_0': 2,  # _j_f_0_0 13483
    '_j_f_0_1': 6,  # _j_f_0_1 5352
    '_j_f_1_1': 6,  # _j_f_1_1 1465
    '_j_f_1_0': 12,  # _j_f_1_0 1662

    '_j_0_0': 2,  # _j_0_0 900
    '_j_0_1': 4,  # _j_0_1 232
    '_j_1_0': 4,  # _j_1_0 151
    '_j_1_1': 6,  # _j_1_1 113

    '_a_0_1': 4,  # _a_0_1 5452
    '_a_0_0': 1,  # _a_0_0 11310
    '_a_1_1': 4,  # _a_1_1 1915
    '_a_1_0': 4,  # _a_1_0 2591

    '_j_a_0_0': 1,  # _j_a_0_0 776
    '_j_a_0_1': 2,  # _j_a_0_1 234
    '_j_a_1_1': 2,  # _j_a_1_1 100
    '_j_a_1_0': 2,  # _j_a_1_0 154

    'place_crafting_table': 2,  # place_crafting_table 164
    'place_furnace': 2,  # place_furnace 170
    'place_torch': 2,  # place_torch 1024
    'place_dirt': 1,  # place_dirt 179
    'place_stone': 1,  # place_stone 276
    'place_cobblestone': 1,  # place_cobblestone 975

    'equip_wooden_pickaxe': 1,  # equip_wooden_pickaxe 140
    'equip_stone_pickaxe': 1,  # equip_stone_pickaxe 214
    'equip_iron_pickaxe': 1,  # equip_iron_pickaxe 162

    'equip_wooden_axe': 'equip_wooden_pickaxe',  # equip_wooden_axe 96
    'equip_stone_axe': 'equip_stone_pickaxe',  # equip_stone_axe 71
    'equip_iron_axe': 'equip_iron_pickaxe',  # equip_iron_axe 10

    'nearbyCraft_wooden_axe': 'nearbyCraft_wooden_pickaxe',  # equip_iron_axe 10
    'nearbyCraft_stone_axe': 'nearbyCraft_stone_pickaxe',  # equip_iron_axe 10
    'nearbyCraft_iron_axe': 'nearbyCraft_iron_pickaxe',  # equip_iron_axe 10
}

counter = 0
for k,v in knn_number.items():
    if type(v) == int:
        counter += v
print(counter)


counter = 0
final_key_with_index = {}
knn_action =[]
for k,v in dict_keys.items():
    print(k)
    if k in knn_number:
        if type(knn_number[k]) == int:
            kmeans = KMeans(n_clusters=knn_number[k])
            kmeans.fit(np.array(v))
            action_centroids = kmeans.cluster_centers_
            index = []
            for a in action_centroids:
                knn_action.append(a)
                index.append(counter)
                counter += 1
            final_key_with_index[k]= index

        # if type(knn_number[k]) == str:
        #     final_key_with_index[k] = final_key_with_index[knn_number[k]]

    else:
        assert len(v) == 1
        knn_action.append(v[0])
        final_key_with_index[k] = [counter]
        counter += 1


print(counter)

knn_action = np.array(knn_action)
knn_action.shape

counter = 0
for k,v in knn_number.items():
    if k not in final_key_with_index:
        print(k)
print(counter)
final_key_with_in

with open('final_key_with_index.pkl', 'wb') as f:
    pickle.dump(final_key_with_index, f)


with open('knn_action.pkl', 'wb') as f:
    pickle.dump(knn_action, f)

for k,v in final_key_with_index.items():
    for i in v:
        if i == 42:
            print(k)
            break
final_key_with_index['_j_f_0_0']
final_key_with_index['_0_0']

















