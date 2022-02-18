#
# import random
#
# from tqdm import tqdm
# import numpy as np
# from sklearn.cluster import KMeans
# import torch as th
# from torch import nn
# import gym
# import minerl
# import cv2
# import matplotlib.pyplot as plt
# import pickle
# # Parameters:
# fail_data = ['v3_absolute_grape_changeling-17_418-2303',
#              'v3_equal_olive_chimera-9_25021-26077',
#              'v3_agonizing_kale_tree_nymph-20_58203-59745',
#              'v3_kindly_lemon_mummy-11_2752-6180',
#              'v3_juvenile_apple_angel-26_862-2906']
# DATA_DIR = "data"  # path to MineRL dataset (should contain "MineRLObtainIronPickaxeVectorObf-v0" directory).
# EPOCHS = 2  # how many times we train over dataset.
# LEARNING_RATE = 0.0001  # learning rate for the neural network.
# BATCH_SIZE = 32
# NUM_ACTION_CENTROIDS = 100  # number of KMeans centroids used to cluster the data.
#
# DATA_SAMPLES = 500000
#
# TRAIN_MODEL_NAME = 'research_potato.pth'  # name to use when saving the trained agent.
# TEST_MODEL_NAME = 'research_potato.pth'  # name to use when loading the trained agent.
# TRAIN_KMEANS_MODEL_NAME = 'centroids_for_research_potato.npy'  # name to use when saving the KMeans model.
# TEST_KMEANS_MODEL_NAME = 'centroids_for_research_potato.npy'  # name to use when loading the KMeans model.
#
# TEST_EPISODES = 10  # number of episodes to test the agent for.
# MAX_TEST_EPISODE_LEN = 18000  # 18k is the default for Mi
# data = minerl.data.make("MineRLObtainDiamondVectorObf-v0", data_dir=DATA_DIR, num_workers=1)
#
# action_state = ['log', 'plank', 'stick', 'table', 'wood', 'c_stone', 'furnace', 'stone',
#                 'ore', 'ingot', 'iron', 'diamond']
#
# all_actions = {}
# for s in action_state:
#     all_actions[s] = []
#
# print("Loading data")
# trajectory_names = data.get_trajectory_names()
# random.shuffle(trajectory_names)
#
# # Add trajectories to the data until we reach the required DATA_SAMPLES.
# find = True
# for trajectory_name in trajectory_names:
#     if find:
#         if trajectory_name == 'v3_alarming_arugula_medusa-25_287-18941':
#             find = False
#         else:
#             continue
#     trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)
#     total_rewards = 0
#     for dataset_observation, dataset_action, r, _, _ in trajectory:
#         total_rewards += r
#     trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)
#     counter = 0
#     if total_rewards >= 32:
#         for dataset_observation, dataset_action, r, _, _ in trajectory:
#             all_actions[action_state[counter]].append(dataset_action['vector'])
#             # all_states.append(counter)
#             if r != 0:
#                 counter += 1
# all_actions = np.array(all_actions)
#
# for s in action_state:
#     all_actions[s] = np.array(all_actions[s])
#
# # Run k-means clustering using scikit-learn.
# all_action_centroids = {}
# for s in action_state:
#     print(s)
#     print(all_actions[s].shape)
#     kmeans = KMeans(n_clusters=NUM_ACTION_CENTROIDS)
#     kmeans.fit(all_actions[s])
#     all_action_centroids[s] = kmeans.cluster_centers_
#     print(all_action_centroids[s].shape)
#
# with open('action_centroids.pkl', 'wb') as f:
#     pickle.dump(all_action_centroids, f)
#
# all_action_centroids = {}
# with open('action_centroids.pkl', 'rb') as f:
#     all_action_centroids = pickle.load(f)
#
# action_centroids = np.zeros((len(action_state), NUM_ACTION_CENTROIDS, 64), dtype=np.float64)
# action_centroids[0, :] = all_action_centroids['log']
# for i in range(100):
#     counter = 1
#     a = action_centroids[0, i]
#     for j in action_state:
#         if j != 'log':
#             print(j)
#             x = all_action_centroids[j]
#             distances = np.sum(np.power((x - a), 2), axis=1)
#             arg_min = np.argmin(distances)
#             action_centroids[counter, i] = x[arg_min]
#             all_action_centroids[j] = np.delete(x, arg_min)
#             counter += 1
#
# np.save(TRAIN_KMEANS_MODEL_NAME, action_centroids)


import os
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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # data visualization
import seaborn as sns  # statistical data visualization
import pickle
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


import time

# Parameters:
MOVING_ACTIONS = ['action$forward', 'action$left', 'action$back', 'action$right', 'action$jump', 'action$sneak',
                  'action$sprint', 'action$attack']
OTHER_ACTIONS = ['action$camera', 'action$place', 'action$equip', 'action$craft', 'action$nearbyCraft',
                 'action$nearbySmelt']

DATA_DIR = "data"  # path to MineRL dataset (should contain "MineRLObtainIronPickaxeVectorObf-v0" directory).


data = minerl.data.make("MineRLObtainDiamondVectorObf-v0", data_dir=DATA_DIR, num_workers=1)
trajectory_names = data.get_trajectory_names()
trajectory_names.sort()


with open('action_key_1.pkl', 'rb') as f:
    action_key = pickle.load(f)
list_key_index = np.array(list(action_key.keys()))
print(len(list_key_index))

env = gym.make('MineRLObtainDiamond-v0')

action_noop = env.action_space.noop()
action_connect_to_vector = {}

for k, v in action_key.items():
    action_connect_to_vector[k] = None



founded = 0
check_2 = False
a_1_s = []
a_2_s = []
main_obs = []
vector_obs = []


import_key = {}
for k_a in action_noop.keys():
    if k_a != 'camera':
        k_in_a = 'action' + '$' + k_a
        import_key[k_in_a] = []

for i in range(len(trajectory_names)):
    trajectory_name = trajectory_names[i]
    print(trajectory_name, i)
    obs_vector = np.load(os.path.join(DATA_DIR, 'MineRLObtainDiamondVectorObf-v0', trajectory_name, 'rendered.npz'))
    obs_normal = np.load(os.path.join(DATA_DIR, 'MineRLObtainDiamond-v0', trajectory_name, 'rendered.npz'))
    if len(a_1_s) == 0:
        a_1_s = np.array(obs_normal['action$camera'])
        vector_obs = np.array(obs_vector['action$vector'])
        for k in import_key.keys():
            import_key[k] = np.array(obs_normal[k])
    else:
        a_1_s = np.concatenate((a_1_s,np.array(obs_normal['action$camera'])), axis=0)
        vector_obs = np.concatenate((vector_obs,np.array(obs_vector['action$vector'])), axis=0)

        for k in import_key.keys():
            import_key[k] = np.concatenate((import_key[k],np.array(obs_normal[k])), axis=0)


action_connect_to_vector = {}

for k, v in action_key.items():
    action_connect_to_vector[k] = None
for key, v in action_key.items():
    check_1 = True
    another_key = {}
    a_1_b = 0
    a_1_u = 0
    a_1_d = 0

    a_2_b = 0
    a_2_u = 0
    a_2_d = 0
    if action_connect_to_vector[key] is None:
        for k_a in action_noop.keys():
            k_in_a = 'action' + '$' + k_a
            if k_a == 'camera':
                if k_a in v:
                    a_1_b = v[k_a][0]
                    a_2_b = v[k_a][1]

                if a_1_b < 0:
                    a_1_u = -13
                    a_1_d = -17

                elif a_1_b > 0:
                    a_1_u = 17
                    a_1_d = 13

                if a_2_b < 0:
                    a_2_u = -13
                    a_2_d = -17

                elif a_2_b > 0:
                    a_2_u = 17
                    a_2_d = 13
            else:
                a_main = action_noop[k_a]
                if k_a in v:
                    a_main = v[k_a]
                another_key[k_in_a] = a_main

    index_s = ((a_1_s[:, 0] <= a_1_u) & (a_1_s[:, 0] >= a_1_d) &
                 (a_1_s[:, 1] <= a_2_u) & (a_1_s[:, 1] >= a_2_d))

    for k, v in another_key.items():
        if np.sum(index_s) != 0:
            index_s = (import_key[k] == v) & index_s
        else:
            break

    if np.sum(index_s) != 0:
        print("founded", key)
        ve_obs = vector_obs[index_s]
        angle = a_1_s[index_s]
        a = np.sqrt(a_1_s[:, 0][index_s] - a_1_b) ** 2 + (a_1_s[:, 1][index_s] - a_2_b) ** 2
        index = np.argmin(a)
        action_connect_to_vector[key] = [ve_obs[index], angle[index]]
    else:
        print("NOT FOUND", key)


delta_up = 17
delta_down = 13
for delta_down in [13,12,11,10,9]:
    for key, v in action_key.items():
        if action_connect_to_vector[key] is None:
            another_key = {}
            a_1_b = 0
            a_1_u = 0
            a_1_d = 0

            a_2_b = 0
            a_2_u = 0
            a_2_d = 0
            if action_connect_to_vector[key] is None:
                for k_a in action_noop.keys():
                    k_in_a = 'action' + '$' + k_a
                    if k_a == 'camera':
                        if k_a in v:
                            a_1_b = v[k_a][0]
                            a_2_b = v[k_a][1]

                        if a_1_b < 0:
                            a_1_u = -delta_down
                            a_1_d = -17

                        elif a_1_b > 0:
                            a_1_u = 17
                            a_1_d = delta_down

                        if a_2_b < 0:
                            a_2_u = -delta_down
                            a_2_d = -17

                        elif a_2_b > 0:
                            a_2_u = 17
                            a_2_d = delta_down
                    else:
                        a_main = action_noop[k_a]
                        if k_a in v:
                            a_main = v[k_a]
                        another_key[k_in_a] = a_main

            index_s = ((a_1_s[:, 0] <= a_1_u) & (a_1_s[:, 0] >= a_1_d) &
                       (a_1_s[:, 1] <= a_2_u) & (a_1_s[:, 1] >= a_2_d))


            for k, v in another_key.items():
                if np.sum(index_s) != 0:
                    index_s = (import_key[k] == v) & index_s
                    # print(np.sum(index_s), k)

                else:
                    # print(np.sum(index_s), k)
                    break

            if np.sum(index_s) != 0:
                print("founded", key)
                ve_obs = vector_obs[index_s]
                angle = a_1_s[index_s]
                a = np.sqrt(a_1_s[:, 0][index_s] - a_1_b) ** 2 + (a_1_s[:, 1][index_s] - a_2_b) ** 2
                index = np.argmin(a)
                action_connect_to_vector[key] = [ve_obs[index], angle[index]]
            else:
                print("NOT FOUND", key)



for key, v in action_connect_to_vector.items():
    # print(v[1])
    if v is None:
        print(key)

x = vector_obs[action_connect_to_vector['_f_0_0']]

delta = []
for i in range(len(x) -1):
    delta.append(np.sum(x[i] - x[i+1]))
np.sum(delta)


action_connect_to_vector['attack_j_-1_0'] = action_connect_to_vector['attack_j_f_-1_0']
action_connect_to_vector['attack_j_0_-1'] = action_connect_to_vector['attack_j_f_0_-1']
action_connect_to_vector['attack_j_0_1'] = action_connect_to_vector['attack_0_1']
action_connect_to_vector['attack_j_1_-1'] = action_connect_to_vector['attack_j_f_1_-1']
action_connect_to_vector['attack_j_1_0'] = action_connect_to_vector['attack_j_f_1_0']
action_connect_to_vector['attack_j_f_-1_-1'] = action_connect_to_vector['_j_f_-1_-1']
action_connect_to_vector['attack_j_f_0_1'] = action_connect_to_vector['_j_f_0_1']

for key, v in action_connect_to_vector.items():
    if v is None:
        print(key)

x = {}
for key, v in action_connect_to_vector.items():
    x[key] = [vector_obs[v][0], a_1_s[v][0]]

#
# with open('action_connect_to_vector.pkl', 'wb') as f:
#     pickle.dump(action_connect_to_vector, f)
invtory_obs = []
inventory_list = ['coal', 'cobblestone', 'crafting_table', 'dirt', 'furnace', 'iron_ore', 'iron_ingot', 'iron_pickaxe',
                  'log', 'planks', 'stick', 'stone', 'stone_pickaxe', 'torch', 'wooden_pickaxe', 'mainhand']

# for k_a in inventory_list:
#     if k_a != 'mainhand':
#         k_in_a = 'observation' + '$' + 'inventory$' + k_a
#         import_key[k_in_a] = []
#     else:
#         k_in_a = 'observation$equipped_items.mainhand.type'
#         import_key[k_in_a] = []
import_key = {}

for k_a in inventory_list:
    import_key[k_a] = []

for i in range(len(trajectory_names)):
    trajectory_name = trajectory_names[i]
    print(trajectory_name, i)
    obs_vector = np.load(os.path.join(DATA_DIR, 'MineRLObtainDiamondVectorObf-v0', trajectory_name, 'rendered.npz'))
    obs_normal = np.load(os.path.join(DATA_DIR, 'MineRLObtainDiamond-v0', trajectory_name, 'rendered.npz'))
    if len(invtory_obs) == 0:
        invtory_obs = np.array(obs_vector['observation$vector'])
        for k_a in inventory_list:
            if k_a != 'mainhand':
                k_in_a = 'observation' + '$' + 'inventory$' + k_a
            else:
                k_in_a = 'observation$equipped_items.mainhand.type'
            import_key[k_a] = np.array(obs_normal[k_in_a])
    else:
        invtory_obs = np.concatenate((invtory_obs,np.array(obs_vector['observation$vector'])), axis=0)
        for k_a in inventory_list:
            if k_a != 'mainhand':
                k_in_a = 'observation' + '$' + 'inventory$' + k_a
            else:
                k_in_a = 'observation$equipped_items.mainhand.type'
            import_key[k_a] = np.concatenate((import_key[k_a], np.array(obs_normal[k_in_a])), axis=0)
    delta = obs_vector['observation$vector']
    x =np.roll(delta, 1, axis=0)
    a = np.sum((delta[1:] - x[1:]), axis=1)
    index = np.sum(a == 0)


model_class = {}
for k_a in inventory_list:
    model_class[k_a] = None

x = import_key['mainhand']
np.unique(x)
with open('Inventory_MAX.pkl', 'rb') as f:
    Inventory_MAX = pickle.load(f)

for k, v in import_key.items():

    if model_class[k] is None:
        X = np.array(invtory_obs)

        if k != 'mainhand':
            y = np.clip(v, 0, Inventory_MAX[k])
            # if k in ['iron_pickaxe', 'stone_pickaxe', 'wooden_pickaxe']:
            #     index = np.where(y == 0)[0]
            #     print(k, np.sum(y == 0)/len(y))
            #     delta = 0.7
            #     X_train, _, y_train, _ = train_test_split(X[index], y[index], test_size=delta, random_state=10)
            #     index = np.where(y != 0)[0]
            #     X = np.concatenate((X_train, X[index]), axis=0)
            #     y = np.concatenate((y_train, y[index]), axis=0)


        else:
            y = np.zeros(len(v))

            index = np.where(v == 'iron_axe')[0]
            y[index] = 3
            index = np.where(v == 'iron_pickaxe')[0]
            y[index] = 0

            index = np.where(v == 'stone_axe')[0]
            y[index] = 3
            index = np.where(v == 'stone_pickaxe')[0]
            y[index] = 1

            index = np.where(v == 'wooden_axe')[0]
            y[index] = 3
            index = np.where(v == 'wooden_pickaxe')[0]
            y[index] = 2

            index = np.where(v == 'other')[0]
            y[index] = 3
            index = np.where(v == 'none')[0]
            y[index] = 3



        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10)
        clf = lgb.LGBMClassifier(device='cpu', boosting_type='gbdt', num_leaves=80, max_depth=8, learning_rate=0.01,
                                 n_estimators=60,
                                 subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0,
                                 min_child_weight=0.001,
                                 min_child_samples=80, subsample=1.0, subsample_freq=0, colsample_bytree=1.0,
                                 reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=12, silent='warn',
                                 importance_type='split')
        clf.fit(X_train, y_train)

        tmp_time = time.time()
        y_pred = clf.predict(X_test)
        model_class[k] = clf
        print(k, accuracy_score(y_pred, y_test))

X_test.shape
for k, v in model_class.items():
    with open(f'./lgb/{k}.pkl', 'wb') as fout:
        pickle.dump(v, fout)






#######################


invtory_obs = []
inventory_list = ['coal', 'cobblestone', 'crafting_table', 'dirt', 'furnace', 'iron_ore', 'log', 'stone']

inventory_list = np.array(inventory_list)
# for k_a in inventory_list:
#     if k_a != 'mainhand':
#         k_in_a = 'observation' + '$' + 'inventory$' + k_a
#         import_key[k_in_a] = []
#     else:
#         k_in_a = 'observation$equipped_items.mainhand.type'
#         import_key[k_in_a] = []
import_key = {}
import_key_s = {}
for k_a in inventory_list:
    import_key[k_a] = []
    import_key_s[k_a] = []
X = []
y = []
x_pov = []
all_obs = []
for i in range(len(trajectory_names)):
    trajectory_name = trajectory_names[i]
    print(trajectory_name, i)
    obs_vector = np.load(os.path.join(DATA_DIR, 'MineRLObtainDiamondVectorObf-v0', trajectory_name, 'rendered.npz'))
    obs_normal = np.load(os.path.join(DATA_DIR, 'MineRLObtainDiamond-v0', trajectory_name, 'rendered.npz'))

    delta_f = obs_vector['observation$vector']
    delta_a = np.roll(delta_f, -1, axis=0)
    error = np.sum((delta_f[:len(delta_f) - 1] - delta_a[ :len(delta_f) - 1]), axis=1)
    index = np.where(error != 0)[0]

    delta_f = delta_f[index]
    delta_a = delta_a[index]
    all_index = index

    stack_map = np.zeros((len(index), len(inventory_list)))
    j = 0
    for k_a in inventory_list:
        k_in_a = 'observation' + '$' + 'inventory$' + k_a
        import_key[k_a] = np.array(obs_normal[k_in_a][index])
        import_key_s[k_a] = np.roll(obs_normal[k_in_a][index], -1, axis=0)
        error = import_key_s[k_a] - import_key[k_a]
        stack_map[:, j] = error
        j += 1

    max = np.max(stack_map, axis=1)
    index = np.where(max > 0)[0]
    delta_f = delta_f[index]
    delta_a = delta_a[index]
    all_index = all_index[index]
    index = np.argmax(stack_map[index], axis=1)
    # index = inventory_list[index]
    # index.shape

    trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)

    counter = 0

    obs_m = None
    del_tal = []
    del_tal_2 = []
    del_tal_index = []

    for dataset_observation, dataset_action, r, _, _ in trajectory:
        if counter in all_index:
            del_tal.append(dataset_observation["pov"])
            del_tal_index.append(counter + 1)
        if counter in del_tal_index:
            del_tal_2.append(dataset_observation["pov"])
        counter += 1

    obs_m = np.concatenate((del_tal, del_tal_2), axis=-1)


    if len(y) == 0:
        y = np.array(index)
        X = np.concatenate((delta_f, delta_a), axis=1)
        all_obs = obs_m
    else:
        y = np.concatenate((y, index), axis=0)
        tmp = np.concatenate((delta_f, delta_a), axis=1)
        X = np.concatenate((X, tmp), axis=0)
        all_obs = np.concatenate((all_obs, obs_m), axis=0)



X.shape
all_obs_X = np.array(all_obs)
y_T = np.array(y)
all_obs.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10)
clf = lgb.LGBMClassifier(device='cpu', boosting_type='gbdt', num_leaves=80, max_depth=8, learning_rate=0.01,
                         n_estimators=60,
                         subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0,
                         min_child_weight=0.001,
                         min_child_samples=80, subsample=1.0, subsample_freq=0, colsample_bytree=1.0,
                         reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=12, silent='warn',
                         importance_type='split')
clf.fit(X_train, y_train)

tmp_time = time.time()
y_pred = clf.predict(X_test)
np.unique(y_pred)
print(k, accuracy_score(y_pred, y_test))
with open(f'./lgb/train.pkl', 'wb') as fout:
    pickle.dump(clf, fout)