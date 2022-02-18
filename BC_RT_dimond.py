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
import os
import time
# import tqdm
#
# def nop(it, *a, **k):
#     return it
#
# tqdm.tqdm = nop

# Parameters:
fail_data = ['v3_absolute_grape_changeling-17_418-2303',
             'v3_equal_olive_chimera-9_25021-26077',
             'v3_agonizing_kale_tree_nymph-20_58203-59745',
             'v3_kindly_lemon_mummy-11_2752-6180',
             'v3_juvenile_apple_angel-26_862-2906']
DATA_DIR = "data"  # path to MineRL dataset (should contain "MineRLObtainIronPickaxeVectorObf-v0" directory).
EPOCHS = 2  # how many times we train over dataset.
LEARNING_RATE = 0.0001  # learning rate for the neural network.
BATCH_SIZE = 32
NUM_ACTION_CENTROIDS = 100  # number of KMeans centroids used to cluster the data.

# Adjust DATA_SAMPLES to fit your RAM, extra 100k samples is about 1.2 GB RAM.
# Example RAM usage and training time for training with different DATA_SAMPLES on a mid-range PC:
# (using the default parameters)
# |----------------------------------------------|
# | DATA_SAMPLES | RAM Usage, MB | Time, minutes |
# |------------------------------|---------------|
# |      100,000 |         3,854 |           1.9 |
# |      200,000 |         5,135 |           3.6 |
# |      500,000 |         8,741 |           8.1 |
# |    1,000,000 |        14,833 |          17.0 |
# |    1,528,808 |        21,411 |          28.1 | <- full MineRLObtainIronPickaxeVectorObf-v0 dataset
# |----------------------------------------------|
DATA_SAMPLES = 1000000

TRAIN_MODEL_NAME = 'research_potato.pth'  # name to use when saving the trained agent.
TEST_MODEL_NAME = 'research_potato.pth'  # name to use when loading the trained agent.
TRAIN_KMEANS_MODEL_NAME = 'centroids_for_research_potato.npy'  # name to use when saving the KMeans model.
TEST_KMEANS_MODEL_NAME = 'centroids_for_research_potato.npy'  # name to use when loading the KMeans model.

TEST_EPISODES = 10  # number of episodes to test the agent for.
MAX_TEST_EPISODE_LEN = 18000  # 18k is the default for MineRLObtainDiamondVectorObf.

with open('knn_action.pkl', 'rb') as f:
    knn_action = pickle.load(f)
knn_action = np.array(knn_action)
with open('final_key_with_index.pkl', 'rb') as f:
    final_key_with_index = pickle.load(f)
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

from threading import Thread

none_move_action = np.array(['attack', 'craft_crafting_table',
                             'craft_planks', 'craft_stick', 'equip_iron_pickaxe',
                             'equip_stone_pickaxe', 'equip_wooden_pickaxe',
                             'nearbyCraft_furnace', 'nearbyCraft_iron_pickaxe',
                             'nearbyCraft_stone_pickaxe', 'nearbyCraft_wooden_pickaxe',
                             'nearbySmelt_iron_ingot', 'place_crafting_table', 'place_furnace'])

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

def process_inventory(obs, attack, t, full_previous, angle_1, angle_2, inven_process):

    data = np.zeros(22)

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
        items['planks'] -= 1
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


MOVING_ACTIONS = ['action$forward', 'action$left', 'action$back', 'action$right', 'action$jump', 'action$sneak',
                  'action$sprint', 'action$attack']
OTHER_ACTIONS = ['action$camera', 'action$place', 'action$equip', 'action$craft', 'action$nearbyCraft',
                 'action$nearbySmelt']


def find_key(byWhat, action_with_key, key, index):
    first_angle = action_with_key['action$camera'][index][0]
    second_angle = action_with_key['action$camera'][index][1]
    if byWhat == 'camera':
        if np.abs(first_angle) > 3 and np.abs(first_angle) < 15:
            key += '_1'
        else:
            key += '_0'
        if np.abs(second_angle) > 3 and np.abs(second_angle) < 15:
            key += '_1'
        else:
            key += '_0'
        return key
    elif byWhat == 'jump':
        if action_with_key['action$jump'][index] == 1:
            key += '_j'
        return find_key('move', action_with_key, key, index)
    elif byWhat == 'move':
        if action_with_key['action$forward'][index] == 1:
            key += '_f'
        elif action_with_key['action$left'][index] == 1 or action_with_key['action$back'][index] == 1 \
                or action_with_key['action$right'][index] == 1:
            key += '_a'
        return find_key('camera', action_with_key, key, index)
    return key


def process_ation(action_d, ation_vec, index):
    # action_d = f_obs
    # ation_vec = action_vec
    first_angle = action_d['action$camera'][index][0]
    second_angle = action_d['action$camera'][index][1]
    if action_d['action$craft'][index] != 'none':
        craft = action_d['action$craft'][index]
        key = f'craft_{craft}'
    elif action_d['action$equip'][index] != 'none':
        equip = action_d['action$equip'][index]
        key = f'equip_{equip}'
    elif action_d['action$nearbyCraft'][index] != 'none':
        nearbyCraft = action_d['action$nearbyCraft'][index]
        key = f'nearbyCraft_{nearbyCraft}'
    elif action_d['action$nearbySmelt'][index] != 'none':
        nearbySmelt = action_d['action$nearbySmelt'][index]
        key = f'nearbySmelt_{nearbySmelt}'
    elif action_d['action$place'][index] != 'none':
        place = action_d['action$place'][index]
        key = f'place_{place}'
    elif action_d['action$attack'][index] == 1:
        if action_d['action$forward'][index] == 0 and action_d['action$left'][index] == 0 and action_d['action$back'][index] == 0 and \
                action_d['action$right'][index] == 0 and action_d['action$jump'][index] == 0 and action_d['action$sneak'][index] == 0 and \
                action_d['action$sprint'][index] == 0:
            if np.abs(first_angle) == 0 and np.abs((second_angle)) == 0:
                key = 'attack_0_0'
            else:
                key = find_key('camera', action_d, 'attack', index)
        else:
            key = find_key('jump', action_d, 'attack',  index)

    else:
        key = find_key('jump', action_d, '', index)

    # actions = final_key_with_index[key][0]
    # if len(actions) == 1:
    #     return actions[0]
    # else:
    #     # x = (ation_vec - knn_action[actions][:, None]) ** 2
    #     distances = np.sum((ation_vec - knn_action[actions][:, None]) ** 2, axis=2)
    #     a = np.argmin(distances)
    #     return actions[a]
    return key

def action_with_key(ation_vec, key):

    actions = final_key_with_index[key]
    if len(actions) == 1:
        return actions[0]
    else:
        # x = (ation_vec - knn_action[actions][:, None]) ** 2
        distances = np.sum((ation_vec - knn_action[actions][:, None]) ** 2, axis=2)
        a = np.argmin(distances)
        return actions[a]

# output_file = f'MineRLObtainDiamond-v0.pkl'
# with open(output_file, 'rb') as file:
#     data_map = pickle.load(file)
# data_map



def train():
    data = minerl.data.make("MineRLObtainDiamondVectorObf-v0", data_dir=DATA_DIR, num_workers=10)
    # i = 95

    # all_pov_obs = []
    # all_actions = []
    # print("Loading data")
    # trajectory_names = data.get_trajectory_names()
    # trajectory_names.sort()
    # d = 100
    # for i in range(len(trajectory_names)):
    #     if i not in [96,21] and d <= i < d + 20:
    #         trajectory_name = trajectory_names[i]
    #         print(trajectory_name, i)
    #
    #         names = np.load(os.path.join(DATA_DIR, 'MineRLObtainDiamondVectorObf-v0', trajectory_name, 'name.npy'))
    #
    #         dict_name = []
    #         trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)
    #         f_obs = np.load(os.path.join(DATA_DIR, 'MineRLObtainDiamond-v0', trajectory_name, 'rendered.npz'))
    #         total_reward = 0
    #         index = 0
    #         # if trajectory_name not in fail_data:
    #         #     name.append(trajectory_name)
    #         for dataset_observation, dataset_action, r, _, _ in trajectory:
    #             action_vec = dataset_action["vector"]
    #             # all_actions.append(process_ation(f_obs, action_vec, index))
    #             # all_pov_obs.append(dataset_observation["pov"])
    #             if f_obs['action$forward'][index] == 1 and f_obs['action$jump'][index] == 0:
    #                 dict_name.append(process_ation(f_obs, action_vec, index))
    #             else:
    #                 dict_name.append(names[index])
    #             index += 1
    #         np.save(os.path.join(DATA_DIR, 'MineRLObtainDiamondVectorObf-v0', trajectory_name, 'name.npy'), np.array(dict_name))

    all_pov_obs = []
    all_actions = []
    all_rs = []
    print("Loading data")
    trajectory_names = data.get_trajectory_names()
    random.seed(10)
    random.shuffle(trajectory_names)
    # Add trajectories to the data until we reach the required DATA_SAMPLES.

    for trajectory_name in trajectory_names:
        # if trajectory_name not in fail_data:

        trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)
        names = np.load(os.path.join(DATA_DIR, 'MineRLObtainDiamondVectorObf-v0', trajectory_name, 'name.npy'))
        total_reward = 0
        index = 0
        # if trajectory_name not in fail_data:
        #     name.append(trajectory_name)
        counter = 0
        for dataset_observation, dataset_action, r, _, _ in trajectory:
            action_vec = dataset_action["vector"]
            a = action_with_key(action_vec, names[index])
            # if a not in [22, 23, 24, 25, 26, 27, 28, 29, 30, 31]:
            # obs = np.concatenate((dataset_observation["pov"], invetory * (counter * 255/13)), axis=2)/255
            if names[index] != '_0_0':
                all_actions.append(a)
                all_pov_obs.append(dataset_observation["pov"])
                all_rs.append(counter)
            index += 1
            if r != 0:
                counter += 1

        if len(all_actions) >= DATA_SAMPLES:
            break

    all_actions = np.array(all_actions)
    all_pov_obs = np.array(all_pov_obs)
    all_rs = np.array(all_rs)
    # len(all_actions)
    # for i in range(64):
    #     print(np.max(all_actions[:, i]), np.min(all_actions[:, i]))
    # all_actions.shape
    # print("Running KMeans on the action vectors")
    # kmeans = KMeans(n_clusters=NUM_ACTION_CENTROIDS)
    # kmeans.fit(all_actions)
    # action_centroids = kmeans.cluster_centers_
    #
    # print("KMeans done")

    # Now onto behavioural cloning itself.
    # Much like with intro track, we do behavioural cloning on the discrete actions,
    # where we turn the original vectors into discrete choices by mapping them to the closest
    # centroid (based on Euclidian distance).


    network = NatureCNN((4, 64, 64), len(knn_action)).cuda()
    optimizer = th.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    num_samples = all_actions.shape[0]
    update_count = 0
    losses = []
    # We have the data loaded up already in all_actions and all_pov_obs arrays.
    # Let's do a manual training loop
    print("Training")
    for _ in range(EPOCHS):
        # Randomize the order in which we go over the samples
        epoch_indices = np.arange(num_samples)
        np.random.shuffle(epoch_indices)

        for batch_i in range(0, num_samples, BATCH_SIZE):
            # NOTE: this will cut off incomplete batches from end of the random indices
            batch_indices = epoch_indices[batch_i:batch_i + BATCH_SIZE]

            # Load the inputs and preprocess
            obs = all_pov_obs[batch_indices].astype(np.float32)
            # Transpose observations to be channel-first (BCHW instead of BHWC)
            obs = obs.transpose(0, 3, 1, 2)
            r_dim = np.expand_dims(all_rs[batch_indices],axis=1)
            invetory = np.ones((obs.shape[0], 1,64, 64), dtype=np.uint8) * r_dim[:,None,None] * 255/13
            obs = np.concatenate((obs, invetory), axis=1)
            obs = th.from_numpy(obs).float().cuda()

            # Normalize observations. Do this here to avoid using too much memory (images are uint8 by default)
            obs /= 255.0

            # Map actions to their closest centroids
            # actions = all_actions[batch_indices]

            actions = all_actions[batch_indices]
            # distances = np.sum((action_vectors - action_centroids[:, None]) ** 2, axis=2)
            # actions = np.argmin(distances, axis=0)

            # Obtain logits of each action
            logits = network(obs)

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
    print("Training done")

    # Save network and the centroids into separate files
    # np.save(TRAIN_KMEANS_MODEL_NAME, action_centroids)
    th.save(network.state_dict(), TRAIN_MODEL_NAME)
    del data


final_key_with_index.keys()
x = final_key_with_index['attack_0_0']
def test():
    print("Running episodes")
    action_centroids = np.load(TEST_KMEANS_MODEL_NAME)
    network = NatureCNN((4, 64, 64), len(knn_action)).cuda()
    network.load_state_dict(th.load(TEST_MODEL_NAME))

    env = gym.make('MineRLObtainDiamondVectorObf-v0')
    cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('image', 1000, 1000)
    cv2.moveWindow('image', -30, 30)

    # num_actions = action_centroids.shape[0]
    action_list = np.arange(len(knn_action))

    for episode in range(TEST_EPISODES):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        counter = 0
        while not done:
            # Process the action:
            #   - Add/remove batch dimensions
            #   - Transpose image (needs to be channels-last)
            #   - Normalize image
            cv2.imshow('image', cv2.cvtColor(obs['pov'], cv2.COLOR_BGR2RGB))
            cv2.resizeWindow('image', 950, 950)
            obs = obs['pov'].transpose(2, 0, 1)
            invetory = np.ones((1, 64, 64), dtype=np.uint8) * counter * 255/13

            obs = np.concatenate((obs, invetory), axis=0)
            obs = th.from_numpy(obs[None].astype(np.float32)).cuda()
            # Turn logits into probabilities
            #obs = rgb_to_hsv(obs)


            # Normalize observations. Do this here to avoid using too much memory (images are uint8 by default)
            obs /= 255.0
            probabilities = th.softmax(network(obs), dim=1)[0]
            # Into numpy
            probabilities = probabilities.detach().cpu().numpy()
            # Sample action according to the probabilities
            discrete_action = np.random.choice(action_list, p=probabilities)

            # Map the discrete action to the corresponding action centroid (vector)
            action = knn_action[discrete_action]
            minerl_action = {"vector": action}

            obs, reward, done, info = env.step(minerl_action)
            if reward != 0:
                counter += 1
            total_reward += reward
            steps += 1
            if steps >= MAX_TEST_EPISODE_LEN:
                break

            if done:
                break
            if cv2.waitKey(10) & 0xFF == ord('o'):
                break
        print(f'Episode reward: {total_reward}, episode length: {steps}')

    cv2.destroyAllWindows()


def shift_image(X, dx, dy):
    X = np.array(X, dtype=np.float64)
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy>0:
        X[:dy, :] = 0
    elif dy<0:
        X[dy:, :] = 0
    if dx>0:
        X[:, :dx] = 0
    elif dx<0:
        X[:, dx:] = 0
    return X

neighbor_pixels = [[-1, 0, 1, 0, -2, 2, 2, -2], [0, 1, 0, -1, -2, -2, 2, 2]]

neighbor_pixels = np.array(neighbor_pixels)
with open('pixel_location.pkl', 'rb') as f:
    pixel_location = pickle.load(f)
with open('first_model.pkl', 'rb') as fin:
    clf = pickle.load(fin)
cv2.namedWindow('mask', cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow('mask', 1000, 1000)
cv2.moveWindow('mask', 1200, 30)
np.random.random_integers(0,179,11)
def display():
    class_color = np.load('name_classes_.npy')
    color = {
        'coal': [92,40,223],
        'crafting_table' : [69,119,130],
        'dirt' : [169,47,142],
        'grass' : [105,161,106],
        'not_common': [162,80,116],
        'rock' : [115,160,142],
        'sand': [15,124,116],
        'sky': [111,215,101],
        'tree_chunk': [49,185,153],
        'tree_leave': [131,215,161],
        'water': [74,171,209]}

    data = minerl.data.make("MineRLObtainDiamondVectorObf-v0", data_dir=DATA_DIR, num_workers=10)


    trajectory_names = data.get_trajectory_names()
    trajectory_names.sort()

    for i in range(len(trajectory_names)):
        if i not in [88,96,21,106,51,119,57] and d <= i < d + 133:
            trajectory_name = trajectory_names[21]
            names = np.load(os.path.join(DATA_DIR, 'MineRLObtainDiamondVectorObf-v0', trajectory_name, 'name.npy'))

            dict_name = []
            trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)
            f_obs = np.load(os.path.join(DATA_DIR, 'MineRLObtainDiamond-v0', trajectory_name, 'rendered.npz'))
            total_reward = 0
            index = 0
            # if trajectory_name not in fail_data:
            #     name.append(trajectory_name)
            for dataset_observation, dataset_action, r, _, _ in trajectory:
                index += 1

                if index <= 0:
                    continue
                obs = dataset_observation["pov"]
                hsv_64 = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
                hsv_64 = np.array(hsv_64, dtype=np.float64)
                Hsv_nei = np.zeros((64, 64, 3, len(neighbor_pixels[0]) + 1), dtype=np.float64)
                Hsv_nei[:, :, :, 0] = hsv_64

                for counter in range(len(neighbor_pixels[0])):
                    new_hsv = shift_image(hsv_64, neighbor_pixels[0][counter],
                                          neighbor_pixels[1][counter])
                    Hsv_nei[:, :, :, counter + 1] = new_hsv
                Hsv_nei = Hsv_nei.reshape(64 * 64, Hsv_nei.shape[-1] * 3)
                Hsv_nei = np.concatenate((Hsv_nei, pixel_location[0]), axis=1)
                Hsv_nei = np.concatenate((Hsv_nei, pixel_location[1]), axis=1)
                y_pred = clf.predict(Hsv_nei)
                k1 = y_pred.reshape(64, 64)
                tmp_mask = np.zeros((64,64,3), dtype=np.uint8)
                for k,v in color.items():
                    indexS = np.where(k1 == k)
                    tmp_mask[indexS[0], indexS[1]] = v

                action_vec = dataset_action["vector"]
                # all_actions.append(process_ation(f_obs, action_vec, index))
                print(names[index])
                cv2.imshow('image', cv2.cvtColor(obs, cv2.COLOR_BGR2RGB))
                cv2.resizeWindow('image', 950, 950)
                cv2.imshow('mask', cv2.cvtColor(tmp_mask, cv2.COLOR_HSV2RGB))
                cv2.resizeWindow('image', 950, 950)
                if cv2.waitKey(10) & 0xFF == ord('o'):
                    break

