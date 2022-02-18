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
from sklearn.preprocessing import LabelEncoder
import io
from tempfile import TemporaryFile
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
neighbor_pixels = [[-1, 0, 1, 0, -2, 2, 2, -2], [0, 1, 0, -1, -2, -2, 2, 2]]
le = LabelEncoder()

neighbor_pixels = np.array(neighbor_pixels)
with open('pixel_location.pkl', 'rb') as f:
    pixel_location = pickle.load(f)

with open('first_model.pkl', 'rb') as fin:
    clf = pickle.load(fin)

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


class NatureCNN(nn.Module):
    """
    CNN from DQN nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
    Nicked from stable-baselines3:
        https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py
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

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


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
        if action_with_key['action$jump'][index] == 1:
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
def shift_image(X, dx, dy):
    X = torch.roll(X, dy, dims=-2)
    X = torch.roll(X, dx, dims=-1)
    if dy>0:
        X[:, :, dy, :] = 0
    elif dy<0:
        X[:, :, dy:, :] = 0
    if dx>0:
        X[:, :, :, :dx] = 0
    elif dx<0:
        X[:, :, :, dx:] = 0
    return X

def convert(obs):
    shape_1 = obs.shape[0]
    hsv = rgb_to_hsv(obs)
    H_s = torch.zeros((shape_1, 64, 64, len(neighbor_pixels[0])))
    Hsv_nei = np.zeros((shape_1, 64, 64, 3, len(neighbor_pixels[0]) + 1), dtype=np.float64)

    Hsv_nei[:, :, :, :, 0] = hsv.cpu().numpy().transpose(0, 2, 3, 1)

    for counter in range(len(neighbor_pixels[0])):
        new_hsv = shift_image(hsv, neighbor_pixels[0][counter],
                              neighbor_pixels[1][counter])
        H_s[:, :, :, counter] = new_hsv[:, 0, :, :]
        Hsv_nei[:, :, :, :, counter + 1] = new_hsv.cpu().numpy().transpose(0, 2, 3, 1)

    Hsv_nei = Hsv_nei.reshape(64 * 64 * shape_1, Hsv_nei.shape[-1] * 3)
    Hsv_nei = np.concatenate((Hsv_nei,  np.repeat(pixel_location[0], shape_1, axis=0)), axis=1)
    Hsv_nei = np.concatenate((Hsv_nei, np.repeat(pixel_location[1], shape_1, axis=0)), axis=1)

    y_pred = clf.predict(Hsv_nei)
    y_pred = np.argmax(y_pred, axis=1)
    # y_pred = le.fit_transform(y_pred)

    y_pred = y_pred.reshape(shape_1, 64, 64)
    return y_pred


# For demonstration purposes, we will only use ObtainPickaxe data which is smaller,
# but has the similar steps as ObtainDiamond in the beginning.
# "VectorObf" stands for vectorized (vector observation and action), where there is no
# clear mapping between original actions and the vectors (i.e. you need to learn it)
data = minerl.data.make("MineRLObtainDiamondVectorObf-v0", data_dir=DATA_DIR, num_workers=10)



trajectory_names = data.get_trajectory_names()
trajectory_names.sort()

d = 0
max_stack = 16
for i in range(len(trajectory_names)):
    if i >= d and i < d + 20:
        trajectory_name = trajectory_names[0]
        trajectory = data.load_data(trajectory_name, skip_interval=0, include_metadata=False)
        total_reward = 0
        index = 0
        # if trajectory_name not in fail_data:
        #     name.append(trajectory_name)
        counter = 0
        all_pov_obs = None
        max_obs = None
        all_obs = []
        for dataset_observation, dataset_action, r, _, _ in trajectory:

            obs = dataset_observation["pov"]
            all_obs.append(obs)
            obs = obs.transpose(2, 0, 1).astype(np.float32)
            obs = torch.from_numpy(obs)

            if max_obs is None:
                max_obs = torch.unsqueeze(obs, dim=0)
            else:
                max_obs = torch.vstack((max_obs, torch.unsqueeze(obs, dim=0)))
                if max_obs.shape[0] == 32:
                    if all_pov_obs is None:
                        all_pov_obs = convert(max_obs)
                    else:
                        all_pov_obs = np.concatenate((all_pov_obs, convert(max_obs)), axis=0)
                    max_obs = None

        if max_obs is not None:
            if all_pov_obs is None:
                all_pov_obs = convert(max_obs)
            else:
                all_pov_obs = np.concatenate((all_pov_obs, convert(max_obs)), axis=0)
        np.save(os.path.join(DATA_DIR, 'MineRLObtainDiamondVectorObf-v0', trajectory_name, 'predicted.npy'),
                np.array(all_pov_obs).astype(np.uint8))

