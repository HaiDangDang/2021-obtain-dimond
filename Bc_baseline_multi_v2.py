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
from hummingbird.ml import convert
import matplotlib.pyplot as plt

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

        hsv[:,:,0] = 100
        hsv[:,:,1] = 2

        obs = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        obs = np.expand_dims(obs, axis=-1)

        tree_obs = np.expand_dims(tree_obs,axis=-1)
        obs = np.concatenate((obs,tree_obs), axis=-1)

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

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

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
    def __init__(self, env, camera_angle=10, always_attack=False):
        super().__init__(env)

        self.camera_angle = camera_angle
        self.always_attack = always_attack
        self._actions = [
            [('attack', 1)],
            [('forward', 1)],
            # [('back', 1)],
            # [('left', 1)],
            # [('right', 1)],
            # [('jump', 1)],
            # [('forward', 1), ('attack', 1)],
            # [('craft', 'planks')],
            [('forward', 1), ('jump', 1)],
            [('camera', [-self.camera_angle, 0])],
            [('camera', [self.camera_angle, 0])],
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

        #self.action_space = gym.spaces.Discrete(len(self.actions))
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(len(self.actions),), dtype=np.uint8)

    def action(self, action):
        action_new = self.env.action_space.noop()

        if action[0] == 1 and action[1] == 0:
            action_new['camera'][0] = -self.camera_angle

        if action[0] == 0 and action[1] == 1:
            action_new['camera'][0] = self.camera_angle

        if action[2] == 1 and action[3] == 0:
            action_new['camera'][1] = self.camera_angle

        if action[2] == 0 and action[3] == 1:
            action_new['camera'][1] = -self.camera_angle

        if action[4] == 1:
            action_new['forward'] = 1
        if action[5] == 1:
            action_new['jump'] = 1
        if action[6] == 1:
            action_new['attack'] = 1

        return action_new

def dataset_action_batch_to_actions(dataset_actions, camera_margin=5):
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
    # There are dummy dimensions of shape one
    # camera_actions = dataset_actions["camera"].squeeze()
    # attack_actions = dataset_actions["attack"].squeeze()
    # forward_actions = dataset_actions["forward"].squeeze()
    # jump_actions = dataset_actions["jump"].squeeze()
    # batch_size = len(camera_actions)
    # actions = np.zeros((batch_size,), dtype=np.int)
    #
    # for i in range(len(camera_actions)):
    #     # Moving camera is most important (horizontal first)
    #     if camera_actions[i][0] < -camera_margin:
    #         actions[i] = 3
    #     elif camera_actions[i][0] > camera_margin:
    #         actions[i] = 4
    #     elif camera_actions[i][1] > camera_margin:
    #         actions[i] = 5
    #     elif camera_actions[i][1] < -camera_margin:
    #         actions[i] = 6
    #     elif forward_actions[i] == 1:
    #         if jump_actions[i] == 1:
    #             actions[i] = 2
    #         else:
    #             actions[i] = 1
    #     elif attack_actions[i] == 1:
    #         actions[i] = 0
    #     else:
    #         # No reasonable mapping (would be no-op)
    #         actions[i] = -1
    camera_actions = dataset_actions["camera"].squeeze()
    attack_actions = dataset_actions["attack"].squeeze()
    forward_actions = dataset_actions["forward"].squeeze()
    jump_actions = dataset_actions["jump"].squeeze()
    batch_size = len(camera_actions)
    actions = np.zeros((batch_size, 7), dtype=np.int)

    for i in range(len(camera_actions)):
        # Moving camera is most important (horizontal first)
        if camera_actions[i][0] < -camera_margin:
            actions[i][0] = 1
        elif camera_actions[i][0] > camera_margin:
            actions[i][1] = 1
        if camera_actions[i][1] > camera_margin:
            actions[i][2] = 1
        elif camera_actions[i][1] < -camera_margin:
            actions[i][3] = 1

        if forward_actions[i] == 1:
            actions[i][4] = 1

        if jump_actions[i] == 1:
            actions[i][5] = 1
        if attack_actions[i] == 1:
            actions[i][6] = 1
    return actions

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

    return torch.stack([h, s * 255 , v], dim=-3)

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


def shift_image_1(X, dx, dy):
    X = torch.roll(X, dy, dims=-2)
    X = torch.roll(X, dx, dims=-1)
    if dy>0:
        X[:, dy, :] = 0
    elif dy<0:
        X[:, dy:, :] = 0
    if dx>0:
        X[ :, :, :dx] = 0
    elif dx<0:
        X[ :, :, dx:] = 0
    return X

def train():
    print("Prepare_Data")
    data = minerl.data.make("MineRLTreechop-v0",  data_dir='data', num_workers=4)
    tranf = torchvision.transforms.Grayscale(num_output_channels=1)

    # We know ActionShaping has seven discrete actions, so we create
    # a network to map images to seven values (logits), which represent
    # likelihoods of selecting those actions
    network = NatureCNN((3, 64, 64), 7).cuda()
    optimizer = th.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    loss_function = nn.BCELoss()

    iter_count = 0
    losses = []
    time_stone = time.time()
    for dataset_obs, dataset_actions, _, _, _ in \
            tqdm(data.batch_iter(num_epochs=EPOCHS, batch_size=32, seq_len=1)):
        # We only use pov observations (also remove dummy dimensions)
        iter_count += 1

        # if iter_count <= 6000:
        #     if (iter_count % 100) == 0:
        #         print('aa')
        #     continue
        actions = dataset_action_batch_to_actions(dataset_actions)

        obs = dataset_obs["pov"].squeeze().astype(np.float32)

        sum_x = np.sum(actions, axis=1)
        mask = sum_x != 0
        obs = obs[mask]
        actions = actions[mask]

        obs = obs.transpose(0, 3, 1, 2)
        obs = th.from_numpy(obs).float().cuda()/255
        #grey = tranf.forward(obs)
        #grey = torch.squeeze(grey, dim=1)/255
        # grey = torch.zeros((obs.shape[0], 4, 64, 64)).float().cuda()
        # grey[:,[0,1,2],:,:] = obs/255
        # hsv = rgb_to_hsv(obs)
        # # H_s = torch.zeros((32, 64, 64, len(neighbor_pixels[0])))
        # # Hsv_nei = np.zeros((32, 64, 64, 3, len(neighbor_pixels[0]) + 1), dtype=np.float64)
        # # Hsv_nei[:, :, :, :, 0] = hsv.cpu().numpy().transpose(0, 2, 3, 1)
        #
        # # for counter in range(len(neighbor_pixels[0])):
        # #     new_hsv = shift_image(hsv, neighbor_pixels[0][counter],
        # #                           neighbor_pixels[1][counter])
        # #     H_s[:, :, :, counter] = new_hsv[:, 0, :, :]
        # #     Hsv_nei[:,:, :, :, counter+1] = new_hsv.cpu().numpy().transpose(0, 2, 3, 1)
        # #
        # #
        # # Hsv_nei = Hsv_nei.reshape(64 * 64 * 32, Hsv_nei.shape[-1] * 3)
        # # Hsv_nei = np.concatenate((Hsv_nei, pixel_location[0]), axis=1)
        # # Hsv_nei = np.concatenate((Hsv_nei, pixel_location[1]), axis=1)
        #
        # #y_pred = pkl_bst.predict(Hsv_nei)
        # # y_pred = torch.argmax(th.from_numpy(y_pred).cuda(), dim=1)
        # # y_pred = y_pred.reshape(32, 64, 64)
        # # tree = torch.where(y_pred == 6)
        # # tree_obs = torch.zeros((32, 64, 64), dtype=torch.float64).cuda().float()
        # # tree_obs[tree[0], tree[1], tree[2]] = 1
        # # obs = torch.stack([grey, tree_obs,y_pred/9], dim=1)
        #
        # # y_pred = y_pred.reshape(32, 64, 64)
        # # tree = np.where(y_pred == 'tree_chunk')
        # # tree_obs = torch.zeros((32, 64, 64), dtype=torch.float64).cuda().float()
        # # tree_obs[tree[0], tree[1], tree[2]] = 1
        # # imsg= plt.imshow(tree_obs[2].cpu().numpy())
        # # plt.show()
        # # imsg= plt.imshow(grey[2].cpu().numpy())
        # # plt.show()
        # #
        # # imsg= plt.imshow(obs[2]/255)
        # # plt.show()
        #
        # h = torch.unsqueeze(hsv[:, 0, :, :], -1)
        # s = torch.unsqueeze(hsv[:, 1, :, :], -1)
        # v = torch.unsqueeze(hsv[:, 2, :, :], -1)
        #
        # h_error = torch.abs(h - h_range_f)
        # index = torch.where(h_error >= 180)
        # h_error[index[0], index[1], index[2], index[3]] = 360 - h_error[index[0], index[1], index[2], index[3]]
        # d_error = torch.sqrt(torch.pow(h_error, 2) + torch.pow((s / 4 - s_range_f / 4), 2)
        #                + torch.pow((v / 8 - v_range_f / 8), 2))
        #
        # # Hs_error = torch.abs(H_s - Hs_range)
        # # index = torch.where(Hs_error >= 180)
        # # Hs_error[index[0], index[1], index[2], index[3]] = 360 - \
        # #             Hs_error[index[0], index[1], index[2], index[3]]
        #
        # min_v, min_k1 = torch.min(d_error, axis=-1)
        # # min_v += 1
        # # min_v = torch.unsqueeze(min_v, -1)
        #
        # # sum_right = torch.sum((Hs_error <= 3), axis=-1)
        # # index = torch.where(d_error <= min_v)
        # # sum_right[index[0], index[1], index[2]] = -1
        # # min_k1 = torch.argmax(sum_right, axis=-1)
        #
        # tree_obs = kind_f[min_k1] / 255
        # #tree_obs =grey
        # # obs = torch.stack([grey, tree_obs], dim=1)
        #
        # grey[:,3,:,:] = tree_obs
        # obs = grey

        # Transpose observations to be channel-first (BCHW instead of BHWC)
        # Normalize observations
        #obs /= 255.0

        # Actions need bit more work


        m = nn.Sigmoid()

        # Obtain logits of each action
        logits = network(obs)

        # Minimize cross-entropy with target labels.
        # We could also compute the probability of demonstration actions and
        # maximize them.
        loss = loss_function(m(logits), th.from_numpy(actions).float().cuda())

        # Standard PyTorch update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(iter_count)
        losses.append(loss.item())
        if (iter_count % 100) == 0:
            mean_loss = sum(losses) / len(losses)
            tqdm.write("Iteration {}. Loss {:<10.3f}".format(iter_count, mean_loss))
            losses.clear()
            print(time.time() - time_stone)
            time_stone = time.time()


    th.save(network.state_dict(), TRAIN_MODEL_NAME)
    del data

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

def get_action_sequence():
    """
    Specify the action sequence for the agent to execute.
    """
    # make planks, sticks, crafting table and wooden pickaxe:
    action_sequence = []
    action_sequence += [''] * 100
    action_sequence += ['craft:planks'] * 4
    action_sequence += ['craft:stick'] * 2
    action_sequence += ['craft:crafting_table']
    action_sequence += ['camera:[10,0]'] * 18
    action_sequence += ['attack'] * 20
    action_sequence += [''] * 10
    action_sequence += ['jump']
    action_sequence += [''] * 5
    action_sequence += ['place:crafting_table']
    action_sequence += [''] * 10

    # bug: looking straight down at a crafting table doesn't let you craft. So we look up a bit before crafting.
    action_sequence += ['camera:[-1,0]']
    action_sequence += ['nearbyCraft:wooden_pickaxe']
    action_sequence += ['camera:[1,0]']
    action_sequence += [''] * 10
    action_sequence += ['equip:wooden_pickaxe']
    action_sequence += [''] * 10

    # dig down:
    action_sequence += ['attack'] * 600
    action_sequence += [''] * 10

    return action_sequence



# Parameters:
EPOCHS = 1  # How many times we train over the dataset.
LEARNING_RATE = 0.0001  # Learning rate for the neural network.

TRAIN_MODEL_NAME = 'another_potato_3.pth'  # name to use when saving the trained agent.
TEST_MODEL_NAME = 'another_potato_3.pth'  # name to use when loading the trained agent.

TEST_EPISODES = 5  # number of episodes to test the agent for.
MAX_TEST_EPISODE_LEN = 5000  # 18k is the default for MineRLObtainDiamond.
TREECHOP_STEPS = 2000  # number of steps to run BC lumberjack for in evaluations

with open('first_model.pkl', 'rb') as fin:
    pkl_bst = pickle.load(fin)

dict_data = {}
with open('data_1.pkl', 'rb') as f:
    dict_data = pickle.load(f)
h_range = []
s_range = []
v_range = []
Hs_range = []

names = []
kind = []
counter = 0
number_of_key = len(dict_data.keys()) - 3
number_ = np.linspace(20, 255, number_of_key - 3)
for k, v in dict_data.items():
    if k != 'animal' and k != 'crafting_table' and k != 'black':
        for i in range(len(v['H'])):
            h_range.append(v['H'][i])
            s_range.append(v['S'][i])
            v_range.append(v['V'][i])
            Hs_range.append(v['Hs'][i])

            names.append(v['name'][i])
            if k == "tree_chunk":
                kind.append(255)
            else:
                kind.append(0)
            # kind.append(number_[counter])
        counter += 1
    # if k == tre
    # counter += 1
h_range = np.expand_dims(h_range, axis=0)
h_range = np.repeat(h_range, 64, axis=0)
h_range = np.expand_dims(h_range, axis=0)
h_range = np.repeat(h_range, 64, axis=0)
h_range = np.array(h_range, dtype=np.float64)
h_range = torch.Tensor(h_range)
h_range_f = torch.unsqueeze(h_range, 0)
h_range_f = h_range_f.cuda()

s_range = np.expand_dims(s_range, axis=0)
s_range = np.repeat(s_range, 64, axis=0)
s_range = np.expand_dims(s_range, axis=0)
s_range = np.repeat(s_range, 64, axis=0)
s_range = np.array(s_range, dtype=np.float64)
s_range = torch.Tensor(s_range)
s_range_f =  torch.unsqueeze(s_range, 0)
s_range_f = s_range_f.cuda()

v_range = np.expand_dims(v_range, axis=0)
v_range = np.repeat(v_range, 64, axis=0)
v_range = np.expand_dims(v_range, axis=0)
v_range = np.repeat(v_range, 64, axis=0)
v_range = np.array(v_range, dtype=np.float64)
v_range = torch.Tensor(v_range)
v_range_f = torch.unsqueeze(v_range, 0)
v_range_f = v_range_f.cuda()
kind = np.array(kind)
kind_f = th.from_numpy(kind).float().cuda()

Hs_range = np.expand_dims(Hs_range, axis=0)
Hs_range = np.repeat(Hs_range, 64, axis=0)
Hs_range = np.expand_dims(Hs_range, axis=0)
Hs_range = np.repeat(Hs_range, 64, axis=0)
Hs_range = np.array(Hs_range, dtype=np.float64)
Hs_range = torch.Tensor(Hs_range)
# Hs_range = Hs_range.cuda()
Hs_range = torch.unsqueeze(Hs_range, 0)

neighbor_pixels = [[-1, 0, 1, 0, -2, 2, 2, -2], [0, 1, 0, -1, -2, -2, 2, 2]]
with open('pixel_location.pkl', 'rb') as f:
    pixel_location = pickle.load(f)

# pixel_location[0] = np.expand_dims(pixel_location[0], axis=0)
# pixel_location[1] = np.expand_dims(pixel_location[1], axis=0)

pixel_location[0] = np.repeat(pixel_location[0], 32, axis=0)
pixel_location[1] = np.repeat(pixel_location[1], 32, axis=0)

train()

env = gym.make('MineRLTreechop-v0')
# env1 = Recorder(env, './video', fps=60)  # saving environment before action shaping to use with scripted part
env = ActionShaping(env, always_attack=True)

network = NatureCNN((2, 64, 64), 7).cuda()
network.load_state_dict(th.load(TEST_MODEL_NAME))
#
# num_actions = env.action_space.n
# action_list = np.arange(num_actions)

# action_sequence = get_action_sequence()
cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow('image', 1000, 1000)
cv2.moveWindow('image', -30, 30)

cv2.namedWindow('mask', cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow('mask', 1000, 1000)
cv2.moveWindow('mask', 1100, 30)
rewards = []
tranf = torchvision.transforms.Grayscale(num_output_channels=1)
with open('pixel_location.pkl', 'rb') as f:
    pixel_location = pickle.load(f)
for episode in range(20):
    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0
    # BC part to get some logs:
    for i in tqdm(range(TREECHOP_STEPS)):
        # Process the action:
        #   - Add/remove batch dimensions
        #   - Transpose image (needs to be channels-last)
        #   - Normalize image
        obs = obs['pov']
        obs = obs.transpose(2, 0, 1).astype(np.float32)
        obs = th.from_numpy(obs).float().cuda()
        # grey = tranf.forward(obs)
        # grey = torch.squeeze(grey, dim=0)/255
        # grey = torch.zeros((4, 64, 64)).float().cuda()
        # grey[[0,1,2],:,:] = obs/255
        #
        # hsv = rgb_to_hsv(obs)
        #
        # h = torch.unsqueeze(hsv[ 0, :, :], -1)
        # s = torch.unsqueeze(hsv[ 1, :, :], -1)
        # v = torch.unsqueeze(hsv[ 2, :, :], -1)
        # h_error = torch.abs(h - h_range_f[0])
        # index = torch.where(h_error >= 180)
        # h_error[index[0], index[1], index[2]] = 360 - h_error[index[0], index[1], index[2]]
        # d_error = torch.sqrt(torch.pow(h_error, 2) + torch.pow((s / 4 - s_range_f[0] / 4), 2)
        #                + torch.pow((v / 8 - v_range_f[0] / 8), 2))
        # d_error = torch.argmin(d_error, axis=-1)
        # tree_obs = kind_f[d_error]/255
        #
        # Hsv_nei = np.zeros((64, 64, 3, len(neighbor_pixels[0]) + 1), dtype=np.float64)
        # Hsv_nei[:, :, :, 0] = hsv.cpu().numpy().transpose(1, 2, 0)
        #
        # for counter in range(len(neighbor_pixels[0])):
        #     new_hsv = shift_image_1(hsv, neighbor_pixels[0][counter],
        #                           neighbor_pixels[1][counter])
        #     Hsv_nei[:, :, :, counter+1] = new_hsv.cpu().numpy().transpose(1, 2, 0)
        #
        #
        # Hsv_nei = Hsv_nei.reshape(64 * 64, Hsv_nei.shape[-1] * 3)
        #
        # Hsv_nei = np.concatenate((Hsv_nei, pixel_location[0]), axis=1)
        # Hsv_nei = np.concatenate((Hsv_nei, pixel_location[1]), axis=1)
        #
        # y_pred = pkl_bst.predict(Hsv_nei)
        #
        # # y_pred = torch.argmax(th.from_numpy(y_pred).cuda(), dim=1)
        # # y_pred = y_pred.reshape(64, 64)
        # # tree = torch.where(y_pred == 6)
        # # tree_obs = torch.zeros((64, 64), dtype=torch.float64).cuda().float()
        # # tree_obs[tree[0], tree[1]] = 1
        # # obs = torch.stack([grey, tree_obs, y_pred / 9], dim=0)
        # # y_pred =y_pred / 9
        #
        # y_pred = y_pred.reshape(64, 64)
        # tree = np.where(y_pred == 'tree_chunk')
        # tree_obs = torch.zeros((64,64), dtype=torch.float64).cuda().float()
        # tree_obs[tree[0], tree[1]] = 1
        #
        # # obs = torch.stack([grey, tree_obs], dim=0)
        # # obs = torch.unsqueeze(obs, 0)
        # grey[3,:,:] = tree_obs
        #
        # obs = grey
        obs = torch.unsqueeze(obs, 0)/255

        obs.shape

        probabilities = th.sigmoid(network(obs))
        # Into numpy
        probabilities = probabilities.detach().cpu().numpy()[0]
        #np.sum(probabilities)
        # Sample action according to the probabilities
        #action = np.argmax(probabilities)
        action = np.zeros(7, dtype=np.uint8)
        for j in range(7):
            p = probabilities[j]
            if p >= 0.5:
                p_1 = p
                p_0 = 1 - p
            else:
                p_1 = p
                p_0 = 1 - p
            action[j] = np.random.choice(2, 1, p=[p_0,p_1])[0]

        obs, reward, done, info = env.step(action)
        cv2.imshow('image', cv2.cvtColor(obs['pov'], cv2.COLOR_BGR2RGB))
        cv2.resizeWindow('image', 950, 950)
        #cv2.imshow('mask', np.array(tree_obs.cpu().numpy(), dtype=(np.float64)))
        #cv2.imshow('mask', np.array(grey.cpu().numpy(), dtype=(np.float64)))
        #cv2.resizeWindow('mask', 950, 950)

        total_reward += reward
        steps += 1
        if done:
            break
        if cv2.waitKey(10) & 0xFF == ord('o'):
            break
    rewards.append(total_reward)

    # scripted part to use the logs:
    # if not done:
    #     for i, action in enumerate(tqdm(action_sequence[:MAX_TEST_EPISODE_LEN - TREECHOP_STEPS])):
    #         obs, reward, done, _ = env1.step(str_to_act(env1, action))
    #         total_reward += reward
    #         steps += 1
    #         if done:
    #             break

    # env1.release()
    # env1.play()
    print(f'Episode #{episode + 1} reward: {total_reward}\t\t episode length: {steps}\n')

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



