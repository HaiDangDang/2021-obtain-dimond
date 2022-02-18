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
m = nn.Softmax(dim=0)
input = torch.randn(2, 3)
output = m(input)
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

        if action[7] == 1:
            action_new['craft'] = 'crafting_table'
        if action[8] == 1:
            action_new['craft'] = 'planks'
        if action[9] == 1:
            action_new['craft'] = 'stick'
        if action[10] == 1:
            action_new['craft'] = 'torch'
        if action[11] == 1:
            action_new['nearbyCraft'] = 'furnace'
        if action[12] == 1:
            action_new['nearbyCraft'] = 'iron_pickaxe'
        if action[13] == 1:
            action_new['nearbyCraft'] = 'stone_pickaxe'
        if action[14] == 1:
            action_new['nearbyCraft'] = 'wooden_pickaxe'
        if action[15] == 1:
            action_new['nearbySmelt'] = 'coal'
        if action[16] == 1:
            action_new['nearbySmelt'] = 'iron_ingot'
        if action[17] == 1:
            action_new['place'] = 'crafting_table'
        return action_new

def dataset_action_batch_to_actions(dataset_actions, camera_margin=5):

    camera_actions = dataset_actions["camera"].squeeze()
    attack_actions = dataset_actions["attack"].squeeze()
    forward_actions = dataset_actions["forward"].squeeze()
    jump_actions = dataset_actions["jump"].squeeze()
    craft_actions = dataset_actions["craft"].squeeze()
    nearbyCraft_actions = dataset_actions["nearbyCraft"].squeeze()
    nearbySmelt_actions = dataset_actions["nearbySmelt"].squeeze()
    place_actions = dataset_actions["place"].squeeze()

    batch_size = len(camera_actions)
    actions = np.zeros((batch_size, 18), dtype=np.int)

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

        if craft_actions[i] == 'crafting_table':
            actions[i][7] = 1
        elif craft_actions[i] == 'planks':
            actions[i][8] = 1
        elif craft_actions[i] == 'stick':
            actions[i][9] = 1
        elif craft_actions[i] == 'torch':
            actions[i][10] = 1

        if nearbyCraft_actions[i] == 'furnace':
            actions[i][11] = 1
        elif nearbyCraft_actions[i] == 'iron_pickaxe':
            actions[i][12] = 1
        elif nearbyCraft_actions[i] == 'stone_pickaxe':
            actions[i][13] = 1
        elif nearbyCraft_actions[i] == 'wooden_pickaxe':
            actions[i][14] = 1

        if nearbySmelt_actions[i] == 'coal':
            actions[i][15] = 1
        elif nearbySmelt_actions[i] == 'iron_ingot':
            actions[i][16] = 1

        if place_actions[i] == 'crafting_table':
            actions[i][17] = 1
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
    data = minerl.data.make("MineRLObtainDiamond-v0",  data_dir='data', num_workers=4)
    tranf = torchvision.transforms.Grayscale(num_output_channels=1)

    # We know ActionShaping has seven discrete actions, so we create
    # a network to map images to seven values (logits), which represent
    # likelihoods of selecting those actions
    network = NatureCNN((3, 64, 64), 18).cuda()
    optimizer = th.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    loss_function = nn.BCELoss()

    iter_count = 0
    losses = []
    time_stone = time.time()
    camera = None
    for dataset_obs, dataset_actions, _, _, _ in \
            tqdm(data.batch_iter(num_epochs=EPOCHS, batch_size=64, seq_len=1)):
        # We only use pov observations (also remove dummy dimensions)
        iter_count += 1

        # if iter_count <= 6000:
        #     if (iter_count % 100) == 0:
        #         print('aa')
        #     continue
        obs = dataset_obs["pov"].squeeze().astype(np.float32)

        obs = obs.transpose(0, 3, 1, 2)

        obs = th.from_numpy(obs).float().cuda()

        # Actions need bit more work
        actions = dataset_action_batch_to_actions(dataset_actions)
        if camera is None:
            camera = dataset_actions['camera']
        else:
            camera = np.concatenate((camera, dataset_actions['camera']), axis=0)
        m = nn.Sigmoid()

        # Obtain logits of each action
        logits = network(obs)

        # We could also compute the probability of demonstration actions and
        # maximize them.
        loss = loss_function(m(logits), th.from_numpy(actions).float().cuda())

        # Standard PyTorch update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(iter_count)
        losses.append(loss.item())
        if (iter_count % 1000) == 0:
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

neighbor_pixels = [[-1, 0, 1, 0, -2, 2, 2, -2], [0, 1, 0, -1, -2, -2, 2, 2]]
with open('pixel_location.pkl', 'rb') as f:
    pixel_location = pickle.load(f)

# pixel_location[0] = np.expand_dims(pixel_location[0], axis=0)
# pixel_location[1] = np.expand_dims(pixel_location[1], axis=0)

pixel_location[0] = np.repeat(pixel_location[0], 32, axis=0)
pixel_location[1] = np.repeat(pixel_location[1], 32, axis=0)

train()

env = gym.make('MineRLObtainDiamond-v0')
# env1 = Recorder(env, './video', fps=60)  # saving environment before action shaping to use with scripted part
env = ActionShaping(env, always_attack=True)
# env.seed(21)
# network = NatureCNN((3, 64, 64), 18).cuda()
# network.load_state_dict(th.load(TEST_MODEL_NAME))
# #
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

        obs = torch.unsqueeze(obs, 0)

        probabilities = th.sigmoid(network(obs))
        # Into numpy
        probabilities = probabilities.detach().cpu().numpy()[0]
        #np.sum(probabilities)
        # Sample action according to the probabilities
        #action = np.argmax(probabilities)
        action = np.zeros(18, dtype=np.uint8)
        for j in range(18):
            p = probabilities[j]
            if p >= 0.5:
                p_1 = p
                p_0 = 1 - p
            else:
                p_1 = p
                p_0 = 1 - p
            action[j] = np.random.choice(2, 1, p=[p_0,p_1])[0]
        #action = np.round(probabilities[0])


        obs, reward, done, info = env.step(action)
        cv2.imshow('image', cv2.cvtColor(obs['pov'], cv2.COLOR_BGR2RGB))
        cv2.resizeWindow('image', 950, 950)
        # cv2.imshow('mask', np.array(tree_obs.cpu().numpy(), dtype=(np.float64)))
        #cv2.imshow('mask', np.array(grey.cpu().numpy(), dtype=(np.float64)))
        # cv2.resizeWindow('mask', 950, 950)

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



