import time
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common import env_checker
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
import minerl  # it's important to import minerl after SB3, otherwise model.save doesn't work...
import cv2
import numpy as np
import pickle
import torch

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import copy
from collections import OrderedDict
import wandb
# from ray.rllib.agents import ppo

#
# try:
#     wandb = None
#     import wandb
# except ImportError:
#     pass
from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, save_to_zip_file
from imitation.data.types import TrajectoryWithRew
import collections
from imitation.algorithms.adversarial import GAIL
from gym import spaces
from torch.nn import functional as F



with open('action_connect_to_vector.pkl', 'rb') as f:
    action_vector_all = pickle.load(f)

with open('action_key_2.pkl', 'rb') as f:
    action_key = pickle.load(f)

list_key_index = np.array(list(action_key.keys()))
print(len(list_key_index))


def process_inventory(obs, attack, angle_1, angle_2):

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


class MyCustomObs(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        # 64 x 64 + 64 x64 + 18 [cpa;, cobblestone]

        # self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 2), dtype=np.uint8)
        self.observation_space = gym.spaces.Dict({
             'pov': gym.spaces.Box(low=-1, high=1, shape=(7, 64, 64)),
             'inventory': gym.spaces.Box(low=0, high=1, shape=(23,)),
        })

        self.action_noop = self.env.action_space.noop()

        self.action_space = gym.spaces.Discrete(len(list_key_index))
        self.list_obs = [[-10, 0], [10, 0], [0, -10], [0, 10]]


        self.reset_all_value()

    def observation(self, observation):
        pov = observation['pov']
        grey = np.dot(pov[..., :3], [0.2989, 0.5870, 0.1140])

        angle_1_fix = (np.round(int(self.angle_1)/10)*10)
        angle_2_fix = (np.round(int(self.angle_2)/10)*10)

        self.stack_grey[tuple((angle_1_fix, angle_2_fix))] = [grey, self.time_add]

        for location in self.list_obs:
            a1 = angle_1_fix + location[0]
            a2 = angle_2_fix + location[1]
            if a2 > 360:
                a2 = a2 - 360
            elif a2 < 0:
                a2 = 360 + a2

            new_tuple = tuple((a1, a2))
            if new_tuple in self.stack_grey:
                grey_1 = self.stack_grey[new_tuple]
                if grey_1[1] >= self.time_add - 1000:
                    pov = np.concatenate((pov, grey_1[0][:, :, None]), axis=-1)
                else:
                    pov = np.concatenate((pov, np.zeros((64, 64, 1))), axis=-1)
            else:
                pov = np.concatenate((pov, np.zeros((64, 64, 1))), axis=-1)

        inventory = process_inventory(observation, self.time_attack_no_new, self.angle_1, self.angle_2)

        pov = pov.transpose(2, 0, 1).astype(np.float32)
        # pov = th.from_numpy(pov).float()
        pov /= 255.0
        # inventory = th.from_numpy(inventory).float()
        obs = {'pov': pov, 'inventory': inventory}
        return obs

    def step(self, action):
        action_key = list_key_index[action]
        action = self.process_a(action)
        obs, reward, done, info = self.env.step(action)

        slipt = action_key.split('_')
        if 'attack' in slipt:
            if self.time_attack_no_new == 0:
                self.current_item = 0
                for items_k in ['planks', 'log', 'cobblestone', 'stone', 'iron_ore', 'coal', 'dirt',
                                'crafting_table', 'furnace']:
                    self.current_item += obs['inventory'][items_k]
                self.time_attack_no_new += 1

            else:
                check_new = 0
                for items_k in ['planks', 'log', 'cobblestone', 'stone', 'iron_ore', 'coal', 'dirt', 'crafting_table',
                                'furnace']:
                    check_new += obs['inventory'][items_k]
                if check_new != self.current_item:
                    self.time_attack_no_new = 0
                else:
                    self.time_attack_no_new += 1
        else:
            self.time_attack_no_new = 0

        if len(slipt) >= 2:
            if slipt[-1] == '-1':
                self.angle_2 -= 5
            elif slipt[-1] == '1':
                self.angle_2 += 5

            if self.angle_2 > 360:
                self.angle_2 = self.angle_2 - 360
            elif self.angle_2 < 0:
                self.angle_2 = 360 + self.angle_2

            if slipt[-2] == '-1':
                self.angle_1 -= 5
            elif slipt[-2] == '1':
                self.angle_1 += 5
            self.angle_1 = np.clip(self.angle_1, -90, 90)

        self.time_add += 1
        if reward != 0:
            reward = np.log2(reward)
        return self.observation(obs), reward, done, info

    def reset(self, **kwargs):
        self.reset_all_value()
        observation = self.env.reset(**kwargs)

        return self.observation(observation)

    def process_a(self, action):
        action_noop = copy.deepcopy(self.action_noop)
        key = list_key_index[action]
        for k, v in action_key[key].items():
            action_noop[k] = v
        return action_noop

    def reset_all_value(self):
        self.angle_1 = 0
        self.angle_2 = 0
        self.time_add = 0

        self.current_item = 0
        self.time_attack_no_new = 0
        self.stack_grey = {}


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super(CustomCNN, self).__init__(observation_space, features_dim=256)
        n_input_channels = observation_space['pov'].shape[0]
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
            n_flatten = self.cnn(th.zeros(1, *observation_space['pov'].shape)).shape[1]

        self.linear_stack = nn.Sequential(
            nn.Linear(23, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

        self.linear = nn.Sequential(
            nn.Linear(n_flatten + 512, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, observations) -> th.Tensor:
        x1 = self.cnn(observations['pov'])
        x2 = self.linear_stack(observations['inventory'])
        x = torch.cat((x1, x2), dim=1)
        return self.linear(x)


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


class CustomCNN_V2(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super(CustomCNN_V2, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = feature_dim
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return features, self.value_net(features)

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        # network = CustomCNN(input_shape=(7, 64, 64), output_dim=len(list_key_index))
        # network.load_state_dict(th.load('another_potato_1.pth'))
        self.mlp_extractor = CustomCNN_V2(self.features_dim)


# def track_exp(project_name=None):
#     config = {
#         "TRAIN_TIMESTEPS": 1000000,  # number of steps to train the agent for. At 70 FPS 2m steps take about 8 hours.
#         "TRAIN_ENV": 'MineRLTreechop-v0',
#         # training environment for the RL agent. Could use MineRLObtainDiamondDense-v0 here.
#         "TRAIN_MODEL_NAME": 'potato',  # name to use when saving the trained agent.
#         "TEST_MODEL_NAME": 'potato',  # name to use when loading the trained agent.
#         "TEST_EPISODES": 10,  # number of episodes to test the agent for.
#         "MAX_TEST_EPISODE_LEN": 18000,  # 18k is the default for MineRLObtainDiamond.
#         "TREECHOP_STEPS": 2000,  # number of steps to run RL lumberjack for in evaluations.
#         "RECORD_TRAINING_VIDEOS": False,  # if True, records videos of all episodes done during training.
#         "RECORD_TEST_VIDEOS": False,  # if True, records videos of all episodes done during evaluation.
#     }
#     wandb.init(
#         anonymous="allow",
#         project=project_name,
#         config=config,
#         sync_tensorboard=True,
#         name='v1',
#         monitor_gym=True,
#         save_code=True,
#     )

def make_env(idx):
    def thunk():
        env = gym.make('MineRLObtainDiamond-v0')
        env = MyCustomObs(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)  # record stats such as returns

        return env
    return thunk


# track_exp(project_name="minerl")

env = DummyVecEnv([make_env(i) for i in range(5)])
# env = gym.make('MineRLObtainDiamond-v0')
# env = MyCustomObs(env)
# env.observation_space['pov'].shape[0]
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(),
    net_arch=[dict(pi=[128,128], vf=[128, 128])],
    activation_fn=nn.ReLU,
    normalize_images=False,
)


def track_exp(project_name=None):
    config = {
        "TRAIN_TIMESTEPS": 2000000,  # number of steps to train the agent for. At 70 FPS 2m steps take about 8 hours.
        "TRAIN_ENV": 'MineRLTreechop-v0',
        # training environment for the RL agent. Could use MineRLObtainDiamondDense-v0 here.
        "TRAIN_MODEL_NAME": 'potato',  # name to use when saving the trained agent.
        "TEST_MODEL_NAME": 'potato',  # name to use when loading the trained agent.
        "TEST_EPISODES": 10,  # number of episodes to test the agent for.
        "MAX_TEST_EPISODE_LEN": 18000,  # 18k is the default for MineRLObtainDiamond.
        "TREECHOP_STEPS": 2000,  # number of steps to run RL lumberjack for in evaluations.
        "RECORD_TRAINING_VIDEOS": False,  # if True, records videos of all episodes done during training.
        "RECORD_TEST_VIDEOS": False,  # if True, records videos of all episodes done during evaluation.
    }
    wandb.init(
        anonymous="allow",
        project=project_name,
        config=config,
        sync_tensorboard=True,
        name='v1',
        monitor_gym=True,
        save_code=True,
    )



track_exp(project_name="minerl")

model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=f"runs/{'v1'}")
# model.policy.action_net = nn.Sequential()
model.vf_coef = 0.3
# model = PPO(CustomActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=f"runs/{'v1'}")
model.policy.state_dict()
model.save('ppo_first')
model.rollout_buffer.rewards.shape
stated_dict = th.load('another_potato_3.pth')
data, params, pytorch_variables = load_from_zip_file("ppo_first")
policy_l = list(params['policy'].keys())
values_k = ['cnn.0.weight',
 'cnn.0.bias',
 'cnn.2.weight',
 'cnn.2.bias',
 'cnn.4.weight',
 'cnn.4.bias',
 'linear_stack.0.weight',
 'linear_stack.0.bias',
 'linear_stack.2.weight',
 'linear_stack.2.bias',
 'linear.0.weight',
 'linear.0.bias',
 'linear.2.weight',
 'linear.2.bias',
'policy.0.weight',
'policy.0.bias',
'policy.2.weight',
'policy.2.bias',
 'value.0.weight',
 'value.0.bias',
 'value.2.weight',
 'value.2.bias',
 'action.0.weight',
 'action.0.bias',
'value_f.0.weight',
'value_f.0.bias']

for i in range(len(values_k)):
    params['policy'][policy_l[i]] = stated_dict[values_k[i]]

model.set_parameters(params)
# model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=f"runs/{'v1'}")

def track_exp(project_name=None):
    config = {
        "TRAIN_TIMESTEPS": 2000000,  # number of steps to train the agent for. At 70 FPS 2m steps take about 8 hours.
        "TRAIN_ENV": 'MineRLTreechop-v0',
        # training environment for the RL agent. Could use MineRLObtainDiamondDense-v0 here.
        "TRAIN_MODEL_NAME": 'potato',  # name to use when saving the trained agent.
        "TEST_MODEL_NAME": 'potato',  # name to use when loading the trained agent.
        "TEST_EPISODES": 10,  # number of episodes to test the agent for.
        "MAX_TEST_EPISODE_LEN": 18000,  # 18k is the default for MineRLObtainDiamond.
        "TREECHOP_STEPS": 2000,  # number of steps to run RL lumberjack for in evaluations.
        "RECORD_TRAINING_VIDEOS": False,  # if True, records videos of all episodes done during training.
        "RECORD_TEST_VIDEOS": False,  # if True, records videos of all episodes done during evaluation.
    }
    wandb.init(
        anonymous="allow",
        project=project_name,
        config=config,
        sync_tensorboard=True,
        name='v1',
        monitor_gym=True,
        save_code=True,
    )



track_exp(project_name="minerl")
model.save('ppo_second')

model.learn(total_timesteps=2000000)  # 2m steps is about 8h at 70 FPS


len(model.ep_info_buffer)
stated_dict = th.load('another_potato_1.pth')
for k,v in stated_dict.items():
    name = 'features_extractor.' + k
    stated_dict[k] = model.policy.state_dict()[name]

th.save(stated_dict, 'another_potato_1.pth')
# env_checker.check_env(env, warn=True, skip_render_check=True)
# MineRL might throw an exception when closing on Windows, but it can be ignored (the environment does close).
stated_dict[k]

model.load('ppo_first')

env_t = gym.make('MineRLObtainDiamond-v0')
env_t = MyCustomObs(env_t)

network = NatureCNN((7, 64, 64), len(list_key_index)).cuda()
network.load_state_dict(th.load('another_potato_4.pth'))
action_list = np.arange(len(list_key_index))

rewards = []
for episode in range(3):
    obs = env_t.reset()
    done = False
    total_reward = 0
    steps = 0
    cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('image', 1000, 1000)
    # RL part to get some logs:
    for i in range(10000):
        action = model.predict(obs.copy())[0]
        a = obs.copy()
        a = {key: th.as_tensor(_obs[None]).to(0) for (key, _obs) in a.items()}
        # a['inventory'] = th.from_numpy(obs['inventory'])
        # a['pov'] = th.from_numpy(obs['pov']) * 255
        #
        # action = model.predict(a)[0]
        #
        x = model.policy.extract_features(a)
        c, d = model.policy.mlp_extractor(x)
        # model.policy.action_net = nn.Sequential()
        model.policy.value_net(d)
        model.policy.action_net(c)

        # obs['pov'].shape
        x = model.rollout_buffer.get(4)
        # len(model.rollout_buffer.pos)
        # model.rollout_buffer.full = True
        model.policy.optimizer
        clip_range = model.clip_range(model._current_progress_remaining)
        for rollout_data in  model.rollout_buffer.get(4):
            actions = rollout_data.actions
            isinstance(model.action_space, spaces.Discrete)
            actions = rollout_data.actions.long().flatten()
            values, log_prob, entropy = model.policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()

            advantages = rollout_data.advantages
            rollout_data.old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            ratio = th.exp(log_prob - rollout_data.old_log_prob)

            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
            values_pred = values
            value_loss = F.mse_loss(rollout_data.returns, values_pred)
            loss = policy_loss + 0.5 * value_loss

            break
        inven = th.from_numpy(obs['inventory'][None]).float().cuda()
        hsv = th.from_numpy(obs['pov'][None]).float().cuda()
        probabilities = th.softmax(network(hsv, inven), dim=1)[0]
        probabilities = probabilities.detach().cpu().numpy()
        action = np.random.choice(action_list, p=probabilities)


        obs, reward, done, _ = env_t.step(action)
        total_reward += reward
        # env.render()
        steps += 1
        image = obs['pov'][:3]
        image = image.transpose(1, 2, 0)

        cv2.imshow('image', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        cv2.resizeWindow('image', 950, 950)
        if done:
            rewards.append(total_reward)
            break
        if cv2.waitKey(10) & 0xFF == ord('o'):
            break
        # time.sleep(0.1)
    # scripted part to use the logs:
    # if not done:
    #     for i, action in enumerate(action_sequence[:config["MAX_TEST_EPISODE_LEN"] - config["TREECHOP_STEPS"]]):
    #         obs, reward, done, _ = env1.step(str_to_act(env1, action))
    #         total_reward += reward
    #         steps += 1
    #         if done:
    #             break
    #
    # print(f'Episode #{episode + 1} return: {total_reward}\t\t episode length: {steps}')
    # writer.add_scalar("return", total_reward, global_step=episode)
    cv2.destroyAllWindows()

list_key_index[6]
# env.close()

img = plt.imshow(obs['pov'][3])
plt.show()






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
from torch.nn import functional as F

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

class NatureCNNV2(nn.Module):
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
            nn.Linear(512, 256)
        )

        self.value = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.policy = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.value_f = nn.Sequential(
            nn.Linear(128, 1),
        )

        self.action = nn.Sequential(
            nn.Linear(128, output_dim),
        )
    def forward(self, observations: th.Tensor, data: th.Tensor) -> th.Tensor:
        x1 = self.cnn(observations)
        x2 = self.linear_stack(data)
        x = torch.cat((x1, x2), dim=1)
        final = self.linear(x)
        v = self.value(final)
        v = self.value_f(v)

        a = self.policy(final)
        a = self.action(a)
        return a, v



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
        if np.abs(first_angle) > 5:
            if first_angle < 0:
                key += '_-1'
                camera.append(-8)
            else:
                key += '_1'
                camera.append(8)
        else:
            key += '_0'
            camera.append(0)

        if np.abs(second_angle) > 5:
            if second_angle < 0:
                key += '_-1'
                camera.append(-8)
            else:
                key += '_1'
                camera.append(8)
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

print("Prepare_Data")


data = minerl.data.make("MineRLObtainDiamond-v0",  data_dir='data', num_workers=4)

all_actions = []
all_pov_obs = []
all_data_obs = []
all_values = []
gamma = 0.99
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

    reward_eps = []
    store_all = {'obs': [], 'data': [], 'a': [], 'values': []}
    index = 0
    list_selected = []
    t_r = 0
    for obs, action, r, _, _ in trajectory:
        b_inventory = np.array(list(obs['inventory'].values()))
        b_inventory = b_inventory[item_by_attack_index]
        final = obs['pov']
        grey = np.dot(obs['pov'][..., :3], [0.2989, 0.5870, 0.1140])
        t_r += r
        if r != 0:
            r = np.log2(r)
        reward_eps.append(r)
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
                final = np.concatenate((final, grey_1[0][:, :, None]), axis=-1)
            else:
                final = np.concatenate((final, np.zeros((64, 64, 1))), axis=-1)
        final = final.astype(np.uint8)
        action_key[key] = dict_action
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
                # stack_index.append([final, a , after
                stack_index.append(index)
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
                            stack_index_max.append(stack_index)
                            stack_index = []
                            if len(stack_index_max) > 2:
                                del stack_index_max[0]
                            # trash_counter += 1
                if add:
                    for v1 in stack_index_max:
                        for v2 in v1:
                            list_selected.append(v2)
                            # all_pov_obs.append(v2[0])
                            # all_actions.append(v2[1])
                            # all_data_obs.append(v2[2])
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
                list_selected.append(index)
                # all_pov_obs.append(final)
                # all_actions.append(a)
                # all_data_obs.append(after_proces)
            # len(index_selected) / 655850

        last_inventory = b_inventory
        index += 1
        # all_pov_obs.append(final)
        # all_actions.append(a)
        # all_data_obs.append(after_proces)
        store_all['obs'].append(final)
        store_all['data'].append(after_proces)
        store_all['a'].append(a)
        store_all['values'].append(r)

    current_v = 0
    for i in range(len(reward_eps)):
        r = reward_eps[len(reward_eps) - 1 - i]
        current_v = r + current_v * gamma
        store_all['values'][len(reward_eps) - 1 - i] = current_v

    for i in list_selected:
        all_pov_obs.append(store_all['obs'][i])
        all_actions.append(store_all['a'][i])
        all_data_obs.append(store_all['data'][i])
        all_values.append(store_all['values'][i])
    print(t_r, len(all_values), np.sum(np.array(all_values) != 0))

    if len(all_actions) >= DATA_SAMPLES:
        break

np.power(1024,1/2)
np.power(2,10)
np.power(2,10)
10 * (0.999**1000)
a = Counter(keyss)
a['_0_0']
2/2
np.power(1024, -2)
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
all_values = np.array(all_values)

# all_actions_r = np.array(all_actions_r)
# np.bincount(all_actions)
# print(len(all_actions)/ 1916597)
# np.sum(all_actions == 80)

network = NatureCNNV2((7, 64, 64), len(list_key_index)).cuda()
optimizer = th.optim.Adam(network.parameters(), lr=LEARNING_RATE)
loss_function = nn.CrossEntropyLoss()

num_samples = all_actions.shape[0]
update_count = 0
losses = []
np.max(all_values)
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
        v = all_values[batch_indices]

        # distances = np.sum((action_vectors - action_centroids[:, None]) ** 2, axis=2)
        # actions = np.argmin(distances, axis=0)
        # Obtain logits of each action
        inven = th.from_numpy(all_data_obs[batch_indices]).float().cuda()
        logits, values = network(obs, inven)

        # Minimize cross-entropy with target labels.
        # We could also compute the probability of demonstration actions and
        # maximize them.
        loss = loss_function(logits, th.from_numpy(actions).long().cuda())
        loss_v = F.mse_loss(values, th.from_numpy(v).float().cuda())
        loss = loss + loss_v * 0.3
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


TRAIN_MODEL_NAME = 'another_potato_3.pth'  # name to use when saving the trained agent.

th.save(network.state_dict(), TRAIN_MODEL_NAME)
