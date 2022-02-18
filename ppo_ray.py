import time
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common import env_checker
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

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
import torch.nn as nn

import ray
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class CustomTorchModel(TorchModelV2):
    def __init__(self, *args, **kwargs):
        TorchModelV2.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)
        self._hidden_layers = nn.Sequential(...)
        self._logits = ...
        self._value_branch = ...

    def forward(self, input_dict, state, seq_lens): ...
    def value_function(self): ...

ModelCatalog.register_custom_model("my_torch_model", CustomTorchModel)

ray.init()
trainer = ppo.PPOTrainer(env="CartPole-v0", config={
    "framework": "torch",
    "model": {
        "custom_model": "my_torch_model",
        # Extra kwargs to be passed to your model's c'tor.
        "custom_model_config": {},
    },
})

#
# try:
#     wandb = None
#     import wandb
# except ImportError:
#     pass

with open('action_connect_to_vector.pkl', 'rb') as f:
    action_vector_all = pickle.load(f)

with open('action_key_1.pkl', 'rb') as f:
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
        self.list_obs = [[-15, 0], [15, 0], [0, -15], [0, 15]]

        self.reset_all_value()

    def observation(self, observation):
        pov = observation['pov']
        grey = np.dot(pov[..., :3], [0.2989, 0.5870, 0.1140])
        self.stack_grey[tuple((self.angle_1, self.angle_2))] = [grey, self.time_add]

        for location in self.list_obs:
            a1 = self.angle_1 + location[0]
            a2 = ((int(self.angle_2 + location[1]) + 180) % 360)
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
        observation, reward, done, info = self.env.step(action)

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
                self.angle_2 -= 15
            elif slipt[-1] == '1':
                self.angle_2 += 15

            if slipt[-2] == '-1':
                self.angle_1 -= 15
            elif slipt[-2] == '1':
                self.angle_1 += 15
            self.angle_1 = np.clip(self.angle_1, -90, 90)

        self.time_add += 1

        return self.observation(observation), reward, done,

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
        super(CustomCNN, self).__init__(observation_space, features_dim=len(list_key_index))
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
            nn.Linear(512, len(list_key_index))
        )

    def forward(self, observations) -> th.Tensor:
        x1 = self.cnn(observations['pov'])
        x2 = self.linear_stack(observations['inventory'])
        x = torch.cat((x1, x2), dim=1)
        return self.linear(x)

    # def __init__(
    #     self,
    #     feature_dim: int,
    #     last_layer_dim_pi: int = 64,
    #     last_layer_dim_vf: int = 64,
    # ):
    #     super(CustomNetwork, self).__init__()
    #
    #     # IMPORTANT:
    #     # Save output dimensions, used to create the distributions
    #     self.latent_dim_pi = last_layer_dim_pi
    #     self.latent_dim_vf = last_layer_dim_vf
    #
    #     # Policy network
    #     self.policy_net = nn.Sequential(
    #         nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
    #     )
    #     # Value network
    #     self.value_net = nn.Sequential(
    #         nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
    #     )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)


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
        network = CustomCNN(input_shape=(7, 64, 64), output_dim=len(list_key_index))
        network.load_state_dict(th.load('another_potato_4.pth'))
        self.mlp_extractor = network


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
        return env
    return thunk


# track_exp(project_name="minerl")

env = DummyVecEnv([make_env(i) for i in range(1)])
# env = gym.make('MineRLObtainDiamond-v0')
# env = MyCustomObs(env)
# env.observation_space['pov'].shape[0]
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(),
)

model = PPO(CustomActorCriticPolicy, "CartPole-v1", verbose=1)

model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=f"runs/{'v1'}")

# model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=f"runs/{'v1'}")
model.learn(total_timesteps=2000000)  # 2m steps is about 8h at 70 FPS
model.save('ppo_first')

# env_checker.check_env(env, warn=True, skip_render_check=True)
# MineRL might throw an exception when closing on Windows, but it can be ignored (the environment does close).
try:
    env.close()
except Exception:
    pass

model.load('ppo_first')

env = gym.make('MineRLObtainDiamond-v0')
env = PovOnlyObservation(env)
env = ActionShaping(env, always_attack=True)
env1 = env.env

for episode in range(3):
    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0
    cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('image', 1000, 1000)
    # RL part to get some logs:
    for i in range(10000):
        action = model.predict(obs.copy())
        print(action)
        obs, reward, done, _ = env.step(action[0])
        total_reward += reward
        env.render()
        steps += 1
        cv2.imshow('image', obs[:, :, 1])
        cv2.resizeWindow('image', 950, 950)
        if done:
            break
        if cv2.waitKey(10) & 0xFF == ord('o'):
            break
        time.sleep(0.1)
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

env.close()
