import gym
import torch as th
from torch import nn
import numpy
import pickle
import numpy as np
import copy
import lightgbm
class EpisodeDone(Exception):
    pass

MAX_TEST_EPISODE_LEN = 18000  # 18k is the default for MineRLObtainDiamond.

class Episode(gym.Env):
    """A class for a single episode."""
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._done = False

    def reset(self):
        if not self._done:
            return self.env.reset()

    def step(self, action):
        s, r, d, i = self.env.step(action)
        if d:
            self._done = True
            raise EpisodeDone()
        else:
            return s, r, d, i

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
        x = th.cat((x1, x2), dim=1)

        return self.linear(x)

# def process_a(action, action_new):
#     key = list_key_index[action]
#     for k, v in action_key[key].items():
#         action_new[k] = v
#     # action_new['attack'] = 1
#     return

def process_inventory(obs, attack, t, full_previous, angle_1, angle_2, inven_process):
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

    # data[15] = attack
    # data = np.concatenate((data, full_previous), axis=0)

    # a2 = int(a2)
    # data[15] = np.clip((a1 + 180)/ 180, 0, 1)
    # data[16] = np.clip(((a2 + 180)% 360)/360, 0, 1)
    # data[15] = np.clip((a1 + 180)/ 180, 0, 1)
    # data[16] = np.clip(((a2 + 180)% 360)/360, 0, 1)
    return data

class MineRLAgent():
    """
    To compete in the competition, you are required to implement the two
    functions in this class:
        - load_agent: a function that loads e.g. network models
        - run_agent_on_episode: a function that plays one game of MineRL

    By default this agent behaves like a random agent: pick random action on
    each step.

    NOTE:
        This class enables the evaluator to run your agent in parallel in Threads,
        which means anything loaded in load_agent will be shared among parallel
        agents. Take care when tracking e.g. hidden state (this should go to run_agent_on_episode).
    """

    def load_agent(self):
        """
        This method is called at the beginning of the evaluation.
        You should load your model and do any preprocessing here.
        THIS METHOD IS ONLY CALLED ONCE AT THE BEGINNING OF THE EVALUATION.
        DO NOT LOAD YOUR MODEL ANYWHERE ELSE.
        """
        # This is a random agent so no need to do anything
        # YOUR CODE GOES HERE

        # np.save("all_key_actions.npy", action_key)
        # with open('action_key_1.pkl', 'wb') as f:
        #     pickle.dump(action_key, f)
        with open('./train/action_key_1.pkl', 'rb') as f:
            self.action_key = pickle.load(f)
        self.list_key_index = np.array(list(self.action_key.keys()))

        self.network = NatureCNN((7, 64, 64),  len(self.list_key_index)).cuda()
        self.network.load_state_dict(th.load("./train/another_potato_4.pth"))

        inventory_list = ['coal', 'cobblestone', 'crafting_table', 'dirt', 'furnace', 'iron_ore', 'iron_ingot',
                          'iron_pickaxe',
                          'log', 'planks', 'stick', 'stone', 'stone_pickaxe', 'torch', 'wooden_pickaxe', 'mainhand']

        self.all_model_inven = {}
        for k_a in inventory_list:
            with open(f'./train/{k_a}.pkl', 'rb') as fin:
                self.all_model_inven[k_a] = pickle.load(fin)

        with open('./train/action_connect_to_vector.pkl', 'rb') as f:
            self.action_vector_all = pickle.load(f)

        pass

    def run_agent_on_episode(self, single_episode_env: Episode):
        """This method runs your agent on a SINGLE episode.

        You should just implement the standard environment interaction loop here:
            obs  = env.reset()
            while not done:
                env.step(self.agent.act(obs))
                ...

        NOTE:
            This method will be called in PARALLEL during evaluation.
            So, only store state in LOCAL variables.
            For example, if using an LSTM, don't store the hidden state in the class
            but as a local variable to the method.

        Args:
            env (gym.Env): The env your agent should interact with.
        """
        # An implementation of a random agent
        # YOUR CODE GOES HERE
        env = single_episode_env

        obs = env.reset()
        last_inventory = obs['vector']

        list_obs = [[-15, 0], [15, 0], [0, -15], [0, 15]]

        inventory_list = ['coal', 'cobblestone', 'crafting_table', 'dirt', 'furnace', 'iron_ore', 'iron_ingot',
                          'iron_pickaxe',
                          'log', 'planks', 'stick', 'stone', 'stone_pickaxe', 'torch', 'wooden_pickaxe', 'mainhand']

        done = False
        total_reward = 0
        steps = 0

        action_list = np.arange(len(self.list_key_index))

        angle_1 = 0
        angle_2 = 0

        time_add = 0

        time_attack_no_new = 0
        current_item = 0
        counter_0_0 = 0


        previous = None
        check_attack = False

        stack_grey = {}
        for i in range(MAX_TEST_EPISODE_LEN):

            hsv = obs['pov']
            grey = np.dot(obs['pov'][..., :3], [0.2989, 0.5870, 0.1140])

            angle_2_fix = ((int(angle_2) + 180) % 360)
            stack_grey[tuple((angle_1, angle_2_fix))] = grey

            for location in list_obs:
                a1 = angle_1 + location[0]
                a2 = ((int(angle_2 + location[1]) + 180) % 360)
                new_tuple = tuple((a1, a2))
                if new_tuple in stack_grey:
                    hsv = np.concatenate((hsv, stack_grey[new_tuple][:, :, None]), axis=-1)
                else:
                    hsv = np.concatenate((hsv, np.zeros((64, 64, 1))), axis=-1)
            # hsv = np.concatenate((hsv, first_obs, second_obs), axis=-1)
            hsv = hsv.transpose(2, 0, 1).astype(np.float32)
            hsv = th.from_numpy(hsv[None]).float().cuda()
            hsv /= 255.0

            inven_process = {}
            for k_a in inventory_list:
                inven_process[k_a] = self.all_model_inven[k_a].predict(obs['vector'][None])[0]

            inven = process_inventory(obs, time_attack_no_new, time_add, [], angle_1, angle_2, inven_process)
            inven = th.from_numpy(inven[None]).float().cuda()

            probabilities = th.softmax(self.network(hsv, inven), dim=1)[0]
            probabilities = probabilities.detach().cpu().numpy()
            action = np.random.choice(action_list, p=probabilities)

            key = self.list_key_index[action]
            if key == 'attack_0_0' and (check_attack or time_attack_no_new >= 300):
                action = np.random.choice([0, 1, 2, 3, 5, 6, 7, 8], p=np.repeat(0.125, 8))
                key = self.list_key_index[action]
                time_attack_no_new = 0

            slipt = key.split('_')
            if 'attack' in slipt:
                if time_attack_no_new == 0:
                    current_item = copy.deepcopy(last_inventory)

                    time_attack_no_new += 1

                else:
                    if np.sum(current_item - last_inventory) != 0.0:
                        time_attack_no_new = 0
                    else:
                        time_attack_no_new += 1
            else:
                time_attack_no_new = 0

            time_add += 1

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

                angle_1 = np.clip(angle_1, -90, 90)

            action = env.action_space.noop()
            action['vector'] = self.action_vector_all[key][0]

            obs, reward, done, info = env.step(action)

            if previous is not None:
                delta = previous - obs['pov']
                delta = np.sum(delta)
                if delta == 0:
                    check_attack = True
                else:
                    check_attack = False
            previous = obs['pov']

            total_reward += reward
            steps += 1
            if done:
                break

