import minerl
import gym
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np, cv2
from pynput.keyboard import Key, Listener
from pynput.mouse import Button
from pynput import mouse
from threading import Thread
import pickle
import time
import random
import imutils
from pairing import pair, depair
import torch
from colorsys import rgb_to_hsv, hsv_to_rgb
torch.cuda.set_device(0)
import torchvision
import math
import copy

a = 3
#
#
#
# env = gym.make('MineRLObtainDiamond-v0')
# env.seed(21)
#
# obs = env.reset()

# hsv_bond = {}

key_list.stop()
mouse_list.stop()

#


def nothing(x):
    pass


def trafer_hsv(dict):
    cv2.setTrackbarPos('HMin', 'image', dict['hMin'])
    cv2.setTrackbarPos('SMin', 'image', dict['sMin'])
    if 'vMin' in dict:
        cv2.setTrackbarPos('VMin', 'image', dict['vMin'])
    else:
        cv2.setTrackbarPos('VMin', 'image', 0)

    cv2.setTrackbarPos('HMax', 'image', dict['hMax'])
    cv2.setTrackbarPos('SMax', 'image', dict['sMax'])

    if 'vMax' in dict:
        cv2.setTrackbarPos('VMax', 'image', dict['vMax'])
    else:
        cv2.setTrackbarPos('VMax', 'image', 255)


def on_move(x, y):
    global attack_b


def on_click(x, y, button, pressed):
    global attack_b
    global action
    # print('{0} at {1}'.format(
    #     button if pressed else 'Released',
    #     (x, y)))
    if pressed:
        try:
            if button == Button.left:
                action['attack'] = 1
        except AttributeError:
            pass
    else:
        action['attack'] = 0
    global key_on_release
    if key_on_release == 'esc':
        return False


def on_scroll(x, y, dx, dy):
    global attack_b


def on_press(key):
    global key_on_press
    global action
    try:
        key_on_press = key.char
        if key_on_press == 'w' and action['back'] != 1:
            action['forward'] = 1

        if key_on_press == 's' and action['forward'] != 1:
            action['back'] = 1

        if key_on_press == 'a' and action['right'] != 1:
            action['left'] = 1
        if key_on_press == 'd' and action['left'] != 1:
            action['right'] = 1

        if key_on_press == 'q' and action['camera'][1] == 0:
            action['camera'][1] = -15

        if key_on_press == 'e' and action['camera'][1] == 0:
            action['camera'][1] = 15

        if key_on_press == 'r' and action['camera'][0] == 0:
            action['camera'][0] = -15
        if key_on_press == 'f' and action['camera'][0] == 0:
            action['camera'][0] = 15


    except AttributeError:
        # do something when a certain key is pressed, using key, not key.char
        pass
    if key == Key.space:
        action['jump'] = 1


def on_release(key):
    global key_on_release
    global jump
    global action
    global pasue
    global lock
    try:
        key_on_release = key.char
        if key_on_release == 'w':
            action['forward'] = 0
        if key_on_release == 's':
            action['back'] = 0
        if key_on_release == 'a':
            action['left'] = 0
        if key_on_release == 'd':
            action['right'] = 0

        if key_on_release == 'q':
            action['camera'][1] = 0
        if key_on_release == 'e':
            action['camera'][1] = 0
        if key_on_release == 'r':
            action['camera'][0] = 0
        if key_on_release == 'f':
            action['camera'][0] = 0
        if key_on_release == 'p':
            pasue = not pasue
            print("pause ", pasue)
        if key_on_release == 'l':
            lock = not lock
            print(lock)
        # if key.char == 'o':
        #     return False
    except AttributeError:
        # do something when a certain key is pressed, using key, not key.char
        pass

    if key == Key.esc:
        # Stop listener
        print("Stop lister ")
        key_on_release = 'esc'
        return False
    if key == Key.space:
        action['jump'] = 0

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

# def shift_image(X, dx, dy):
#     X = torch.roll(X, dy, dims=-2)
#     X = torch.roll(X, dx, dims=-1)
#     if dy>0:
#         X[ :, dy, :] = 1500
#     elif dy<0:
#         X[:, dy:, :] = 1500
#     if dx>0:
#         X[:, :, :dx] = 1500
#     elif dx<0:
#         X[:, :, dx:] = 1500
#     return X

def Life2CodingRGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  # checks mouse moves
        global hsv
        #image_2 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        colorsBGR = hsv[y, x]
        # colorsRGB=cv2.cvtColor(colorsBGR,cv2.COLOR_BGR2HSV)
        global min_k1
        global names
        global dict_data
        global h_range
        global s_range
        global v_range
        global X_VALUE
        global Y_VALUE
        global hsv_torch
        global k1
        X_VALUE = y
        Y_VALUE = x

        index = min_k1[y,x]

        print("RGB Value at ({},{}):{} {} {}".format(x, y, hsv_torch[:,y,x], k1[y,x],
                                                     [h_range[y,x,index],
                                                      s_range[y,x,index],
                                                      v_range[y,x,index]]), end='\r')

def Life2CodingRGB2(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  # checks mouse moves
        image_2  =cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        colorsBGR = image_2[y, x]
        # colorsRGB=cv2.cvtColor(colorsBGR,cv2.COLOR_BGR2HSV)
        global tmp_33
        global names
        print("RGB Value at ({},{}):{} {}".format(x,y,colorsBGR, names[tmp_33[y,x]]), end='\r')

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


cv2.namedWindow('mask', cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow('mask', 1000, 1000)
cv2.moveWindow('mask', 1100, 30)
cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow('image', 1000, 1000)
cv2.moveWindow('image', -30, 30)



cv2.namedWindow('color_range')
cv2.moveWindow('color_range', 40, 1000)


# Initialize HSV min/max values
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

index = 1
action = env.action_space.noop()
key_list = Listener(on_press=on_press, on_release=on_release)
mouse_list = mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
attack_b = False
key_list.start()
mouse_list.start()

key_on_press = ''
key_on_release = ''
pasue = False
lock = True
levels = [-1, -1, -1]
current_level = -1
indexs = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
with open('pixel_location.pkl', 'rb') as f:
    pixel_location = pickle.load(f)


pool_size = 20
pool_colors = np.zeros([pool_size,3])
np.random.seed(10)
x = list(np.arange(0, 179, 179/pool_size))
pool_colors[:,0] =np.random.permutation(x)
pool_colors[:,1] = np.random.choice(range(50, 200), size=pool_size)
pool_colors[:,2] = np.random.choice(range(100, 200), size=pool_size)

names_stack = []
t_R = 0
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (400, 400)
fontScale = 0.6
fontColor = (0, 0, 255)
lineType = 1

max_size = 128 * 15

x = np.linspace(-1, 1, 3, dtype=np.int64)
# x = [-2, -1, 1, 2]
# neighbor_pixels = [[], []]

neighbor_pixels = [[-1, 0, 1, 0, -2, 2, 2, -2], [0, 1, 0, -1, -2, -2, 2, 2]]
# for i in range(len(x)):
#     for j in range(len(x)):
#         if not (x[i] == 0 and x[j] == 0):
#             neighbor_pixels[0].append(x[i])
#             neighbor_pixels[1].append(x[j])
neighbor_pixels = np.array(neighbor_pixels)
H_VALUE = 0
S_VALUE = 0
V_VALUE = 0
X_VALUE = 0
Y_VALUE = 0

with open('first_model.pkl', 'rb') as fin:
    clf = pickle.load(fin)

dict_data = {}
with open('data_1.pkl', 'rb') as f:
    dict_data = pickle.load(f)

h_range = []
s_range = []
v_range = []

names = []
kind = []
tree_chunk = []
Hs_range = []

for k, v in dict_data.items():
    for i in range(len(v['H'])):
        h_range.append(v['H'][i])
        s_range.append(v['S'][i])
        v_range.append(v['V'][i])
        Hs_range.append(v['Hs'][i])

        names.append(v['name'][i])
        kind.append(k)
        if k == "tree_chunk":
            tree_chunk.append(255)
        else:
            tree_chunk.append(0)

kind = np.array(kind)
x = np.sort(h_range)
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
s_range_f = torch.unsqueeze(s_range, 0)
s_range_f = s_range_f.cuda()

v_range = np.expand_dims(v_range, axis=0)
v_range = np.repeat(v_range, 64, axis=0)
v_range = np.expand_dims(v_range, axis=0)
v_range = np.repeat(v_range, 64, axis=0)
v_range = np.array(v_range, dtype=np.float64)
v_range = torch.Tensor(v_range)
v_range_f = torch.unsqueeze(v_range, 0)
v_range_f = v_range_f.cuda()

kind_f = np.expand_dims(kind, axis=0)
kind_f = np.repeat(kind_f, 64, axis=0)
kind_f = np.expand_dims(kind_f, axis=0)
kind_f = np.repeat(kind_f, 64, axis=0)
kind_f = np.array(kind_f)

Hs_range = np.expand_dims(Hs_range, axis=0)
Hs_range = np.repeat(Hs_range, 64, axis=0)
Hs_range = np.expand_dims(Hs_range, axis=0)
Hs_range = np.repeat(Hs_range, 64, axis=0)
Hs_range = np.array(Hs_range, dtype=np.float64)
Hs_range = torch.Tensor(Hs_range).cuda()

tree_chunk = np.array(tree_chunk, dtype=np.uint8)

keyboard_map = {'c': 'tree_chunk', 't': 'tree_leave', 'r': 'rock', 's': 'sand', 'd': 'dirt',
                'u': 'not_common', 'g':'grass','w': 'water', 'k': 'sky', 'n':'snow',
                'm':'monster', 'a':'animal', 'h': 'coal', 'i': 'iron','b': 'crafting_table'}
key_selected = ''
add_done = False
x = np.linspace(-2, 2, 5, dtype=np.int64)
neighbor_pixels_2 = [[], []]
for i in range(len(x)):
    for j in range(len(x)):
        if not (x[i] == 0 and x[j] == 0):
            neighbor_pixels_2[0].append(x[i])
            neighbor_pixels_2[1].append(x[j])
neighbor_pixels_2 = [[-1, 0, 1, 0, -2, 2, 2, -2], [0, 1, 0, -1, -2, -2, 2, 2]]

neighbor_pixels_2 = np.array(neighbor_pixels_2)
while (True):
    # Display result image
    color_range = np.zeros([128, max_size, 3], dtype=np.uint8)
    tmp_time = time.time()

    if pasue:
        #action['place'] = 'none'
        #action['equip'] = 'wooden_pickaxe'

        #action['nearbyCraft'] = 'none'
        #action['nearbyCraft'] = 'wooden_axe'

        obs, reward, done, _ = env.step(action)
        t_R += reward
        # obs['inventory']
        #print(obs['inventory']['wooden_pickaxe'], end='\r')

        # record_p.append(obs['pov'])
    else:
        action = env.action_space.noop()
    # if action['jump'] == 1:
    #     action['jump'] = 0
    image = obs['pov']
    image = image.transpose(2, 0, 1).astype(np.float32)
    image = torch.from_numpy(image)
    hsv_torch = rgb_to_hsv(image)
    # hsv_torch = hsv_torch.cpu().numpy()

    image[0, 0, 0] = 254

    image = obs['pov']

    # Convert to HSV format and color threshold
    X = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    tmp_mask = np.zeros([64, 64, 3], dtype=np.uint8)

    hsv_64 = hsv_torch
    pixels_maps = {}
    keys_mask = {}
    key_index = 0

    H_s = np.zeros((64, 64,len(neighbor_pixels[0])), dtype=np.float64)
    hsv_64_numpy = hsv_torch.cpu().numpy().transpose(1,2, 0).astype(np.float32)
    Hsv_nei = np.zeros((64, 64, 3, len(neighbor_pixels[0]) + 1), dtype=np.float64)
    Hsv_nei[:, :, :, 0] = hsv_64_numpy

    for counter in range(len(neighbor_pixels[0])):
        new_hsv = shift_image(hsv_64_numpy, neighbor_pixels[0][counter],
                              neighbor_pixels[1][counter])
        H_s[:, :,counter] = new_hsv[:,:,0]
        Hsv_nei[:, :, :, counter + 1] = new_hsv

    Hsv_nei.shape
    H_s_copy = np.copy(H_s)
    H_s = np.expand_dims(H_s, axis=-2)
    H_s = np.repeat(H_s, h_range.shape[-1], axis=-2)
    H_s = torch.from_numpy(H_s).cuda()
    tmp_time = time.time()
    # h = np.expand_dims(hsv_64[:, :, 0], axis=-1)
    # s = np.expand_dims(hsv_64[:, :, 1], axis=-1)
    # v = np.expand_dims(hsv_64[:, :, 2], axis=-1)
    h = torch.unsqueeze(hsv_64[0,:,:], -1)
    s = torch.unsqueeze(hsv_64[1,:,:], -1)
    v = torch.unsqueeze(hsv_64[2,:,:], -1)

    hsv_torch = hsv_torch.cpu().numpy()

    # H1_range.shape

    if len(dict_data['rock']['H']) != 0:
        tmp_time = time.time()

        Hsv_nei = Hsv_nei.reshape(64 * 64, Hsv_nei.shape[-1] * 3)
        Hsv_nei = np.concatenate((Hsv_nei, pixel_location[0]), axis=1)
        Hsv_nei = np.concatenate((Hsv_nei, pixel_location[1]), axis=1)
        # with open('first_model.pkl', 'rb') as fin:
        #     clf = pickle.load(fin)
        y_pred = clf.predict(Hsv_nei)
        # np.unique(y_pred)

        k1 = y_pred.reshape(64,64)
        index_color = 0
        # tmp_33_kind = kind[tmp_33]
        for i in range(64):
            for j in range(64):
                kins = k1[i,j]

                if kins in names_stack:
                    index_1 = np.where(np.array(names_stack) == kins)[0][0]
                else:
                    names_stack.append(kins)
                    index_1 = len(names_stack)
                color = pool_colors[index_1]
                tmp_mask[i, j, :] = color

                if (index_1 + 1) * 128 < max_size:
                    if color_range[0, index_1 * 128 +10, 0] == 0:
                        color_range[:, index_1 * 128: (index_1 + 1) * 128, :] = color
                        cv2.putText(color_range, kins,
                                    (index_1 * 128 + 20, 64),
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)
        # print(time.time() - tmp_time)

    # tmp_33[29,22]
    if not lock and not pasue:
        a = 3
        if key_on_press in keyboard_map and key_on_release != 'l':
            key_selected = keyboard_map[key_on_press]
            print("new: key {0} - {1} - {2}".format(key_selected, [hsv_torch[0, X_VALUE,Y_VALUE],
                                                             hsv_torch[1, X_VALUE,Y_VALUE],
                                                             hsv_torch[2, X_VALUE,Y_VALUE]], [X_VALUE, Y_VALUE]))
            key_on_press = ''
            add_done = False
        if key_selected != '':
            if key_on_press in indexs:
                print("aa")
                if int(key_on_press) == 7:
                    Hsv_nei[:, :, 0, :] = Hsv_nei[:, :, 0, :] / 360
                    Hsv_nei[:, :, 1, :] = Hsv_nei[:, :, 1, :] / 255
                    Hsv_nei[:, :, 2, :] = Hsv_nei[:, :, 2, :] / 255
                    Hsv_nei = Hsv_nei.reshape(64 * 64, Hsv_nei.shape[-1] * 3)
                    key_selected = ''
                    data_train = {}
                    k_11 = k1.reshape(64 * 64)
                    with open('data_train.pkl', 'rb') as f:
                        data_train = pickle.load(f)
                    if len(data_train['x']) == 0:
                        data_train['x'] = Hsv_nei
                        data_train['y'] = k_11
                    else:
                        data_train['x'] = np.concatenate((data_train['x'], Hsv_nei), axis=0)
                        data_train['y'] = np.concatenate((data_train['y'], k_11), axis=0)
                    print("ADD: ", data_train['x'].shape[0])

                    with open('data_train.pkl', 'wb') as f:
                        pickle.dump(data_train, f)
                if int(key_on_press) == 1:
                    # hsv_torch = hsv_torch.cpu().numpy()
                    dict_data[key_selected]['H'].append(hsv_torch[0, X_VALUE,Y_VALUE])
                    dict_data[key_selected]['S'].append(hsv_torch[1, X_VALUE,Y_VALUE])
                    dict_data[key_selected]['V'].append(hsv_torch[2, X_VALUE,Y_VALUE])
                    dict_data[key_selected]['Hs'].append(H_s_copy[X_VALUE,Y_VALUE,:])

                    name = '{0}_{1}'.format(key_selected, len(dict_data[key_selected]['H']))

                    dict_data[key_selected]['name'].append(name)

                    print("ADDED : key {0} - {1} - {2} - {3}".format(key_selected, [hsv_torch[0, X_VALUE,Y_VALUE],
                                                                              hsv_torch[1, X_VALUE,Y_VALUE],
                                                                              hsv_torch[2, X_VALUE,Y_VALUE]],
                                                                [X_VALUE, Y_VALUE],
                                                                     H_s_copy[X_VALUE,Y_VALUE,:]))
                    with open('data_1.pkl', 'wb') as f:
                        pickle.dump(dict_data, f)


                if int(key_on_press) == 2:
                    print("STOP")
                    key_selected = ''
                    add_done = False
                if int(key_on_press) == 4:
                    h_del = dict_data[key_selected]['H'][-1]
                    s_del = dict_data[key_selected]['S'][-1]
                    v_del = dict_data[key_selected]['V'][-1]
                    name_del = dict_data[key_selected]['name'][-1]
                    dict_data[key_selected]['H'] = dict_data[key_selected]['H'][:-1]
                    dict_data[key_selected]['S'] = dict_data[key_selected]['S'][:-1]
                    dict_data[key_selected]['V'] = dict_data[key_selected]['V'][:-1]
                    dict_data[key_selected]['Hs'] = dict_data[key_selected]['Hs'][:-1]


                    print("DEL : key {0} - {1}".format(key_selected, [h_del, s_del, v_del]))
                    # key_selected = ''
                    add_done = False
                    with open('data_1.pkl', 'wb') as f:
                        pickle.dump(dict_data, f)
                dict_data = {}
                with open('data_1.pkl', 'rb') as f:
                    dict_data = pickle.load(f)

                h_range = []
                s_range = []
                v_range = []

                names = []
                kind = []
                tree_chunk = []
                Hs_range = []

                for k, v in dict_data.items():
                    for i in range(len(v['H'])):
                        h_range.append(v['H'][i])
                        s_range.append(v['S'][i])
                        v_range.append(v['V'][i])
                        Hs_range.append(v['Hs'][i])

                        names.append(v['name'][i])
                        kind.append(k)
                        if k == "tree_chunk":
                            tree_chunk.append(255)
                        else:
                            tree_chunk.append(0)

                kind = np.array(kind)
                print(len(h_range))

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
                s_range_f = torch.unsqueeze(s_range, 0)
                s_range_f = s_range_f.cuda()

                v_range = np.expand_dims(v_range, axis=0)
                v_range = np.repeat(v_range, 64, axis=0)
                v_range = np.expand_dims(v_range, axis=0)
                v_range = np.repeat(v_range, 64, axis=0)
                v_range = np.array(v_range, dtype=np.float64)
                v_range = torch.Tensor(v_range)
                v_range_f = torch.unsqueeze(v_range, 0)
                v_range_f = v_range_f.cuda()

                kind_f = np.expand_dims(kind, axis=0)
                kind_f = np.repeat(kind_f, 64, axis=0)
                kind_f = np.expand_dims(kind_f, axis=0)
                kind_f = np.repeat(kind_f, 64, axis=0)
                kind_f = np.array(kind_f)

                Hs_range = np.expand_dims(Hs_range, axis=0)
                Hs_range = np.repeat(Hs_range, 64, axis=0)
                Hs_range = np.expand_dims(Hs_range, axis=0)
                Hs_range = np.repeat(Hs_range, 64, axis=0)
                Hs_range = np.array(Hs_range, dtype=np.float64)
                Hs_range = torch.Tensor(Hs_range).cuda()



                add_done = True
                key_on_press = ''
        #image = cv2.imread("./block_image/Mega_Spruce_Tree.png")
    if key_on_press in indexs:
        if int(key_on_press) == 7:
            Hsv_nei[:, :, 0, :] = Hsv_nei[:, :, 0, :] / 360
            Hsv_nei[:, :, 1, :] = Hsv_nei[:, :, 1, :] / 255
            Hsv_nei[:, :, 2, :] = Hsv_nei[:, :, 2, :] / 255
            Hsv_nei = Hsv_nei.reshape(64 * 64, Hsv_nei.shape[-1] * 3)
            key_selected = ''
            data_train = {}
            k_11 = k1.reshape(64 * 64)
            with open('data_train.pkl', 'rb') as f:
                data_train = pickle.load(f)
            if len(data_train['x']) == 0:
                data_train['x'] = Hsv_nei
                data_train['y'] = k_11
            else:
                data_train['x'] = np.concatenate((data_train['x'], Hsv_nei), axis=0)
                data_train['y'] = np.concatenate((data_train['y'], k_11), axis=0)
            print("ADD: ", data_train['x'].shape[0])

            with open('data_train.pkl', 'wb') as f:
                pickle.dump(data_train, f)
            key_on_press = ''

    # print(np.count_nonzero(names == 32)/4096, end='\r')
    # mask = 255 - mask
    # hsv[:,:,2] = 255
    # X = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)



    # X = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # dst = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 0.7,
    #                       cv2.cvtColor(tmp_mask, cv2.COLOR_HSV2RGB), 0.7, 0)
    #hsv = np.array(hsv_64,dtype=np.uint8)
    # hsv[48,19,0] = hsv[48,18,0]
    # hsv[48,19,1] = hsv[48,18,1]
    #hsv[48,19,2]=  hsv[48,18,2]

    tmp_hsv = np.array(hsv)
    #tmp_hsv[X_VALUE,Y_VALUE,0] = 50

    # tmp_hsv[X_VALUE,Y_VALUE,1] = 255
    #
    tmp_hsv[X_VALUE,Y_VALUE,2] = 230
    #tmp_hsv[:,:,2] = 255

    # index = np.where((hsv[:, :, 0] >=20) |( hsv[:, :, 0] <= 16))
    #hsv[index[0], index[1], :] = [0,0, 255]
    # tmp_hsv[:,:,0] = 179
    # tmp_hsv[:,:,1] = 2
    # tmp_hsv[:,:,2] = tmp_hsv[:,:,2] + 50

    tmp_image = cv2.cvtColor(tmp_hsv, cv2.COLOR_HSV2BGR)
    # tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2GRAY)

    # tmp_image[0,1]
    # tmp_image[X_VALUE,Y_VALUE,:] = [100,100,100]
    tmp_mask[X_VALUE,Y_VALUE,2] = 230

    cv2.imshow('image',tmp_image)
    cv2.resizeWindow('image', 950, 950)
    cv2.setMouseCallback('image', Life2CodingRGB)

    cv2.imshow('mask', cv2.cvtColor(tmp_mask, cv2.COLOR_HSV2RGB))
    cv2.resizeWindow('mask', 950, 950)
    cv2.setMouseCallback('mask', Life2CodingRGB)


    cv2.imshow('color_range', cv2.cvtColor(color_range, cv2.COLOR_HSV2RGB))
    # cv2.resizeWindow('mask', 800, 800)

    # for c in cnts:
    #     x, y, w, h = cv2.boundingRect(c)
    #
    #     print(cv2.contourArea(c))
    # print("---")
    # image = obs['pov']
    # cv2.imshow('image', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # print(time.time() - tmp_time)

    if cv2.waitKey(10) & 0xFF == ord('o'):
        break
    # time.sleep(0.1)

key_list.stop()
mouse_list.stop()

cv2.destroyAllWindows()
# data_train = {'y': np.array([]),
#               'x': np.array([])}
# with open('data_train.pkl', 'wb') as f:
#     pickle.dump(data_train, f)

hsv[16,60]
# with open('data_1.pkl', 'wb') as f:
#     pickle.dump(dict_data, f)
#

# pixels_maps = {}
# keys_mask = {}
# key_index = 0
# hsv_64 = np.array(hsv, dtype=np.float64)
# for i in range(64):
#     for j in range(64):
#
#         pixel_id = tuple((i, j))
#         pixel_hsv = hsv_64[i, j, :]
#
#         neigh_bor = [neighbor_pixels[0] + i, neighbor_pixels[1] + j]
#         index = np.where(((neigh_bor[0] <= 63) & (neigh_bor[0] > 0)))
#         neigh_bor[0] = neigh_bor[0][index]
#         neigh_bor[1] = neigh_bor[1][index]
#         index = np.where(((neigh_bor[1] <= 63) & (neigh_bor[1] > 0)))
#         neigh_bor[0] = neigh_bor[0][index]
#         neigh_bor[1] = neigh_bor[1][index]
#
#         neigh_hs = hsv_64[neigh_bor[0], neigh_bor[1], : 2]
#
#         HS_error = np.sum(np.abs(neigh_hs - pixel_hsv[:2]), axis=1) / 2
#         HS_error_min = np.min(HS_error)
#         if HS_error_min <= 25:
#             for ins in range(len(HS_error)):
#                 if np.abs(HS_error[ins] - HS_error_min) <= 5:
#                     next_pixel = tuple((neigh_bor[0][ins], neigh_bor[1][ins]))
#                     if next_pixel in pixels_maps:
#                         key = pixels_maps[next_pixel]
#                     else:
#                         key = key_index
#                         keys_mask[key] = [[], [], {}]
#                         key_index += 1
#                     pixels_maps[pixel_id] = key
#                     pixels_maps[next_pixel] = key
#                     if pixel_id not in keys_mask[key][2]:
#                         keys_mask[key][2][pixel_id] = 0
#                         keys_mask[key][0].append(pixel_id[0])
#                         keys_mask[key][1].append(pixel_id[1])
#                     if next_pixel not in keys_mask[key][2]:
#                         keys_mask[key][2][next_pixel] = 0
#                         keys_mask[key][0].append(next_pixel[0])
#                         keys_mask[key][1].append(next_pixel[1])
# keys_mask[0]
# for k,v in keys_mask.items():
#     print(len(v[0]))
#     # tmp_mask[v[0], v[1],0] = np.mean(hsv[v[0], v[1],0])
#     # tmp_mask[v[0], v[1],1] = np.mean(hsv[v[0], v[1],1])
#     # tmp_mask[v[0], v[1],2] = np.mean(hsv[v[0], v[1],2])
a = 3
a = 3
# dict_data['dirt']
# dict_data['tree_chunk']

# dict_data = {}
# for k,v in keyboard_map.items():
#     kind = v
#     if kind not in dict_data:
#         dict_data[kind] = {'H': [], 'S': [], 'V': [],
#                            'Hs': [],
#                            'name': []}
# with open('data_1.pkl', 'wb') as f:
#     pickle.dump(dict_data, f)
    # dict_data[kind]['H'].append(v['H'])
    # dict_data[kind]['S'].append(v['S'])
    # dict_data[kind]['V'].append(v['v'])
    # name = '{0}_{1}'.format(kind, len(dict_data[kind]['H']))
    # dict_data[kind]['name'].append(name)
#


# with open('data.pkl', 'rb') as f:
#     output = pickle.load(f)
#
# h_range = []
# s_range = []
# v_range = []
# names = []
# kind = []
# for k,v in output.items():
#     for i in range(len(v['H'])):
#         h_range.append(v['H'][i])
#         s_range.append(v['S'][i])
#         v_range.append(v['V'][i])
#         names.append(v['name'][i])
#         kind.append(k)
# h_range = np.expand_dims(h_range,axis=0)
# h_range = np.repeat(h_range, 64, axis=0)
# h_range = np.expand_dims(h_range,axis=0)
# h_range = np.repeat(h_range, 64, axis=0)
#
# s_range = np.expand_dims(s_range,axis=0)
# s_range = np.repeat(s_range, 64, axis=0)
# s_range = np.expand_dims(s_range,axis=0)
# s_range = np.repeat(s_range, 64, axis=0)
# h_range = np.array(h_range, dtype=np.float64)
# s_range = np.array(s_range, dtype=np.float64)
#
# v_range = np.expand_dims(v_range,axis=0)
# v_range = np.repeat(v_range, 64, axis=0)
# v_range = np.expand_dims(v_range,axis=0)
# v_range = np.repeat(v_range, 64, axis=0)
# v_range = np.array(v_range, dtype=np.float64)

# key_del = 'tree_chunk'
# item_del = 'tree_chunk_4'
# tmp_key = {'H': [], 'S': [], 'V': [],
#                            'D': [],
#                            'name': []}
# v = dict_data[key_del]
# for i in range(len(dict_data[key_del]['H'])):
#     if v['name'][i] != item_del:
#         tmp_key['H'].append(v['H'][i])
#         tmp_key['S'].append(v['S'][i])
#         tmp_key['V'].append(v['V'][i])
#         tmp_key['D'].append(v['D'][i])
#         name = '{0}_{1}'.format(kind, len(tmp_key['H']))
#         tmp_key['name'].append(name)
#
# dict_data[key_del] = tmp_key
