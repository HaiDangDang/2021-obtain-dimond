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

# class PixelConnect():
#     def __init__(self):
#         super.__init__()
#
# class CompareNeighbor():
#     def __init__(self, max_neighbor_pixel=1, image_size=64):
#         super.__init__()
#         x = np.linspace(-max_neighbor_pixel, max_neighbor_pixel, max_neighbor_pixel * 2 + 1)
#         self.neighbor_pixels =  [[], []]
#         for i in range(len(x)):
#             for j in range(len(y)):
#                 if x[i] != 0 and x[j] != 0:
#                     self.neighbor_pixels[0].append(x[i])
#                     self.neighbor_pixels[1].append(x[j])
#
#
#         x = np.arange(1, image_size, 2)
#         self.pixels_map = {}
#         for i in range(len(x)):
#             for j in range(len(x)):
#                 self.pixels_map[tuple((x[i], x[j]))] = PixelConnect()
#
#     def make_pixel_connect(self, image):
#         hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
#         for pixel in self.pixels_map:
#

a = 3

# obs = env.reset()


# env = gym.make('MineRLObtainDiamond-v0')
# obs = env.reset()
# record_p = []

# hsv_bond = {}
#
key_list.stop()
mouse_list.stop()




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
        X[:dy, :] = 1500
    elif dy<0:
        X[dy:, :] = 1500
    if dx>0:
        X[:, :dx] = 1500
    elif dx<0:
        X[:, dx:] = 1500
    return X

def Life2CodingRGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  # checks mouse moves
        image_2  =cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        colorsBGR = image_2[y, x]
        # colorsRGB=cv2.cvtColor(colorsBGR,cv2.COLOR_BGR2HSV)
        global tmp_33
        global names
        global H_VALUE
        global S_VALUE
        global V_VALUE
        global X_VALUE
        global Y_VALUE
        X_VALUE = y
        Y_VALUE = x
        H_VALUE = colorsBGR[0]
        S_VALUE = colorsBGR[1]
        V_VALUE = colorsBGR[2]

        print("RGB Value at ({},{}):{} {}".format(x,y,colorsBGR, names[tmp_33[y,x]]), end='\r')

def Life2CodingRGB2(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  # checks mouse moves
        image_2  =cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        colorsBGR = image_2[y, x]
        # colorsRGB=cv2.cvtColor(colorsBGR,cv2.COLOR_BGR2HSV)
        global tmp_33
        global names
        print("RGB Value at ({},{}):{} {}".format(x,y,colorsBGR, names[tmp_33[y,x]]), end='\r')


cv2.namedWindow('mask', cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow('mask', 800, 800)
cv2.moveWindow('mask', 770, 30)
cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow('image', 800, 800)
cv2.moveWindow('image', 0, 30)



cv2.namedWindow('color_range')
cv2.moveWindow('color_range', 40, 1000)

# cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
# cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
# cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
# cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
# cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
# cv2.createTrackbar('VMax', 'image', 0, 255, nothing)
# # Set default value for Max HSV trackbars
# cv2.setTrackbarPos('HMax', 'image', 179)
# cv2.setTrackbarPos('SMax', 'image', 255)
# cv2.setTrackbarPos('VMax', 'image', 255)

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

pool_size = 20
pool_colors = np.zeros([pool_size,3])
np.random.seed(10)
x = list(np.arange(0, 179, 179/pool_size))
pool_colors[:,0] =np.random.permutation(x)
pool_colors[:,1] = np.random.choice(range(50, 200), size=pool_size)
pool_colors[:,2] = np.random.choice(range(150, 255), size=pool_size)

names_stack = []
t_R = 0
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (400, 400)
fontScale = 0.6
fontColor = (0, 0, 255)
lineType = 1

max_size = 128 * 15

x = np.linspace(-2, 2, 2 * 2 + 1, dtype=np.int64)
neighbor_pixels = [[], []]
for i in range(len(x)):
    for j in range(len(x)):
        if not (x[i] == 0 and x[j] == 0):
            neighbor_pixels[0].append(x[i])
            neighbor_pixels[1].append(x[j])
neighbor_pixels = np.array(neighbor_pixels)
H_VALUE = 0
S_VALUE = 0
V_VALUE = 0
X_VALUE = 0
Y_VALUE = 0
dict_data = {}
with open('data_1.pkl', 'rb') as f:
    dict_data = pickle.load(f)
h_range = []
s_range = []
v_range = []

h_mean_range = []
s_mean_range = []
v_mean_range = []

h_std_range = []
s_std_range = []
v_std_range = []

names = []
kind = []
for k, v in dict_data.items():
    for i in range(len(v['H'])):
        h_range.append(v['H'][i])
        s_range.append(v['S'][i])
        v_range.append(v['V'][i])

        h_mean_range.append(v['H_mean'][i])
        s_mean_range.append(v['S_mean'][i])
        v_mean_range.append(v['V_mean'][i])

        h_std_range.append(v['H_std'][i])
        s_std_range.append(v['S_std'][i])
        v_std_range.append(v['V_std'][i])

        names.append(v['name'][i])
        kind.append(k)
h_range = np.expand_dims(h_range, axis=0)
h_range = np.repeat(h_range, 64, axis=0)
h_range = np.expand_dims(h_range, axis=0)
h_range = np.repeat(h_range, 64, axis=0)
h_range = np.array(h_range, dtype=np.float64)

s_range = np.expand_dims(s_range, axis=0)
s_range = np.repeat(s_range, 64, axis=0)
s_range = np.expand_dims(s_range, axis=0)
s_range = np.repeat(s_range, 64, axis=0)
s_range = np.array(s_range, dtype=np.float64)

v_range = np.expand_dims(v_range, axis=0)
v_range = np.repeat(v_range, 64, axis=0)
v_range = np.expand_dims(v_range, axis=0)
v_range = np.repeat(v_range, 64, axis=0)
v_range = np.array(v_range, dtype=np.float64)

h_mean_range = np.expand_dims(h_mean_range, axis=0)
h_mean_range = np.repeat(h_mean_range, 64, axis=0)
h_mean_range = np.expand_dims(h_mean_range, axis=0)
h_mean_range = np.repeat(h_mean_range, 64, axis=0)
h_mean_range = np.array(h_mean_range, dtype=np.float64)

s_mean_range = np.expand_dims(s_mean_range, axis=0)
s_mean_range = np.repeat(s_mean_range, 64, axis=0)
s_mean_range = np.expand_dims(s_mean_range, axis=0)
s_mean_range = np.repeat(s_mean_range, 64, axis=0)
s_mean_range = np.array(s_mean_range, dtype=np.float64)

v_mean_range = np.expand_dims(v_mean_range, axis=0)
v_mean_range = np.repeat(v_mean_range, 64, axis=0)
v_mean_range = np.expand_dims(v_mean_range, axis=0)
v_mean_range = np.repeat(v_mean_range, 64, axis=0)
v_mean_range = np.array(v_mean_range, dtype=np.float64)


h_std_range = np.expand_dims(h_std_range, axis=0)
h_std_range = np.repeat(h_std_range, 64, axis=0)
h_std_range = np.expand_dims(h_std_range, axis=0)
h_std_range = np.repeat(h_std_range, 64, axis=0)
h_std_range = np.array(h_std_range, dtype=np.float64)


s_std_range = np.expand_dims(s_std_range, axis=0)
s_std_range = np.repeat(s_std_range, 64, axis=0)
s_std_range = np.expand_dims(s_std_range, axis=0)
s_std_range = np.repeat(s_std_range, 64, axis=0)
s_std_range = np.array(s_std_range, dtype=np.float64)

v_std_range = np.expand_dims(v_std_range, axis=0)
v_std_range = np.repeat(v_std_range, 64, axis=0)
v_std_range = np.expand_dims(v_std_range, axis=0)
v_std_range = np.repeat(v_std_range, 64, axis=0)
v_std_range = np.array(v_std_range, dtype=np.float64)

keyboard_map = {'c': 'tree_chunk', 't': 'tree_leave', 'r': 'rock', 's': 'sand', 'd': 'dirt',
                'u': 'not_common', 'g':'grass','b':'black','w': 'water', 'k':'sky'}
key_selected = ''
add_done = False
while (True):
    # Display result image
    color_range = np.zeros([128, max_size, 3], dtype=np.uint8)

    if pasue:
        # action['place'] = 'furnace'
        # action['place'] = 'torch'

        # action['nearbyCraft'] = 'iron_pickaxe'
        # action['nearbyCraft'] = 'stone_pickaxe'

        obs, reward, done, _ = env.step(action)
        t_R += reward
        # obs['inventory']
        # print(obs['inventory']['iron_ore'], end='\r')

        # record_p.append(obs['pov'])
    else:
        action = env.action_space.noop()
    # if action['jump'] == 1:
    #     action['jump'] = 0
    image = obs['pov']

    # Convert to HSV format and color threshold
    X = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


    tmp_mask = np.zeros([64, 64, 3], dtype=np.uint8)

    hsv_64 = np.array(hsv, dtype=np.float64)
    H_nei = np.zeros((64, 64, len(neighbor_pixels[0])), dtype=np.float64)
    S_nei = np.zeros((64, 64, len(neighbor_pixels[0])), dtype=np.float64)
    V_nei = np.zeros((64, 64, len(neighbor_pixels[0])), dtype=np.float64)

    for counter in range(len(neighbor_pixels[0])):
        new_hsv = shift_image(hsv_64, neighbor_pixels[0][counter],
                              neighbor_pixels[1][counter])
        zero_index = np.where(new_hsv[:, :, 0] == 1500)
        new_hsv[zero_index[0], zero_index[1], :] = hsv_64[zero_index[0], zero_index[1], :]
        x = np.abs(new_hsv[:,:,0] - hsv_64[:,:,0])
        y = np.abs(new_hsv[:,:,1] - hsv_64[:,:,1])
        zero_index = np.where((x >= 10) | (y >= 10))
        new_hsv[zero_index[0], zero_index[1], :] = hsv_64[zero_index[0], zero_index[1], :]

        H_nei[:, :, counter] = new_hsv[:, :, 0]
        S_nei[:, :, counter] = new_hsv[:, :, 1]
        V_nei[:, :, counter] = new_hsv[:, :, 2]

    h_mean_m = np.mean(H_nei, axis=-1)
    h_std_m = np.std(H_nei, axis=-1)

    s_mean_m = np.mean(S_nei, axis=-1)
    s_std_m = np.std(S_nei, axis=-1)

    v_mean_m = np.mean(V_nei, axis=-1)
    v_std_m = np.std(V_nei, axis=-1)

    h_mean = np.expand_dims(h_mean_m, axis=-1)
    s_mean = np.expand_dims(s_mean_m, axis=-1)
    v_mean = np.expand_dims(v_mean_m, axis=-1)

    h_std = np.expand_dims(h_std_m, axis=-1)
    s_std = np.expand_dims(s_std_m, axis=-1)
    v_std = np.expand_dims(v_std_m, axis=-1)

    h = np.expand_dims(hsv_64[:, :, 0], axis=-1)
    s = np.expand_dims(hsv_64[:, :, 1], axis=-1)
    v = np.expand_dims(hsv_64[:, :, 2], axis=-1)

    h_error = np.abs(h - h_range)
    index = np.where(h_error >= 89.5)
    h_error[index[0], index[1], index[2]] = 179 - h_error[index[0], index[1], index[2]]

    h_mean_error = np.abs(h_mean - h_mean_range)
    index = np.where(h_mean_error >= 89.5)
    h_mean_error[index[0], index[1], index[2]] = 179 - h_mean_error[index[0], index[1], index[2]]
    # h_error[29, 23]
    # s_range.shape
    # x = h_error + np.abs(s - s_range)/2 + np.abs(v - v_range)/2
    # x = h_mean_error + np.abs(s_mean - s_mean_range)/2 + np.abs(v_mean - v_mean_range)/2
    # x = np.abs(h_std - h_std_range) + np.abs(s_std - s_std_range)  + np.abs(v_std - v_std_range)
    # x[30,21]
    # # H_nei.shape
    # # H_nei.shape
    # v_mean_m[40,26]
    # # hsv[43,25]
    # # h_range[0,0,:]
    # # np.min(x[43,25])
    # s_mean[29,35]
    # s_range.shape
    # s_mean_range[29,35]
    # v_mean_range[0,0]
    # dict_data['dirt']['V_mean'][-1]
    # len(dict_data['dirt']['H'])
    # dict_data.keys()
    tmp_33 = h_error + np.abs(s - s_range)/2 + np.abs(v - v_range)/2 + \
             h_mean_error + np.abs(s_mean - s_mean_range)/2 + np.abs(v - v_mean_range)/2 + \
             (np.abs(h_std - h_std_range) + \
             np.abs(s_std - s_std_range) + np.abs(v_std - v_std_range) )

    if len(dict_data['rock']['H']) != 0:
        tmp_33 = np.argmin(tmp_33, axis=2)
        index_color = 0
        for i in range(64):
            for j in range(64):
                pixel = hsv[i, j, :]
                index_x = tmp_33[i,j]
                kins = kind[index_x]
                # if kins not in kindss:
                #     kindss[kins] = index_color
                #     index_color += 1

                # tmp_mask[i, j, :] = np.array([hsv_r[names[index_x]]['H'],
                #                               hsv_r[names[index_x]]['S'],
                #                               hsv_r[names[index_x]]['v']], dtype=np.uint8)

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
    # tmp_33[29,22]
    if not lock and not pasue:
        a = 3
        if key_on_press in keyboard_map and key_on_release != 'l':
            key_selected = keyboard_map[key_on_press]
            print("new: key {0} - {1} - {2}".format(key_selected, [hsv_64[X_VALUE,Y_VALUE, 0],
                                                             hsv_64[X_VALUE,Y_VALUE, 1],
                                                             hsv_64[X_VALUE,Y_VALUE, 2]], [X_VALUE, Y_VALUE]))
            key_on_press = ''
            add_done = False

        if key_selected != '':
            if key_on_press in indexs:
                if int(key_on_press) == 1:
                    dict_data[key_selected]['H'].append(hsv_64[X_VALUE,Y_VALUE,0])
                    dict_data[key_selected]['S'].append(hsv_64[X_VALUE,Y_VALUE,1])
                    dict_data[key_selected]['V'].append(hsv_64[X_VALUE,Y_VALUE,2])

                    dict_data[key_selected]['H_mean'].append(h_mean_m[X_VALUE,Y_VALUE])
                    dict_data[key_selected]['S_mean'].append(s_mean_m[X_VALUE,Y_VALUE])
                    dict_data[key_selected]['V_mean'].append(v_mean_m[X_VALUE,Y_VALUE])

                    dict_data[key_selected]['H_std'].append(h_std_m[X_VALUE,Y_VALUE])
                    dict_data[key_selected]['S_std'].append(s_std_m[X_VALUE,Y_VALUE])
                    dict_data[key_selected]['V_std'].append(v_std_m[X_VALUE,Y_VALUE])

                    name = '{0}_{1}'.format(key_selected, len(dict_data[key_selected]['H']))

                    dict_data[key_selected]['name'].append(name)

                    print("ADDED : key {0} - {1} - {2} - {3}".format(key_selected, [H_VALUE, S_VALUE, V_VALUE],
                                                               [X_VALUE, Y_VALUE], [h_mean_m[X_VALUE,Y_VALUE],
                                                                                    s_mean_m[X_VALUE,Y_VALUE],
                                                                                    v_mean_m[X_VALUE,Y_VALUE]]))
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
                    dict_data[key_selected]['name'] = dict_data[key_selected]['name'][:-1]

                    dict_data[key_selected]['H_mean']= dict_data[key_selected]['H_mean'][:-1]
                    dict_data[key_selected]['S_mean']= dict_data[key_selected]['S_mean'][:-1]
                    dict_data[key_selected]['V_mean']= dict_data[key_selected]['V_mean'][:-1]

                    dict_data[key_selected]['H_std']= dict_data[key_selected]['H_std'][:-1]
                    dict_data[key_selected]['S_std']= dict_data[key_selected]['S_std'][:-1]
                    dict_data[key_selected]['V_std']= dict_data[key_selected]['V_std'][:-1]

                    print("DEL : key {0} - {1}".format(key_selected, [h_del, s_del, v_del]))
                    # key_selected = ''
                    add_done = False
                    with open('data_1.pkl', 'wb') as f:
                        pickle.dump(dict_data, f)

                h_range = []
                s_range = []
                v_range = []

                h_mean_range = []
                s_mean_range = []
                v_mean_range = []

                h_std_range = []
                s_std_range = []
                v_std_range = []

                names = []
                kind = []
                for k, v in dict_data.items():
                    # if k == 'dirt':
                    for i in range(len(v['H'])):
                        h_range.append(v['H'][i])
                        s_range.append(v['S'][i])
                        v_range.append(v['V'][i])

                        h_mean_range.append(v['H_mean'][i])
                        s_mean_range.append(v['S_mean'][i])
                        v_mean_range.append(v['V_mean'][i])

                        h_std_range.append(v['H_std'][i])
                        s_std_range.append(v['S_std'][i])
                        v_std_range.append(v['V_std'][i])

                        names.append(v['name'][i])
                        kind.append(k)
                        # print(v['H_mean'][i])

                print("---- {0} ---".format(len(v_std_range)))

                h_range = np.expand_dims(h_range, axis=0)
                h_range = np.repeat(h_range, 64, axis=0)
                h_range = np.expand_dims(h_range, axis=0)
                h_range = np.repeat(h_range, 64, axis=0)
                h_range = np.array(h_range, dtype=np.float64)

                s_range = np.expand_dims(s_range, axis=0)
                s_range = np.repeat(s_range, 64, axis=0)
                s_range = np.expand_dims(s_range, axis=0)
                s_range = np.repeat(s_range, 64, axis=0)
                s_range = np.array(s_range, dtype=np.float64)

                v_range = np.expand_dims(v_range, axis=0)
                v_range = np.repeat(v_range, 64, axis=0)
                v_range = np.expand_dims(v_range, axis=0)
                v_range = np.repeat(v_range, 64, axis=0)
                v_range = np.array(v_range, dtype=np.float64)

                h_mean_range = np.expand_dims(h_mean_range, axis=0)
                h_mean_range = np.repeat(h_mean_range, 64, axis=0)
                h_mean_range = np.expand_dims(h_mean_range, axis=0)
                h_mean_range = np.repeat(h_mean_range, 64, axis=0)
                h_mean_range = np.array(h_mean_range, dtype=np.float64)

                s_mean_range = np.expand_dims(s_mean_range, axis=0)
                s_mean_range = np.repeat(s_mean_range, 64, axis=0)
                s_mean_range = np.expand_dims(s_mean_range, axis=0)
                s_mean_range = np.repeat(s_mean_range, 64, axis=0)
                s_mean_range = np.array(s_mean_range, dtype=np.float64)

                v_mean_range = np.expand_dims(v_mean_range, axis=0)
                v_mean_range = np.repeat(v_mean_range, 64, axis=0)
                v_mean_range = np.expand_dims(v_mean_range, axis=0)
                v_mean_range = np.repeat(v_mean_range, 64, axis=0)
                v_mean_range = np.array(v_mean_range, dtype=np.float64)

                h_std_range = np.expand_dims(h_std_range, axis=0)
                h_std_range = np.repeat(h_std_range, 64, axis=0)
                h_std_range = np.expand_dims(h_std_range, axis=0)
                h_std_range = np.repeat(h_std_range, 64, axis=0)
                h_std_range = np.array(h_std_range, dtype=np.float64)

                s_std_range = np.expand_dims(s_std_range, axis=0)
                s_std_range = np.repeat(s_std_range, 64, axis=0)
                s_std_range = np.expand_dims(s_std_range, axis=0)
                s_std_range = np.repeat(s_std_range, 64, axis=0)
                s_std_range = np.array(s_std_range, dtype=np.float64)

                v_std_range = np.expand_dims(v_std_range, axis=0)
                v_std_range = np.repeat(v_std_range, 64, axis=0)
                v_std_range = np.expand_dims(v_std_range, axis=0)
                v_std_range = np.repeat(v_std_range, 64, axis=0)
                v_std_range = np.array(v_std_range, dtype=np.float64)

                add_done = True
                key_on_press = ''
        #image = cv2.imread("./block_image/Mega_Spruce_Tree.png")

    # pixels_maps = {}
    # keys_mask = {}
    # key_index = 0
    #
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
    #         V_error = np.abs(hsv_64[neigh_bor[0], neigh_bor[1], 2] - pixel_hsv[2])
    #         if HS_error_min <= 3:
    #             key = None
    #             for ins in range(len(HS_error)):
    #                 if np.abs(HS_error[ins] - HS_error_min) <= 3:
    #
    #                     next_pixel = tuple((neigh_bor[0][ins], neigh_bor[1][ins]))
    #                     if key is None:
    #                         if pixel_id in pixels_maps:
    #                             key = pixels_maps[pixel_id]
    #
    #                         elif next_pixel in pixels_maps:
    #                             key = pixels_maps[next_pixel]
    #                         else:
    #                             key = key_index
    #                             keys_mask[key] = [[], [], {}, [], []]
    #                             key_index += 1
    #                     # print(key)
    #                     pixels_maps[pixel_id] = key
    #                     pixels_maps[next_pixel] = key
    #                     if pixel_id not in keys_mask[key][2]:
    #                         keys_mask[key][2][pixel_id] = 0
    #                         if np.abs(V_error[ins]) <= 255:
    #                             keys_mask[key][0].append(pixel_id[0])
    #                             keys_mask[key][1].append(pixel_id[1])
    #                         else:
    #                             keys_mask[key][3].append(pixel_id[0])
    #                             keys_mask[key][4].append(pixel_id[1])
    #
    #                     if next_pixel not in keys_mask[key][2]:
    #                         keys_mask[key][2][next_pixel] = 0
    #                         if np.abs(V_error[ins]) <= 255:
    #                             keys_mask[key][0].append(next_pixel[0])
    #                             keys_mask[key][1].append(next_pixel[1])
    #                         else:
    #                             keys_mask[key][3].append(next_pixel[0])
    #                             keys_mask[key][4].append(next_pixel[1])
    #
    # #print(len(keys_mask))
    # max = 0
    # k_m = 0
    # for k,v in keys_mask.items():
    #
    #     tmp_mask[v[0], v[1],0] = np.mean(hsv[v[0], v[1],0])
    #     tmp_mask[v[0], v[1],1] = np.mean(hsv[v[0], v[1],1])
    #     tmp_mask[v[0], v[1],2] = np.mean(hsv[v[0], v[1],2])
    #     tmp_mask[v[3], v[4], 0] = np.mean(hsv[v[3], v[4], 0])
    #     tmp_mask[v[3], v[4], 1] = np.mean(hsv[v[3], v[4], 1])
    #     tmp_mask[v[3], v[4], 2] = np.mean(hsv[v[3], v[4], 2]) + 20

    # print(np.count_nonzero(names == 32)/4096, end='\r')
    # mask = 255 - mask
    # hsv[:,:,2] = 255
    # X = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)



    # X = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # dst = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 0.7,
    #                       cv2.cvtColor(tmp_mask, cv2.COLOR_HSV2RGB), 0.7, 0)
    tmp_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tmp_image[X_VALUE,Y_VALUE,:] = [255,255,255]
    cv2.imshow('image',tmp_image)
    cv2.resizeWindow('image', 800, 800)

    cv2.setMouseCallback('image', Life2CodingRGB)

    cv2.imshow('mask', cv2.cvtColor(tmp_mask, cv2.COLOR_HSV2RGB))
    cv2.resizeWindow('mask', 800, 800)
    cv2.setMouseCallback('mask', Life2CodingRGB)


    cv2.imshow('color_range', cv2.cvtColor(color_range, cv2.COLOR_HSV2RGB))
    cv2.resizeWindow('mask', 800, 800)

    # for c in cnts:
    #     x, y, w, h = cv2.boundingRect(c)
    #
    #     print(cv2.contourArea(c))
    # print("---")
    # image = obs['pov']
    # cv2.imshow('image', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if cv2.waitKey(10) & 0xFF == ord('o'):
        break
    time.sleep(0.1)

key_list.stop()
mouse_list.stop()

cv2.destroyAllWindows()
hsv[16,60]
with open('data_1.pkl', 'wb') as f:
    pickle.dump(dict_data, f)


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

hsv_r = {}
hsv_r['sky'] = {'H': 109, 'range_H': [48, 54], 'S': 80, 'range_S': [125, 140], 'v':255,'kind': 'sky'}
hsv_r['sky_2'] = {'H': 112, 'range_H': [48, 54], 'S': 29, 'range_S': [125, 140], 'v':255,'kind': 'sky'}
hsv_r['sky_3'] = {'H': 109.75, 'S': 117.36,'v':175, 'kind': 'sky'}
hsv_r['sky_sunset'] = {'H': 4.03, 'S': 97.36,'v':94, 'kind': 'sky'}
hsv_r['sky_sunset_2'] = {'H': 122, 'S': 57,'v':103.3, 'kind': 'sky'}
hsv_r['sky_sunset_3'] = {'H': 144, 'S': 41,'v':89.77, 'kind': 'sky'}
hsv_r['sky_sunset_4'] = {'H': 5.69, 'S': 140,'v':172.77, 'kind': 'sky'}
hsv_r['sky_sunset_5'] = {'H': 170, 'S': 65,'v':122.8, 'kind': 'sky'}

hsv_r['dirt'] = {'H': 13, 'range_H': [48, 54], 'S': 126, 'range_S': [125, 140],'v': 130, 'kind': 'dirt'}
hsv_r['dirt_2'] = {'H': 13, 'S': 126.266,'v': 76.806, 'kind': 'dirt'}
hsv_r['dirt_3'] = {'H': 14, 'S': 134.0,'v': 76.806, 'kind': 'dirt'}
hsv_r['dirt_4'] = {'H': 15, 'S': 130.266,'v': 76.806, 'kind': 'dirt'}
hsv_r['dirt_5'] = {'H': 18.0, 'S': 126.4, 'v': 68.9, 'kind': 'dirt'}
hsv_r['dirt_6'] = {'H': 23.0, 'S': 136.4, 'v': 55.9, 'kind': 'dirt'}
hsv_r['dirt_7'] = {'H': 12.0, 'S': 85.4, 'v': 54.9, 'kind': 'dirt'}
hsv_r['dirt_8'] = {'H': 15.0, 'S': 125.4, 'v': 49.9, 'kind': 'dirt'}
hsv_r['dirt_10'] = {'H': 15.0, 'S': 131.4, 'v': 39.9, 'kind': 'dirt'}
hsv_r['dirt_11'] = {'H': 17.0, 'S': 128.0, 'v': 40, 'kind': 'dirt'}
hsv_r['dirt_12'] = {'H': 12.0, 'S': 108.0, 'v': 68, 'kind': 'dirt'}
hsv_r['dirt_13'] = {'H': 13, 'S': 104,'v': 86, 'kind': 'dirt'}
hsv_r['dirt_14'] = {'H': 13, 'S': 102,'v': 65, 'kind': 'dirt'}
hsv_r['dirt_15'] = {'H': 12, 'S': 104,'v': 54, 'kind': 'dirt'}
hsv_r['dirt_16'] = {'H': 13, 'S': 117,'v': 61, 'kind': 'dirt'}
hsv_r['dirt_17'] = {'H': 13, 'S': 107,'v': 50, 'kind': 'dirt'}
hsv_r['dirt_18'] = {'H': 17, 'S': 129,'v': 87, 'kind': 'dirt'}
hsv_r['dirt_20'] = {'H': 18, 'S': 133,'v': 50, 'kind': 'dirt'}
hsv_r['dirt_21'] = {'H': 17, 'S': 125,'v': 57, 'kind': 'dirt'}
hsv_r['dirt_22'] = {'H': 12, 'S': 68,'v': 56, 'kind': 'dirt'}
hsv_r['dirt_23'] = {'H': 12, 'S': 58,'v': 57, 'kind': 'dirt'}
hsv_r['dirt_24'] = {'H': 6, 'S': 96,'v': 37, 'kind': 'dirt'}
hsv_r['dirt_25'] = {'H': 10, 'S': 107,'v': 57, 'kind': 'dirt'}
hsv_r['dirt_26'] = {'H': 8, 'S': 99,'v': 77, 'kind': 'dirt'}
hsv_r['dirt_27'] = {'H': 150, 'S': 70,'v': 22, 'kind': 'dirt'}
hsv_r['dirt_28'] = {'H': 172, 'S': 74,'v': 38, 'kind': 'dirt'}

hsv_r['dirt_29'] = {'H': 4, 'S': 88,'v': 61, 'kind': 'dirt'}


hsv_r['water_2'] = {'H': 115.89, 'S': 149.04,'v': 129.66, 'kind': 'water'}
hsv_r['water_3'] = {'H': 119.89, 'S': 125.04,'v': 134.66, 'kind': 'water'}

hsv_r['water'] = {'H': 116.89, 'range_H': [48, 54], 'S': 173.04, 'range_S': [125, 140],'v': 228.66, 'kind': 'water'}
hsv_r['sand'] = {'H': 25.98, 'range_H': [48, 54], 'S': 68.58, 'range_S': [125, 140],'v': 201.44, 'kind': 'sand'}
hsv_r['sand_2'] = {'H': 28, 'S': 19,'v': 248, 'kind': 'sand'}
hsv_r['sand_3'] = {'H': 10, 'S': 13,'v': 57, 'kind': 'sand'}
hsv_r['sand_4'] = {'H': 18, 'S': 21,'v': 61, 'kind': 'sand'}
hsv_r['sand_5'] = {'H': 135, 'S': 25,'v': 41, 'kind': 'sand'}
hsv_r['sand_6'] = {'H': 165, 'S': 10,'v': 49, 'kind': 'sand'}
hsv_r['sand_7'] = {'H': 0, 'S': 10,'v': 51, 'kind': 'sand'}
hsv_r['sand_8'] = {'H': 126, 'S': 38,'v': 34, 'kind': 'rock'}

# hsv_r['dirt_water'] = {'H': 11.2, 'range_H': [48, 54], 'S': 11.6, 'range_S': [125, 140],'v': 48, 'kind': 'dirt'}
hsv_r['dirt_far'] = {'H':21.94, 'range_H': [48, 54], 'S': 123.6, 'range_S': [125, 140],'v': 75.49, 'kind': 'dirt'}
hsv_r['rock'] = {'H': 9.15, 'S': 0.825,'v': 58.4, 'kind': 'rock'}
hsv_r['b'] = {'H': 0, 'S': 0,'v': 0, 'kind': 'black'}
# hsv_r['sand_water'] = {'H': 117, 'range_H': [48, 54], 'S': 142, 'range_S': [125, 140],'v': 216.8, 'kind': 'sand_water'}
hsv_r['rock_water'] = {'H': 116.3, 'range_H': [48, 54], 'S': 164.11, 'range_S': [125, 140],'v': 222.0, 'kind': 'water'}
hsv_r['dirt_water'] = {'H': 119.7, 'range_H': [48, 54], 'S': 157.2, 'range_S': [125, 140],'v': 188.0, 'kind': 'water'}


hsv_r['Cactus'] = {'H': 63, 'range_H': [47, 54], 'S': 221, 'range_S': [168, 194], 'v': 71, 'kind': 'not_common'}
hsv_r['bamboo'] = {'H': 37.39, 'range_H': [47, 54], 'S': 195.71, 'range_S': [168, 194], 'v': 136.89, 'kind': 'not_common'}
hsv_r['yellow_flower'] = {'H': 31, 'range_H': [47, 54], 'S': 253.25, 'range_S': [168, 194], 'v': 219, 'kind': 'not_common'}
hsv_r['pink_flower'] = {'H': 146.63, 'range_H': [47, 54], 'S': 54.1, 'range_S': [168, 194], 'v': 185.5, 'kind': 'not_common'}
hsv_r['red_flower'] = {'H': 0, 'range_H': [47, 54], 'S': 247, 'range_S': [168, 194], 'v': 163, 'kind': 'not_common'}
hsv_r['red_flower_under'] = {'H':55, 'range_H': [47, 54], 'S':255, 'range_S': [168, 194], 'v': 51.26, 'kind': 'not_common'}


hsv_r['tree_50'] = {'H': 50.69, 'range_H': [47, 54], 'S': 180.93, 'range_S': [168, 194], 'v': 44, 'kind': 'tree_leave'}

hsv_r['chunk_50'] = {'H': 18, 'range_H': [16, 20], 'S': 131.2, 'range_S': [120, 140], 'v': 57.5, 'range_v': [30, 70],
                        'kind': 'tree_chunk'}

hsv_r['tree_44'] = {'H':44.25, 'range_H': [47, 54], 'S':125.7, 'range_S': [168, 194], 'v': 51.26, 'kind': 'tree_leave'}
hsv_r['tree_44_185'] = {'H': 44.46, 'S': 185.68, 'v': 39.63, 'kind': 'tree_leave'}
hsv_r['chunk44_185'] = {'H': 17.5, 'S': 144.7, 'v': 35.4, 'range_v': [30, 70],
                        'kind': 'tree_chunk'}
hsv_r['chunk44_185_inside'] = {'H': 17.0, 'S': 131.4, 'v': 32.4+20, 'range_v': [30, 70],
                        'kind': 'tree_chunk'}
hsv_r['chunk44_185_inside_2'] = {'H': 17.2, 'S': 125.4, 'v': 32.4, 'range_v': [30, 70], 'kind': 'tree_chunk'}
hsv_r['tree_green_not'] = {'H':17.83, 'S':104.0, 'v': 99.9, 'kind': 'tree_leave'}
hsv_r['tree_green_not_chunk'] = {'H':22.1, 'S':17.1, 'v': 131.5, 'kind': 'tree_chunk'}
hsv_r['tree_swamp_oak'] = {'H': 33.4, 'S': 125.58, 'v': 25.83, 'kind': 'tree_leave'}
hsv_r['chunk_swamp_oak'] = {'H': 18.0, 'S': 131.4, 'v': 76.9, 'kind': 'tree_chunk'}
hsv_r['leave_36_swamp'] = {'H': 33.177, 'S': 126.5, 'v': 75.13,'kind': 'tree_leave'}
hsv_r['tree_spruce'] = {'H': 60, 'S': 93.94, 'v': 33.3, 'kind': 'tree_leave'}
hsv_r['tree_spruce_chunk'] = {'H': 15.36,'S': 183.39, 'v': 24.82, 'kind': 'tree_chunk'}
hsv_r['tree_spruce_2'] = {'H': 70, 'S': 83, 'v': 37, 'kind': 'tree_leave'}
hsv_r['tree_spruce_3'] = {'H': 95, 'S': 65, 'v': 73, 'kind': 'tree_leave'}
hsv_r['tree_spruce_4'] = {'H': 60, 'S': 73, 'v': 7, 'kind': 'tree_leave'}
hsv_r['tree_spruce_5'] = {'H': 60, 'S': 128, 'v': 2, 'kind': 'tree_leave'}
hsv_r['tree_spruce_chunk_2'] = {'H': 12,'S': 54, 'v': 47, 'kind': 'tree_chunk'}
hsv_r['tree_spruce_chunk_3'] = {'H': 15,'S': 107, 'v': 38, 'kind': 'tree_chunk'}

hsv_r['grass_36_swamp'] = {'H': 36.177, 'S': 126.5, 'v': 64.13,'kind': 'grass'}
hsv_r['grass_51_swamp'] = {'H': 51.76, 'S': 127.105, 'v': 64.848, 'kind': 'grass'}
hsv_r['grass_44'] = {'H': 44.177, 'range_H': [48, 54], 'S': 138.7, 'range_S': [125, 140], 'v': 86.33,'kind': 'grass'}
hsv_r['grass_51'] = {'H': 50.97, 'range_H': [48, 54], 'S': 132.5, 'range_S': [125, 140], 'v': 102,'kind': 'grass'}
hsv_r['grass_taiga'] = {'H': 58.24, 'S': 73.26, 'v': 101.75,'kind': 'grass'}
hsv_r['grass_taiga_2'] = {'H': 58.24, 'S': 73.26, 'v': 47.07,'kind': 'grass'}
hsv_r['grass_1'] = {'H': 28, 'S': 127, 'v': 52,'kind': 'grass'}
hsv_r['grass_2'] = {'H': 13, 'S': 107, 'v': 62,'kind': 'grass'}
hsv_r['grass_3'] = {'H': 51, 'S': 94, 'v': 84,'kind': 'grass'}
hsv_r['grass_4'] = {'H': 35, 'S': 124, 'v': 80,'kind': 'grass'}
hsv_r['grass_5'] = {'H': 32, 'S': 88, 'v': 78,'kind': 'grass'}
hsv_r['grass_6'] = {'H': 34, 'S': 131, 'v': 68,'kind': 'grass'}
hsv_r['grass_7'] = {'H': 33, 'S': 126, 'v': 71,'kind': 'grass'}
hsv_r['grass_8'] = {'H': 21, 'S': 108, 'v': 71,'kind': 'grass'}
hsv_r['grass_9'] = {'H': 60, 'S': 70, 'v': 33,'kind': 'grass'}
hsv_r['grass_10'] = {'H': 43, 'S': 134, 'v': 40,'kind': 'grass'}


hsv_r['rock_1'] = {'H': 15, 'S': 8,'v': 122, 'kind': 'rock'}
hsv_r['rock_2'] = {'H': 30, 'S': 2,'v': 103, 'kind': 'rock'}
hsv_r['rock_3'] = {'H': 17, 'S': 19, 'v': 93, 'kind': 'rock'}
hsv_r['rock_4'] = {'H': 23, 'S': 8,'v': 131, 'kind': 'rock'}
hsv_r['rock_5'] = {'H': 20, 'S': 15,'v': 104, 'kind': 'rock'}
hsv_r['rock_6'] = {'H': 4, 'S': 22,'v': 93, 'kind': 'rock'}
hsv_r['rock_7'] = {'H': 6, 'S': 46,'v': 88, 'kind': 'rock'}
hsv_r['rock_8'] = {'H': 123, 'S': 45,'v': 57, 'kind': 'rock'}
hsv_r['rock_9'] = {'H': 170, 'S': 9,'v': 89, 'kind': 'rock'}
hsv_r['rock_10'] = {'H': 120, 'S': 68,'v': 41, 'kind': 'rock'}

hsv_r['sand_8'] = {'H': 126, 'S': 38,'v': 34, 'kind': 'sand'}
hsv_r['sand_9'] = {'H': 125, 'S': 49,'v': 31, 'kind': 'sand'}
hsv_r['sand_10'] = {'H': 23, 'S': 34,'v': 89, 'kind': 'sand'}
hsv_r['sand_11'] = {'H': 123, 'S': 79,'v': 29, 'kind': 'sand'}
hsv_r['dirt_30'] = {'H': 139, 'S': 102,'v': 20, 'kind': 'dirt'}
hsv_r['sand_12'] = {'H': 120, 'S': 41,'v': 31, 'kind': 'sand'}
hsv_r['rock_11'] = {'H': 120, 'S': 108,'v': 26, 'kind': 'rock'}
hsv_r['dirt_31'] = {'H': 143, 'S': 93,'v': 22, 'kind': 'dirt'}
hsv_r['dirt_32'] = {'H': 141, 'S': 78,'v': 23, 'kind': 'dirt'}
hsv_r['dirt_33'] = {'H': 146, 'S': 85,'v': 21, 'kind': 'dirt'}
hsv_r['dirt_34'] = {'H': 141, 'S': 89,'v': 20, 'kind': 'dirt'}
hsv_r['dirt_35'] = {'H': 141, 'S': 85,'v': 21, 'kind': 'dirt'}
hsv_r['rock_12'] = {'H': 120, 'S': 43,'v': 59, 'kind': 'rock'}
hsv_r['rock_13'] = {'H': 123, 'S': 83,'v': 37, 'kind': 'rock'}
hsv_r['rock_14'] = {'H': 123, 'S': 74,'v': 38, 'kind': 'rock'}
hsv_r['rock_15'] = {'H': 123, 'S': 58,'v': 53, 'kind': 'rock'}
hsv_r['rock_16'] = {'H': 120, 'S': 78,'v': 36, 'kind': 'rock'}
hsv_r['grass_11'] = {'H': 69, 'S': 58, 'v': 31,'kind': 'grass'}
hsv_r['grass_12'] = {'H': 95, 'S': 70, 'v': 22,'kind': 'grass'}
hsv_r['rock_17'] = {'H': 120, 'S': 20,'v': 50, 'kind': 'rock'}
hsv_r['rock_18'] = {'H': 135, 'S': 15,'v': 34, 'kind': 'rock'}
hsv_r['dirt_36'] = {'H': 3, 'S': 56,'v': 41, 'kind': 'dirt'}
hsv_r['dirt_37'] = {'H': 30, 'S': 56,'v': 32, 'kind': 'dirt'}
hsv_r['dirt_38'] = {'H': 16, 'S': 131,'v': 64, 'kind': 'dirt'}
hsv_r['dirt_39'] = {'H': 17, 'S': 137,'v': 52, 'kind': 'dirt'}
hsv_r['rock_19'] = {'H': 40, 'S': 28,'v': 81, 'kind': 'rock'}
hsv_r['dirt_40'] = {'H': 18, 'S': 130,'v': 63, 'kind': 'dirt'}
hsv_r['grass_13'] = {'H': 53, 'S': 93, 'v': 33,'kind': 'grass'}
hsv_r['sand_13'] = {'H': 27, 'S': 92,'v': 166, 'kind': 'sand'}
hsv_r['grass_14'] = {'H': 28, 'S': 142, 'v': 108,'kind': 'grass'}
hsv_r['tree_acacia'] = {'H': 27, 'S': 197, 'v': 44, 'kind': 'tree_leave'}
hsv_r['tree_acacia_chunk'] = {'H': 19, 'S': 41, 'v': 68, 'kind': 'tree_chunk'}
hsv_r['tree_acacia_chunk_2'] = {'H': 21, 'S': 38, 'v': 88, 'kind': 'tree_chunk'}
hsv_r['tree_acacia_chunk_3'] = {'H': 27, 'S': 68, 'v': 97, 'kind': 'tree_chunk'}
hsv_r['tree_acacia_chunk_3_inside'] = {'H': 8, 'S': 151, 'v': 145, 'kind': 'tree_chunk'}
hsv_r['dirt_41'] = {'H': 27, 'S': 147,'v': 33, 'kind': 'dirt'}
hsv_r['dirt_42'] = {'H': 17, 'S': 130,'v': 57, 'kind': 'dirt'}
hsv_r['sand_14'] = {'H': 25, 'S': 70,'v': 134, 'kind': 'sand'}
hsv_r['sand_15'] = {'H': 25, 'S': 68,'v': 94, 'kind': 'sand'}
hsv_r['tree_44_chunk'] = {'H':30, 'S':7, 'v': 174, 'kind': 'tree_chunk'}
hsv_r['tree_birch_2'] = {'H':44, 'S':132, 'v': 29, 'kind': 'tree_leave'}
hsv_r['rock_20'] = {'H': 0, 'S': 0,'v': 86, 'kind': 'rock'}
hsv_r['tree_birch_chunk_2'] = {'H':30, 'S':6, 'v': 125, 'kind': 'tree_chunk'}
hsv_r['tree_birch_chunk_3'] = {'H':24, 'S':8, 'v': 169, 'kind': 'tree_chunk'}
hsv_r['tree_birch_chunk_4'] = {'H':30, 'S':6, 'v': 80, 'kind': 'tree_chunk'}
hsv_r['tree_birch_chunk_5'] = {'H':0, 'S':0, 'v': 207, 'kind': 'tree_chunk'}
hsv_r['tree_birch_chunk_6'] = {'H':15, 'S':11, 'v': 183, 'kind': 'tree_chunk'}
hsv_r['tree_birch_chunk_7'] = {'H':73, 'S':35, 'v': 51, 'kind': 'tree_chunk'}
hsv_r['tree_birch_chunk_8'] = {'H':43, 'S':82, 'v': 28, 'kind': 'tree_chunk'}
hsv_r['tree_birch_chunk_9'] = {'H':20, 'S':8, 'v': 99, 'kind': 'tree_chunk'}
hsv_r['tree_birch_chunk_10'] = {'H':24, 'S':10, 'v': 134, 'kind': 'tree_chunk'}
hsv_r['tree_birch_chunk_11'] = {'H':10, 'S':29, 'v': 52, 'kind': 'tree_chunk'}
hsv_r['tree_birch_chunk_12'] = {'H':173, 'S':13, 'v': 52, 'kind': 'tree_chunk'}
hsv_r['tree_birch_chunk_13'] = {'H':15, 'S':10, 'v': 49, 'kind': 'tree_chunk'}
hsv_r['tree_birch_chunk_14'] = {'H':53, 'S':17, 'v': 184, 'kind': 'tree_chunk'}
hsv_r['tree_birch_chunk_15'] = {'H':18, 'S':9, 'v': 148, 'kind': 'tree_chunk'}
hsv_r['tree_birch_chunk_16'] = {'H':170, 'S':6, 'v': 120, 'kind': 'tree_chunk'}
hsv_r['tree_birch_chunk_17'] = {'H':20, 'S':12, 'v': 126, 'kind': 'tree_chunk'}
hsv_r['tree_birch_chunk_18'] = {'H':0, 'S':0, 'v': 144, 'kind': 'tree_chunk'}
hsv_r['tree_birch_chunk_19'] = {'H':23, 'S':10, 'v': 103, 'kind': 'tree_chunk'}
hsv_r['chunk_oak_2'] = {'H': 18, 'S': 130.7, 'v': 53.5,'kind': 'tree_chunk'}
hsv_r['chunk_oak_3'] = {'H': 18, 'S': 131, 'v': 66,'kind': 'tree_chunk'}
hsv_r['chunk_oak_4'] = {'H': 18, 'S': 127, 'v': 46,'kind': 'tree_chunk'}
hsv_r['chunk_oak_5'] = {'H': 18, 'S': 129, 'v': 67,'kind': 'tree_chunk'}
hsv_r['chunk_oak_6'] = {'H': 18, 'S': 130, 'v': 57,'kind': 'tree_chunk'}
hsv_r['chunk_oak_7'] = {'H': 18, 'S': 132, 'v': 52,'kind': 'tree_chunk'}
hsv_r['chunk_oak_8'] = {'H': 19, 'S': 134, 'v': 55,'kind': 'tree_chunk'}
hsv_r['chunk_oak_9'] = {'H': 18, 'S': 134, 'v': 42,'kind': 'tree_chunk'}
hsv_r['chunk_oak_10'] = {'H': 17, 'S': 130, 'v': 51,'kind': 'tree_chunk'}
hsv_r['rock_21'] = {'H': 0, 'S': 0,'v': 133, 'kind': 'rock'}
hsv_r['rock_22'] = {'H': 0, 'S': 0,'v': 41, 'kind': 'rock'}
hsv_r['rock_23'] = {'H': 30, 'S': 17,'v': 88, 'kind': 'rock'}



h_range = []
s_range = []
v_range = []
names = []
kind = []
for k,v in hsv_r.items():
    h_range.append(v['H'])
    s_range.append(v['S'])
    v_range.append(v['v'])
    names.append(k)
    kind.append(v['kind'])
h_range = np.expand_dims(h_range,axis=0)
h_range = np.repeat(h_range, 64, axis=0)
h_range = np.expand_dims(h_range,axis=0)
h_range = np.repeat(h_range, 64, axis=0)

s_range = np.expand_dims(s_range,axis=0)
s_range = np.repeat(s_range, 64, axis=0)
s_range = np.expand_dims(s_range,axis=0)
s_range = np.repeat(s_range, 64, axis=0)
h_range = np.array(h_range, dtype=np.float64)
s_range = np.array(s_range, dtype=np.float64)

v_range = np.expand_dims(v_range,axis=0)
v_range = np.repeat(v_range, 64, axis=0)
v_range = np.expand_dims(v_range,axis=0)
v_range = np.repeat(v_range, 64, axis=0)
v_range = np.array(v_range, dtype=np.float64)

len(hsv_r)

#

# dict_data = {}
# for k,v in hsv_r.items():
#     kind = v['kind']
#     if v['kind'] not in dict_data:
#         dict_data[kind] = {'H': [], 'S': [], 'V': [],
#                            'H_mean': [], 'S_mean': [], 'V_mean': [],
#                            'H_std': [], 'S_std': [], 'V_std': [],
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




