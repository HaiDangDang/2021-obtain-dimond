import numpy as np

hsv_bond = {'normal':  {}, 'dark': {}, 'mix': {}}

hsv_bond['normal'] = {'tree': {}, 'tree_2': {}, 'tree_2_2': {},'mine': {}, 'land': {}, 'land_3': {}, 'tree_3': {}, 'item': {}, 'land_2': {}, 'useless_tree': {},  'animal': {},'monster': {}, 'land_5': {}}

hsv_bond['normal']['land_2']['black'] = {'hMin': 0, 'sMin': 0, 'hMax': 0, 'sMax': 0, 'vMin': 0, 'vMax': 0}
hsv_bond['normal']['land_2']['white'] = {'hMin': 0, 'sMin': 0, 'hMax': 0, 'sMax': 0, 'vMin': 255, 'vMax': 255}
hsv_bond['normal']['land_2']['magma'] = {'hMin': 8, 'sMin': 200, 'hMax': 23, 'sMax': 248, 'vMin': 120, 'vMax': 240}

hsv_bond['normal']['land_2']['sky'] = {'hMin': 104, 'sMin': 78, 'hMax': 112, 'sMax': 129}
hsv_bond['normal']['land_2']['dirt'] = {'hMin': 11, 'sMin': 49, 'hMax': 14, 'sMax': 140}
hsv_bond['normal']['land_2']['dirt_far'] = {'hMin': 16, 'sMin': 112, 'hMax': 30, 'sMax': 138, 'vMin': 24, 'vMax': 106}
hsv_bond['normal']['land_2']['dirt_water'] = {'hMin': 145, 'sMin': 45, 'hMax': 179, 'sMax': 78, 'vMin': 5, 'vMax': 62}

hsv_bond['normal']['land_2']['sand'] = {'hMin': 23, 'sMin': 49, 'hMax': 29, 'sMax': 83,'vMin': 68, 'vMax': 255}
hsv_bond['normal']['land_2']['sand_water'] = {'hMin': 23, 'sMin': 7, 'hMax': 29, 'sMax': 67,'vMin': 15, 'vMax': 155}

hsv_bond['normal']['land_2']['snow'] = {'hMin': 0, 'sMin': 0, 'hMax': 91, 'sMax': 19, 'vMin': 132, 'vMax': 255}
hsv_bond['normal']['land_2']['water'] = {'hMin': 110, 'sMin': 125, 'hMax': 121, 'sMax': 206}
hsv_bond['normal']['land_2']['water_deep'] = {'hMin': 116, 'sMin': 186, 'hMax': 121, 'sMax': 210}
hsv_bond['normal']['land_2']['water_under'] = {'hMin': 115, 'sMin': 206, 'hMax': 127, 'sMax': 225, 'vMin': 7, 'vMax': 113}

hsv_bond['normal']['land_2']['red_dirt'] = {'hMin': 2, 'sMin': 42, 'hMax': 10, 'sMax': 126}

hsv_bond['normal']['land_3']['rock'] = {'hMin': 0, 'sMin': 1, 'hMax': 23, 'sMax': 38,'vMin': 71, 'vMax': 165}
hsv_bond['normal']['land_3']['rock_2'] = {'hMin': 51, 'sMin': 0, 'hMax': 121, 'sMax': 22, 'vMin': 0, 'vMax': 240}
hsv_bond['normal']['land_3']['rock_3'] = {'hMin': 0, 'sMin': 0, 'hMax': 31, 'sMax': 22, 'vMin': 27, 'vMax': 195}
hsv_bond['normal']['land_3']['rock_dark_yellow'] = {'hMin': 13, 'sMin': 47, 'hMax': 27, 'sMax': 106, 'vMin': 0, 'vMax': 175}
hsv_bond['normal']['land_3']['rock_dark_brown'] = {'hMin': 27, 'sMin': 23, 'hMax': 32, 'sMax': 62, 'vMin': 0, 'vMax': 175}

# hsv_bond['normal']['item']['table'] = {'hMin': 7, 'sMin': 128, 'hMax': 17, 'sMax': 172, 'vMin': 6, 'vMax': 255}
# hsv_bond['normal']['item']['torch_1'] = {'hMin': 17, 'sMin': 252, 'hMax': 27, 'sMax': 255, 'vMin': 243, 'vMax': 255}
# hsv_bond['normal']['item']['torch_2'] = {'hMin': 28, 'sMin': 98, 'hMax': 30, 'sMax': 105, 'vMin': 250, 'vMax': 255}

hsv_bond['normal']['mine']['coal'] = {'hMin': 0, 'sMin': 0, 'hMax': 2, 'sMax': 2, 'vMin': 1, 'vMax': 26}
# hsv_bond['normal']['mine']['iron'] = {'hMin': 11, 'sMin': 22, 'hMax': 16, 'sMax': 86, 'vMin': 0, 'vMax': 255}
# hsv_bond['normal']['mine']['iron_2'] = {'hMin': 13, 'sMin': 104, 'hMax': 19, 'sMax': 113, 'vMin': 40, 'vMax': 255}


#hsv_bond['normal']['land'] = {'sky': {}, 'dirt': {}, 'water': {}, 'sand': {}, 'rock': {}}
#
hsv_bond['normal']['land']['grass'] = {'hMin': 42, 'sMin': 109, 'hMax': 48, 'sMax': 126, 'vMin': 65, 'vMax': 255} #129
hsv_bond['normal']['land']['grass_1'] = {'hMin': 44, 'sMin': 144, 'hMax': 49, 'sMax': 158, 'vMin': 66, 'vMax': 255}

hsv_bond['normal']['land']['grass_2'] = {'hMin': 35, 'sMin': 116, 'hMax': 41, 'sMax': 135, 'vMin': 20, 'vMax': 170} #40
hsv_bond['normal']['land']['grass_3'] = {'hMin': 29, 'sMin': 114, 'hMax': 34, 'sMax': 129, 'vMin': 50, 'vMax': 255}
hsv_bond['normal']['land']['grass_4'] = {'hMin': 46, 'sMin': 114, 'hMax': 54, 'sMax': 138, 'vMin': 61, 'vMax': 255}
hsv_bond['normal']['land']['grass_5'] = {'hMin': 50, 'sMin': 60, 'hMax': 63, 'sMax': 81, 'vMin': 17, 'vMax': 150}


hsv_bond['normal']['land']['grass_6'] = {'hMin': 25, 'sMin': 133, 'hMax': 38, 'sMax': 145, 'vMin': 34, 'vMax': 170}
hsv_bond['normal']['land']['grass_7'] = {'hMin': 54, 'sMin': 175, 'hMax': 55, 'sMax': 183}
hsv_bond['normal']['land']['grass_8'] = {'hMin': 42, 'sMin': 106, 'hMax': 47, 'sMax': 121}
hsv_bond['normal']['land']['grass_9'] = {'hMin': 45, 'sMin': 94, 'hMax': 52, 'sMax': 118}
hsv_bond['normal']['land_3']['grass_10'] = {'hMin': 47, 'sMin': 127, 'hMax': 55, 'sMax': 161, 'vMin': 13, 'vMax': 79}
hsv_bond['normal']['land_3']['grass_11'] = {'hMin': 33, 'sMin': 110, 'hMax': 41, 'sMax': 121, 'vMin': 82, 'vMax': 135}
hsv_bond['normal']['land_3']['grass_12'] = {'hMin': 40, 'sMin': 129, 'hMax': 46, 'sMax': 142, 'vMin': 7, 'vMax': 145}



hsv_bond['normal']['tree_2_2']['leaf_1'] = {'hMin': 40, 'sMin': 79, 'hMax': 62, 'sMax': 103, 'vMin': 4, 'vMax': 158}
hsv_bond['normal']['tree_2_2']['chunk_1_brown'] = {'hMin': 13, 'sMin': 162, 'hMax': 19, 'sMax': 204, 'vMin': 0, 'vMax': 116}

# tree 2 - have dark grey + brown chunk - low hue leaf - like swarm

hsv_bond['normal']['tree_2_2']['leaf_2'] = {'hMin': 26, 'sMin': 189, 'hMax': 30, 'sMax': 202}
hsv_bond['normal']['tree_3']['chunk_2_grey'] = {'hMin': 15, 'sMin': 36, 'hMax': 22, 'sMax': 44}
hsv_bond['normal']['tree_2_2']['chunk_2_brown'] = {'hMin': 17, 'sMin': 130, 'hMax': 18, 'sMax': 134}

# tree 3 - have brown tree - leaf green high + a little black dot
hsv_bond['normal']['tree']['leaf_3'] = {'hMin': 40, 'sMin': 161, 'hMax': 53, 'sMax': 191, 'vMin': 17, 'vMax': 255}
hsv_bond['normal']['tree']['leaf_3_far'] = {'hMin': 66, 'sMin': 43, 'hMax': 107, 'sMax': 81}

hsv_bond['normal']['tree']['chunk_3_brown'] = {'hMin': 16, 'sMin': 127, 'hMax': 19, 'sMax': 137}
hsv_bond['normal']['tree']['chunk_3_brown_dark'] = {'hMin': 16, 'sMin': 138, 'hMax': 19, 'sMax': 153}


# tree 4 - have light grey tree + black  dot- low hue leaf + black dot
hsv_bond['normal']['tree']['leaf_4'] = {'hMin': 42, 'sMin': 123, 'hMax': 47, 'sMax': 128, 'vMin': 0, 'vMax': 60}

hsv_bond['normal']['tree']['leaf_4_extra'] = {'hMin': 42, 'sMin': 144, 'hMax': 49, 'sMax': 177, 'vMin': 0, 'vMax': 60}
hsv_bond['normal']['tree']['leaf_4_extra_2'] = {'hMin': 35, 'sMin': 115, 'hMax': 49, 'sMax': 165, 'vMin': 0, 'vMax': 56}
hsv_bond['normal']['tree']['leaf_4_extra_3'] = {'hMin': 29, 'sMin': 119, 'hMax': 41, 'sMax': 160, 'vMin': 12, 'vMax': 46}
hsv_bond['normal']['tree']['leaf_4_extra_4'] = {'hMin': 37, 'sMin': 142, 'hMax': 44, 'sMax': 161, 'vMin': 41, 'vMax': 74}

hsv_bond['normal']['tree_3']['chunk_4_grey'] = {'hMin': 8, 'sMin': 3, 'hMax': 38, 'sMax': 9}

# tree 5 - high tree + light brown tree + green leaf

hsv_bond['normal']['tree_2']['leaf_5'] = {'hMin': 41, 'sMin': 175, 'hMax': 44, 'sMax': 194}
hsv_bond['normal']['tree_2']['chunk_5_grey'] = {'hMin': 17, 'sMin': 125, 'hMax': 19, 'sMax': 135}

# tree 6 - high S green tree + brown chunk
hsv_bond['normal']['tree_2']['leaf_6'] = {'hMin': 52, 'sMin': 234, 'hMax': 54, 'sMax': 249}
#hsv_bond['normal']['tree_2']['chunk_6_brown'] = {'hMin': 17, 'sMin': 134, 'hMax': 28, 'sMax': 182}

hsv_bond['normal']['tree_2']['leaf_7'] = {'hMin': 53, 'sMin': 230, 'hMax': 54, 'sMax': 234}
hsv_bond['normal']['tree_2']['chunk_7_brown'] = {'hMin': 17, 'sMin': 134, 'hMax': 23, 'sMax': 182}

hsv_bond['normal']['tree_2']['inside_chunk'] = {'hMin': 19, 'sMin': 127, 'hMax': 22, 'sMax': 137}
hsv_bond['normal']['tree_2']['inside_chunk_1'] = {'hMin': 14, 'sMin': 114, 'hMax': 16, 'sMax': 132}
hsv_bond['normal']['tree_2']['inside_chunk_2'] = {'hMin': 5, 'sMin': 141, 'hMax': 12, 'sMax': 178, 'vMin': 8, 'vMax': 234}


hsv_bond['normal']['useless_tree']['Cactus'] = {'hMin': 59, 'sMin': 216, 'hMax': 66, 'sMax': 226, 'vMin': 23, 'vMax': 137}


hsv_bond['normal']['land_5']['reset'] = {'hMin': 0, 'sMin': 0, 'hMax': 179, 'sMax': 255}

HMins_bond = []
SMins_bond = []
VMins_bond = []
HMaxs_bond = []
SMaxs_bond = []
VMaxs_bond = []
all_names = []

for k,v in hsv_bond['normal'].items():
    for k1,v1 in v.items():
        all_names.append(k1)

        HMins_bond.append(v1['hMin'])
        HMaxs_bond.append(v1['hMax'])

        SMins_bond.append(v1['sMin'])
        SMaxs_bond.append(v1['sMax'])


        if 'vMin' in v1:
            VMins_bond.append(v1['vMin'])
        else:
            VMins_bond.append(0)

        if 'vMax' in v1:
            VMaxs_bond.append(v1['vMax'])
        else:
            VMaxs_bond.append(255)

HMins_bond = np.expand_dims(HMins_bond,axis=0)
HMins_bond = np.repeat(HMins_bond, 64, axis=0)
HMins_bond = np.expand_dims(HMins_bond,axis=0)
HMins_bond = np.repeat(HMins_bond, 64, axis=0)

SMins_bond = np.expand_dims(SMins_bond,axis=0)
SMins_bond = np.repeat(SMins_bond, 64, axis=0)
SMins_bond = np.expand_dims(SMins_bond,axis=0)
SMins_bond = np.repeat(SMins_bond, 64, axis=0)

VMins_bond = np.expand_dims(VMins_bond,axis=0)
VMins_bond = np.repeat(VMins_bond, 64, axis=0)
VMins_bond = np.expand_dims(VMins_bond,axis=0)
VMins_bond = np.repeat(VMins_bond, 64, axis=0)


HMaxs_bond = np.expand_dims(HMaxs_bond,axis=0)
HMaxs_bond = np.repeat(HMaxs_bond, 64, axis=0)
HMaxs_bond = np.expand_dims(HMaxs_bond,axis=0)
HMaxs_bond = np.repeat(HMaxs_bond, 64, axis=0)

SMaxs_bond = np.expand_dims(SMaxs_bond,axis=0)
SMaxs_bond = np.repeat(SMaxs_bond, 64, axis=0)
SMaxs_bond = np.expand_dims(SMaxs_bond,axis=0)
SMaxs_bond = np.repeat(SMaxs_bond, 64, axis=0)

VMaxs_bond = np.expand_dims(VMaxs_bond,axis=0)
VMaxs_bond = np.repeat(VMaxs_bond, 64, axis=0)
VMaxs_bond = np.expand_dims(VMaxs_bond,axis=0)
VMaxs_bond = np.repeat(VMaxs_bond, 64, axis=0)

# all_names = np.expand_dims(all_names,axis=0)
# all_names = np.repeat(all_names, 64, axis=0)
# all_names = np.expand_dims(all_names,axis=0)
# all_names = np.repeat(all_names, 64, axis=0)




