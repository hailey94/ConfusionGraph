
import json
import numpy as np
from itertools import product

model_info = 'VGG16-R-Adam-92.628' #'ResNet50-P-SGD-94.48' #'VGG16-R-Adam-92.628'
with open('/shared/hailey/layer_infos/{}-per-layer-changes.json'.format(model_info), 'r') as f:
    layer_dict = json.load(f)

mode = 'corrections'
layer_dict = {l: i for l, i in layer_dict.items() if ((mode in l))} # ('block' in l) and
for i, (layer, info) in enumerate(layer_dict.items()):
    # print()
    # print(layer, info)
    if i == 0:
        common = set(info)
    else:
        common = common.intersection(set(info))



with open('/shared/hailey/layer_infos/{}-per-layer-statistics.json'.format(model_info), 'r') as f:
    layer_dict = json.load(f)

layer_dict = {l: i for l, i in layer_dict.items() if ((mode in l))} # ('block' in l) and

if model_info.split('-')[0] =='VGG16':
    orders = ['block1_conv2', 'block2_conv2','block3_conv3','block4_conv3','block5_conv3',
              'global_average_pooling2d','dense','dense_1', 'dense_2']
    channel_info = {'block1_conv2': 64, 'block2_conv2': 128, 'block3_conv3': 256, 'block4_conv3': 512,
                    'block5_conv3': 512,
                    'global_average_pooling2d': 512, 'dense': 4096, 'dense_1': 2048, 'dense_2': 1024}


elif model_info.split('-')[0] =='ResNet50':
    orders = ['pool1_pool', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out','conv5_block3_out', 'avg_pool', 'dense','dense_1',  'dense_2']
    channel_info = {'pool1_pool': 64, 'conv2_block3_out': 256, 'conv3_block4_out': 512, 'conv4_block6_out': 1024,
                    'conv5_block3_out': 2048,
                    'avg_pool': 2048, 'dense': 4096, 'dense_1': 2048, 'dense_2': 1024}


ordered_dict = {}
for layer in orders:
    for l, i in layer_dict.items():
        layer_name = l.split('.')[1].split('-')[-2]
        if layer == layer_name:
            ordered_dict[l] = i

# print(ordered_dict.keys())
# exit()

layer_U = {l: i for l, i in ordered_dict.items() if l in list(ordered_dict.keys())[1:]}
layer_L = {l: i for l, i in ordered_dict.items() if l in list(ordered_dict.keys())[:-1]}

# print(layer_U.keys())
# print(layer_L.keys())
# exit()


tmp = []

layer_tmp = []
for i, (U, L) in enumerate(zip(layer_U.items(), layer_L.items())):
    u_layer, u_info = U
    l_layer, l_info = L

    upperLayer = u_layer.split('.')[1].split('-')[-2]
    lowerLayer = l_layer.split('.')[1].split('-')[-2]

    u_channel, l_channel = channel_info[upperLayer], channel_info[lowerLayer]
    u_changes, l_changes = u_info['change_dict_channels'], l_info['change_dict_channels']
    neighbor_mat = np.zeros([l_channel,u_channel])

    for c in common:
        combi = list(product(l_changes[c],u_changes[c]))
        for (l, u) in combi:
            neighbor_mat[int(l), int(u)] += 1
    # print(lowerLayer, upperLayer, np.max(neighbor_mat))

    print(i, lowerLayer, upperLayer)
    tmp_n, tmp_N1 = np.where(neighbor_mat==np.max(neighbor_mat))
    print(tmp_n, tmp_N1)
    if i == 0:
        layer_tmp.append(set(tmp_n))
        tmp.extend(tmp_N1)
    else:
        tmp.extend(list(tmp_n))
        layer_tmp.append(set(tmp))
        tmp = []
        tmp.extend(list(tmp_N1))

    if i == len(layer_U)-1:
        layer_tmp.append(set(tmp))

    #
    np.savetxt('/shared/hailey/neighbors/{}-{}_{}_to_{}.csv'.format(u_layer.split('.')[0],mode,\
                                                                    lowerLayer, upperLayer),neighbor_mat)
#