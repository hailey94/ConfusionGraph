"""ACE library.

Library for discovering and testing concept activation vectors. It contains
ConceptDiscovery class that is able to discover the concepts belonging to one
of the possible classification labels of the classification task of a network
and calculate each concept's TCAV score..
"""
from multiprocessing import dummy as multiprocessing
import sys
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from confusion_helpers import *
import matplotlib.pyplot as plt
import pandas as pd
import json
import cv2
from tqdm import tqdm

# np.random.seed(0)
class PathOffCheck(object):
    def __init__(self,
                 model,
                 datagenerator,
                 neighbor_matrixes,
                 layer_orders,
                 off_nums_per_layer,
                 ):

        self.model = model
        self.datagenerator = datagenerator
        self.neighbor_matrixes = neighbor_matrixes
        self.layer_orders = layer_orders

        self.layer_idxs = {}
        for layer in self.layer_orders:
            self.layer_idxs[layer] = self.model.layers.index(self.model.get_layer(layer))

        self.predictions = []
        self.probabilities = []
        self.labels = []
        self.off_nums = off_nums_per_layer
        self.label_dict={}


    def prediction_wo_off(self):
        save_orig_output = np.empty(0)
        files = []
        for batch, (img_batch, label_batch, fns) in tqdm(enumerate(self.datagenerator)):
            output = self.model(img_batch)
            prediction = np.argmax(output,axis=1)
            save_orig_output = np.concatenate([save_orig_output,prediction],axis=0)

            files.extend(fns)
            self.labels.append(np.where(label_batch == 1)[-1].item())

        return save_orig_output, np.array(files)

    def return_neighbors(self, additional_neighbors=None, skip_rate=None):
        neighbors = {layer:[] for layer in self.layer_orders}
        for i, layer in enumerate(self.layer_orders):
            if i < len(self.neighbor_matrixes):
                Ns, Ms = np.where(self.neighbor_matrixes[i] == np.max(self.neighbor_matrixes[i]))
                Ns, Ms = set(Ns), set(Ms)

                if isinstance(additional_neighbors, list):
                    add_Ns, add_Ms = np.where(additional_neighbors[i] == np.max(additional_neighbors[i]))
                    Ns, Ms = Ns.union(set(add_Ns)), Ms.union(set(add_Ms))

                neighbors[layer].extend(list(Ns))
                neighbors[self.layer_orders[i+1]].extend(list(Ms))
        cnt = 0
        candidate= []
        for layer, n_list in neighbors.items():
            n_list = set(n_list)
            neighbors[layer] = n_list
            cnt += len(n_list)
            candidate.extend(n_list)

        if skip_rate > 0:
            skip_num = int(skip_rate*cnt)
            skip_N = np.random.choice([x for x in range(len(candidate))], skip_num, replace=False)
            skip_N.sort()

            cnt = 0
            for layer, n_list in neighbors.items():
                neighbor_list = []
                for elt in n_list:
                    if cnt in skip_N:
                        neighbor_list.append(elt)
                    cnt += 1
                neighbors[layer] = neighbor_list
        return neighbors

    def _path_attack(self, ablation_channels, random=False):

        for batch, (img_batch, label_batch, fns) in tqdm(enumerate(self.datagenerator)):
            convOutputs = get_1st_acts_from_images(img_batch, self.model, self.layer_orders[0])

            for i, layer in enumerate(self.layer_orders):
                t_idx = self.layer_idxs[layer]
                if i == len(self.layer_orders) - 1:
                    next_idx = len(self.model.layers)
                    next_layer = None
                else:
                    next_layer = self.layer_orders[i + 1]
                    next_idx = self.layer_idxs[self.layer_orders[i + 1]]

                ablation_logit_model = create_consecutives(t_idx, next_idx,
                                                           layer, next_layer,
                                                           self.model, input_shape=self.model.layers[t_idx + 1].input_shape)

                Ns = ablation_channels[layer]
                if random:
                    Ns = np.random.choice([x for x in range(convOutputs.shape[-1])], len(Ns), replace=False)

                ## self.neighbor_matrixes[i] = n x m 에서 n channel 끄고
                for channel in Ns:
                    zeros = np.zeros(convOutputs.shape[:-1])
                    try:
                        convOutputs = convOutputs.copy()

                    except AttributeError:
                        convOutputs = np.asarray(convOutputs).copy()

                    if len(convOutputs.shape) == 4:
                        convOutputs[:, :, :, channel] = zeros
                    elif len(convOutputs.shape) == 2:
                        convOutputs[:,  channel] = zeros
                ## forward
                convOutputs = ablation_logit_model(convOutputs)

            pred = tf.nn.softmax(convOutputs)
            self.probabilities.append(pred)
            self.labels.append(np.where(label_batch == 1)[-1].item())
            self.predictions.append(np.argmax(pred, axis=1).item())

        self.probabilities = np.asarray(self.probabilities).squeeze()


    def _return_uni_directional_path(self, skip_rate=None):

        for batch, (img_batch, label_batch, fns) in tqdm(enumerate(self.datagenerator)):
            convOutputs = get_1st_acts_from_images(img_batch, self.model, self.layer_orders[0])
            # c, p = get_acts_from_images(img_batch, self.model, self.layer_orders[0], return_pred=True, preprocess=False)
            # print((c == convOutputs.squeeze()).all())

            for i, layer in enumerate(self.layer_orders):
                t_idx = self.layer_idxs[layer]
                if i == len(self.layer_orders) - 1:
                    next_idx = len(self.model.layers)
                    next_layer = None
                else:
                    next_layer = self.layer_orders[i + 1]
                    next_idx = self.layer_idxs[self.layer_orders[i + 1]]

                if i < len(self.neighbor_matrixes):
                    Ns, Ms = np.where(self.neighbor_matrixes[i] == np.max(self.neighbor_matrixes[i]))
                    Ns, Ms = set(Ns), set(Ms)


                if isinstance(skip_rate,float):
                    skip_N = np.random.choice(list(Ns), int(len(Ns) * skip_rate), replace=False)
                    skip_M = np.random.choice(list(Ms), int(len(Ms) * skip_rate), replace=False)
                else: skip_N, skip_M = [], []

                ablation_logit_model = create_consecutives(t_idx, next_idx,
                                                           layer, next_layer,
                                                           self.model, input_shape=self.model.layers[t_idx + 1].input_shape)

                print(layer, Ns)
                print(layer, Ms)

                ## self.neighbor_matrixes[i] = n x m 에서 n channel 끄고
                if i < len(self.neighbor_matrixes):
                    for channel in Ns:
                        if channel in skip_N: continue
                        zeros = np.zeros(convOutputs.shape[:-1])
                        try:
                            convOutputs = convOutputs.copy()

                        except AttributeError:
                            convOutputs = np.asarray(convOutputs).copy()

                        if len(convOutputs.shape) == 4:
                            convOutputs[:, :, :, channel] = zeros
                        elif len(convOutputs.shape) == 2:
                            convOutputs[:,  channel] = zeros
                ## forward
                convOutputs = ablation_logit_model(convOutputs)

                ## self.neighbor_matrixes[i] = n x m 에서 m channel 끄고
                if i < len(self.neighbor_matrixes):
                    for channel in Ms:
                        if channel in skip_M: continue
                        zeros = np.zeros(convOutputs.shape[:-1])
                        try:
                            convOutputs = convOutputs.copy()

                        except AttributeError:
                            convOutputs = np.asarray(convOutputs).copy()

                        if len(convOutputs.shape) == 4:
                            convOutputs[:, :, :, channel] = zeros
                        elif len(convOutputs.shape) == 2:
                            convOutputs[:,  channel] = zeros

            exit(1)

            # print((p == np.asarray(convOutputs).squeeze()).all())
            pred = tf.nn.softmax(convOutputs)
            self.probabilities.append(pred)
            self.labels.append(np.where(label_batch == 1)[-1].item())
            self.predictions.append(np.argmax(pred, axis=1).item())

        self.probabilities = np.asarray(self.probabilities).squeeze()


    def _return_predictions_multi_neighbors(self, additional_neighbors):

        for batch, (img_batch, label_batch, fns) in tqdm(enumerate(self.datagenerator)):
            convOutputs = get_1st_acts_from_images(img_batch, self.model, self.layer_orders[0])
            # c, p = get_acts_from_images(img_batch, self.model, self.layer_orders[0], return_pred=True, preprocess=False)
            # print((c == convOutputs.squeeze()).all())

            for i, layer in enumerate(self.layer_orders):
                t_idx = self.layer_idxs[layer]
                if i == len(self.layer_orders) - 1:
                    next_idx = len(self.model.layers)
                    next_layer = None
                else:
                    next_layer = self.layer_orders[i + 1]
                    next_idx = self.layer_idxs[self.layer_orders[i + 1]]

                if i < len(self.neighbor_matrixes):
                    Ns, Ms = np.where(self.neighbor_matrixes[i] == np.max(self.neighbor_matrixes[i]))
                    Ns, Ms = set(Ns), set(Ms)

                    add_Ns, add_Ms = np.where(additional_neighbors[i] == np.max(additional_neighbors[i]))
                    Ns, Ms = Ns.union(set(add_Ns)), Ms.union(set(add_Ms))

                ablation_logit_model = create_consecutives(t_idx, next_idx,
                                                           layer, next_layer,
                                                           self.model, input_shape=self.model.layers[t_idx + 1].input_shape)

                # print(i, layer, len(Ns), Ns)
                # print(i, layer, len(Ms), Ms)

                ## self.neighbor_matrixes[i] = n x m 에서 n channel 끄고
                if i < len(self.neighbor_matrixes):
                    for channel in Ns:
                        zeros = np.zeros(convOutputs.shape[:-1])
                        try:
                            convOutputs = convOutputs.copy()

                        except AttributeError:
                            convOutputs = np.asarray(convOutputs).copy()

                        if len(convOutputs.shape) == 4:
                            convOutputs[:, :, :, channel] = zeros
                        elif len(convOutputs.shape) == 2:
                            convOutputs[:,  channel] = zeros
                ## forward
                convOutputs = ablation_logit_model(convOutputs)

                ## self.neighbor_matrixes[i] = n x m 에서 m channel 끄고
                if i < len(self.neighbor_matrixes):
                    for channel in Ms:
                        zeros = np.zeros(convOutputs.shape[:-1])
                        try:
                            convOutputs = convOutputs.copy()

                        except AttributeError:
                            convOutputs = np.asarray(convOutputs).copy()

                        if len(convOutputs.shape) == 4:
                            convOutputs[:, :, :, channel] = zeros
                        elif len(convOutputs.shape) == 2:
                            convOutputs[:,  channel] = zeros

            # print((p == np.asarray(convOutputs).squeeze()).all())
            pred = tf.nn.softmax(convOutputs)
            self.probabilities.append(pred)
            self.labels.append(np.where(label_batch == 1)[-1].item())
            self.predictions.append(np.argmax(pred, axis=1).item())

        self.probabilities = np.asarray(self.probabilities).squeeze()



    def _return_consecutive_results(self, additional_neighbors):

        for batch, (img_batch, label_batch, fns) in tqdm(enumerate(self.datagenerator)):
            convOutputs = get_1st_acts_from_images(img_batch, self.model, self.layer_orders[0])
            # c, p = get_acts_from_images(img_batch, self.model, self.layer_orders[0], return_pred=True, preprocess=False)
            # print((c == convOutputs.squeeze()).all())

            for i, layer in enumerate(self.layer_orders):
                t_idx = self.layer_idxs[layer]
                if i == len(self.layer_orders) - 1:
                    next_idx = len(self.model.layers)
                    next_layer = None
                else:
                    next_layer = self.layer_orders[i + 1]
                    next_idx = self.layer_idxs[self.layer_orders[i + 1]]

                if i < len(self.neighbor_matrixes):
                    Ns, Ms = np.where(self.neighbor_matrixes[i] == np.max(self.neighbor_matrixes[i])) # violation꺼
                    Ns, Ms = set(Ns), set(Ms)

                    add_Ns, add_Ms = np.where(additional_neighbors[i] == np.max(additional_neighbors[i])) # correction꺼
                    Ns, Ms = Ns.union(set(add_Ns)), Ms.union(set(add_Ms))

                ablation_logit_model = create_consecutives(t_idx, next_idx,
                                                           layer, next_layer,
                                                           self.model, input_shape=self.model.layers[t_idx + 1].input_shape)


                ## self.neighbor_matrixes[i] = n x m 에서 n channel 끄고
                if (i == 2) or (i==4):# len(self.neighbor_matrixes):
                    for channel in Ns:
                        zeros = np.zeros(convOutputs.shape[:-1])
                        try:
                            convOutputs = convOutputs.copy()

                        except AttributeError:
                            convOutputs = np.asarray(convOutputs).copy()

                        if len(convOutputs.shape) == 4:
                            convOutputs[:, :, :, channel] = zeros
                        elif len(convOutputs.shape) == 2:
                            convOutputs[:,  channel] = zeros
                ## forward
                convOutputs = ablation_logit_model(convOutputs)

                ## self.neighbor_matrixes[i] = n x m 에서 m channel 끄고
                if (i == 1) or (i == 3):# len(self.neighbor_matrixes) :
                    for channel in Ms:
                        zeros = np.zeros(convOutputs.shape[:-1])
                        try:
                            convOutputs = convOutputs.copy()

                        except AttributeError:
                            convOutputs = np.asarray(convOutputs).copy()

                        if len(convOutputs.shape) == 4:
                            convOutputs[:, :, :, channel] = zeros
                        elif len(convOutputs.shape) == 2:
                            convOutputs[:,  channel] = zeros

            pred = tf.nn.softmax(convOutputs)
            self.probabilities.append(pred)
            self.labels.append(np.where(label_batch == 1)[-1].item())
            self.predictions.append(np.argmax(pred, axis=1).item())
        self.probabilities = np.asarray(self.probabilities).squeeze()
        ### print accs ###


    def path_random_attack(self, additional_neighbors, skip_rate=0.5):

        for batch, (img_batch, label_batch, fns) in tqdm(enumerate(self.datagenerator)):
            convOutputs = get_1st_acts_from_images(img_batch, self.model, self.layer_orders[0])

            for i, layer in enumerate(self.layer_orders):
                t_idx = self.layer_idxs[layer]
                if i == len(self.layer_orders) - 1:
                    next_idx = len(self.model.layers)
                    next_layer = None
                else:
                    next_layer = self.layer_orders[i + 1]
                    next_idx = self.layer_idxs[self.layer_orders[i + 1]]

                if i < len(self.neighbor_matrixes):
                    Ns, Ms = np.where(self.neighbor_matrixes[i] == np.max(self.neighbor_matrixes[i]))
                    Ns, Ms = set(Ns), set(Ms)

                    add_Ns, add_Ms = np.where(additional_neighbors[i] == np.max(additional_neighbors[i]))
                    Ns, Ms = Ns.union(set(add_Ns)), Ms.union(set(add_Ms))

                ablation_logit_model = create_consecutives(t_idx, next_idx,
                                                           layer, next_layer,
                                                           self.model, input_shape=self.model.layers[t_idx + 1].input_shape)

                skip_N = np.random.choice(list(Ns), int(len(Ns)*skip_rate), replace=False)
                skip_M = np.random.choice(list(Ms), int(len(Ms)*skip_rate), replace=False)


                ## self.neighbor_matrixes[i] = n x m 에서 n channel 끄고
                if i < len(self.neighbor_matrixes):
                    for channel in Ns:
                        if channel in skip_N: continue
                        zeros = np.zeros(convOutputs.shape[:-1])
                        try:
                            convOutputs = convOutputs.copy()

                        except AttributeError:
                            convOutputs = np.asarray(convOutputs).copy()

                        if len(convOutputs.shape) == 4:
                            convOutputs[:, :, :, channel] = zeros
                        elif len(convOutputs.shape) == 2:
                            convOutputs[:,  channel] = zeros
                ## forward
                convOutputs = ablation_logit_model(convOutputs)

                ## self.neighbor_matrixes[i] = n x m 에서 m channel 끄고
                if i < len(self.neighbor_matrixes):
                    for channel in Ms:
                        if channel in skip_M: continue
                        zeros = np.zeros(convOutputs.shape[:-1])
                        try:
                            convOutputs = convOutputs.copy()

                        except AttributeError:
                            convOutputs = np.asarray(convOutputs).copy()

                        if len(convOutputs.shape) == 4:
                            convOutputs[:, :, :, channel] = zeros
                        elif len(convOutputs.shape) == 2:
                            convOutputs[:,  channel] = zeros

            # print((p == np.asarray(convOutputs).squeeze()).all())
            pred = tf.nn.softmax(convOutputs)
            self.probabilities.append(pred)
            self.labels.append(np.where(label_batch == 1)[-1].item())
            self.predictions.append(np.argmax(pred, axis=1).item())

        self.probabilities = np.asarray(self.probabilities).squeeze()

    def _random_offs(self):

        for batch, (img_batch, label_batch, fns) in tqdm(enumerate(self.datagenerator)):
            convOutputs = get_1st_acts_from_images(img_batch, self.model, self.layer_orders[0])

            for i, layer in enumerate(self.layer_orders):

                t_idx = self.layer_idxs[layer]
                if i == len(self.layer_orders) - 1:
                    next_idx = len(self.model.layers)
                    next_layer = None
                else:
                    next_layer = self.layer_orders[i + 1]
                    next_idx = self.layer_idxs[self.layer_orders[i + 1]]

                ablation_logit_model = create_consecutives(t_idx, next_idx,
                                                           layer, next_layer,
                                                           self.model, input_shape=self.model.layers[t_idx + 1].input_shape)

                ## self.neighbor_matrixes[i] = n x m 에서 n channel 끄고
                if i < len(self.off_nums):
                    Ns = self.off_nums[self.layer_orders[i]]
                    for channel in Ns:
                        zeros = np.zeros(convOutputs.shape[:-1])
                        try:
                            convOutputs = convOutputs.copy()

                        except AttributeError:
                            convOutputs = np.asarray(convOutputs).copy()

                        if len(convOutputs.shape) == 4:
                            convOutputs[:, :, :, channel] = zeros
                        elif len(convOutputs.shape) == 2:
                            convOutputs[:,  channel] = zeros
                ## forward
                convOutputs = ablation_logit_model(convOutputs)

            # print((p == np.asarray(convOutputs).squeeze()).all())
            pred = tf.nn.softmax(convOutputs)
            self.probabilities.append(pred)
            self.labels.append(np.where(label_batch == 1)[-1].item())
            self.predictions.append(np.argmax(pred, axis=1).item())

        self.probabilities = np.asarray(self.probabilities).squeeze()

    def sigmoid(self,x, a, b, c):
        return c / (1 + np.exp(-a * (x - b)))

    def sigmoid2(self,x, a, b, c):
        return c / (1 + np.exp(-a * (b-x)))


    def superimpose(self, fn, cam, emphasize=False):
        img_bgr = cv2.imread(fn, cv2.IMREAD_COLOR)
        img_bgr = cv2.resize(img_bgr, (224,224), interpolation=cv2.INTER_LINEAR)
        if np.max(cam) != np.min(cam):
            cam = (cam - np.min(cam))/(np.max(cam)- np.min(cam))

        heatmap = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))

        if emphasize:
            heatmap = self.sigmoid(heatmap, 50, 0.5, 1)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        hif = .8
        superimposed_img = heatmap * hif + img_bgr
        superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)  # scale 0 to 255
        superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

        return superimposed_img_rgb, img_bgr

    def extract_CAM(self, additional_neighbors, class_names, extract=True):

        for batch, (img_batch, label_batch, fns) in tqdm(enumerate(self.datagenerator)):

            cam = {}
            label = np.where(label_batch==1)[1].item()
            label_class =class_names[label]
            if os.path.exists('/shared/hailey/figures2/{}/'.format(label_class)) == False:
                os.mkdir('/shared/hailey/figures2/{}/'.format(label_class))

            orig_prd = self.model(img_batch)
            orig_prd = np.argmax(orig_prd)
            orig_prd = class_names[orig_prd]
            # print(label, np.argmax(orig_prd))
            # exit(1)
            # continue

            if label not in self.label_dict.keys():
                self.label_dict[label] = 1
            else:
                if self.label_dict[label] >= 10:
                    continue
                else:
                    self.label_dict[label] += 1

            convOutputs = get_1st_acts_from_images(img_batch, self.model, self.layer_orders[0])

            if extract:
                for i, layer in enumerate(self.layer_orders):

                    t_idx = self.layer_idxs[layer]
                    if i == len(self.layer_orders) - 1:
                        next_idx = len(self.model.layers)
                        next_layer = None
                    else:
                        next_layer = self.layer_orders[i + 1]
                        next_idx = self.layer_idxs[self.layer_orders[i + 1]]

                    if i < len(self.neighbor_matrixes):
                        Ns, Ms = np.where(self.neighbor_matrixes[i] == np.max(self.neighbor_matrixes[i]))
                        Ns, Ms = set(Ns), set(Ms)

                        add_Ns, add_Ms = np.where(additional_neighbors[i] == np.max(additional_neighbors[i]))
                        Ns, Ms = Ns.union(set(add_Ns)), Ms.union(set(add_Ms))

                        if layer not in cam.keys():
                            cam[layer] = {}
                        if next_layer not in cam.keys():
                            cam[next_layer] = {}

                    ablation_logit_model = create_consecutives(t_idx, next_idx,
                                                               layer, next_layer,
                                                               self.model, input_shape=self.model.layers[t_idx + 1].input_shape,
                                                               summary=False)

                    ## self.neighbor_matrixes[i] = n x m 에서 n channel 끄고
                    if i < len(self.neighbor_matrixes):
                        for channel in Ns:
                            zeros = np.zeros(convOutputs.shape[:-1])
                            try:
                                convOutputs = convOutputs.copy()

                            except AttributeError:
                                convOutputs = np.asarray(convOutputs).copy()

                            if len(convOutputs.shape) == 4:
                                cam[layer][channel] = convOutputs[:, :, :, channel].copy()
                                convOutputs[:, :, :, channel] = zeros
                            elif len(convOutputs.shape) == 2:
                                cam[layer][channel] = convOutputs[:,channel].copy()
                                convOutputs[:,  channel] = zeros
                    ## forward
                    convOutputs = ablation_logit_model(convOutputs)

                    ## self.neighbor_matrixes[i] = n x m 에서 m channel 끄고
                    if i < len(self.neighbor_matrixes):
                        for channel in Ms:
                            # self.cams.append((self.layer_orders[i+1], channel, convOutputs[:, :, :, channel]))
                            zeros = np.zeros(convOutputs.shape[:-1])
                            try:
                                convOutputs = convOutputs.copy()

                            except AttributeError:
                                convOutputs = np.asarray(convOutputs).copy()

                            if len(convOutputs.shape) == 4:
                                cam[next_layer][channel] = convOutputs[:, :, :, channel].copy()
                                convOutputs[:, :, :, channel] = zeros
                            elif len(convOutputs.shape) == 2:
                                cam[next_layer][channel] = convOutputs[:, channel].copy()
                                convOutputs[:,  channel] = zeros

                pred = tf.nn.softmax(convOutputs)
                pred = np.argmax(pred)
                pred = class_names[pred]
                for layer_name, cam_infos in cam.items():
                    for j, (channel_num, channel_arr) in enumerate(cam_infos.items()):
                        if j == 0 : layer_cam = channel_arr
                        else: layer_cam += channel_arr

                    layer_cam = np.maximum(0,layer_cam)

                    layer_cam = np.squeeze(layer_cam)

                    if len(layer_cam.shape) < 2: continue
                    superimposed_img, orig_img = self.superimpose(fns[0], layer_cam, False)
                    superimposed_img_emp, _ = self.superimpose(fns[0], layer_cam, True)

                    b, g, r = cv2.split(orig_img)  # img파일을 b,g,r로 분리
                    orig_img = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge

                    b, g, r = cv2.split(superimposed_img)  # img파일을 b,g,r로 분리
                    superimposed_img = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge

                    b, g, r = cv2.split(superimposed_img_emp)  # img파일을 b,g,r로 분리
                    superimposed_img_emp = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge


                    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
                    fig.suptitle('from {} to {}'.format(orig_prd,pred))

                    ax[0].imshow(orig_img)
                    ax[0].set_title("image")
                    ax[1].imshow(superimposed_img)
                    ax[1].set_title("activation")
                    ax[2].imshow(superimposed_img_emp)
                    ax[2].set_title("emhasized")
                    file_name = fns[0].split('/')[-1]
                    plt.savefig('/shared/hailey/figures2/{}/{}-{}'.format(label_class,layer_name,file_name))
                    plt.close()

            else:
                for i, layer in enumerate(self.layer_orders):

                    t_idx = self.layer_idxs[layer]
                    if i == len(self.layer_orders) - 1:
                        next_idx = len(self.model.layers)
                        next_layer = None
                    else:
                        next_layer = self.layer_orders[i + 1]
                        next_idx = self.layer_idxs[self.layer_orders[i + 1]]
                    try:
                        convOutputs = convOutputs.copy()

                    except AttributeError:
                        convOutputs = np.asarray(convOutputs).copy()
                    for i in range(convOutputs.shape[-1]):
                        if i == 0:
                            if len(convOutputs.shape) == 4:
                                layer_cam = convOutputs[:, :, :, i].copy()
                            elif len(convOutputs.shape) == 2:
                                layer_cam = convOutputs[:, i].copy()
                        else:
                            if len(convOutputs.shape) == 4:
                                layer_cam += convOutputs[:, :, :, i].copy()
                            elif len(convOutputs.shape) == 2:
                                layer_cam += convOutputs[:, i].copy()

                    ablation_logit_model = create_consecutives(t_idx, next_idx,
                                                               layer, next_layer,
                                                               self.model,
                                                               input_shape=self.model.layers[t_idx + 1].input_shape,
                                                               summary=False)


                    convOutputs = ablation_logit_model(convOutputs)

                    layer_cam = np.maximum(0, layer_cam)

                    layer_cam = np.squeeze(layer_cam)

                    if len(layer_cam.shape) < 2: continue
                    superimposed_img, orig_img = self.superimpose(fns[0], layer_cam, False)
                    superimposed_img_emp, _ = self.superimpose(fns[0], layer_cam, True)

                    b, g, r = cv2.split(orig_img)  # img파일을 b,g,r로 분리
                    orig_img = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge

                    b, g, r = cv2.split(superimposed_img)  # img파일을 b,g,r로 분리
                    superimposed_img = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge

                    b, g, r = cv2.split(superimposed_img_emp)  # img파일을 b,g,r로 분리
                    superimposed_img_emp = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge

                    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
                    fig.suptitle('{}'.format(orig_prd))

                    ax[0].imshow(orig_img)
                    ax[0].set_title("image")
                    ax[1].imshow(superimposed_img)
                    ax[1].set_title("activation")
                    ax[2].imshow(superimposed_img_emp)
                    ax[2].set_title("emhasized")
                    file_name = fns[0].split('/')[-1]
                    plt.savefig('/shared/hailey/figures2/{}/All-{}-{}'.format(label_class, layer, file_name))
                    plt.close()


    def separate_CAM(self, additional_neighbors, class_names, extract=True):

        for batch, (img_batch, label_batch, fns) in tqdm(enumerate(self.datagenerator)):

            cam = {}
            label = np.where(label_batch==1)[1].item()
            label_class =class_names[label]
            if os.path.exists('/shared/hailey/figures3/{}/'.format(label_class)) == False:
                os.mkdir('/shared/hailey/figures3/{}/'.format(label_class))

            orig_prd = self.model(img_batch)
            orig_prd = np.argmax(orig_prd)
            orig_prd = class_names[orig_prd]
            # print(label, np.argmax(orig_prd))
            # exit(1)
            # continue

            if label not in self.label_dict.keys():
                self.label_dict[label] = 1
            else:
                if self.label_dict[label] >= 10:
                    continue
                else:
                    self.label_dict[label] += 1

            convOutputs = get_1st_acts_from_images(img_batch, self.model, self.layer_orders[0])

            if extract:
                for i, layer in enumerate(self.layer_orders):

                    t_idx = self.layer_idxs[layer]
                    if i == len(self.layer_orders) - 1:
                        next_idx = len(self.model.layers)
                        next_layer = None
                    else:
                        next_layer = self.layer_orders[i + 1]
                        next_idx = self.layer_idxs[self.layer_orders[i + 1]]

                    if i < len(self.neighbor_matrixes):
                        Ns, Ms = np.where(self.neighbor_matrixes[i] == np.max(self.neighbor_matrixes[i]))
                        Ns, Ms = set(Ns), set(Ms)

                        add_Ns, add_Ms = np.where(additional_neighbors[i] == np.max(additional_neighbors[i]))
                        Ns, Ms = Ns.union(set(add_Ns)), Ms.union(set(add_Ms))

                        if layer not in cam.keys():
                            cam[layer] = {}
                        if next_layer not in cam.keys():
                            cam[next_layer] = {}

                    ablation_logit_model = create_consecutives(t_idx, next_idx,
                                                               layer, next_layer,
                                                               self.model, input_shape=self.model.layers[t_idx + 1].input_shape,
                                                               summary=False)

                    ## self.neighbor_matrixes[i] = n x m 에서 n channel 끄고
                    if i < len(self.neighbor_matrixes):
                        for channel in Ns:
                            zeros = np.zeros(convOutputs.shape[:-1])
                            try:
                                convOutputs = convOutputs.copy()

                            except AttributeError:
                                convOutputs = np.asarray(convOutputs).copy()

                            if len(convOutputs.shape) == 4:
                                layer_cam = convOutputs[:, :, :, channel].copy()
                                convOutputs[:, :, :, channel] = zeros
                            elif len(convOutputs.shape) == 2:
                                layer_cam = convOutputs[:,channel].copy()
                                convOutputs[:,  channel] = zeros

                            layer_cam = np.maximum(0, layer_cam)
                            layer_cam = np.squeeze(layer_cam)

                            if len(layer_cam.shape) < 2: continue
                            superimposed_img, orig_img = self.superimpose(fns[0], layer_cam, False)
                            superimposed_img_emp, _ = self.superimpose(fns[0], layer_cam, True)

                            b, g, r = cv2.split(orig_img)  # img파일을 b,g,r로 분리
                            orig_img = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge

                            b, g, r = cv2.split(superimposed_img)  # img파일을 b,g,r로 분리
                            superimposed_img = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge

                            b, g, r = cv2.split(superimposed_img_emp)  # img파일을 b,g,r로 분리
                            superimposed_img_emp = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge

                            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
                            fig.suptitle('label {}, pred {}'.format(label_class, orig_prd))

                            ax[0].imshow(orig_img)
                            ax[0].set_title("image")
                            ax[1].imshow(superimposed_img)
                            ax[1].set_title("activation")
                            ax[2].imshow(superimposed_img_emp)
                            ax[2].set_title("emhasized")
                            file_name = fns[0].split('/')[-1]
                            plt.savefig('/shared/hailey/figures3/{}/{}-ch{}-{}'.format(label_class, layer, channel,file_name))
                            plt.close()

                    ## forward
                    convOutputs = ablation_logit_model(convOutputs)

                    ## self.neighbor_matrixes[i] = n x m 에서 m channel 끄고
                    if i < len(self.neighbor_matrixes):
                        for channel in Ms:
                            # self.cams.append((self.layer_orders[i+1], channel, convOutputs[:, :, :, channel]))
                            zeros = np.zeros(convOutputs.shape[:-1])
                            try:
                                convOutputs = convOutputs.copy()

                            except AttributeError:
                                convOutputs = np.asarray(convOutputs).copy()

                            if len(convOutputs.shape) == 4:
                                layer_cam = convOutputs[:, :, :, channel].copy()
                                convOutputs[:, :, :, channel] = zeros
                            elif len(convOutputs.shape) == 2:
                                layer_cam = convOutputs[:, channel].copy()
                                convOutputs[:,  channel] = zeros

                            layer_cam = np.maximum(0, layer_cam)
                            layer_cam = np.squeeze(layer_cam)

                            if len(layer_cam.shape) < 2: continue
                            superimposed_img, orig_img = self.superimpose(fns[0], layer_cam, False)
                            superimposed_img_emp, _ = self.superimpose(fns[0], layer_cam, True)

                            b, g, r = cv2.split(orig_img)  # img파일을 b,g,r로 분리
                            orig_img = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge

                            b, g, r = cv2.split(superimposed_img)  # img파일을 b,g,r로 분리
                            superimposed_img = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge

                            b, g, r = cv2.split(superimposed_img_emp)  # img파일을 b,g,r로 분리
                            superimposed_img_emp = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge

                            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
                            fig.suptitle('label {}, pred {}'.format(label_class, orig_prd))

                            ax[0].imshow(orig_img)
                            ax[0].set_title("image")
                            ax[1].imshow(superimposed_img)
                            ax[1].set_title("activation")
                            ax[2].imshow(superimposed_img_emp)
                            ax[2].set_title("emhasized")
                            file_name = fns[0].split('/')[-1]
                            plt.savefig('/shared/hailey/figures3/{}/{}-ch{}-{}'.format(label_class, next_layer, channel,
                                                                                       file_name))
                            plt.close()


    def block5_CAM(self, additional_neighbors, bottleneck, thres=0):

        iou_score=0
        for batch, (img_batch, label_batch, fns) in tqdm(enumerate(self.datagenerator)):

            convOutputs, pred = get_acts_from_images(img_batch, self.model, bottleneck, return_pred=True)
            convOutputs = convOutputs.copy()
            pred = tf.nn.softmax(pred)

            ## make gradcam
            cls = np.argmax(self.model.predict(img_batch))
            grads_val, output = get_grads(img_batch, self.model, bottleneck, cls, return_scores=False, return_logit=False)
            weights = np.mean(np.squeeze(grads_val), axis=(0, 1))

            layer_cam = np.dot(output, weights)
            layer_cam = np.maximum(layer_cam, 0)  # Passing through ReLU
            layer_cam = np.squeeze(layer_cam)

            key_channels = np.zeros(layer_cam.shape)
            for i in range(len(self.neighbor_matrixes)):
                Ns, Ms = np.where(self.neighbor_matrixes[i] == np.max(self.neighbor_matrixes[i]))
                Ns, Ms = set(Ns), set(Ms)

                add_Ns, add_Ms = np.where(additional_neighbors[i] == np.max(additional_neighbors[i]))
                Ns, Ms = Ns.union(set(add_Ns)), Ms.union(set(add_Ms))

                if i == 0:
                    for channel in Ms:
                        try:
                            convOutputs = convOutputs.copy()

                        except AttributeError:
                            convOutputs = np.asarray(convOutputs).copy()

                        key_channels += convOutputs[:, :, channel].copy()


                if i == 1:
                    for channel in Ns:
                        try:
                            convOutputs = convOutputs.copy()

                        except AttributeError:
                            convOutputs = np.asarray(convOutputs).copy()

                        key_channels += convOutputs[:, :, channel].copy()

            key_channels = np.maximum(key_channels, 0)  # Passing through ReLU


            layer_cam /= np.max(layer_cam)  # Passing through ReLU
            key_channels /= np.max(key_channels)  # Passing through ReLU

            layer_cam = (layer_cam > thres)
            key_channels = (key_channels > thres)

            intersection = np.logical_and(layer_cam, key_channels)
            union = np.logical_or(layer_cam, key_channels)
            iou_score += np.sum(intersection) / np.sum(union)
        print(len(self.datagenerator))
        print(iou_score/len(self.datagenerator))

    def GradCam(self, class_names, layer_name):
        for batch, (img_batch, label_batch, fns) in tqdm(enumerate(self.datagenerator)):
            label = np.where(label_batch==1)[1].item()
            label_class =class_names[label]

            if os.path.exists('/shared/hailey/figures4/{}/'.format(label_class)) == False:
                os.mkdir('/shared/hailey/figures4/{}/'.format(label_class))

            cls = np.argmax(self.model.predict(img_batch))

            grads_val, output = get_grads(img_batch, self.model, layer_name, cls, return_scores=False, return_logit=False)
            weights = np.mean(np.squeeze(grads_val), axis=(0, 1))

            layer_cam = np.dot(output, weights)
            layer_cam = np.maximum(layer_cam, 0)  # Passing through ReLU

            layer_cam = np.squeeze(layer_cam)

            if len(layer_cam.shape) < 2: continue
            superimposed_img, orig_img = self.superimpose(fns[0], layer_cam, False)
            superimposed_img_emp, _ = self.superimpose(fns[0], layer_cam, True)

            b, g, r = cv2.split(orig_img)  # img파일을 b,g,r로 분리
            orig_img = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge

            b, g, r = cv2.split(superimposed_img)  # img파일을 b,g,r로 분리
            superimposed_img = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge

            b, g, r = cv2.split(superimposed_img_emp)  # img파일을 b,g,r로 분리
            superimposed_img_emp = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge

            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
            fig.suptitle('label {}, pred {}'.format(label_class, class_names[cls]))

            ax[0].imshow(orig_img)
            ax[0].set_title("image")
            ax[1].imshow(superimposed_img)
            ax[1].set_title("activation")
            ax[2].imshow(superimposed_img_emp)
            ax[2].set_title("emhasized")
            file_name = fns[0].split('/')[-1]
            plt.savefig('/shared/hailey/figures4/{}/GradCAM-{}-{}'.format(label_class, layer_name,file_name))
            plt.close()
