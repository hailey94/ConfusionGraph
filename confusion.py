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

class ConfusiontDiscovery(object):
  """Discovering and testing concepts of a class.

  For a trained network, it first discovers the concepts as areas of the iamges
  in the class and then calculates the TCAV score of each concept. It is also
  able to transform images from pixel space into concept space.
  """

  def __init__(self,
               model,
               datagenerator,
               layer_orders,
               bottlenecks,
               sub_model=None,
               num_channels=None,
               ):

    self.model = model
    self.layer_orders = layer_orders
    self.datagenerator = datagenerator
    if isinstance(bottlenecks, str):
      self.bottlenecks = [bottlenecks]
    else: self.bottlenecks = bottlenecks
    self.sub_model = sub_model

    self.layer_idxs = {}
    for layer in self.layer_orders:
      self.layer_idxs[layer] = self.model.layers.index(self.model.get_layer(layer))

    self.feature_votes_allpass = {x:{} for x in range(num_channels)}
    self.feature_votes_correction = {x: {} for x in range(num_channels)}
    self.feature_votes_violation = {x: {} for x in range(num_channels)}

    self.cnt_cor = 0
    self.cnt_all_p = 0
    self.cnt_viol = 0

    self.class_names = {0: 'grizzly+bear', 1: 'bobcat', 2: 'elephant', 3: 'gorilla', 4: 'rhinoceros', 5: 'dalmatian',
                   6: 'dolphin', 7: 'antelope', 8: 'otter', 9: 'german+shepherd', 10: 'cow', 11: 'hamster', 12: 'collie',
                   13: 'chimpanzee', 14: 'rabbit', 15: 'deer', 16: 'wolf', 17: 'fox', 18: 'humpback+whale', 19: 'siamese+cat', 20: 'ox',
                   21: 'giraffe', 22: 'seal', 23: 'tiger', 24: 'polar+bear'}

  def jensen_shannon_div(self,p,q):
    m = 0.5*(p+q)
    return(0.5*tf.keras.metrics.kl_divergence(p, m) + 0.5*tf.keras.metrics.kl_divergence(q, m))

  def cosine_similarity(self,a, b):
    """Cosine similarity of two vectors."""
    assert a.shape == b.shape, 'Two vectors must have the same dimensionality'
    a_norm, b_norm = np.linalg.norm(a), np.linalg.norm(b)
    if a_norm * b_norm == 0:
      return 0.
    cos_sim = np.sum(a * b) / (a_norm * b_norm)
    return cos_sim

  def prediction_collection(self, prediction=False,
                            return_dist=False, feature_vote=False, return_feature=False):

    for bottleneck in self.bottlenecks:
      t_layer = self.model.get_layer(bottleneck)
      t_idx = self.model.layers.index(t_layer)

      # ablation_logit_model = model for ablation research on target layer
      self.sub_model = create_submodel(t_idx, bottleneck, self.model, input_shape=self.model.layers[t_idx + 1].input_shape)
      bottleneck_dist = [0, 0, 0]
      cnt = 0

      for epoch, (img_batch, label_batch, fns) in tqdm(enumerate(self.datagenerator)):

        if prediction:
          self._return_predictions(img_batch, label_batch, fns, bottleneck, epoch,
                                                           feature_vote=feature_vote)

        if return_dist:
          jensen_shannon_div, zero_out_to_total, cosine, count_for_one_image_in_one_layer = self._return_distances(img_batch, bottleneck)
          cnt += count_for_one_image_in_one_layer
          bottleneck_dist[0] += jensen_shannon_div
          bottleneck_dist[1] += zero_out_to_total
          bottleneck_dist[2] += cosine

        if return_feature:
          self._return_features(img_batch, label_batch, fns, bottleneck, feature_vote=feature_vote)

      print('\n\n{}'.format(bottleneck))
      print('jsd: {}'.format(bottleneck_dist[0] / cnt))
      print('zero_out_to_total: {}'.format(bottleneck_dist[1] / cnt))
      print('cosine sim: {}'.format(bottleneck_dist[2] / cnt))
      print('cnt: {}'.format(cnt))

  def _return_predictions(self, img, label_batch, fns, bottleneck, epoch, feature_vote=False):
    convOutputs, pred = get_acts_from_images(img, self.model, bottleneck, return_pred=True)
    convOutputs = convOutputs.copy()
    pred = tf.nn.softmax(pred)

    for i in range(convOutputs.shape[-1]):
      if len(convOutputs.shape) == 4:
        fwd_tensor = convOutputs[:, :, :, i].copy()
        zeros = np.zeros(fwd_tensor.shape)
        convOutputs[:, :, :, i] = zeros
        logits = self.sub_model(convOutputs)
        logits = tf.nn.softmax(logits)
        convOutputs[:, :, :, i] = fwd_tensor

      elif len(convOutputs.shape) == 2:
        fwd_tensor = convOutputs[:, i].copy()
        zeros = np.zeros(fwd_tensor.shape)
        convOutputs[:, i] = zeros
        logits = self.sub_model(convOutputs)
        logits = tf.nn.softmax(logits)
        convOutputs[:, i] = fwd_tensor


      if feature_vote:
        for (label_np, prediction_np, prediction_a_np, fn) in zip(label_batch, pred, logits, fns):
          prediction = int(np.argmax(prediction_np))
          prediction_a = int(np.argmax(prediction_a_np))
          ground_truth = int(np.argmax(label_np))

          fn = fn.split('/')[-1]
          key = '%d_%d_to_%d' %(ground_truth, prediction, prediction_a)

          J_gt_pred = float(self.jensen_shannon_div(label_np, prediction_np).numpy())
          J_gt_abl = float(self.jensen_shannon_div(label_np, prediction_a_np).numpy())
          J_pred_abl = float(self.jensen_shannon_div(prediction_np, prediction_a_np).numpy())

          if (ground_truth != prediction) and(prediction_a != prediction) and (prediction_a != ground_truth):
            continue

          if (ground_truth == prediction) and (prediction_a == prediction):
            if key not in self.feature_votes_allpass[i].keys():
              self.feature_votes_allpass[i][key] = {}  # 1
              self.feature_votes_allpass[i][key] = {'count': 1}
              self.feature_votes_allpass[i][key]['image'] = [fn]
              self.feature_votes_allpass[i][key]['jsd'] = [[J_gt_pred, J_gt_abl, J_pred_abl]]
            else:
              self.feature_votes_allpass[i][key]['count'] += 1
              self.feature_votes_allpass[i][key]['image'].append(fn)
              self.feature_votes_allpass[i][key]['jsd'].append([J_gt_pred, J_gt_abl, J_pred_abl])

          if (ground_truth != prediction) and (prediction_a == ground_truth):
            if key not in self.feature_votes_correction[i].keys():
              self.feature_votes_correction[i][key] = {} #1
              self.feature_votes_correction[i][key] = {'count':1}
              self.feature_votes_correction[i][key]['image'] = [fn]
              self.feature_votes_correction[i][key]['jsd'] = [[J_gt_pred, J_gt_abl, J_pred_abl]]

            else:
              self.feature_votes_correction[i][key]['count'] += 1
              self.feature_votes_correction[i][key]['image'].append(fn)
              self.feature_votes_correction[i][key]['jsd'].append([J_gt_pred, J_gt_abl, J_pred_abl])

          if (ground_truth == prediction) and (prediction_a != ground_truth):
            if key not in self.feature_votes_violation[i].keys():
              self.feature_votes_violation[i][key] = {}  # 1
              self.feature_votes_violation[i][key] = {'count': 1}
              self.feature_votes_violation[i][key]['image'] = [fn]
              self.feature_votes_violation[i][key]['jsd'] = [[J_gt_pred, J_gt_abl, J_pred_abl]]
            else:
              self.feature_votes_violation[i][key]['count'] += 1
              self.feature_votes_violation[i][key]['image'].append(fn)
              self.feature_votes_violation[i][key]['jsd'].append([J_gt_pred, J_gt_abl, J_pred_abl])

  def _return_distances(self, img, bottleneck):
    from copy import deepcopy
    convOutputs, pred = get_acts_from_images(img, self.model, bottleneck, return_pred=True)
    orig_conv = deepcopy(convOutputs)
    convOutputs = convOutputs.copy()

    jensen_shannon_div = 0
    zero_out_to_total =0
    cosine = 0
    count_for_one_image_in_one_layer = 0
    if len(pred.shape) > 1:
      p = np.argmax(pred, axis=1)
    else:
      p = np.argmax(pred)

    for b in range(convOutputs.shape[-1]):
      if len(convOutputs.shape) > 2:
        fwd_tensor = convOutputs[:, :, b].copy()
        zeros = np.zeros(fwd_tensor.shape)
        convOutputs[:, :, b] = zeros
        logits = self.sub_model(np.expand_dims(convOutputs,0))
        abl_pred = np.argmax(logits, axis=1)

      elif len(convOutputs.shape) == 2:
        fwd_tensor = convOutputs[:, b].copy()
        zeros = np.zeros(fwd_tensor.shape)
        convOutputs[:, b] = zeros
        logits = self.sub_model(np.expand_dims(convOutputs, 0))
        abl_pred = np.argmax(logits, axis=1)

      else:
        fwd_tensor = convOutputs[b].copy()
        zeros = np.zeros(fwd_tensor.shape)
        convOutputs[b] = zeros
        logits = self.sub_model(np.expand_dims(convOutputs, 0))
        abl_pred = np.argmax(logits, axis=1)


      if abl_pred != p:
        jensen_shannon_div += np.sum(self.jensen_shannon_div(orig_conv,convOutputs))
        zero_out_to_total += np.sum(np.abs(fwd_tensor)) /  np.sum(np.abs(orig_conv))
        cosine += self.cosine_similarity(orig_conv,convOutputs)
        count_for_one_image_in_one_layer += 1

      if len(convOutputs.shape) > 2:
        convOutputs[:, :, b] = fwd_tensor
      elif len(convOutputs.shape) == 2:
        convOutputs[:, b] = fwd_tensor
      else:
        convOutputs[b] = fwd_tensor

    return jensen_shannon_div, zero_out_to_total, cosine, count_for_one_image_in_one_layer



  def _return_features(self):
    frm = 20 # ox
    to = 10 # cow
    for epoch, (img_batch, label_batch, fns) in tqdm(enumerate(self.datagenerator)):

      assert img_batch.shape[0] == 1
      label_class = np.argmax(label_batch)
      if (label_class != frm):
        pass
      convOutputs, pred = get_acts_from_images(img_batch, self.model, self.bottlenecks[0], return_pred=True)
      convOutputs = convOutputs.copy()
      pred = tf.nn.softmax(pred)
      prediction_class = np.argmax(pred)
      convOutputs = np.expand_dims(convOutputs,axis=0)

      for i in range(convOutputs.shape[-1]):
        if len(convOutputs.shape) == 4:
          fwd_tensor = convOutputs[:, :, :, i].copy()
          zeros = np.zeros(fwd_tensor.shape)
          convOutputs[:, :, :, i] = zeros
          logits = self.sub_model(convOutputs)
          logits = tf.nn.softmax(logits)
          convOutputs[:, :, :, i] = fwd_tensor

        elif len(convOutputs.shape) == 2:
          fwd_tensor = convOutputs[:, i].copy()
          zeros = np.zeros(fwd_tensor.shape)
          convOutputs[:, i] = zeros
          logits = self.sub_model(convOutputs)
          logits = tf.nn.softmax(logits)
          convOutputs[:, i] = fwd_tensor

        logits = np.squeeze(logits,axis=0)
        ablation_class = np.argmax(logits)
        # non changed
        if label_class == prediction_class == ablation_class:
          pass
          # print('\nStrict on class {}, layer {}'.format(self.class_names[frm], self.bottlenecks[0]))
          # print('model prediction: on {} = {:.4f}, {} = {:.4f}'.format(self.class_names[prediction_class], pred[prediction_class], self.class_names[to], pred[to]))

        # correction
        elif (label_class == frm) and (prediction_class == to) and (ablation_class == frm):
            print('\nCorrection from class {} to {}, layer {}'.format(self.class_names[to],self.class_names[frm], self.bottlenecks[0]))
            print('model prediction: on {} = {:.4f}, {} = {:.4f}'.format(self.class_names[prediction_class],
                                                                         pred[prediction_class], self.class_names[to],
                                                                         pred[to]))
            print('ablation prediction: on {} = {:.4f}, {} = {:.4f}'.format(self.class_names[prediction_class],
                                                                         logits[prediction_class], self.class_names[label_class],
                                                                         logits[label_class]))

        # violation
        elif (label_class == frm) and (prediction_class==frm) and (ablation_class == to):
            print('\nViolation from class {} to {}, layer {}'.format(self.class_names[frm],self.class_names[to], self.bottlenecks[0]))
            print('model prediction: on {} = {:.4f}, {} = {:.4f}'.format(self.class_names[prediction_class],
                                                                         pred[prediction_class], self.class_names[to],
                                                                         pred[to]))
            print('ablation prediction: on {} = {:.4f}, {} = {:.4f}'.format(self.class_names[prediction_class],
                                                                            logits[prediction_class],
                                                                            self.class_names[to],
                                                                            logits[to]))

  def _return_consecutive_features(self):
    frm = 20 # ox
    to = 10 # cow
    for epoch, (img_batch, label_batch, fns) in tqdm(enumerate(self.datagenerator)):

      assert img_batch.shape[0] == 1
      label_class = np.argmax(label_batch)
      if (label_class != frm):
        pass
      convOutputs, pred = get_acts_from_images(img_batch, self.model, self.bottlenecks[0], return_pred=True)
      convOutputs = convOutputs.copy()
      pred = tf.nn.softmax(pred)
      prediction_class = np.argmax(pred)
      convOutputs = np.expand_dims(convOutputs,axis=0)

      for i in range(convOutputs.shape[-1]):
        if len(convOutputs.shape) == 4:
          fwd_tensor = convOutputs[:, :, :, i].copy()
          zeros = np.zeros(fwd_tensor.shape)
          convOutputs[:, :, :, i] = zeros
          logits = self.sub_model(convOutputs)
          logits = tf.nn.softmax(logits)
          convOutputs[:, :, :, i] = fwd_tensor

        elif len(convOutputs.shape) == 2:
          fwd_tensor = convOutputs[:, i].copy()
          zeros = np.zeros(fwd_tensor.shape)
          convOutputs[:, i] = zeros
          logits = self.sub_model(convOutputs)
          logits = tf.nn.softmax(logits)
          convOutputs[:, i] = fwd_tensor

        logits = np.squeeze(logits,axis=0)
        ablation_class = np.argmax(logits)
        # non changed
        if label_class == prediction_class == ablation_class:
          pass
          # print('\nStrict on class {}, layer {}'.format(self.class_names[frm], self.bottlenecks[0]))
          # print('model prediction: on {} = {:.4f}, {} = {:.4f}'.format(self.class_names[prediction_class], pred[prediction_class], self.class_names[to], pred[to]))

        # correction
        elif (label_class == frm) and (prediction_class == to) and (ablation_class == frm):
            print('\nCorrection from class {} to {}, layer {}'.format(self.class_names[to],self.class_names[frm], self.bottlenecks[0]))
            print('model prediction: on {} = {:.4f}, {} = {:.4f}'.format(self.class_names[prediction_class],
                                                                         pred[prediction_class], self.class_names[to],
                                                                         pred[to]))
            print('ablation prediction: on {} = {:.4f}, {} = {:.4f}'.format(self.class_names[prediction_class],
                                                                         logits[prediction_class], self.class_names[label_class],
                                                                         logits[label_class]))

        # violation
        elif (label_class == frm) and (prediction_class==frm) and (ablation_class == to):
            print('\nViolation from class {} to {}, layer {}'.format(self.class_names[frm],self.class_names[to], self.bottlenecks[0]))
            print('model prediction: on {} = {:.4f}, {} = {:.4f}'.format(self.class_names[prediction_class],
                                                                         pred[prediction_class], self.class_names[to],
                                                                         pred[to]))
            print('ablation prediction: on {} = {:.4f}, {} = {:.4f}'.format(self.class_names[prediction_class],
                                                                            logits[prediction_class],
                                                                            self.class_names[to],
                                                                            logits[to]))

  def sigmoid(self, x, a, b, c):
    return c / (1 + np.exp(-a * (x - b)))

  def sigmoid2(self, x, a, b, c):
    return c / (1 + np.exp(-a * (b - x)))

  def superimpose(self, fn, cam, emphasize=False):
    img_bgr = cv2.imread(fn, cv2.IMREAD_COLOR)
    img_bgr = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
    if np.max(cam) != np.min(cam):
      cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))

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

  def superimpose2(self, fn, cam, emphasize=False):
    img_bgr = cv2.imread(fn, cv2.IMREAD_COLOR)
    img_bgr = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
    if np.max(cam) != np.min(cam):
      cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))

    heatmap = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))

    if emphasize:
      heatmap = self.sigmoid2(heatmap, 50, 0.5, 1)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    hif = .8
    superimposed_img = heatmap * hif + img_bgr
    superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)  # scale 0 to 255
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    return superimposed_img_rgb, img_bgr


  def _return_heatmaps(self):
    for bottleneck in self.bottlenecks:
      t_layer = self.model.get_layer(bottleneck)
      t_idx = self.model.layers.index(t_layer)

      # ablation_logit_model = model for ablation research on target layer
      self.sub_model = create_submodel(t_idx, bottleneck, self.model,
                                       input_shape=self.model.layers[t_idx + 1].input_shape)


      for epoch, (img_batch, label_batch, fns) in tqdm(enumerate(self.datagenerator)):
        convOutputs, pred = get_acts_from_images(img_batch, self.model, bottleneck, return_pred=True)
        convOutputs = convOutputs.copy()

        if len(pred.shape) > 1:
          p = np.argmax(pred, axis=1)
        else:
          p = np.argmax(pred)

        if self.class_names[p] not in ['ox', 'cow']: continue

        for b in range(convOutputs.shape[-1]):
          if len(convOutputs.shape) > 2:
            fwd_tensor = convOutputs[:, :, b].copy()
            zeros = np.zeros(fwd_tensor.shape)
            convOutputs[:, :, b] = zeros
            logits = self.sub_model(np.expand_dims(convOutputs,0))
            abl_pred = np.argmax(logits, axis=1)
            convOutputs[:, :, b] = fwd_tensor

          elif len(convOutputs.shape) == 2:
            fwd_tensor = convOutputs[:, b].copy()
            zeros = np.zeros(fwd_tensor.shape)
            convOutputs[:, b] = zeros
            logits = self.sub_model(np.expand_dims(convOutputs, 0))
            abl_pred = np.argmax(logits, axis=1)
            convOutputs[:, b] = fwd_tensor

          else:
            print(convOutputs.shape)
            exit(1)

          if abl_pred != p:
            layer_cam = np.maximum(fwd_tensor, 0)  # Passing through ReLU
            layer_cam = np.squeeze(layer_cam)

            if len(layer_cam.shape) < 2: continue
            superimposed_img, orig_img = self.superimpose(fns[0], layer_cam, False)
            superimposed_img_emp, _ = self.superimpose(fns[0], layer_cam, True)

            blue, g, r = cv2.split(orig_img)  # img파일을 b,g,r로 분리
            orig_img = cv2.merge([r, g, blue])  # b, r을 바꿔서 Merge

            bblue, g, r = cv2.split(superimposed_img)  # img파일을 b,g,r로 분리
            superimposed_img = cv2.merge([r, g, blue])  # b, r을 바꿔서 Merge

            bblue, g, r = cv2.split(superimposed_img_emp)  # img파일을 b,g,r로 분리
            superimposed_img_emp = cv2.merge([r, g, blue])  # b, r을 바꿔서 Merge

            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

            fig.suptitle('from {}, to {}'.format(self.class_names[p], self.class_names[abl_pred[0]]))

            ax[0].imshow(orig_img)
            ax[0].set_title("image")
            ax[1].imshow(superimposed_img)
            ax[1].set_title("activation")
            ax[2].imshow(superimposed_img_emp)
            ax[2].set_title("emhasized")

            file_name = '{}-{}-frm_{}-to-{}-{}.jpg'.format(bottleneck,b,p,abl_pred[0],fns[0].split('/')[-1].split('.')[0])
            plt.savefig('/shared/hailey/ox-cow/{}'.format(file_name))
            plt.close()


  def save_cams(self, layer_cam, fns, file_name):

    layer_cam = np.squeeze(layer_cam)
    if len(layer_cam.shape) < 2: pass
    superimposed_img, orig_img = self.superimpose(fns[0], layer_cam, False)
    superimposed_img_emp, _ = self.superimpose(fns[0], layer_cam, True)

    blue, g, r = cv2.split(orig_img)  # img파일을 b,g,r로 분리
    orig_img = cv2.merge([r, g, blue])  # b, r을 바꿔서 Merge

    bblue, g, r = cv2.split(superimposed_img)  # img파일을 b,g,r로 분리
    superimposed_img = cv2.merge([r, g, blue])  # b, r을 바꿔서 Merge

    bblue, g, r = cv2.split(superimposed_img_emp)  # img파일을 b,g,r로 분리
    superimposed_img_emp = cv2.merge([r, g, blue])  # b, r을 바꿔서 Merge

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    ax[0].imshow(orig_img)
    ax[0].set_title("image")
    ax[1].imshow(superimposed_img)
    ax[1].set_title("activation")
    ax[2].imshow(superimposed_img_emp)
    ax[2].set_title("emhasized")

    plt.savefig('{}'.format(file_name))
    plt.close()

  def _return_heatmaps_sep(self):
    for bottleneck in self.bottlenecks:
      t_layer = self.model.get_layer(bottleneck)
      t_idx = self.model.layers.index(t_layer)

      # ablation_logit_model = model for ablation research on target layer
      self.sub_model = create_submodel(t_idx, bottleneck, self.model,
                                       input_shape=self.model.layers[t_idx + 1].input_shape)


      for epoch, (img_batch, label_batch, fns) in tqdm(enumerate(self.datagenerator)):
        convOutputs, pred = get_acts_from_images(img_batch, self.model, bottleneck, return_pred=True)
        convOutputs = convOutputs.copy()

        if len(pred.shape) > 1:
          p = np.argmax(pred, axis=1)
        else:
          p = np.argmax(pred)

        prob1 = pred[p]

        if self.class_names[p] not in ['ox', 'cow']: continue
        layer_cam_conf = np.zeros(convOutputs.shape[:-1])
        layer_cam_robust = np.zeros(convOutputs.shape[:-1])

        for b in range(convOutputs.shape[-1]):
          if len(convOutputs.shape) > 2:
            fwd_tensor = convOutputs[:, :, b].copy()
            zeros = np.zeros(fwd_tensor.shape)
            convOutputs[:, :, b] = zeros
            logits = self.sub_model(np.expand_dims(convOutputs,0))
            abl_pred = np.argmax(logits, axis=1)
            convOutputs[:, :, b] = fwd_tensor

          elif len(convOutputs.shape) == 2:
            fwd_tensor = convOutputs[:, b].copy()
            zeros = np.zeros(fwd_tensor.shape)
            convOutputs[:, b] = zeros
            logits = self.sub_model(np.expand_dims(convOutputs, 0))
            abl_pred = np.argmax(logits, axis=1)
            convOutputs[:, b] = fwd_tensor

          else:
            print(convOutputs.shape)
            exit(1)

          if self.class_names[p] not in ['ox', 'cow']: continue

          prob2 = logits[:,abl_pred[0]]
          delta = (prob1 - prob2) / prob1

          if abl_pred != p:
            layer_cam_conf += np.maximum(delta*fwd_tensor, 0)  # Passing through ReLU

          if abl_pred == p:
            layer_cam_robust += np.maximum(delta*layer_cam_robust, 0)  # Passing through ReLU

          file_name = '/shared/hailey/robust/{}-robust-{}-{}.jpg'.format(bottleneck, self.class_names[p], fns[0].split('/')[-1].split('.')[0])
          self.save_cams(layer_cam_robust, fns, file_name)

          file_name = '/shared/hailey/confused/{}-confused-{}-{}.jpg'.format(bottleneck, self.class_names[p], fns[0].split('/')[-1].split('.')[0])
          self.save_cams(layer_cam_conf, fns, file_name)

  def _return_consecutive_results(self):
      frm = 20  # ox 20
      to = 10  # cow 1-
      for epoch, (img_batch, label_batch, fns) in tqdm(enumerate(self.datagenerator)):
          file_name = fns[0].split('/')[-1]
          assert img_batch.shape[0] == 1
          label_class = np.argmax(label_batch)
          if (label_class != frm):
            continue
          for i, layer in enumerate(self.layer_orders):
              t_idx = self.layer_idxs[layer]
              if not os.path.exists('/shared/hailey/ox-cow/{}/'.format(self.class_names[frm])):
                os.mkdir('/shared/hailey/ox-cow/{}/'.format(self.class_names[frm]))

              savedir='/shared/hailey/ox-cow/{}/{}'.format(self.class_names[frm], layer)
              if not os.path.exists(savedir): os.mkdir(savedir)

              convOutputs, pred = get_acts_from_images(img_batch, self.model, layer, return_pred=True)
              convOutputs = np.expand_dims(convOutputs, axis=0)
              convOutputs = convOutputs.copy()
              pred = tf.nn.softmax(pred)
              ablation_logit_model = create_submodel(t_idx, layer, self.model,
                                                     input_shape=self.model.layers[t_idx + 1].input_shape)
              correction_count = 0
              violation_count = 0
              all_same_count = 0

              for i in range(convOutputs.shape[-1]):
                  if len(convOutputs.shape) == 4:
                      fwd_tensor = convOutputs[:, :, :, i].copy()
                      zeros = np.zeros(fwd_tensor.shape)
                      convOutputs[:, :, :, i] = zeros
                      logits = ablation_logit_model(convOutputs)
                      logits = tf.nn.softmax(logits)
                      convOutputs[:, :, :, i] = fwd_tensor

                  else:
                    continue

                  prediction = int(np.argmax(pred))
                  prediction_a = int(np.argmax(logits))

                  if (label_class == prediction) and (prediction_a == prediction):
                    all_same_count += 1

                  if (label_class != prediction) and (prediction_a == label_class):
                    correction_count += 1
                    np.save('{}/correction_{}-to-{}_{}.npy'.format(savedir,self.class_names[prediction], self.class_names[prediction_a],file_name),fwd_tensor)

                  if (label_class == prediction) and (prediction_a != label_class):
                    violation_count += 1
                    np.save('{}/violation_{}-to-{}_{}.npy'.format(savedir,self.class_names[label_class], self.class_names[prediction_a],file_name),fwd_tensor)

              if all_same_count == convOutputs.shape[-1]:
                grads_val, output = get_grads(img_batch, self.model, layer, label_class, return_scores=False,
                                              return_logit=False)
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
                fig.suptitle('label {}, pred {}'.format(label_class, self.class_names[label_class]))

                ax[0].imshow(orig_img)
                ax[0].set_title("image")
                ax[1].imshow(superimposed_img)
                ax[1].set_title("activation")
                ax[2].imshow(superimposed_img_emp)
                ax[2].set_title("emhasized")
                plt.savefig('/shared/hailey/ox-cow/{}/GradCAM-{}-{}'.format(self.class_names[label_class], layer, file_name))
                plt.close()

                superimposed_img, orig_img = self.superimpose2(fns[0], layer_cam, False)
                superimposed_img_emp, _ = self.superimpose2(fns[0], layer_cam, True)

                b, g, r = cv2.split(orig_img)  # img파일을 b,g,r로 분리
                orig_img = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge

                b, g, r = cv2.split(superimposed_img)  # img파일을 b,g,r로 분리
                superimposed_img = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge

                b, g, r = cv2.split(superimposed_img_emp)  # img파일을 b,g,r로 분리
                superimposed_img_emp = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge

                fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
                fig.suptitle('label {}, pred {}'.format(label_class, self.class_names[label_class]))

                ax[0].imshow(orig_img)
                ax[0].set_title("image")
                ax[1].imshow(superimposed_img)
                ax[1].set_title("activation")
                ax[2].imshow(superimposed_img_emp)
                ax[2].set_title("emhasized")
                file_name = fns[0].split('/')[-1]
                plt.savefig('/shared/hailey/ox-cow/{}/GradCAM2-{}-{}'.format(self.class_names[label_class], layer, file_name))
                plt.close()