""" collection of various helper functions for running ACE"""

from multiprocessing import dummy as multiprocessing
import sys
import os
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
# import tcav.model as model
import numpy as np
from PIL import Image
from skimage.segmentation import mark_boundaries
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import tensorflow as tf

def make_model(model_to_run, model_path,input_shape=(32, 32, 3)):
  """Make an instance of a model.

  Args:
    sess: tf session instance.
    model_to_run: a string that describes which model to make.
    model_path: Path to models saved graph.
    randomize: Start with random weights
    labels_path: Path to models line separated class names text file.

  Returns:
    a model instance.

  Raises:
    ValueError: If model name is not valid.
  """
  mymodel = model_to_run.build_graph(input_shape)
  mymodel.load_weights(model_path)

  return mymodel

def create_submodel(t_idx, bottleneck,  model, input_shape=None, classifier_activation=None,save_name=None, summary=False):
    cfg = model.get_config()

    layers = cfg['layers']
    for i, l in enumerate(layers):
        if i == 0: layers[i]['config']['batch_input_shape'] = input_shape
        if i >= t_idx+1:
            for elt in layers[i]['inbound_nodes']:
                for e in elt:
                    if e[0] == bottleneck:
                        e[0] = 'input_1'

    sub_layers = []
    for i, l in enumerate(layers):
        if i  == 0 or i > t_idx: sub_layers.append(l)
        else:
            continue
    cfg['layers'] = sub_layers
    subModel = tf.keras.Model.from_config(cfg)
    if summary:
        subModel.summary()

    orig_layers = model.layers[t_idx+1:]

    for i,layer in enumerate(subModel.layers):
        if i == 0: continue
        subModel.layers[i].set_weights(orig_layers[i-1].get_weights())
    return subModel


def create_consecutives(t_idx, next_idx, bottleneck, next_layer, model, input_shape=None, summary=False, save_name=None):
  cfg = model.get_config()

  layers = cfg['layers']

  for i, l in enumerate(layers):
    if i == 0: layers[i]['config']['batch_input_shape'] = input_shape #input shape 바꾸고
    if i >= t_idx + 1:
      for elt in layers[i]['inbound_nodes']:
        for e in elt:
          if e[0] == bottleneck: # bottleneck layer를 input_1 layer로 바꾸기
            e[0] = 'input_1'

  sub_layers = []
  for i, l in enumerate(layers):
    if i == 0 or (i > t_idx and i <= next_idx):
      sub_layers.append(l)
    else:
      continue

  if isinstance(next_layer,str):
      cfg['output_layers'] = [[next_layer,0,0]]

  cfg['layers'] = sub_layers
  subModel = tf.keras.Model.from_config(cfg)
  if summary:
      subModel.summary()

  orig_layers = model.layers[t_idx + 1:]

  for i, layer in enumerate(subModel.layers):
    if i == 0: continue
    subModel.layers[i].set_weights(orig_layers[i - 1].get_weights())
  return subModel


def get_acts_from_images(imgs, model, bottleneck_name, return_pred=False, preprocess=False):
  """Run images in the model to get the activations.
  Args:
    imgs: a list of images
    model: a model instance
    bottleneck_name: bottleneck name to get the activation from
  Returns:
    numpy array of activations.
  """
  if preprocess:
    imgs = tf.keras.applications.vgg19.preprocess_input(imgs)  # converted to BGR

  gradModel = tf.keras.Model(
    inputs=[model.inputs],
    outputs=[model.get_layer(bottleneck_name).output,
             model.output])

  with tf.GradientTape() as tape:
    if imgs.shape[0] == 1:
      images = tf.cast(imgs, tf.float32)
      (convOutputs, predictions) = gradModel(images)
    else:
      convOutputs = []
      predictions = []
      for i in range(imgs.shape[0]):
        img = np.expand_dims(imgs[i,:,:,:],0)
        images = tf.cast(img, tf.float32)
        (conv, pred) = gradModel(images)
        convOutputs.append(conv)
        predictions.append(pred)
        del conv
        del pred
  del gradModel
  tf.keras.backend.clear_session()
  if return_pred:
    return np.asarray(convOutputs).squeeze(), np.asarray(predictions).squeeze()
  else: return np.asarray(convOutputs).squeeze()



def get_1st_acts_from_images(imgs, model, first_layer):
  gradModel = tf.keras.Model(
    inputs=[model.inputs],
    outputs=[model.get_layer(first_layer).output,
             model.output])

  with tf.GradientTape() as tape:
    if imgs.shape[0] == 1:
      images = tf.cast(imgs, tf.float32)
      (convOutputs, predictions) = gradModel(images)
    else:
      convOutputs = []
      predictions = []
      for i in range(imgs.shape[0]):
        img = np.expand_dims(imgs[i,:,:,:],0)
        images = tf.cast(img, tf.float32)
        (conv, pred) = gradModel(images)
        convOutputs.append(conv)
        predictions.append(pred)
        del conv
        del pred
  del gradModel
  tf.keras.backend.clear_session()
  return np.asarray(convOutputs)


def get_grads(imgs, model, bottleneck_name, idx, return_scores=False, return_logit=False):
  """Run images in the model to get the activations.
  Args:
    imgs: a list of images
    model: a model instance
    bottleneck_name: bottleneck name to get the activation from
  Returns:
    numpy array of activations.
  """
  imgs = tf.keras.applications.vgg19.preprocess_input(imgs)  # converted to BGR

  gradModel = tf.keras.Model(
    inputs=[model.inputs],
    outputs=[model.get_layer(bottleneck_name).output,
             model.get_layer('dense_3').output,
             model.output])
  with tf.GradientTape() as tape:
    inputs = tf.cast(imgs, tf.float32)
    (convOutputs, predictions, cls_scores) = gradModel(inputs)
    loss = predictions[:, idx]
    # use automatic differentiation to compute the gradients
  grads = tape.gradient(loss, convOutputs)
  del gradModel
  tf.keras.backend.clear_session()

  if return_scores==True and return_logit==True: return grads, convOutputs, cls_scores, loss
  if return_scores==True and return_logit==False: return grads, convOutputs, cls_scores
  if return_scores==False and return_logit==True: return grads, convOutputs, loss
  if return_scores==False and return_logit==False: return grads, convOutputs

def get_sens(imgs, model, bottleneck_name, idx, return_scores=False, return_logit=False):
  """Run images in the model to get the activations.
  Args:
    imgs: a list of images
    model: a model instance
    bottleneck_name: bottleneck name to get the activation from
  Returns:
    numpy array of activations.
  """

  imgs = tf.keras.applications.vgg19.preprocess_input(imgs)  # converted to BGR

  gradModel = tf.keras.Model(
    inputs=[model.inputs],
    outputs=[model.get_layer(bottleneck_name).output,
             model.get_layer('predictions').output,
             model.output])
  with tf.GradientTape() as tape:
    inputs = tf.cast(imgs, tf.float32)
    (convOutputs, predictions, cls_scores) = gradModel(inputs)
    loss =  tf.math.reduce_mean(
          input_tensor=tf.nn.softmax_cross_entropy_with_logits(
              labels=tf.one_hot(idx, predictions.shape[-1]),
              logits=predictions))
    # use automatic differentiation to compute the gradients
  grads = tape.gradient(loss, convOutputs)
  del gradModel
  tf.keras.backend.clear_session()

  if return_scores==True and return_logit==True: return grads, convOutputs, cls_scores, loss
  if return_scores==True and return_logit==False: return grads, convOutputs, cls_scores
  if return_scores==False and return_logit==True: return grads, convOutputs, loss
  if return_scores==False and return_logit==False: return grads, convOutputs

