from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import json
import tensorflow as tf
from tqdm import tqdm

from model import build_model
from confusion import ConfusiontDiscovery
from confusion_helpers import *

print("you are working on the tensorflow with version ",tf.__version__)

from data import DataGenerator


def save_cof(c_matrix, fn, cof_labels):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import pandas as pd

    c_matrix = pd.DataFrame(c_matrix, columns=cof_labels, index=cof_labels)
    c_matrix.to_csv('./files/{:s}.csv'.format(fn))

    plt.rcParams['figure.figsize'] = (35.0, 20.0)
    plt.rcParams['font.family'] = "serif"
    plt.rcParams['font.size'] = 15
    bg_color = (0.88, 0.85, 0.95)
    plt.rcParams['figure.facecolor'] = bg_color
    plt.rcParams['axes.facecolor'] = bg_color
    fig, ax = plt.subplots(1)
    # Setting the font scale
    sns.heatmap(c_matrix, annot=True, cmap='coolwarm',fmt='.0f',ax=ax, annot_kws={'size':15})
    plt.savefig('./files/{:s}.png'.format(fn), dpi=200)
    plt.close()


def draw_bar(data, num_classes, fname):
    # data = {'data_name':nd_array, 'data_name':nd_array, 'data_name':nd_array, 'data_name':nd_array,}
    xloc = [j for j in range(num_classes)]
    for i in range(num_classes):
        fig = plt.figure(figsize=(20, 40))  ## Figure 생성 사이즈는 10 by 10
        for idx, (x_name, arr) in enumerate(data.items()):
            ax = fig.add_subplot(4, 1, idx+1)  ## Axes 추가
            ith_vals = arr[i,:]
            bars = ax.bar(xloc, ith_vals, align='edge', color='silver')
            plt.xticks(xloc, xloc, fontsize=13)
            for k, b in enumerate(bars):
                ax.text(b.get_x() + b.get_width() * (1 / 2), b.get_height() + 0.1,
                        '{:,.3f}'.format(ith_vals[k]), ha='center', fontsize=13, rotation=45 )

        plt.savefig('./files/%s-class_%d.png' % (fname, i))
        plt.close()


def main(args):
    x_test = ['{:s}{:s}/{:s}'.format(args.data_dir,'val', x) for x in os.listdir(args.data_dir+'val/')]
    with open('{:s}{:s}'.format(args.data_dir,'AwA2-subset-labels.json'),'r') as f:
        labels = json.load(f)
        cof_labels={}
        for l in labels:
            idx=labels[l]
            cls = l.split('/')[-1].split('_')[0]
            cof_labels[idx]=cls
    cof_labels = [cls for idx, cls in cof_labels.items()]

    IMG_SIZE = 224
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
    params = {'batch_size': args.batch,
              'n_classes': 25,
              'shuffle': False,
              'model': args.model}

    validation_generator = DataGenerator(x_test, labels, **params)

    paths = ['VGG16-06-15_17:31-R/Adam-028-0.92628.hdf5', 'VGG16-06-15_14:07-R/SGD-043-0.91827.hdf5',
             'VGG16-06-15_11:13-P/SGD-020-0.92708.hdf5', 'VGG16-06-15_11:12-P/Adam-043-0.91146.hdf5',
             'ResNet50-06-28_21:41-P/SGD-007-0.92480.hdf5','ResNet50-06-28_21:41-P/SGD-038-0.94480.hdf5',
             'ResNet50-06-29_15:05-P/Adam-010-0.91120.hdf5']

    fn = ['VGG16-R-Adam-92.628', 'VGG16-R-SGD-91.827',
          'VGG16-P-SGD-92.708', 'VGG16-P-Adam-91.146',
          'ResNet50-P-SGD-92.48', 'ResNet50-P-SGD-94.48',
          'ResNet50-P-Adam-91.120']

    model_path = '/shared/hailey/%s' %paths[args.selection-1]

    model = build_model(args.mode, model_path=model_path, num_classes = 25, classifier_activation=None,
                        base_freeze=False)
    model.summary()

    print('build fin!')

    bottleneck = args.bottlenecks.split(',')
    if len(bottleneck) == 1: bottleneck = bottleneck[0]
    model_layers = model.layers
    t_layer = model.get_layer(bottleneck)
    t_idx = model.layers.index(t_layer)

    # ablation_logit_model = model for ablation research on target layer
    ablation_logit_model = create_submodel(t_idx, bottleneck, model, input_shape=model_layers[t_idx+1].input_shape)

    num_channels = model_layers[t_idx+1].input_shape[-1]

    # Creating the ConceptDiscovery class instance
    cd = ConceptDiscovery(
        model=model,
        datagenerator=validation_generator,
        num_classes=25,
        bottlenecks=args.bottlenecks.split(','),
        sub_model=ablation_logit_model,
        num_channels=num_channels,
    )

    cd.prediction_collection(feature_vote=True, prediction=True)

    #### feature vote
    feature_votes_correction = cd.feature_votes_correction
    feature_votes_violation = cd.feature_votes_violation
    file_name = '/shared/hailey/softmax-files/{}-corrections-{}-softmax.json'.format(fn[args.selection - 1], args.bottlenecks)
    with open(file_name, 'w') as f:
        json.dump(feature_votes_correction, f)

    file_name = '/shared/hailey/softmax-files/{}-violations-{}-softmax.json'.format(fn[args.selection - 1], args.bottlenecks)
    with open(file_name, 'w') as f:
        json.dump(feature_votes_violation, f)




def parse_arguments(argv):
  """Parses the arguments passed to the run.py script."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str,
      help='model selection for validation', default='VGG16')
  parser.add_argument('--batch', type=int, default=1, help='batch size to train ')
  parser.add_argument('--data_dir', type=str, default='/shared/AwA_sub/', help='data directory')
  parser.add_argument('--mode', type=str, default='inference', help='inference')
  parser.add_argument('--bottlenecks', type=str, default='block5_conv3', help='target layer')
  parser.add_argument('--selection', type=int, default=2, help='range 0 ~ 6')
  parser.add_argument('--act', type=str, default='softmax', help='None or softmax')

  return parser.parse_args(argv)

if __name__ == '__main__':
    print(time.strftime('%c', time.localtime(time.time())))

    main(parse_arguments(sys.argv[1:]))

