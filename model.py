import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.applications import ResNet50, ResNet101, MobileNet, Xception, VGG19, VGG16



def build_model(mode, model_name = None, model_path = None, num_classes=5, classifier_activation=None,
                base_freeze=False, weights='imagenet'):

    # clear_session()
    if model_path :
        model_name = model_path.split('/')[-2]
        model_name = model_name.split('-')[0]

    img = Input(shape = (None,None, 3)) #224,224,3

    if model_name == 'MobileNet':

        model = MobileNet(include_top=False,
                          weights=weights,
                          input_tensor=img,
                          input_shape=None,
                          pooling='avg')

    elif model_name == 'ResNet50':

        model = ResNet50(include_top=False,
                         weights= weights,#'imagenet',
                         input_tensor=img,
                         input_shape=None,
                         pooling='avg')

    elif model_name == 'ResNet101':

        model = ResNet101(include_top=False,
                         weights=None,#'imagenet',
                         input_tensor=img,
                         input_shape=None,
                         pooling='avg')

    elif model_name == 'Xception':

        model = Xception(include_top=False,
                         weights=weights,
                         input_tensor=img,
                         input_shape=None,
                         pooling='avg')

    elif model_name == 'VGG19':

        model = VGG19(include_top=False,
                         weights= weights,# [None,'imagenet']
                         input_tensor=img,
                         input_shape=None,
                         pooling='avg')

    elif model_name == 'VGG16':

        model = VGG16(include_top=False,
                         weights= weights,#'imagenet',
                         input_tensor=img,
                         input_shape=None,
                         pooling='avg')

    model.trainable = base_freeze

    final_layer = model.layers[-1].output
    dense_layer_1 = Dense(4096, activation = 'relu')(final_layer)
    dense_layer_2 = Dense(2048, activation = 'relu')(dense_layer_1)
    dense_layer_3 = Dense(1024, activation = 'relu')(dense_layer_2)
    output_layer = Dense(num_classes, activation=classifier_activation)(dense_layer_3)
    model = Model(inputs = img, outputs = output_layer)

    if mode == 'inference':
        model.load_weights(model_path)
        model.trainable = False
    print(mode)
    # m_weights = model.get_weights()
    # for w in m_weights:
    #     print(w.shape)
    # for i,layer in enumerate(model.layers):
    #     if len(layer.weights) != 0:
    #         print(layer.weights[0], layer.weights[1])
    return model
