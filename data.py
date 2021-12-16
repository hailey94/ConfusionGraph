import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, n_classes=1000, shuffle=True, model='VGG16'):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.model = model
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y, fn = self.__data_generation(list_IDs_temp)
        return X, y, fn

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = []
        y = []
        fn = []
        # Generate data
        for i, filename in enumerate(list_IDs_temp):
            # Store class
            fn.append(filename)
            label = self.labels[filename]
            if not tf.io.gfile.exists(filename):
                tf.logging.error('Cannot find file: {}'.format(filename))
                return None
            try:
                img = np.array(Image.open(filename))
                if (len(img.shape) < 3):
                    w, h = img.shape
                    tmp = np.zeros([w, h, 3])
                    for i in range(3):
                        tmp[:, :, i] = img
                    img = tmp
                # Resize
                height, width, _ = img.shape
                new_height = height * 256 // min(img.shape[:2])
                new_width = width * 256 // min(img.shape[:2])

                img = tf.image.resize(img, [new_width, new_height])

                # Crop
                height, width, _ = img.shape
                startx = width // 2 - (224 // 2)
                starty = height // 2 - (224 // 2)
                img = img[starty:starty + 224, startx:startx + 224]
                assert img.shape[0] == 224 and img.shape[1] == 224
                if self.model == 'MobileNet':
                    img =tf.keras.applications.mobilenet.preprocess_input(img)
                elif self.model == 'ResNet50':
                    img =tf.keras.applications.resnet.preprocess_input(img)
                elif self.model == 'ResNet101':
                    img = tf.keras.applications.resnet.preprocess_input(img)
                elif self.model == 'Xception':
                    img = tf.keras.applications.xception.preprocess_input(img)
                elif self.model == 'VGG19':
                    img = tf.keras.applications.vgg19.preprocess_input(img)
                elif self.model == 'VGG16':
                    img = tf.keras.applications.vgg16.preprocess_input(img)

            except Exception as e:
                tf.logging.info(e)
                return None

            X.append(img)
            y.append(label)
        X = np.array(X)
        y = np.array(y)
        y = tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y , fn# keras.utils.to_categorical(y, num_classes=self.n_classes) #One-hot encoding


class DataGenerator_Sample(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, n_classes=1000, shuffle=True, model='VGG16'):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.model = model
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, y, fn = self.__data_generation(list_IDs_temp)
        return X, y, fn

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = []
        y = []
        fn = []
        # Generate data
        for i, filename in enumerate(list_IDs_temp):
            # Store class
            fn.append(filename)
            label = self.labels[filename]

            if not tf.io.gfile.exists(filename):
                tf.logging.error('Cannot find file: {}'.format(filename))
                return None
            try:
                img = np.array(Image.open(filename))
                if (len(img.shape) < 3):
                    w, h = img.shape
                    tmp = np.zeros([w, h, 3])
                    for i in range(3):
                        tmp[:, :, i] = img
                    img = tmp
                # Resize
                height, width, _ = img.shape
                new_height = height * 256 // min(img.shape[:2])
                new_width = width * 256 // min(img.shape[:2])
                
                img = tf.image.resize(img, [new_width, new_height])

                # Crop
                height, width, _ = img.shape
                startx = width // 2 - (224 // 2)
                starty = height // 2 - (224 // 2)
                img = img[starty:starty + 224, startx:startx + 224]
                assert img.shape[0] == 224 and img.shape[1] == 224
                if self.model == 'MobileNet':
                    img =tf.keras.applications.mobilenet.preprocess_input(img)
                elif self.model == 'ResNet50':
                    img =tf.keras.applications.resnet.preprocess_input(img)
                elif self.model == 'ResNet101':
                    img = tf.keras.applications.resnet.preprocess_input(img)
                elif self.model == 'Xception':
                    img = tf.keras.applications.xception.preprocess_input(img)
                elif self.model == 'VGG19':
                    img = tf.keras.applications.vgg19.preprocess_input(img)
                elif self.model == 'VGG16':
                    img = tf.keras.applications.vgg16.preprocess_input(img)

            except Exception as e:
                tf.logging.info(e)
                return None

            X.append(img)
            y.append(label)
        X = np.array(X)
        y = np.array(y)
        y = tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y , fn# keras.utils.to_categorical(y, num_classes=self.n_classes) #One-hot encoding
