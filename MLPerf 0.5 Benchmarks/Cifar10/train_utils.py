import numpy as np
import os
import struct
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization, add
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.regularizers import l2

def lr_schedule(epoch):
    initial_learning_rate = 0.001
    decay_per_epoch = 0.99
    lrate = initial_learning_rate * (decay_per_epoch ** epoch)
    print('Learning rate = %f'%lrate)
    return lrate

def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def load_cifar_10_data(data_dir, negatives=False):
    """
    Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
    """

    # get the meta_data_dict
    # num_cases_per_batch: 1000
    # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # num_vis: :3072

    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    return cifar_train_data, cifar_train_filenames, to_categorical(cifar_train_labels), \
        cifar_test_data, cifar_test_filenames, to_categorical(cifar_test_labels), cifar_label_names


def generate_resnet(nstacks=3,init_num_filter=16,kernel_size=3,
                 use_batch_norm=True, use_act = True, num_classes = 10):
    inputs = Input(shape=[32,32,3])
    x = Conv2D(init_num_filter,
              kernel_size=kernel_size,
              strides=1,
              padding='same',
              kernel_initializer='he_normal',
              kernel_regularizer=l2(1e-4))(inputs)
    if(use_batch_norm == True):
        x = BatchNormalization()(x)
    if(use_act == True):
        x = Activation('relu')(x)
    for i in range(nstacks):
        if(i==0):
            y = Conv2D(init_num_filter*(2**i),
                  kernel_size=kernel_size,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
            if(use_batch_norm == True):
                y = BatchNormalization()(y)
            if(use_act == True):
                y = Activation('relu')(y)
            y = Conv2D(init_num_filter*(2**i),
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-4))(y)
            if(use_batch_norm == True):
                y = BatchNormalization()(y)
            x = add([x, y])
            if(use_act == True):
                x = Activation('relu')(x)
        else:
            y = Conv2D(init_num_filter*(2**i),
              kernel_size=kernel_size,
              strides=2,
              padding='same',
              kernel_initializer='he_normal',
              kernel_regularizer=l2(1e-4))(x)
            if(use_batch_norm == True):
                y = BatchNormalization()(y)
            if(use_act == True):
                y = Activation('relu')(y)
            y = Conv2D(init_num_filter*(2**i),
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(1e-4))(y)
            if(use_batch_norm == True):
                y = BatchNormalization()(y)
            if(kernel_size >= 3):
                x = Conv2D(init_num_filter*(2**i),
                              kernel_size=kernel_size-2,
                              strides=2,
                              padding='same',
                              kernel_initializer='he_normal',
                              kernel_regularizer=l2(1e-4))(x)
            else:
                x = Conv2D(init_num_filter*(2**i),
                              kernel_size=1,
                              strides=2,
                              padding='same',
                              kernel_initializer='he_normal',
                              kernel_regularizer=l2(1e-4))(x)                

            x = add([x, y])
            if(use_act == True):
                x = Activation('relu')(x)
                
    pool_size = int(np.amin(x.shape[1:3]))
    x = AveragePooling2D(pool_size=pool_size)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                activation='softmax',
                kernel_initializer='he_normal')(y)

    return Model(inputs=inputs, outputs=outputs)