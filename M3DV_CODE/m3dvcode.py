#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#####准备input#####
import collections
from itertools import repeat
import numpy as np
import scipy
import matplotlib.pyplot as plt
from skimage.measure import find_contours
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import keras.backend as K
from keras.metrics import binary_accuracy
from keras.models import load_model
#from keras.models import load_weights
from collections.abc import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.utils import np_utils
from keras.optimizers import Adam
from tensorflow.keras.layers import (Conv3D, BatchNormalization, AveragePooling3D, concatenate, Lambda, SpatialDropout3D,
                          Activation, Input, GlobalAvgPool3D, Dense, Conv3DTranspose, add)
from keras.regularizers import l2 as l2_penalty
from tensorflow.keras.models import Model
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
import tensorflow as tf


# In[ ]:


######准备各种工具函数#######
####type1 gpu 调用函数###
def get_gpu_session(ratio=None, interactive=False):
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    if ratio is None:
        config.gpu_options.allow_growth = True
    else:
        config.gpu_options.per_process_gpu_memory_fraction = ratio
    if interactive:
        sess = tf.InteractiveSession(config=config)
    else:
        sess = tf.compat.v1.Session(config=config)
    return sess

def set_gpu_usage(ratio=None):
    sess = get_gpu_session(ratio)
    tf.compat.v1.keras.backend.set_session(sess)


# In[ ]:


######type2  计算各种精度指标的函数  ######
def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    F1 score: https://en.wikipedia.org/wiki/F1_score
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)


def invasion_acc(y_true, y_pred):
    binary_truth = y_true[:, -2] + y_true[:, -1]
    binary_pred = y_pred[:, -2] + y_pred[:, -1]
    return binary_accuracy(binary_truth, binary_pred)


def invasion_precision(y_true, y_pred):
    binary_truth = y_true[:, -2] + y_true[:, -1]
    binary_pred = y_pred[:, -2] + y_pred[:, -1]
    return precision(binary_truth, binary_pred)


def invasion_recall(y_true, y_pred):
    binary_truth = y_true[:, -2] + y_true[:, -1]
    binary_pred = y_pred[:, -2] + y_pred[:, -1]
    return recall(binary_truth, binary_pred)


def invasion_fmeasure(y_true, y_pred):
    binary_truth = y_true[:, -2] + y_true[:, -1]
    binary_pred = y_pred[:, -2] + y_pred[:, -1]
    return fmeasure(binary_truth, binary_pred)


def ia_acc(y_true, y_pred):
    binary_truth = y_true[:, -1]
    binary_pred = y_pred[:, -1]
    return binary_accuracy(binary_truth, binary_pred)


def ia_precision(y_true, y_pred):
    binary_truth = y_true[:, -1]
    binary_pred = y_pred[:, -1]
    return precision(binary_truth, binary_pred)


def ia_recall(y_true, y_pred):
    binary_truth = y_true[:, -1]
    binary_pred = y_pred[:, -1]
    return recall(binary_truth, binary_pred)


def ia_fmeasure(y_true, y_pred):
    binary_truth = y_true[:, -1]
    binary_pred = y_pred[:, -1]
    return fmeasure(binary_truth, binary_pred)


# In[ ]:


#######type3  相关损失函数######
class DiceLoss:
    def __init__(self, beta=1., smooth=1.):
        self.__name__ = 'dice_loss_' + str(int(beta * 100))
        self.beta = beta  # the more beta, the more recall
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        bb = self.beta * self.beta
        y_true_f = K.batch_flatten(y_true)
        y_pred_f = K.batch_flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f, axis=-1)
        weighted_union = bb * K.sum(y_true_f, axis=-1) +                          K.sum(y_pred_f, axis=-1)
        score = -((1 + bb) * intersection + self.smooth) /                 (weighted_union + self.smooth)
        return score


# In[ ]:


######type4 各种工具函数####
def find_edges(mask, level=0.5):
    edges = find_contours(mask, level)[0]
    ys = edges[:, 0]
    xs = edges[:, 1]
    return xs, ys


def plot_contours(arr, aux, level=0.5, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(1, 1, **kwargs)
    ax.imshow(arr, cmap=plt.cm.gray)
    xs, ys = find_edges(aux, level)
    ax.plot(xs, ys)


def crop_at_zyx_with_dhw(voxel, zyx, dhw, fill_with):
    '''Crop and pad on the fly.'''
    shape = voxel.shape
    # z, y, x = zyx
    # d, h, w = dhw
    crop_pos = []
    padding = [[0, 0], [0, 0], [0, 0]]
    for i, (center, length) in enumerate(zip(zyx, dhw)):
        assert length % 2 == 0
        # assert center < shape[i] # it's not necessary for "moved center"
        low = round(center) - length // 2
        high = round(center) + length // 2
        if low < 0:
            padding[i][0] = int(0 - low)
            low = 0
        if high > shape[i]:
            padding[i][1] = int(high - shape[i])
            high = shape[i]
        crop_pos.append([int(low), int(high)])
    cropped = voxel[crop_pos[0][0]:crop_pos[0][1], crop_pos[1]
                                                   [0]:crop_pos[1][1], crop_pos[2][0]:crop_pos[2][1]]
    if np.sum(padding) > 0:
        cropped = np.lib.pad(cropped, padding, 'constant',
                             constant_values=fill_with)
    return cropped


def window_clip(v, window_low=-1024, window_high=400, dtype=np.uint8):
    '''Use lung windown to map CT voxel to grey.'''
    # assert v.min() <= window_low
    return np.round(np.clip((v - window_low) / (window_high - window_low) * 255., 0, 255)).astype(dtype)


def resize(voxel, spacing, new_spacing=[1., 1., 1.]):
    '''Resize `voxel` from `spacing` to `new_spacing`.'''
    resize_factor = []
    for sp, nsp in zip(spacing, new_spacing):
        resize_factor.append(float(sp) / nsp)
    resized = scipy.ndimage.interpolation.zoom(voxel, resize_factor, mode='nearest')
    for i, (sp, shape, rshape) in enumerate(zip(spacing, voxel.shape, resized.shape)):
        new_spacing[i] = float(sp) * shape / rshape
    return resized, new_spacing


def rotation(array, angle):
    '''using Euler angles method.
    @author: renchao
    @params:
        angle: 0: no rotation, 1: rotate 90 deg, 2: rotate 180 deg, 3: rotate 270 deg
    '''
    #
    X = np.rot90(array, angle[0], axes=(0, 1))  # rotate in X-axis
    Y = np.rot90(X, angle[1], axes=(0, 2))  # rotate in Y'-axis
    Z = np.rot90(Y, angle[2], axes=(1, 2))  # rotate in Z"-axis
    return Z


def reflection(array, axis):
    '''
    @author: renchao
    @params:
        axis: -1: no flip, 0: Z-axis, 1: Y-axis, 2: X-axis
    '''
    if axis != -1:
        ref = np.flip(array, axis)
    else:
        ref = np.copy(array)
    return ref


def crop(array, zyx, dhw):
    z, y, x = zyx
    d, h, w = dhw
    cropped = array[z - d // 2:z + d // 2,
              y - h // 2:y + h // 2,
              x - w // 2:x + w // 2]
    return cropped


def random_center(shape, move):
    offset = np.random.randint(-move, move + 1, size=3)
    # offset
    zyx = np.array(shape) // 2 + offset
    #np.array(shape) // 2对应的是中心点，+offset表示的是中心点的偏移量
    #最后得出的是，中心的偏移量
    return zyx


def get_uniform_assign(length, subset):
    assert subset > 0
    per_length, remain = divmod(length, subset)
    total_set = np.random.permutation(list(range(subset)) * per_length)
    remain_set = np.random.permutation(list(range(subset)))[:remain]
    return list(total_set) + list(remain_set)


def split_validation(df, subset, by):
    df = df.copy()
    for sset in df[by].unique():
        length = (df[by] == sset).sum()
        df.loc[df[by] == sset, 'subset'] = get_uniform_assign(length, subset)
    df['subset'] = df['subset'].astype(int)
    return df


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse
_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


# In[ ]:


PARAMS = {
    'activation': lambda: Activation('relu'),  # the activation functions
    'bn_scale': True,  # whether to use the scale function in BN
    'weight_decay': 0.,  # l2 weight decay
    'kernel_initializer': 'he_uniform',  # initialization
    'first_scale': lambda x: x / 128. - 1.,  # the first pre-processing function
    'dhw': [32, 32, 32],  # the input shape
    'k': 16,  # the `growth rate` in DenseNet
    'bottleneck': 4,  # the `bottleneck` in DenseNet
    'compression': 2,  # the `compression` in DenseNet
    'first_layer': 32,  # the channel of the first layer
    'down_structure': [2, 2, 2],  # the down-sample structure
    'output_size': 1,  # the output number of the classification head
    'dropout_rate': None  # whether to use dropout, and how much to use
}


def _conv_block(x, filters):
    bn_scale = PARAMS['bn_scale']
    activation = PARAMS['activation']
    #激活函数
    kernel_initializer = PARAMS['kernel_initializer']
    #初始化内核
    weight_decay = PARAMS['weight_decay']
    #
    bottleneck = PARAMS['bottleneck']
    #bottleneck的结构
    dropout_rate = PARAMS['dropout_rate']
    #droupout 率
    x = BatchNormalization(scale=bn_scale, axis=-1)(x)
    #
    x = activation()(x)
    x = Conv3D(filters * bottleneck, kernel_size=(1, 1, 1), padding='same', use_bias=False,
               kernel_initializer=kernel_initializer,
               kernel_regularizer=l2_penalty(weight_decay))(x)
    if dropout_rate is not None:
        x = SpatialDropout3D(dropout_rate)(x)
    x = BatchNormalization(scale=bn_scale, axis=-1)(x)
    x = activation()(x)
    x = Conv3D(filters, kernel_size=(3, 3, 3), padding='same', use_bias=True,
               kernel_initializer=kernel_initializer,
               kernel_regularizer=l2_penalty(weight_decay))(x)
    return x


def _dense_block(x, n):
    k = PARAMS['k']

    for _ in range(n):
        conv = _conv_block(x, k)
        x = concatenate([conv, x], axis=-1)
    return x


def _transmit_block(x, is_last):
    bn_scale = PARAMS['bn_scale']
    activation = PARAMS['activation']
    kernel_initializer = PARAMS['kernel_initializer']
    weight_decay = PARAMS['weight_decay']
    compression = PARAMS['compression']

    x = BatchNormalization(scale=bn_scale, axis=-1)(x)
    x = activation()(x)
    if is_last:
        x = GlobalAvgPool3D()(x)
    else:
        *_, f = x.get_shape().as_list()
        x = Conv3D(f // compression, kernel_size=(1, 1, 1), padding='same', use_bias=True,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=l2_penalty(weight_decay))(x)
        x = AveragePooling3D((2, 2, 2), padding='valid')(x)
    return x


def get_model(weights=None, verbose=True, **kwargs):
    for k, v in kwargs.items():
        assert k in PARAMS
        PARAMS[k] = v
    if verbose:
        print("Model hyper-parameters:", PARAMS)

    dhw = PARAMS['dhw']
    #input scale 32*32*32
    first_scale = PARAMS['first_scale']
    #first preprocessing function 归一化
    first_layer = PARAMS['first_layer']
    #第一个卷积层的通道数
    kernel_initializer = PARAMS['kernel_initializer']
    #初始化
    weight_decay = PARAMS['weight_decay']
    #
    down_structure = PARAMS['down_structure']
    #降采样结构
    output_size = PARAMS['output_size']
    #output number of classifi
    shape = dhw + [1]

    inputs = Input(shape=shape) #[32,32,32,1]

    if first_scale is not None:
        scaled = Lambda(first_scale)(inputs)
        
    else:
        scaled = inputs
    conv = Conv3D(first_layer, kernel_size=(3, 3, 3), padding='same', use_bias=True,
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer=l2_penalty(weight_decay))(scaled)
    
    downsample_times = len(down_structure)
    #降采样次数对应降采样结构的长度
    top_down = []
    for l, n in enumerate(down_structure):
        db = _dense_block(conv, n)
        #conv对应的是一个卷积网络，n对应的是降采样的的数字，4
        top_down.append(db)
        conv = _transmit_block(db, l == downsample_times - 1)


    if output_size == 1:
        last_activation = 'sigmoid'
    else:
        last_activation = 'softmax'

    clf_head = Dense(output_size, activation=last_activation,
                     kernel_regularizer=l2_penalty(weight_decay),
                     kernel_initializer=kernel_initializer,
                     name='clf')(conv)

    model = Model(inputs, clf_head)
    if verbose:
        model.summary()

    if weights is not None:
        model.load_weights(weights)
    return model


def get_compiled(loss={"clf": 'binary_crossentropy'},
                 optimizer='adam',
                 metrics={'clf': ['accuracy', precision, recall]},
                 loss_weights={"clf": 1.}, weights=None, **kwargs):
    model = get_model(weights=weights, **kwargs)
    model.compile(loss=loss, optimizer=optimizer,
                  metrics=metrics, loss_weights=loss_weights)
    return model


# In[ ]:



# In[ ]:

####关于数据loader的函数#######
def shuffle_iterator(iterator):
    # iterator should have limited size
    #interator是一个范围的数
    index = list(iterator)
    total_size = len(index)
    i = 0
    random.shuffle(index)
    #随机打乱下标
    while True:
        yield index[i]
        i += 1
        if i >= total_size:
            i = 0
            random.shuffle(index)
def _collate_fn(data):
        xs = []
        ys = []
        #segs=[]
        for x, y in data:
            #print("y:",y)
            xs.append(x)
            ys.append(y)#对应的是label，
        xs=np.array(xs)
        ys=np.array(ys)
        xs = xs.reshape(xs.shape[0],32,32,32,1)
        #ys = np_utils.to_categorical(ys,num_classes=2)
        return xs,ys
def get_loader(dataset, batch_size):
    total_size = len(dataset)
    #total_size表示整个数据集的大小
    print('Size', total_size)
    index_generator = shuffle_iterator(range(total_size))
    while True:
        data = []
        for _ in range(batch_size):
            idx = next(index_generator)
            data.append(dataset[idx])
        #print(np.array(data).shape)
        #data 对应的是一个batch_size数据合集
        yield _collate_fn(data)
        


# In[ ]:


