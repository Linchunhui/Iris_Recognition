from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import math

#CLASSES = 108
dense_blocks_num = 6
k = 12
L = 40
layers = 4

#Used for BN
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

'''
(layers * 2) * dense_blocks_num + (dense_blocks_num - 1) + 1(init) + 1(fc)  
= (6 * 2) * 4 + (3 - 1) + 1 + 1
= 52
= L
'''

def weight_variable(shape, stddev=None, name='weight'):
    if stddev == None:
        if len(shape) == 4:
            stddev = math.sqrt(2. / (shape[0] * shape[1] * shape[2]))
        else:
            stddev = math.sqrt(2. / shape[0])
    else:
        stddev = 0.1
    initial = tf.truncated_normal(shape, stddev=stddev)
    W = tf.Variable(initial, name=name)
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, W)
    return W


def bias_variable(shape, name='bias'):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def bn_layer(inputs, training):

  return tf.layers.batch_normalization(
      inputs=inputs, axis=-1,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)


def dense_block_layer(input, k, k0, layer, train, keep_prob):
    '''
    :param input: input feature map
    :param k: output channels
    :param k0: input channels
    :param layer: the layer number in the dense block
    :param train: used for bn layer
    :return: the dense block layer output
    '''
    input_channels = k0 + (layer - 1) * k
    weights_1 = weight_variable(shape=[1, 1, input_channels, 4 * k])
    weights_3 = weight_variable(shape=[3, 3, 4 * k, k])
    with tf.name_scope("dense_bottleneck_layer"):
        input = bn_layer(input, train)
        input = tf.nn.relu(input)
        input = tf.nn.conv2d(input, weights_1, [1, 1, 1, 1], padding='SAME', name='conv1')
        input = tf.nn.dropout(input, keep_prob=keep_prob)
        input = bn_layer(input, train)
        input = tf.nn.relu(input)
        input = tf.nn.conv2d(input, weights_3, [1, 1, 1, 1], padding='SAME', name='conv3')
        input = tf.nn.dropout(input, keep_prob=keep_prob)
    return input


def dense_block(input, k, layers, train, keep_prob):
    '''
    dense block
    :param input: input feature map
    :param k: output channels
    :param layers: the layer numbers of the dense block
    :param train: used fot train
    :return: the dense block output
    '''
    input_shape = input.get_shape().as_list()
    k0 = input_shape[-1]
    #output = input
    for i in range(1, layers + 1):
        #print(i)
        with tf.name_scope("layer_%d" % i):
            output = dense_block_layer(input, k, k0, i, train, keep_prob)
            input = tf.concat(values=[input, output], axis=-1)
    return input

def transition_layer(input, train, keep_prob):
    """
    1x1 conv, 2x2 avegage pool
    :param input:
    :return:
    """
    input_shape = input.get_shape().as_list()
    with tf.name_scope('transition_layer'):
        input = bn_layer(input, train)
        weights = weight_variable(shape = [1, 1, input_shape[-1], input_shape[-1]])
        input = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='SAME')
        input = tf.nn.dropout(input, keep_prob=keep_prob)
        input = tf.nn.avg_pool(input, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    return input

def densenet(input, keep_prob,N_Class, train=True):
    '''
    densenet: k = 12, L = 40
    :param input:
    :param train:
    :return:
    '''

    input_shape = input.get_shape().as_list()
    with tf.name_scope('initial'):
        weights = weight_variable(shape=[3, 3, input_shape[-1], 16])
        input = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='SAME')

    with tf.name_scope("dense_blocks"):
        for i in range(1, 1 + dense_blocks_num):
            with tf.name_scope("dense_block_%d" % i):
                input = dense_block(input, k, layers, train, keep_prob)
                if i != dense_blocks_num:
                    input = transition_layer(input, train, keep_prob)

    print(input)
    with tf.name_scope('classification_layer'):
        #input = tf.nn.avg_pool(input, [1, 8, 8, 1], [1, 8, 8, 1], padding='VALID')
        input = tf.reduce_mean(input, axis=[1, 2])
        print(input)
        input_shape = input.get_shape().as_list()
        weights = weight_variable(shape=[input_shape[-1], N_Class])
        bias = bias_variable([N_Class])
        input = tf.matmul(input, weights) + bias

    output = input
    return output


if __name__ == '__main__':
    input = tf.constant(0.1, shape=[8, 64, 512, 3], dtype=tf.float32)
    output = densenet(input, 1.0,10)
    print(output)