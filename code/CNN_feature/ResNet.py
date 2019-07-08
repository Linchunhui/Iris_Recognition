# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division


import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
import tfplot as tfp


def resnet_arg_scope(
        is_training=True, weight_decay=0.0001, batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5, batch_norm_scale=True):
    '''

    In Default, we do not use BN to train resnet, since batch_size is too small.
    So is_training is False and trainable is False in the batch_norm params.

    '''
    batch_norm_params = {
        'is_training': False, 'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc

def resnet_base(img_batch, scope_name, is_training=False):
    '''
    this code is derived from light-head rcnn.
    https://github.com/zengarden/light_head_rcnn

    It is convenient to freeze blocks. So we adapt this mode.
    '''
    if scope_name == 'resnet_v1_50':
        middle_num_units = 6
    elif scope_name == 'resnet_v1_101':
        middle_num_units = 23
    else:
        raise NotImplementedError('We only support resnet_v1_50 or resnet_v1_101. Check your network name....yjr')

    blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
              resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
              resnet_v1_block('block3', base_depth=256, num_units=9, stride=2),
              resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]
    # when use fpn . stride list is [1, 2, 2]

    with slim.arg_scope(resnet_arg_scope(is_training=False)):
        with tf.variable_scope(scope_name, scope_name):
            # Do the first few layers manually, because 'SAME' padding can behave inconsistently
            # for images of different sizes: sometimes 0, sometimes 1
            net = resnet_utils.conv2d_same(
                img_batch, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(
                net, [3, 3], stride=2, padding='VALID', scope='pool1')

    not_freezed = [False] * 0 + (4-0)*[True]
    # Fixed_Blocks can be 1~3

    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[0]))):
        C2, end_points_C2 = resnet_v1.resnet_v1(net,
                                                blocks[0:1],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)
        #C2=tf.layers.average_pooling2d(inputs=C2, pool_size=3, strides=2,padding="valid")
        #C2=tf.reduce_mean(C2, axis=[1, 2], keep_dims=False, name='global_average_pooling')
    with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[1]))):
        C3, end_points_C3 = resnet_v1.resnet_v1(C2,
                                                blocks[1:2],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)
        C3 = slim.avg_pool2d(C3, 2)
        #C3 = tf.reduce_mean(C3, axis=[1, 2], keep_dims=False, name='global_average_pooling')
    #return C3
    '''with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[2]))):
        C4, end_points_C4 = resnet_v1.resnet_v1(C3,
                                                blocks[2:3],
                                                global_pool=False,
                                                include_root_block=False,
                                                scope=scope_name)'''
    return C3

    # C2 = tf.Print(C2, [tf.shape(C2)], summarize=10, message='C2_shape')
    #add_heatmap(C2, name='Layer2/C2_heat')
def get_restorer():
    checkpoint_path = r"E:\FPN\data\pretrained_weights\resnet_v1_101.ckpt"
    print("model restore from pretrained mode, path is :", checkpoint_path)
    model_variables = slim.get_model_variables()


    def name_in_ckpt_rpn(var):
            return var.op.name

    def name_in_ckpt_fastrcnn_head(var):
        return '/'.join(var.op.name.split('/')[1:])

    nameInCkpt_Var_dict = {}
    for var in model_variables:
        if var.name.startswith('resnet_v1_101'):
            var_name_in_ckpt = name_in_ckpt_rpn(var)
            nameInCkpt_Var_dict[var_name_in_ckpt] = var
    restore_variables = nameInCkpt_Var_dict
    for key, item in restore_variables.items():
            print("var_in_graph: ", item.name)
            print("var_in_ckpt: ", key)
            print(20*"___")
    restorer = tf.train.Saver(restore_variables)
    print(20 * "****")
    print("restore from pretrained_weighs in IMAGE_NET")
    return restorer, checkpoint_path

