from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append(".")
sys.path.append("..")

import tensorflow as tf
import tensorflow.contrib.slim as slim
# import collections

# Define the convolution building block
def conv2(batch_input, kernel=3, output_channel=64, stride=1, use_bias=True, scope='conv'):
    # kernel: An integer specifying the width and height of the 2D convolution window
    with tf.variable_scope(scope):
        if use_bias:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
                            activation_fn=None,
                            weights_initializer=tf.contrib.layers.xavier_initializer())
        else:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
                            activation_fn=None,
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=None)

def deconv2(batch_input, kernel=3, output_channel=64, stride=2, use_bias=True, scope='deconv'):
    with tf.variable_scope(scope):
      if use_bias:
        return slim.layers.conv2d_transpose(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
          activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
      else:
        return slim.layers.conv2d_transpose(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
          activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer()) 

def prelu_tf(inputs, name='Prelu'):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alpha', inputs.get_shape()[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5
    return pos + neg


# The implementation of PixelShuffler
def pixelShuffler(inputs, scale=2):
    size = tf.shape(inputs)
    batch_size = size[0]
    h = size[1]
    w = size[2]
    c = inputs.get_shape().as_list()[-1]

    # Get the target channel size
    channel_target = c // (scale * scale)
    channel_factor = c // channel_target

    shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
    shape_2 = [batch_size, h * scale, w * scale, 1]

    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(inputs, channel_target, axis=3)
    output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

    return output

def batchnorm(inputs, is_training):
    return slim.batch_norm(inputs, decay=0.9, epsilon=0.001, updates_collections=tf.GraphKeys.UPDATE_OPS,
                        scale=False, fused=True, is_training=is_training)

# Definition of the generator
def SRResnet(gen_inputs, gen_output_channels, up_ratio=2, reuse=False, up_type='deconv', is_training=True):
    # # Check the flag
    # if FLAGS is None:
    #     raise  ValueError('No FLAGS is provided for generator')

    # The Bx residual blocks
    def residual_block(inputs, output_channel, stride, scope):
        with tf.variable_scope(scope):
            net = conv2(inputs, 4, output_channel, stride, use_bias=False, scope='conv_1')
            net = batchnorm(net, is_training)
            net = prelu_tf(net)
            net = conv2(net, 3, output_channel, stride, use_bias=False, scope='conv_2')
            net = batchnorm(net, is_training)
            net = net + inputs

        return net

    with tf.variable_scope('generator_unit', reuse=reuse):
        # The input layer
        with tf.variable_scope('input_stage'):
            net = conv2(gen_inputs, 9, 64, 1, scope='conv')
            net = prelu_tf(net)

        stage1_output = net

        # The residual block parts
        for i in range(1, 16+1 , 1):
            name_scope = 'resblock_%d'%(i)
            net = residual_block(net, 64, 1, name_scope)

        with tf.variable_scope('resblock_output'):
            net = conv2(net, 3, 64, 1, use_bias=False, scope='conv')
            net = batchnorm(net, is_training)

        net = net + stage1_output

        if up_type == 'subpixel':
            with tf.variable_scope('subpixelconv_stage1'):
                net = conv2(net, 3, 256, 1, scope='conv')
                net = pixelShuffler(net, scale=2)
                net = prelu_tf(net)

            with tf.variable_scope('subpixelconv_stage2'):
                net = conv2(net, 3, 256, 1, scope='conv')
                net = pixelShuffler(net, scale=2)
                net = prelu_tf(net)

            if up_ratio == 4:
                with tf.variable_scope('subpixelconv_stage3'):
                    net = conv2(net, 3, 256, 1, scope='conv')
                    net = pixelShuffler(net, scale=2)
                    net = prelu_tf(net)

            with tf.variable_scope('output_stage'):
                net = conv2(net, 9, gen_output_channels, 1, scope='conv')

        elif up_type == 'deconv':
            with tf.variable_scope('deconv_stage1'):
                net = conv2(net, 3, 256, 1, scope='conv')
                net = deconv2(net, 3, 256, 2)
                net = prelu_tf(net)

            with tf.variable_scope('deconv_stage2'):
                net = conv2(net, 3, 256, 1, scope='conv')
                net = deconv2(net, 3, 256, 2)
                net = prelu_tf(net)

            if up_ratio == 4:
                with tf.variable_scope('deconv_stage3'):
                    net = conv2(net, 3, 256, 1, scope='conv')
                    net = deconv2(net, 3, 256, 2)
                    net = prelu_tf(net)

            if up_ratio == 8:
                with tf.variable_scope('deconv_stage3'):
                    net = conv2(net, 3, 256, 1, scope='conv')
                    net = deconv2(net, 3, 256, 2)
                    net = prelu_tf(net)
                with tf.variable_scope('deconv_stage4'):
                    net = conv2(net, 3, 256, 1, scope='conv')
                    net = deconv2(net, 3, 256, 2)
                    net = prelu_tf(net)

            with tf.variable_scope('deconv_output_stage'):
                net = conv2(net, 9, gen_output_channels, 1, scope='conv')

    return net
