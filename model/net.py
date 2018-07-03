from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim

def conv(batch_input, in_channels, out_channels, stride):
    with tf.variable_scope("conv"):
        # in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv

def conv2(batch_input, in_channels, out_channels, stride, fsz=4):
    with tf.variable_scope("Conv"):
        # in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [fsz, fsz, in_channels, out_channels], dtype=tf.float32, initializer=tf.truncated_normal_initializer(0.0, 0.01),
            regularizer=slim.l2_regularizer(0.0001))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(input, channels):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        # channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized


def deconv(batch_input, in_channels, out_channels):
    with tf.variable_scope("deconv"):
        # batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        batch = tf.shape(batch_input)[0]
        in_height = tf.shape(batch_input)[1]
        in_width = tf.shape(batch_input)[2]
        # print("out_channels, in_channels: ",out_channels, in_channels)
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
        return conv

def build_unet(input, channel=64, input_channel=3, output_channel=3,reuse=False,num_layer=9):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    layers = []
    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, channel]
    with tf.variable_scope("encoder_1"):
        output = conv(input, in_channels=input_channel, out_channels=channel, stride=2)
        layers.append(output)
    layer_specs = [
        (channel, channel * 2), # encoder_2: [batch, 512, 512, channel] => [batch, 256, 256, channel * 2]
        (channel * 2, channel * 4), # encoder_2: [batch, 256, 256, channel * 2] => [batch, 128, 128, channel * 4]
        (channel * 4, channel * 8), # encoder_2: [batch, 128, 128, channel * 4] => [batch, 64, 64, channel * 8]
        (channel * 8, channel * 8), # encoder_3: [batch, 64, 64, channel * 8] => [batch, 32, 32, channel * 8]
        (channel * 8, channel * 8), # encoder_4: [batch, 32, 32, channel * 8] => [batch, 16, 16, channel * 8]
        (channel * 8, channel * 8), # encoder_5: [batch, 16, 16, channel * 8] => [batch, 8, 8, channel * 8]
        (channel * 8, channel * 8), # encoder_6: [batch, 8, 8, channel * 8] => [batch, 4, 4, channel * 8]
        (channel * 8, channel * 8), # encoder_7: [batch, 4, 4, channel * 8] => [batch, 2, 2, channel * 8]
        (channel * 8, channel * 8), # encoder_8: [batch, 2, 2, channel * 8] => [batch, 1, 1, channel * 8]
    ]

    for iter, (in_channels, out_channels) in enumerate(layer_specs[:num_layer]):
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(rectified, in_channels=in_channels, out_channels=out_channels, stride=2)
            output = batchnorm(convolved, out_channels)
            layers.append(output)

    layer_specs = [
        (channel * 8, channel * 8, 0.5),   # decoder_8: [batch, 1, 1, channel * 8] => [batch, 2, 2, channel * 8 * 2]
        (channel * 8, channel * 8, 0.5),   # decoder_7: [batch, 2, 2, channel * 8 * 2] => [batch, 4, 4, channel * 8 * 2]
        (channel * 8, channel * 8, 0.5),   # decoder_6: [batch, 4, 4, channel * 8 * 2] => [batch, 8, 8, channel * 8 * 2]
        (channel * 8, channel * 8, 0.5),   # decoder_5: [batch, 8, 8, channel * 8 * 2] => [batch, 16, 16, channel * 8 * 2]
        (channel * 8, channel * 8, 0.5),   # decoder_4: [batch, 16, 16, channel * 8 * 2] => [batch, 32, 32, channel * 4 * 2]
        (channel * 8, channel * 8, 0.0),   # decoder_3: [batch, 32, 32, channel * 4 * 2] => [batch, 64, 64, channel * 2 * 2]
        (channel * 8, channel * 4, 0.0),   # decoder_3: [batch, 64, 64, channel * 4 * 2] => [batch, 64, 64, channel * 2 * 2]
        (channel * 4, channel * 2, 0.0),   # decoder_3: [batch, 128, 128, channel * 4 * 2] => [batch, 64, 64, channel * 2 * 2]
        (channel * 2, channel, 0.0),       # decoder_2: [batch, 256, 256, channel * 2 * 2] => [batch, 128, 128, channel * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (in_channels, out_channels, dropout) in enumerate(layer_specs[9-num_layer:]):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                in_layer = layers[-1]
                in_channels = in_channels
            else:
                in_layer = tf.concat([layers[-1], layers[skip_layer]], axis=3)
                in_channels = in_channels * 2
            rectified = tf.nn.relu(in_layer)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = deconv(rectified, in_channels, out_channels)
            output = batchnorm(output, out_channels)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)
    # decoder_1: [batch, 128, 128, channel * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        in_layer = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(in_layer)
        output = deconv(rectified, channel*2, channel)
        output = tf.reshape(output, [1,tf.shape(output)[1],tf.shape(output)[2],channel])

    with tf.variable_scope("decoder_last"):
        output_final = slim.conv2d(output, output_channel, 3, 1, activation_fn=None)

    out = tf.depth_to_space(output_final, 2)

    return out


def buildflownet(input,channel=64,output_channel=3,reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0001)):
      # Define network      
      batch_norm_params = {
        'decay': 0.9997,
        'epsilon': 0.001,
        'is_training': True,
      }
      with slim.arg_scope([slim.batch_norm], is_training = True, updates_collections=None):
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
          net = slim.conv2d(input, channel, [5, 5], stride=1, scope='conv1')
          net = slim.max_pool2d(net, [2, 2], scope='pool1')
          net = slim.conv2d(net, channel*2, [5, 5], stride=1, scope='conv2')
          net = slim.max_pool2d(net, [2, 2], scope='pool2')
          net = slim.conv2d(net, channel*4, [3, 3], stride=1, scope='conv3')
          net = slim.max_pool2d(net, [2, 2], scope='pool3')
          net = slim.conv2d(net, channel*8, [3, 3], stride=1, scope='conv4')
          net = slim.max_pool2d(net, [2, 2], scope='pool4')
          net = tf.image.resize_bilinear(net, [64,64])
          net = slim.conv2d(net, channel*8, [3, 3], stride=1, scope='conv5')
          net = tf.image.resize_bilinear(net, [128,128])
          net = slim.conv2d(net, channel*4, [3, 3], stride=1, scope='conv6')
          net = tf.image.resize_bilinear(net, [256,256])
          net = slim.conv2d(net, channel*2, [3, 3], stride=1, scope='conv7')
          net = tf.image.resize_bilinear(net, [512,512])
          net = slim.conv2d(net, channel, [5, 5], stride=1, scope='conv8')
    net = slim.conv2d(net, output_channel, [5, 5], stride=1, activation_fn=tf.sigmoid, normalizer_fn=None, scope='conv9')
    return net