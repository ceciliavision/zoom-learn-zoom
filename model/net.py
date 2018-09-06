from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append(".")
sys.path.append("..")

import tensorflow as tf
import tensorflow.contrib.slim as slim
import srganops
import loss
import collections, math, os
import loss as losses

def upsample(batch_input, in_channels, out_channels, sp=512, type='deconv'):
    if type == 'deconv':
        upsampled = deconv(batch_input, in_channels, out_channels)
    elif type == 'deconv_conv':
        upsampled = deconv(batch_input, in_channels, out_channels)
        upsampled = conv2(upsampled, out_channels, out_channels, stride=1, fsz=3)
    elif type == 'bilinear':
        upsampled = tf.image.resize_bilinear(batch_input, [sp,sp], align_corners=True)
        upsampled = conv2(upsampled, in_channels, out_channels, stride=1, fsz=3)
    elif type == 'subpixel':
        upsampled = conv(upsampled, in_channels, out_channels*4, stride=1)
        upsampled = tf.depth_to_space(upsampled, 2)
    else:
        print("Not recognized upsample type.")
        exit()
    
    return upsampled

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
        filter = tf.get_variable("filter", [fsz, fsz, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
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

def build_unet(input, channel=64, input_channel=3, output_channel=3, reuse=False, num_layer=9, up_type='deconv'):
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

    sp = int(128//(2**num_layer))
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
            output = upsample(rectified, in_channels, out_channels, int(sp*2**decoder_layer), up_type)
            output = batchnorm(output, out_channels)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)
    # decoder_1: [batch, 128, 128, channel * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        in_layer = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(in_layer)
        output = upsample(rectified, channel*2, channel, 128, up_type)
        output = tf.reshape(output, [1,tf.shape(output)[1],tf.shape(output)[2],channel])

    # output = tf.depth_to_space(output, 4)

    # with tf.variable_scope("decoder_last0"):
    #     output = tf.nn.relu(output)
    #     output = upsample(output, channel, channel, 256, up_type)
    #     output = tf.reshape(output, [1,tf.shape(output)[1],tf.shape(output)[2],channel])
    # with tf.variable_scope("decoder_last1"):
    #     output = tf.nn.relu(output)
    #     output = upsample(output, channel, channel, 512, up_type)
    #     output = tf.reshape(output, [1,tf.shape(output)[1],tf.shape(output)[2],channel])
    
    with tf.variable_scope("decoder_last"):
        output = slim.conv2d(output, output_channel, 3, 1, activation_fn=None)

    output = tf.depth_to_space(output, 4)

    with tf.variable_scope("decoder_refine"):
        output = slim.conv2d(output, channel, 3, 1, 'SAME', activation_fn=None)
        output = tf.nn.relu(output)
        output = slim.conv2d(output, 3, 7, 1, 'SAME', activation_fn=None)

    print("output shape:", output.shape)

    return output


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

# def SRResnet(inputs, targets, FLAGS):
#     # Define the container of the parameter
#     Network = collections.namedtuple('Network', 'content_loss, gen_grads_and_vars, gen_output, target_translated, train, global_step, \
#             learning_rate')

#     # Build the generator part
#     with tf.variable_scope('generator'):
#         output_channel = targets.get_shape().as_list()[-1]
#         gen_output = generator(inputs, output_channel, reuse=False, FLAGS=FLAGS)
#         gen_output.set_shape([FLAGS.batch_size, FLAGS.crop_size, FLAGS.crop_size, 3])

#     # Use the VGG54 feature
#     # if FLAGS.perceptual_mode == 'VGG54':
#     #     with tf.name_scope('vgg19_1') as scope:
#     #         extracted_feature_gen = VGG19_slim(gen_output, FLAGS.perceptual_mode, reuse=False, scope=scope)
#     #     with tf.name_scope('vgg19_2') as scope:
#     #         extracted_feature_target = VGG19_slim(targets, FLAGS.perceptual_mode, reuse=True, scope=scope)

#     # elif FLAGS.perceptual_mode == 'VGG22':
#     #     with tf.name_scope('vgg19_1') as scope:
#     #         extracted_feature_gen = VGG19_slim(gen_output, FLAGS.perceptual_mode, reuse=False, scope=scope)
#     #     with tf.name_scope('vgg19_2') as scope:
#     #         extracted_feature_target = VGG19_slim(targets, FLAGS.perceptual_mode, reuse=True, scope=scope)

#     # elif FLAGS.perceptual_mode == 'MSE':
#     extracted_feature_gen = gen_output
#     extracted_feature_target = targets

#     # else:
#     #     raise NotImplementedError('Unknown perceptual type')

#     # Calculating the generator loss
#     with tf.variable_scope('generator_loss'):
#         # Content loss
#         with tf.variable_scope('content_loss'):
#             # Compute the euclidean distance between the two features
#             # check=tf.equal(extracted_feature_gen, extracted_feature_target)
#             # diff = extracted_feature_gen - extracted_feature_target
#             if FLAGS.perceptual_mode == 'MSE':
#                 # content_loss = tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))
#                 content_loss, target_translated=losses.compute_unalign_loss(extracted_feature_gen, extracted_feature_target,
#                     tar_h=512, tar_w=512, tol=16, losstype='l1', stride=2)
#             # else:
#             #     content_loss = FLAGS.vgg_scaling * tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))

#         gen_loss = content_loss

#     # Define the learning rate and global step
#     with tf.variable_scope('get_learning_rate_and_global_step'):
#         global_step = tf.contrib.framework.get_or_create_global_step()
#         learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step, FLAGS.decay_rate,
#                                                    staircase=FLAGS.stair)
#         incr_global_step = tf.assign(global_step, global_step + 1)

#     with tf.variable_scope('generator_train'):
#         # Need to wait discriminator to perform train step
#         with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
#             gen_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
#             gen_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.beta)
#             gen_grads_and_vars = gen_optimizer.compute_gradients(gen_loss, gen_tvars)
#             gen_train = gen_optimizer.apply_gradients(gen_grads_and_vars)

#     # [ToDo] If we do not use moving average on loss??
#     exp_averager = tf.train.ExponentialMovingAverage(decay=0.99)
#     update_loss = exp_averager.apply([content_loss])

#     return Network(
#         content_loss=exp_averager.average(content_loss),
#         gen_grads_and_vars=gen_grads_and_vars,
#         gen_output=gen_output,
#         target_translated=target_translated,
#         train=tf.group(update_loss, incr_global_step, gen_train),
#         global_step=global_step,
#         learning_rate=learning_rate
#     )


# Definition of the generator
def SRResnet(gen_inputs, gen_output_channels, up_ratio=2, reuse=False, up_type='deconv', is_training=True):
    # # Check the flag
    # if FLAGS is None:
    #     raise  ValueError('No FLAGS is provided for generator')

    # The Bx residual blocks
    def residual_block(inputs, output_channel, stride, scope):
        with tf.variable_scope(scope):
            net = srganops.conv2(inputs, 4, output_channel, stride, use_bias=False, scope='conv_1')
            net = srganops.batchnorm(net, is_training)
            net = srganops.prelu_tf(net)
            net = srganops.conv2(net, 3, output_channel, stride, use_bias=False, scope='conv_2')
            net = srganops.batchnorm(net, is_training)
            net = net + inputs

        return net

    with tf.variable_scope('generator_unit', reuse=reuse):
        # The input layer
        with tf.variable_scope('input_stage'):
            net = srganops.conv2(gen_inputs, 9, 64, 1, scope='conv')
            net = srganops.prelu_tf(net)

        stage1_output = net

        # The residual block parts
        for i in range(1, 16+1 , 1):
            name_scope = 'resblock_%d'%(i)
            net = residual_block(net, 64, 1, name_scope)

        with tf.variable_scope('resblock_output'):
            net = srganops.conv2(net, 3, 64, 1, use_bias=False, scope='conv')
            net = srganops.batchnorm(net, is_training)

        net = net + stage1_output

        if up_type == 'subpixel':
            with tf.variable_scope('subpixelconv_stage1'):
                net = srganops.conv2(net, 3, 256, 1, scope='conv')
                net = srganops.pixelShuffler(net, scale=2)
                net = srganops.prelu_tf(net)

            with tf.variable_scope('subpixelconv_stage2'):
                net = srganops.conv2(net, 3, 256, 1, scope='conv')
                net = srganops.pixelShuffler(net, scale=2)
                net = srganops.prelu_tf(net)

            if up_ratio == 4:
                with tf.variable_scope('subpixelconv_stage3'):
                    net = srganops.conv2(net, 3, 256, 1, scope='conv')
                    net = srganops.pixelShuffler(net, scale=2)
                    net = srganops.prelu_tf(net)

            with tf.variable_scope('output_stage'):
                net = srganops.conv2(net, 9, gen_output_channels, 1, scope='conv')

        elif up_type == 'deconv':
            with tf.variable_scope('deconv_stage1'):
                net = srganops.conv2(net, 3, 256, 1, scope='conv')
                net = srganops.deconv2(net, 3, 256, 2)
                # net = srganops.pixelShuffler(net, scale=2)
                net = srganops.prelu_tf(net)

            with tf.variable_scope('deconv_stage2'):
                net = srganops.conv2(net, 3, 256, 1, scope='conv')
                net = srganops.deconv2(net, 3, 256, 2)
                # net = srganops.pixelShuffler(net, scale=2)
                net = srganops.prelu_tf(net)

            if up_ratio == 4:
                with tf.variable_scope('deconv_stage3'):
                    net = srganops.conv2(net, 3, 256, 1, scope='conv')
                    net = srganops.deconv2(net, 3, 256, 2)
                    # net = srganops.pixelShuffler(net, scale=2)
                    net = srganops.prelu_tf(net)

            with tf.variable_scope('deconv_output_stage'):
                net = srganops.conv2(net, 9, gen_output_channels, 1, scope='conv')

    return net


# Definition of the discriminator
def discriminator(dis_inputs, FLAGS=None):
    if FLAGS is None:
        raise ValueError('No FLAGS is provided for generator')

    # Define the discriminator block
    def discriminator_block(inputs, output_channel, kernel_size, stride, scope):
        with tf.variable_scope(scope):
            net = srganops.conv2(inputs, kernel_size, output_channel, stride, use_bias=False, scope='conv1')
            net = srganops.batchnorm(net, FLAGS.is_training)
            net = srganops.lrelu(net, 0.2)

        return net

    with tf.device('/gpu:0'):
        with tf.variable_scope('discriminator_unit'):
            # The input layer
            with tf.variable_scope('input_stage'):
                net = srganops.conv2(dis_inputs, 3, 64, 1, scope='conv')
                net = srganops.lrelu(net, 0.2)

            # The discriminator block part
            # block 1
            net = discriminator_block(net, 64, 3, 2, 'disblock_1')

            # block 2
            net = discriminator_block(net, 128, 3, 1, 'disblock_2')

            # block 3
            net = discriminator_block(net, 128, 3, 2, 'disblock_3')

            # block 4
            net = discriminator_block(net, 256, 3, 1, 'disblock_4')

            # block 5
            net = discriminator_block(net, 256, 3, 2, 'disblock_5')

            # block 6
            net = discriminator_block(net, 512, 3, 1, 'disblock_6')

            # block_7
            net = discriminator_block(net, 512, 3, 2, 'disblock_7')

            # The dense layer 1
            with tf.variable_scope('dense_layer_1'):
                net = slim.flatten(net)
                net = srganops.denselayer(net, 1024)
                net = srganops.lrelu(net, 0.2)

            # The dense layer 2
            with tf.variable_scope('dense_layer_2'):
                net = srganops.denselayer(net, 1)
                net = tf.nn.sigmoid(net)

    return net

