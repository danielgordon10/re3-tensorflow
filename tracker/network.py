import tensorflow as tf

import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))

from re3_utils.tensorflow_util import tf_util
from re3_utils.tensorflow_util.CaffeLSTMCell import CaffeLSTMCell

from constants import LSTM_SIZE

IMAGENET_MEAN = [123.151630838, 115.902882574, 103.062623801]

msra_initializer = tf.contrib.layers.variance_scaling_initializer()
bias_initializer = tf.zeros_initializer()
prelu_initializer = tf.constant_initializer(0.25)

def alexnet_conv_layers(input, batch_size, num_unrolls):
    input = tf.to_float(input) - IMAGENET_MEAN
    with tf.variable_scope('conv1'):
        conv1 = tf_util.conv_layer(input, 96, 11, 4, padding='VALID')
        pool1 = tf.nn.max_pool(
                conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',
                name='pool1')
        lrn1 = tf.nn.local_response_normalization(pool1, depth_radius=2,
                alpha=2e-5, beta=0.75, bias=1.0, name='norm1')

    with tf.variable_scope('conv1_skip'):
        prelu_skip = tf_util.get_variable('prelu', shape=[16], dtype=tf.float32,
                initializer=prelu_initializer)

        conv1_skip = tf_util.prelu(tf_util.conv_layer(lrn1, 16, 1, activation=None),
                prelu_skip)
        conv1_skip = tf.transpose(conv1_skip, perm=[0,3,1,2])
        conv1_skip_flat = tf_util.remove_axis(conv1_skip, [2,3])

    with tf.variable_scope('conv2'):
        conv2 = tf_util.conv_layer(lrn1, 256, 5, num_groups=2, padding='SAME')
        pool2 = tf.nn.max_pool(
                conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',
                name='pool2')
        lrn2 = tf.nn.local_response_normalization(pool2, depth_radius=2,
                alpha=2e-5, beta=0.75, bias=1.0, name='norm2')

    with tf.variable_scope('conv2_skip'):
        prelu_skip = tf_util.get_variable('prelu', shape=[32], dtype=tf.float32,
                initializer=prelu_initializer)

        conv2_skip = tf_util.prelu(tf_util.conv_layer(lrn2, 32, 1, activation=None),
                prelu_skip)
        conv2_skip = tf.transpose(conv2_skip, perm=[0,3,1,2])
        conv2_skip_flat = tf_util.remove_axis(conv2_skip, [2,3])

    with tf.variable_scope('conv3'):
        conv3 = tf_util.conv_layer(lrn2, 384, 3, padding='SAME')

    with tf.variable_scope('conv4'):
        conv4 = tf_util.conv_layer(conv3, 384, 3, num_groups=2, padding='SAME')

    with tf.variable_scope('conv5'):
        conv5 = tf_util.conv_layer(conv4, 256, 3, num_groups=2, padding='SAME')
        pool5 = tf.nn.max_pool(
                conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',
                name='pool5')
        pool5 = tf.transpose(pool5, perm=[0,3,1,2])
        pool5_flat = tf_util.remove_axis(pool5, [2,3])

    with tf.variable_scope('conv5_skip'):
        prelu_skip = tf_util.get_variable('prelu', shape=[64], dtype=tf.float32,
                initializer=prelu_initializer)

        conv5_skip = tf_util.prelu(tf_util.conv_layer(conv5, 64, 1, activation=None),
                prelu_skip)
        conv5_skip = tf.transpose(conv5_skip, perm=[0,3,1,2])
        conv5_skip_flat = tf_util.remove_axis(conv5_skip, [2,3])

    with tf.variable_scope('big_concat'):
        # Concat all skip layers.
        skip_concat = tf.concat([conv1_skip_flat, conv2_skip_flat, conv5_skip_flat, pool5_flat], 1)
        skip_concat_shape = skip_concat.get_shape().as_list()

        # Split and merge image pairs
        # (BxTx2)xHxWxC
        pool5_reshape = tf.reshape(skip_concat, [batch_size, num_unrolls, 2, skip_concat_shape[-1]])
        # (BxT)x(2xHxWxC)
        reshaped = tf_util.remove_axis(pool5_reshape, [1,3])

        return reshaped

def inference(inputs, num_unrolls, train, batch_size=None, prevLstmState=None, reuse=None):
    # Data should be in order BxTx2xHxWxC where T is the number of unrolls
    # Mean subtraction
    if batch_size is None:
        batch_size = int(inputs.get_shape().as_list()[0] / (num_unrolls * 2))

    variable_list = []

    if reuse is not None and not reuse:
        reuse = None

    with tf.variable_scope('re3', reuse=reuse):
        conv_layers = alexnet_conv_layers(inputs, batch_size, num_unrolls)

        # Embed Fully Connected Layer
        with tf.variable_scope('fc6'):
            fc6_out = tf_util.fc_layer(conv_layers, 1024)

            # (BxT)xC
            fc6_reshape = tf.reshape(fc6_out, tf.stack([batch_size, num_unrolls, fc6_out.get_shape().as_list()[-1]]))

        # LSTM stuff
        swap_memory = num_unrolls > 1
        with tf.variable_scope('lstm1'):
            #lstm1 = CaffeLSTMCell(LSTM_SIZE, initializer=msra_initializer)
            lstm1 = tf.contrib.rnn.LSTMCell(LSTM_SIZE, use_peepholes=True, initializer=msra_initializer, reuse=reuse)
            if prevLstmState is not None:
                state1 = tf.contrib.rnn.LSTMStateTuple(prevLstmState[0], prevLstmState[1])
            else:
                state1 = lstm1.zero_state(batch_size, dtype=tf.float32)
            lstm1_outputs, state1 = tf.nn.dynamic_rnn(lstm1, fc6_reshape, initial_state=state1, swap_memory=swap_memory)
            if train:
                lstmVars = [var for var in tf.trainable_variables() if 'lstm1' in var.name]
                for var in lstmVars:
                    tf_util.variable_summaries(var, var.name[:-2])

        with tf.variable_scope('lstm2'):
            #lstm2 = CaffeLSTMCell(LSTM_SIZE, initializer=msra_initializer)
            lstm2 = tf.contrib.rnn.LSTMCell(LSTM_SIZE, use_peepholes=True, initializer=msra_initializer, reuse=reuse)
            if prevLstmState is not None:
                state2 = tf.contrib.rnn.LSTMStateTuple(prevLstmState[2], prevLstmState[3])
            else:
                state2 = lstm2.zero_state(batch_size, dtype=tf.float32)
            lstm2_inputs = tf.concat([fc6_reshape, lstm1_outputs], 2)
            lstm2_outputs, state2 = tf.nn.dynamic_rnn(lstm2, lstm2_inputs, initial_state=state2, swap_memory=swap_memory)
            if train:
                lstmVars = [var for var in tf.trainable_variables() if 'lstm2' in var.name]
                for var in lstmVars:
                    tf_util.variable_summaries(var, var.name[:-2])
            # (BxT)xC
            outputs_reshape = tf_util.remove_axis(lstm2_outputs, 1)

        # Final FC layer.
        with tf.variable_scope('fc_output'):
            fc_output_out = tf_util.fc_layer(outputs_reshape, 4, activation=None)

    if prevLstmState is not None:
        return fc_output_out, state1, state2
    else:
        return fc_output_out

def get_var_list():
    return tf.trainable_variables()

def loss(outputs, labels):
    with tf.variable_scope('loss'):
        diff = tf.reduce_sum(tf.abs(outputs - labels, name='diff'), axis=1)
        loss = tf.reduce_mean(diff, name='loss')

    # L2 Loss on variables.
    with tf.variable_scope('l2_weight_penalty'):
        l2_weight_penalty = 0.0005 * tf.add_n([tf.nn.l2_loss(v)
            for v in get_var_list()])

    full_loss = loss + l2_weight_penalty

    return full_loss, loss

def training(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    with tf.device('/cpu:0'):
        global_step = tf.train.create_global_step()
    train_op = optimizer.minimize(loss, var_list=get_var_list(), global_step=global_step,
        colocate_gradients_with_ops=True)
    return train_op

