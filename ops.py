import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np
from transformer import spatial_transformer_network as stn

# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init = tf_contrib.layers.variance_scaling_initializer()
weight_regularizer = tf_contrib.layers.l2_regularizer(0.0001)


##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, scope='conv_0'):

    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias, padding=padding)

        return x

def fully_conneted(x, units, use_bias=True, scope='fully_0'):
    with tf.variable_scope(scope):
        x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x

def spatial_transformer_layer(name_scope,
                              input_tensor,
                              img_size,
                              kernel_size,
                              pooling=None,
                              strides=[1, 1, 1, 1],
                              pool_strides=[1, 1, 1, 1],
                              activation=tf.nn.relu,
                              use_bn=False,
                              use_mvn=False,
                              is_training=False,
                              use_lrn=False,
                              keep_prob=1.0,
                              dropout_maps=False,
                              init_opt=0,
                              bias_init=0.1):
    """
        Define spatial transformer network layer
        Args:
        scope_or_name: `string` or `VariableScope`, the scope to open.
        inputs: `4-D Tensor`, it is assumed that `inputs` is shaped `[batch_size, Y, X, Z]`.
        kernel: `4-D Tensor`, [kernel_height, kernel_width, in_channels, out_channels] kernel.
        img_size: 2D array, [image_width. image_height]
        bias: `1-D Tensor`, [out_channels] bias.
        strides: list of `ints`, length 4, the stride of the sliding window for each dimension of `inputs`.
        activation: activation function to be used (default: `tf.nn.relu`).
        use_bn: `bool`, whether or not to include batch normalization in the layer.
        is_training: `bool`, whether or not the layer is in training mode. This is only used if `use_bn` == True.
        use_lrn: `bool`, whether or not to include local response normalization in the layer.
        keep_prob: `double`, dropout keep prob.
        dropout_maps: `bool`, If true whole maps are dropped or not, otherwise single elements.
        padding: `string` from 'SAME', 'VALID'. The type of padding algorithm used in the convolution.
    Returns:
        `4-D Tensor`, has the same type `inputs`.
    """

    img_height = img_size[0]
    img_width = img_size[1]

    with tf.variable_scope(name_scope):
        if init_opt == 0:
            stddev = np.sqrt(2 / (kernel_size[0] * kernel_size[1] * kernel_size[2] * kernel_size[3]))

        elif init_opt == 1:
            stddev = 5e-2

        elif init_opt == 2:
            stddev = min(np.sqrt(2.0 / (kernel_size[0] * kernel_size[1] * kernel_size[2])), 5e-2)

        kernel = tf.get_variable('weights', kernel_size,
                                 initializer=tf.random_normal_initializer(stddev=stddev))

        conv = tf.nn.conv2d(input_tensor, kernel, strides, padding='SAME', name='conv')

        bias = tf.get_variable('bias', kernel_size[3],
                               initializer=tf.constant_initializer(value=bias_init))

        output_tensor = tf.nn.bias_add(conv, bias, name='pre_activation')

        if activation:
            output_tensor = activation(output_tensor, name='activation')

        if use_lrn:
            output_tensor = tf.nn.local_response_normalization(output_tensor, name='local_responsive_normalization')

        if dropout_maps:
            conv_shape = tf.shape(output_tensor)
            n_shape = tf.stack([conv_shape[0], 1, 1, conv_shape[3]])
            output_tensor = tf.nn.dropout(output_tensor, keep_prob=keep_prob, noise_shape=n_shape)
        else:
            output_tensor = tf.nn.dropout(output_tensor, keep_prob=keep_prob)

        if pooling:
            output_tensor = tf.nn.max_pool(output_tensor, ksize=pooling, strides=pool_strides, padding='VALID')

        output_tensor = tf.contrib.layers.flatten(output_tensor)

        output_tensor = tf.contrib.layers.fully_connected(output_tensor, 64, scope='fully_connected_layer_1')
        output_tensor = tf.nn.tanh(output_tensor)

        output_tensor = tf.contrib.layers.fully_connected(output_tensor, 6, scope='fully_connected_layer_2')
        output_tensor = tf.nn.tanh(output_tensor)

        stn_output = stn(input_fmap=input_tensor, theta=output_tensor, out_dims=(img_height, img_width))

        return stn_output, output_tensor

def resblock(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='resblock') :
    with tf.variable_scope(scope) :

        x = batch_norm(x_init, is_training, scope='batch_norm_0')
        x = relu(x)


        if downsample :
            x = conv(x, channels, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
            x_init = conv(x_init, channels, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')

        else :
            x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')

        x = batch_norm(x, is_training, scope='batch_norm_1')
        x = relu(x)
        x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_1')

        return x + x_init

def bottle_resblock(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='bottle_resblock') :
    with tf.variable_scope(scope) :
        x = batch_norm(x_init, is_training, scope='batch_norm_1x1_front')
        shortcut = relu(x)

        x = conv(shortcut, channels, kernel=1, stride=1, use_bias=use_bias, scope='conv_1x1_front')
        x = batch_norm(x, is_training, scope='batch_norm_3x3')
        x = relu(x)

        if downsample :
            x = conv(x, channels, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
            shortcut = conv(shortcut, channels*4, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')

        else :
            x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')
            shortcut = conv(shortcut, channels * 4, kernel=1, stride=1, use_bias=use_bias, scope='conv_init')

        x = batch_norm(x, is_training, scope='batch_norm_1x1_back')
        x = relu(x)
        x = conv(x, channels*4, kernel=1, stride=1, use_bias=use_bias, scope='conv_1x1_back')

        return x + shortcut

def get_residual_layer(res_n) :
    x = []

    if res_n == 18 :
        x = [2, 2, 2, 2]

    if res_n == 34 :
        x = [3, 4, 6, 3]

    if res_n == 50 :
        x = [3, 4, 6, 3]

    if res_n == 101 :
        x = [3, 4, 23, 3]

    if res_n == 152 :
        x = [3, 8, 36, 3]

    return x

##################################################################################
# Sampling
##################################################################################

def flatten(x) :
    return tf.layers.flatten(x)

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    return gap

def avg_pooling(x) :
    return tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='SAME')

##################################################################################
# Activation function
##################################################################################


def relu(x):
    return tf.nn.relu(x)

##################################################################################
# Normalization function
##################################################################################

def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)

##################################################################################
# Loss function
##################################################################################

def classification_loss(labels, theta, org) :
    logits = stn(org, theta)

    n_class = 1
    flat_logits = tf.reshape(logits, [-1])
    flat_labels = tf.reshape(labels, [-1])

    # print(tf.shape(flat_logits))
    # print(tf.shape(flat_labels))

    loss = tf.losses.mean_squared_error(flat_labels, flat_logits)

    # flat_logits = tf.multiply(flat_logits, 255.0)
    # flat_labels = tf.multiply(flat_labels, 255.0)

    # flat_logits = tf.dtypes.cast(flat_logits, dtype=tf.int32)
    # flat_labels = tf.dtypes.cast(flat_labels, dtype=tf.int32)

    # accuracy, update_op = tf.metrics.accuracy(labels=flat_labels[0],
    #                                       predictions=flat_logits[0])

    # return loss, accuracy
    return loss



