import tensorflow as tf
from ops import dense, conv2d, deconv2d, dense_layer, conv_tr_layer,\
    sigmoid_conv_tr_layer, conv_dropout_layer

def conv_discr(images, batch_size):
    """
    Convolutional Discriminator.
    """
    conv1 = tf.nn.relu(conv2d(images, 5, 16, "conv1"))
    conv2 = tf.nn.relu(conv2d(conv1, 5, 32, "conv2"))
    flat = tf.reshape(conv2, [batch_size, 7*7*32])
    
    #dense = tf.nn.relu(dense(flat, self.hidden, "hidden"))
    proba = tf.nn.sigmoid(dense(flat, 1, "dense_sigma"))
    
    return proba

def deconv_gen(z, batch_size):
    """
    Transpose-convolutional Generator.
    """
    z_expand1 = dense(z, 7*7*64, "expand1") # TODO : add a relu here ?
    z_matrix = tf.nn.relu(tf.reshape(z_expand1, [batch_size, 7, 7, 64]))
    deconv1 = tf.nn.relu(deconv2d(z_matrix, 5, 
        [batch_size, 14, 14, 32], "deconv1"))
    deconv2 = deconv2d(deconv1, 5, [batch_size, 28, 28, 1], "deconv2")
    gen_image = tf.nn.sigmoid(deconv2)
    
    return gen_image

def fc_discr(images, units):
    """
    Fully-connected Discriminator.
    """
    flat = tf.reshape(images, shape=[-1, 28*28])
    dense1 = tf.nn.relu(dense(flat, units, "discr/dense1"))
    dense2 = tf.nn.relu(dense(dense1, units, "discr/dense2"))
    dense3 = tf.nn.sigmoid(dense(dense2, 1, "discr/dense3"))
    return dense3

def fc_gen(z, units):
    """
    Fully-connected Generator.
    """
    dense1 = tf.nn.relu(dense(z, units, "gen/dense1"))
    dense2 = tf.nn.relu(dense(dense1, units, "gen/dense2"))
    dense3 = tf.nn.sigmoid(dense(dense2, 28*28, "gen/dense3"))
    deflat = tf.reshape(dense3, shape=[-1, 28, 28, 1])
    return deflat

def deep_gen(z, batch_size):
    """
    A deeper generator. Inspired by the tensorflow implementation.
    
    Args :
        - z (Tensor) : latent vectors.
    """
    with tf.variable_scope('generator'):
        dense1 = dense_layer(z, 7*7*256, "dense1")
        reshape = tf.reshape(dense1, [-1, 7, 7, 256], name='reshape')
        conv_tr1 = conv_tr_layer(reshape, 5, [batch_size, 7, 7, 128],\
            'conv_tr1', strides=1)
        conv_tr2 = conv_tr_layer(conv_tr1, 5, [batch_size, 14, 14, 64], 'conv_tr2')
        conv_tr3 = sigmoid_conv_tr_layer(\
            conv_tr2, 5, [batch_size, 28, 28, 1], 'conv_tr3')
    return conv_tr3

def deep_discr(images, training):
    """
    A deeper discriminator. Inspired by the tensorflow implementation.
    
    Args :
        - images (Tensor) : images.
        - training (bool) : whether we are in training of inference mode.
    """
    with tf.variable_scope('discriminator'):
        conv1 = conv_dropout_layer(images, 5, 64, training, name="conv1")
        conv2 = conv_dropout_layer(conv1, 5, 128, training, name="conv2")
        flat = tf.reshape(conv2, [-1, 7*7*128], name='flat')
        dense1 = tf.sigmoid(dense(flat, 1, 'dense'))
    return dense1
        

































