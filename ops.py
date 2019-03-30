import tensorflow as tf

def conv2d(X, size, num_f, name):
    """
    Conv2D layer
    
    Args :
        - X_in (Tensor): input;
        - size (Integer) : size (height and width) of the filters;
        - num_f (Integer): number of filters;
    Returns :
        - X_out (Tensor): output;
    """
    with tf.variable_scope(name):
        init = tf.truncated_normal_initializer(stddev=0.02)
        W = tf.get_variable('W', [size, size, X.shape[-1], num_f], initializer=init)
        b = tf.get_variable('b', [num_f], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(X, W, strides=[1, 2, 2, 1], padding="SAME") + b
    return conv

def deconv2d(X, size, output_shape, name, strides=2):
    """
    Transpose of Conv2D layer, improperly called deconvolution layer
    
    Args :
        - X (Tensor): input;
        - size (Integer) : size (height and width) of the filters;
        - output_shape (list of Integers): shape of the output Tensor;
        - name (str): name of the operation;
    Returns :
        - X_out (Tensor): output;
    """
    with tf.variable_scope(name):
        init = tf.truncated_normal_initializer(stddev=0.02)
        W = tf.get_variable('W', 
                            [size, size, output_shape[-1], X.shape[-1]],
                            initializer=init)
                            
        b = tf.get_variable('b',
                            [output_shape[-1]],
                            initializer=tf.constant_initializer(0.0))
        # check this !
        deconv = tf.nn.conv2d_transpose(X, W, output_shape,\
            [1, strides, strides, 1]) + b
        
    return deconv

def dense(X, units, name):
    """
    Fully-connected layer, with no linearity.
    
    Args :
        - X (Tensor): Input tensor;
        - units (Integer) : number of hidden units;
        - name (str) : name of the operation;
    """
    with tf.variable_scope(name):
        init = tf.random_normal_initializer(stddev=0.2)
        W = tf.get_variable('W', [X.shape[-1], units], initializer=init)
        b = tf.get_variable('b', [units], initializer=tf.constant_initializer(0.0))
        dense = tf.matmul(X, W) + b
    return dense

def batch_norm(X, name):
    """
    Batch normalization layer.
    
    Args :
        - X (Tensor) : input;
        - name (str) : name of the operation.
    """
    with tf.variable_scope(name):
        raise NotImplementedError

def dense_layer(X, units, name):
    """
    Dense layer, with batch-norm and leaky relu.
    """
    with tf.variable_scope(name):
        fc = dense(X, units, 'dense')
        bn = tf.layers.batch_normalization(fc, name='batch_norm')
        lrelu = tf.nn.leaky_relu(bn, name='leaky_relu')
    return lrelu

def conv_tr_layer(X, size, output_shape, name, strides=2):
    """
    Transpose-convolutional layer.
    
    Args:
        - X (Tensor) : Input;
        - size (int) : size of the sliding window;
        - output_shape (tuple or list of int) : shape of the output tensor;
        - name (str) : name of the op.
    """
    with tf.variable_scope(name):
        conv_tr = deconv2d(X, size, output_shape, "conv_tr", strides=strides)
        bn = tf.layers.batch_normalization(conv_tr, name='batch_norm')
        lrelu = tf.nn.leaky_relu(bn, name='leaky_relu')
    return lrelu

def sigmoid_conv_tr_layer(X, size, output_shape, name):
    """
    Last transpose-convolutional layer, ends with a sigmoid non-linearity.
    
    Args:
        - X (Tensor) : Input;
        - size (int) : size of the sliding window;
        - output_shape (tuple or list of int) : shape of the output tensor;
        - name (str) : name of the op.
    """
    with tf.variable_scope(name):
        conv_tr = deconv2d(X, size, output_shape, "conv_tr")
        sigmoid = tf.nn.sigmoid(conv_tr, name='sigmoid')
    return sigmoid

def conv_dropout_layer(X, size, num_filters, training, name):
    """
    Convolutional layer, with leaky relu non-linearity.
    
    Args:
        - X (Tensor) : input;
        - size (int) : size of the filters;
        - num_filters (int) : number of filters;
        - name (str) : name of the operation.
    """
    with tf.variable_scope(name):
        conv = conv2d(X, size, num_filters, "conv")
        lrelu = tf.nn.leaky_relu(conv, name='leaky_relu')
        dropout = tf.layers.dropout(lrelu, training=training, rate=0.3,
                                    name='dropout')
    return dropout



























