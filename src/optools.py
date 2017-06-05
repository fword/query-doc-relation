import tensorflow as tf
import numpy as np

def init_weights(shape, mean = 0.0001, stddev = 0.01, name=None):
    return tf.Variable(tf.random_normal(shape, mean = mean, stddev = stddev), name = name)

def init_bias(shape, val=0., name=None):
    initial = tf.constant(val, shape=shape)
    return tf.Variable(initial, name = name)

def element_wise_cosin(a, b, a_normed=False, b_normed=False, keep_dims=True):
    if a_normed:
        normalized_a = a 
    else:
        normalized_a = tf.nn.l2_normalize(a, 1)
    if b_normed:
        normalized_b = b 
    else:
        normalized_b = tf.nn.l2_normalize(b, 1)
    #return tf.matmul(normalized_a, normalized_b, transpose_b=True)
    return tf.reduce_sum(tf.multiply(normalized_a, normalized_b), 1, keep_dims=keep_dims)

#[batch_size, y] [x, y] => [batch_size, x]
def cosin(a, b, a_normed=False, b_normed=False):
    if a_normed:
        normalized_a = a 
    else:
        normalized_a = tf.nn.l2_normalize(a, 1)
    if b_normed:
        normalized_b = b 
    else:
        normalized_b = tf.nn.l2_normalize(b, 1)
    return tf.matmul(normalized_a, normalized_b, transpose_b=True)
def sparse_tensor_to_dense(input_tensor, maxlen=0):
  """
  notice maxlen must > your max real index
  otherwise runtime check error like 
  Invalid argument: indices[3] = [0,3] is out of bounds: need 0 <= index < [5,3]
  @FIXME still face this might be tf bug, when running mutlitple tf reading same data ?
  """
  if maxlen <= 0:
    return tf.sparse_tensor_to_dense(input_tensor)
  else:
    return tf.sparse_to_dense(input_tensor.indices, 
                              [input_tensor.dense_shape[0], maxlen], 
                              input_tensor.values)

activation_map = {'sigmoid' :  tf.nn.sigmoid, 'tanh' : tf.nn.tanh, 'relu' : tf.nn.relu}
