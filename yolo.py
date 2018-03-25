from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def leaky_relu(inputs, alpha=.1):
  return tf.maximum(inputs, alpha * inputs)

"""
Usage of arg scope:
  with slim.arg_scope(darknet_arg_scope()):
    logits, end_points = darknet.darknet_19(images, num_classes,
                                                is_training=is_training)
"""
def darknet_arg_scope(weight_decay=0.00004,
                      use_batch_norm=True,
                      batch_norm_decay=0.9997,
                      batch_norm_epsilon=0.001,
                      activation_fn=leaky_relu):
  """Defines the default arg scope for darknet models.
  Args:
    weight_decay: The weight decay to use for regularizing the model.
    use_batch_norm: "If `True`, batch_norm is applied after each convolution.
    batch_norm_decay: Decay for batch norm moving average.
    batch_norm_epsilon: Small float added to variance to avoid dividing by zero
      in batch norm.
    activation_fn: Activation function for conv2d.
  Returns:
    An `arg_scope` to use for the darknet models.
  """
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      # collection containing update_ops.
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      # use fused batch norm if possible.
      'fused': None,
  }
  if use_batch_norm:
    normalizer_fn = slim.batch_norm
    normalizer_params = batch_norm_params
  else:
    normalizer_fn = None
    normalizer_params = {}
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=slim.variance_scaling_initializer(),
        activation_fn=activation_fn,
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params) as sc:
      return sc

def darknet_19_base(inputs,
                    scope='darknet_19_base'):
  """Darknet model from https://arxiv.org/abs/1612.08242
    Args:
      inputs: a tensor of shape [batch_size, height, width, channels].
      scope: Optional variable_scope.
    Returns:
      tensor_out: output tensor corresponding to the final_endpoint.
      end_points: a set of activations for external use, for example summaries or
                  losses.
  """
  with tf.variable_scope(scope, 'darknet_19_base', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.conv2d(inputs, 32, [3, 3], scope='Conv2D_1')
      net = slim.max_pool2d(net, [2, 2], stride=2, scope='MaxPool_1')

      net = slim.conv2d(net,  64, [3, 3], scope='Conv2D_2')
      net = slim.max_pool2d(net, [2, 2], stride=2, scope='MaxPool_2')

      net = slim.conv2d(net, 128, [3, 3], scope='Conv2D_3')
      net = slim.conv2d(net,  64, [1, 1], scope='Conv2D_4')
      net = slim.conv2d(net, 128, [3, 3], scope='Conv2D_5')
      net = slim.max_pool2d(net, [2, 2], stride=2, scope='MaxPool_3')

      net = slim.conv2d(net, 256, [3, 3], scope='Conv2D_6')
      net = slim.conv2d(net, 128, [1, 1], scope='Conv2D_7')
      net = slim.conv2d(net, 256, [3, 3], scope='Conv2D_8')
      net = slim.max_pool2d(net, [2, 2], stride=2, scope='MaxPool_4')

      net = slim.conv2d(net, 512, [3, 3], scope='Conv2D_9')
      net = slim.conv2d(net, 256, [1, 1], scope='Conv2D_10')
      net = slim.conv2d(net, 512, [3, 3], scope='Conv2D_11')
      net = slim.conv2d(net, 256, [1, 1], scope='Conv2D_12')
      net = slim.conv2d(net, 512, [3, 3], scope='Conv2D_13')
      net = slim.max_pool2d(net, [2, 2], stride=2, scope='MaxPool_5')

      net = slim.conv2d(net,1024, [3, 3], scope='Conv2D_14')
      net = slim.conv2d(net, 512, [1, 1], scope='Conv2D_15')
      net = slim.conv2d(net,1024, [3, 3], scope='Conv2D_16')
      net = slim.conv2d(net, 512, [1, 1], scope='Conv2D_17')
      net = slim.conv2d(net,1024, [3, 3], scope='Conv2D_18')

      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      end_points['scope_name'] = sc.name

      return net, end_points

def darknet_19(inputs,
            num_classes=1000,
            is_training=True,
            prediction_fn=slim.softmax,
            spatial_squeeze=True,
            reuse=None,
            scope='darknet_19'):
  """Darknet-19 for classification
    Constructs an Darknet-19 network for classification as described in
    https://arxiv.org/abs/1612.08242
    The default image size used to train this network is 224x224.
    Args:
      inputs: a tensor of shape [batch_size, height, width, channels].
      num_classes: number of predicted classes. If 0 or None, the logits layer
        is omitted and the input features to the logits layer (before dropout)
        are returned instead.
      is_training: whether is training or not.
      prediction_fn: a function to get predictions out of logits.
      spatial_squeeze: if True, logits is of shape [B, C], if false logits is of
          shape [B, 1, 1, C], where B is batch_size and C is number of classes.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
  """
  with tf.variable_scope(scope, 'darknet_19', [inputs], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm], is_training=is_training):
      net, end_points = darknet_19_base(inputs, scope='base')

      with tf.variable_scope('Logits'):
        net = end_points['Conv2D_19'] \
            = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                          normalizer_fn=None, scope='Conv2D_19')
        net = end_points['global_pool'] \
            = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')

        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

      logits = net
      end_points['Logits'] = logits
      end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points

darknet_19.default_image_size = 224


class Yolo():
  """YOLO Feature Extractor using Darknet19 features."""

  def __init__(self,
               options,
               feed_dict,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams,
               batch_norm_trainable=True,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False):
    """Darknet19 Feature Extractor for SSD Models.
    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: Not used in YOLO
      min_depth:        Not used in YOLO
      pad_to_multiple:  Not used in YOLO
      conv_hyperparams: tf slim arg_scope for conv2d and separable_conv2d ops.
      batch_norm_trainable: Whether to update batch norm parameters during
        training or not. When training with a small batch size
        (e.g. 1), it is desirable to disable batch norm update and use
        pretrained batch norm params.
      reuse_weights: Whether to reuse variables. Default is None.
    """
    #super(YOLOv2Darknet19FeatureExtractor, self).__init__(
    #is_training, depth_multiplier, min_depth, pad_to_multiple,
    #conv_hyperparams, batch_norm_trainable, reuse_weights)

    self._is_training = is_training
    self._depth_multiplier = depth_multiplier
    self._min_depth = min_depth
    self._pad_to_multiple = pad_to_multiple
    self._conv_hyperparams = conv_hyperparams
    self._batch_norm_trainable = batch_norm_trainable
    self._reuse_weights = reuse_weights
    self._use_explicit_padding = use_explicit_padding
    self._use_depthwise = use_depthwise
    return self.extract_features(feed_dict['img_inp'])

  def preprocess(self, resized_inputs):
    """Darknet19 preprocessing.
    Maps pixel values to the range [-1, 1].
    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    return (2.0 / 255.0) * resized_inputs - 1.0

  def extract_features(self, preprocessed_inputs):
    """Extract features from preprocessed inputs.
    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    Returns:
      feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]
    """
    preprocessed_inputs.get_shape().assert_has_rank(4)
    shape_assert = tf.Assert(
        tf.logical_and(tf.greater_equal(tf.shape(preprocessed_inputs)[1], 33),
                       tf.greater_equal(tf.shape(preprocessed_inputs)[2], 33)),
        ['image size must at least be 33 in both height and width.'])

    with tf.control_dependencies([shape_assert]):
      with slim.arg_scope(darknet.darknet_arg_scope()):
        with tf.variable_scope('darknet_19',
                               reuse=self._reuse_weights) as scope:
          net, end_points = darknet.darknet_19_base(preprocessed_inputs,
                                                        scope='base')
          net = slim.conv2d(net, 1024, [3, 3], scope='Conv2D_19')
          net = slim.conv2d(net, 1024, [3, 3], scope='Conv2D_20')
          scope_name = end_points['scope_name']
          conv_13 = end_points[scope_name+'/Conv2D_13']
          conv_21 = slim.conv2d(conv_13, 64, [1, 1], scope='Conv2D_21')
          conv_21 = tf.space_to_depth(conv_21, block_size=2)
          net = tf.concat([conv_21, net], axis=-1)
          net = slim.conv2d(net, 1024, [3, 3], scope='Conv2D_22')
          feature_map = net
          pass
        pass
      pass
    print(feature_map.shape)
    exit(1)
    return [feature_map]
