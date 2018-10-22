# Converted to TensorFlow .caffemodel
# with the DeepLab-ResNet configuration.
# The batch normalisation layer is provided by
# the slim library (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim).

from kaffe.tensorflow import Network
import tensorflow as tf

class PlaneNet(Network):
    def setup(self, is_training, options):
        '''Network definition.
        
        Args:
          is_training: whether to update the running mean and variance of the batch normalisation layer.
                       If the batch size is small, it is better to keep the running mean and variance of 
                       the-pretrained model frozen.
          options: contains network configuration parameters
        '''

        if False: # Dilated Residual Networks change the first few layers to deal with the gridding issue
            (self.feed('img_inp')
                 .conv(7, 7, 64, 2, 2, biased=False, relu=False, name='conv1')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1')
                 .max_pool(3, 3, 2, 2, name='pool1'))
            
            (self.feed('pool1')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch1'))
                 
            (self.feed('pool1')
                 .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2a_branch2a')
                 .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2a_branch2b')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c')
                 .batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch2c'))            
        else:
            with tf.variable_scope('degridding'):
                (self.feed('img_inp')
                     .conv(7, 7, 16, 1, 1, biased=False, relu=False, name='conv1')
                     .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn1')
                     .conv(1, 1, 16, 2, 2, biased=False, relu=False, name='conv2_c')
                     .batch_normalization(is_training=is_training, activation_fn=None, name='bn2c'))

                (self.feed('bn1')
                     .conv(3, 3, 16, 1, 1, biased=False, relu=False, name='conv2a')
                     .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2a')
                     .conv(3, 3, 16, 2, 2, biased=False, relu=False, name='conv2b')
                     .batch_normalization(is_training=is_training, activation_fn=None, name='bn2b'))
                
                (self.feed('bn2b',
                           'bn2c')
                     .add(name='add1')
                     .relu(name='relu1')
                     .conv(1, 1, 32, 2, 2, biased=False, relu=False, name='conv3c')
                     .batch_normalization(is_training=is_training, activation_fn=None, name='bn3c'))
                (self.feed('relu1')
                     .conv(3, 3, 32, 1, 1, biased=False, relu=False, name='conv3a')
                     .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3a')
                     .conv(3, 3, 32, 2, 2, biased=False, relu=False, name='conv3b')
                     .batch_normalization(is_training=is_training, activation_fn=None, name='bn3b'))
                (self.feed('bn3b',
                           'bn3c')
                     .add(name='add2')
                     .relu(name='pool1'))

                pass
            pass

        (self.feed('pool1')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch1'))
                 
        (self.feed('pool1')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2a_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2a_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch2c'))

        (self.feed('bn2a_branch1', 
                   'bn2a_branch2c')
             .add(name='res2a')
             .relu(name='res2a_relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2b_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2b_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn2b_branch2c'))

        (self.feed('res2a_relu', 
                   'bn2b_branch2c')
             .add(name='res2b')
             .relu(name='res2b_relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2c_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2c_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2c_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2c_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2c_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn2c_branch2c'))

        (self.feed('res2b_relu', 
                   'bn2c_branch2c')
             .add(name='res2c')
             .relu(name='res2c_relu')
             .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res3a_branch1')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn3a_branch1'))

        (self.feed('res2c_relu')
             .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='res3a_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3a_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3a_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3a_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3a_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn3a_branch2c'))

        (self.feed('bn3a_branch1', 
                   'bn3a_branch2c')
             .add(name='res3a')
             .relu(name='res3a_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b1_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b1_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b1_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b1_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b1_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn3b1_branch2c'))

        (self.feed('res3a_relu', 
                   'bn3b1_branch2c')
             .add(name='res3b1')
             .relu(name='res3b1_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b2_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b2_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b2_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b2_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b2_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn3b2_branch2c'))

        (self.feed('res3b1_relu', 
                   'bn3b2_branch2c')
             .add(name='res3b2')
             .relu(name='res3b2_relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b3_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b3_branch2a')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b3_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b3_branch2b')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b3_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn3b3_branch2c'))

        (self.feed('res3b2_relu', 
                   'bn3b3_branch2c')
             .add(name='res3b3')
             .relu(name='res3b3_relu')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch1')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4a_branch1'))

        (self.feed('res3b3_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4a_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4a_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4a_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4a_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4a_branch2c'))

        (self.feed('bn4a_branch1', 
                   'bn4a_branch2c')
             .add(name='res4a')
             .relu(name='res4a_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b1_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b1_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b1_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b1_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b1_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b1_branch2c'))

        (self.feed('res4a_relu', 
                   'bn4b1_branch2c')
             .add(name='res4b1')
             .relu(name='res4b1_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b2_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b2_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b2_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b2_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b2_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b2_branch2c'))

        (self.feed('res4b1_relu', 
                   'bn4b2_branch2c')
             .add(name='res4b2')
             .relu(name='res4b2_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b3_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b3_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b3_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b3_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b3_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b3_branch2c'))

        (self.feed('res4b2_relu', 
                   'bn4b3_branch2c')
             .add(name='res4b3')
             .relu(name='res4b3_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b4_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b4_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b4_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b4_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b4_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b4_branch2c'))

        (self.feed('res4b3_relu', 
                   'bn4b4_branch2c')
             .add(name='res4b4')
             .relu(name='res4b4_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b5_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b5_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b5_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b5_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b5_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b5_branch2c'))

        (self.feed('res4b4_relu', 
                   'bn4b5_branch2c')
             .add(name='res4b5')
             .relu(name='res4b5_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b6_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b6_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b6_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b6_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b6_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b6_branch2c'))

        (self.feed('res4b5_relu', 
                   'bn4b6_branch2c')
             .add(name='res4b6')
             .relu(name='res4b6_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b7_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b7_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b7_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b7_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b7_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b7_branch2c'))

        (self.feed('res4b6_relu', 
                   'bn4b7_branch2c')
             .add(name='res4b7')
             .relu(name='res4b7_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b8_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b8_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b8_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b8_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b8_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b8_branch2c'))

        (self.feed('res4b7_relu', 
                   'bn4b8_branch2c')
             .add(name='res4b8')
             .relu(name='res4b8_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b9_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b9_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b9_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b9_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b9_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b9_branch2c'))

        (self.feed('res4b8_relu', 
                   'bn4b9_branch2c')
             .add(name='res4b9')
             .relu(name='res4b9_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b10_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b10_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b10_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b10_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b10_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b10_branch2c'))

        (self.feed('res4b9_relu', 
                   'bn4b10_branch2c')
             .add(name='res4b10')
             .relu(name='res4b10_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b11_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b11_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b11_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b11_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b11_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b11_branch2c'))

        (self.feed('res4b10_relu', 
                   'bn4b11_branch2c')
             .add(name='res4b11')
             .relu(name='res4b11_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b12_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b12_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b12_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b12_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b12_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b12_branch2c'))

        (self.feed('res4b11_relu', 
                   'bn4b12_branch2c')
             .add(name='res4b12')
             .relu(name='res4b12_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b13_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b13_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b13_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b13_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b13_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b13_branch2c'))

        (self.feed('res4b12_relu', 
                   'bn4b13_branch2c')
             .add(name='res4b13')
             .relu(name='res4b13_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b14_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b14_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b14_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b14_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b14_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b14_branch2c'))

        (self.feed('res4b13_relu', 
                   'bn4b14_branch2c')
             .add(name='res4b14')
             .relu(name='res4b14_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b15_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b15_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b15_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b15_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b15_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b15_branch2c'))

        (self.feed('res4b14_relu', 
                   'bn4b15_branch2c')
             .add(name='res4b15')
             .relu(name='res4b15_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b16_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b16_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b16_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b16_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b16_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b16_branch2c'))

        (self.feed('res4b15_relu', 
                   'bn4b16_branch2c')
             .add(name='res4b16')
             .relu(name='res4b16_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b17_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b17_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b17_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b17_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b17_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b17_branch2c'))

        (self.feed('res4b16_relu', 
                   'bn4b17_branch2c')
             .add(name='res4b17')
             .relu(name='res4b17_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b18_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b18_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b18_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b18_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b18_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b18_branch2c'))

        (self.feed('res4b17_relu', 
                   'bn4b18_branch2c')
             .add(name='res4b18')
             .relu(name='res4b18_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b19_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b19_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b19_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b19_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b19_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b19_branch2c'))

        (self.feed('res4b18_relu', 
                   'bn4b19_branch2c')
             .add(name='res4b19')
             .relu(name='res4b19_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b20_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b20_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b20_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b20_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b20_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b20_branch2c'))

        (self.feed('res4b19_relu', 
                   'bn4b20_branch2c')
             .add(name='res4b20')
             .relu(name='res4b20_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b21_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b21_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b21_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b21_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b21_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b21_branch2c'))

        (self.feed('res4b20_relu', 
                   'bn4b21_branch2c')
             .add(name='res4b21')
             .relu(name='res4b21_relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b22_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b22_branch2a')
             .atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b22_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b22_branch2b')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b22_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn4b22_branch2c'))

        (self.feed('res4b21_relu', 
                   'bn4b22_branch2c')
             .add(name='res4b22')
             .relu(name='res4b22_relu')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch1')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn5a_branch1'))

        (self.feed('res4b22_relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5a_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5a_branch2a')
             .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5a_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5a_branch2b')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn5a_branch2c'))

        (self.feed('bn5a_branch1', 
                   'bn5a_branch2c')
             .add(name='res5a')
             .relu(name='res5a_relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5b_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5b_branch2a')
             .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5b_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5b_branch2b')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5b_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn5b_branch2c'))

        (self.feed('res5a_relu', 
                   'bn5b_branch2c')
             .add(name='res5b')
             .relu(name='res5b_relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5c_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5c_branch2a')
             .atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5c_branch2b')
             .batch_normalization(activation_fn=tf.nn.relu, name='bn5c_branch2b', is_training=is_training)
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5c_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn5c_branch2c'))

        (self.feed('res5b_relu', 
                   'bn5c_branch2c')
             .add(name='res5c')
             .relu(name='res5c_relu'))

        
        
        (self.feed('res5c_relu')
             .avg_pool(24, 32, 24, 32, name='res5d_pool1')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5d_pool1_conv')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='res5d_pool1_bn')
             .resize_bilinear(size=[24, 32], name='res5d_upsample1'))

        (self.feed('res5c_relu')
             .avg_pool(12, 16, 12, 16, name='res5d_pool2')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5d_pool2_conv')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='res5d_pool2_bn')
             .resize_bilinear(size=[24, 32], name='res5d_upsample2'))

        (self.feed('res5c_relu')
             .avg_pool(6, 8, 6, 8, name='res5d_pool3')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5d_pool3_conv')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='res5d_pool3_bn')
             .resize_bilinear(size=[24, 32], name='res5d_upsample3'))

        (self.feed('res5c_relu')
             .avg_pool(3, 4, 3, 4, name='res5d_pool4')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5d_pool4_conv')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='res5d_pool4_bn')
             .resize_bilinear(size=[24, 32], name='res5d_upsample4'))


        #deep supervision at layers in list options.deepSupervisionLayers
        if len(options.deepSupervisionLayers) > 0:
            with tf.variable_scope('deep_supervision'):
                for layerIndex, layer in enumerate(options.deepSupervisionLayers):
                    (self.feed(layer)
                         .avg_pool(24, 32, 24, 32, name=layer+'_pool1')
                         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name=layer+'_pool1_conv')
                         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name=layer+'_pool1_bn')
                         .resize_bilinear(size=[24, 32], name=layer+'_upsample1'))
            
                    (self.feed(layer)
                         .avg_pool(12, 16, 12, 16, name=layer+'_pool2')
                         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name=layer+'_pool2_conv')
                         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name=layer+'_pool2_bn')
                         .resize_bilinear(size=[24, 32], name=layer+'_upsample2'))
            
                    (self.feed(layer)
                         .avg_pool(6, 8, 6, 8, name=layer+'_pool3')
                         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name=layer+'_pool3_conv')
                         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name=layer+'_pool3_bn')
                         .resize_bilinear(size=[24, 32], name=layer+'_upsample3'))

                    (self.feed(layer)
                         .avg_pool(3, 4, 3, 4, name=layer+'_pool4')
                         .conv(1, 1, 512, 1, 1, biased=False, relu=False, name=layer+'_pool4_conv')
                         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name=layer+'_pool4_bn')
                         .resize_bilinear(size=[24, 32], name=layer+'_upsample4'))

                    
                    (self.feed(layer+'_pool1')
                         .reshape(shape=[-1, 1024], name=layer+'_plane_reshape1')
                         .fc(num_out=options.numOutputPlanes * 3, name=layer+'_plane_fc', relu=False)
                         .reshape(shape=[-1, options.numOutputPlanes, 3], name=layer+'_plane_pred'))

                    if options.predictConfidence == 1 and layerIndex > 0:
                        (self.feed(layer+'_plane_reshape1')
                             .fc(num_out=options.numOutputPlanes, name=layer+'_plane_confidence_fc', relu=False)
                             .reshape(shape=[-1, options.numOutputPlanes, 1], name=layer+'_plane_confidence_pred'))
                        pass
                    
                    (self.feed(layer,
                               layer+'_upsample1',
                               layer+'_upsample2',
                               layer+'_upsample3',
                               layer+'_upsample4')
                         .concat(axis=3, name=layer+'_segmentation_concat')
                         .conv(3, 3, 512, 1, 1, biased=False, relu=False, name=layer+'_segmentation_conv1')
                         .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name=layer+'_segmentation_bn1')
                         .dropout(keep_prob=0.9, name=layer+'_segmentation_dropout')
                         .conv(1, 1, options.numConcaveGroups + options.numConvexGroups, 1, 1, relu=False, name=layer+'_segmentation_conv2')
                         .resize_bilinear(size=[192, 256], name=layer+'_segmentation_pred'))

                    (self.feed(layer+'_segmentation_dropout')
                         .conv(1, 1, 1, 1, 1, relu=False, name=layer+'_non_plane_mask_conv2')
                         .resize_bilinear(size=[192, 256], name=layer+'_non_plane_mask_pred'))
                    # (self.feed(layer+'_segmentation_dropout')
                    #      .conv(1, 1, 1, 1, 1, relu=False, name=layer+'_non_plane_depth_conv2')
                    #      .resize_bilinear(size=[192, 256], name=layer+'_non_plane_depth_pred'))
                    # (self.feed(layer+'_segmentation_dropout')
                    #      .conv(1, 1, 3, 1, 1, relu=False, name=layer+'_non_plane_normal_conv2')
                    #      .resize_bilinear(size=[192, 256], name=layer+'_non_plane_normal_pred'))
                    continue
                pass
            pass
            


        (self.feed('res5d_pool1')
             .reshape(shape=[-1, 2048], name='plane_reshape1')
             .fc(num_out=options.numOutputPlanes * 3, name='plane_fc', relu=False)
             .reshape(shape=[-1, options.numOutputPlanes, 3], name='plane_pred'))

        if options.predictConfidence == 1:
            (self.feed('plane_reshape1')
                 .fc(num_out=options.numOutputPlanes, name='plane_confidence_fc', relu=False)
                 .reshape(shape=[-1, options.numOutputPlanes, 1], name='plane_confidence_pred'))
            pass
             
        (self.feed('res5c_relu',
                   'res5d_upsample1',
                   'res5d_upsample2',
                   'res5d_upsample3',
                   'res5d_upsample4')
             .concat(axis=3, name='segmentation_concat')
             .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='segmentation_conv1')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='segmentation_bn1')
             .dropout(keep_prob=0.9, name='segmentation_dropout')
             .conv(1, 1, options.numConcaveGroups + options.numConvexGroups, 1, 1, relu=False, name='segmentation_conv2')
             .resize_bilinear(size=[192, 256], name='segmentation_pred'))
        

        (self.feed('segmentation_dropout')
             .conv(1, 1, 1, 1, 1, relu=False, name='non_plane_mask_conv2')
             .resize_bilinear(size=[192, 256], name='non_plane_mask_pred'))
        (self.feed('segmentation_dropout')
             .conv(1, 1, 1, 1, 1, relu=False, name='non_plane_depth_conv2')
             .resize_bilinear(size=[192, 256], name='non_plane_depth_pred'))
        (self.feed('segmentation_dropout')
             .conv(1, 1, 3, 1, 1, relu=False, name='non_plane_normal_conv2')
             .resize_bilinear(size=[192, 256], name='non_plane_normal_pred'))


        #boundary prediction
        if options.predictBoundary == 1:
            (self.feed('segmentation_dropout')
                 .conv(1, 1, 1, 1, 1, relu=False, name='boundary_smooth_conv5')
                 .resize_bilinear(size=[192, 256], name='boundary_smooth_upsample5'))
            (self.feed('segmentation_dropout')
                 .conv(1, 1, 1, 1, 1, relu=False, name='boundary_occlusion_conv5')
                 .resize_bilinear(size=[192, 256], name='boundary_occlusion_upsample5'))
            (self.feed('bn1')
                 .conv(1, 1, 1, 1, 1, relu=False, name='boundary_conv0')
                 .resize_bilinear(size=[192, 256], name='boundary_upsample0'))
            (self.feed('relu1')
                 .conv(1, 1, 1, 1, 1, relu=False, name='boundary_conv1')
                 .resize_bilinear(size=[192, 256], name='boundary_upsample1'))
            (self.feed('res2c_relu')
                 .conv(1, 1, 1, 1, 1, relu=False, name='boundary_conv2')
                 .resize_bilinear(size=[192, 256], name='boundary_upsample2'))
            (self.feed('res3b3_relu')
                 .conv(1, 1, 1, 1, 1, relu=False, name='boundary_conv3')
                 .resize_bilinear(size=[192, 256], name='boundary_upsample3'))
            (self.feed('boundary_smooth_upsample5',
                       'boundary_upsample0',                   
                       'boundary_upsample1',
                       'boundary_upsample2',
                       'boundary_upsample3')
                 .concat(axis=3, name='boundary_smooth_concat')         
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='boundary_smooth_bn')
                 .conv(1, 1, 1, 1, 1, relu=False, name='boundary_smooth_pred'))
            (self.feed('boundary_occlusion_upsample5',
                       'boundary_upsample0',
                       'boundary_upsample1',
                       'boundary_upsample2',
                       'boundary_upsample3')
                 .concat(axis=3, name='boundary_occlusion_concat')
                 .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='boundary_occlusion_bn')
                 .conv(1, 1, 1, 1, 1, relu=False, name='boundary_occlusion_pred'))
            (self.feed('boundary_smooth_pred',
                       'boundary_occlusion_pred')
                 .concat(axis=3, name='boundary_pred'))
            pass

        #local prediction
        if options.predictLocal == 1:
            (self.feed('segmentation_dropout')
                 .conv(1, 1, 1, 1, 1, relu=False, name='local_score_pred'))
            (self.feed('segmentation_dropout')
                 .conv(1, 1, 3, 1, 1, relu=False, name='local_plane_pred'))
            (self.feed('segmentation_dropout')
                 .conv(1, 1, 16*16, 1, 1, relu=False, name='local_mask_pred'))
            pass
            
