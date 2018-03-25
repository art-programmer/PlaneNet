import tensorflow as tf
import numpy as np
import threading
import PIL.Image as Image
from functools import partial
from multiprocessing import Pool
import cv2

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules import *


HEIGHT=192
WIDTH=256
NUM_OBJECTS = 10
NUM_THREADS = 4


class RecordReader():
    def __init__(self):
        return

    def getBatch(self, filename_queue, numOutputObjects = 20, batchSize = 16, min_after_dequeue = 1000, random=True, getLocal=False, getSegmentation=False, test=True):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                #'height': tf.FixedLenFeature([], tf.int64),
                #'width': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string),
                'image_path': tf.FixedLenFeature([], tf.string),
                'depth': tf.FixedLenFeature([HEIGHT * WIDTH], tf.float32),
                'info': tf.FixedLenFeature([4 * 4 + 4], tf.float32),
                'objects': tf.FixedLenFeature([NUM_OBJECTS * 13], tf.float32),
            })

        # Convert from a scalar string tensor (whose single string has
        # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
        # [mnist.IMAGE_PIXELS].
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
        image = tf.reshape(image, [HEIGHT, WIDTH, 3])


        depth = features['depth']
        depth = tf.reshape(depth, [HEIGHT, WIDTH, 1])

        # normal = features['normal']
        # normal = tf.reshape(normal, [HEIGHT, WIDTH, 3])
        # normal = tf.nn.l2_normalize(normal, dim=2)

        objects = tf.reshape(features['objects'], [NUM_OBJECTS, 13])
        numObjects = tf.reduce_sum(tf.cast(tf.greater(objects[:, 12], 0), dtype=tf.int32))

        #numPlanes = tf.maximum(numPlanes, 1)
        #planes = tf.slice(planes, [0, 0], [numPlanes, 3])

        if False:
            #shuffle_inds = tf.one_hot(tf.random_shuffle(tf.range(numPlanes)), depth = numPlanes)
            shuffle_inds = tf.one_hot(tf.range(numPlanes), numPlanes)

            planes = tf.transpose(tf.matmul(tf.transpose(planes), shuffle_inds))
            planes = tf.reshape(planes, [numPlanes, 3])
            planes = tf.concat([planes, tf.zeros([numOutputObjects - numPlanes, 3])], axis=0)
            planes = tf.reshape(planes, [numOutputObjects, 3])
            pass




        if random:
            image_inp, object_inp, depth_gt, num_objects_gt, image_path, info = tf.train.shuffle_batch([image, objects, depth, numObjects, features['image_path'], features['info']], batch_size=batchSize, capacity=min_after_dequeue + (NUM_THREADS + 2) * batchSize, num_threads=NUM_THREADS, min_after_dequeue=min_after_dequeue)
        else:
            image_inp, object_inp, depth_gt, num_objects_gt, image_path, info = tf.train.batch([image, objects, depth, numObjects, features['image_path'], features['info']], batch_size=batchSize, capacity=min_after_dequeue + (NUM_THREADS + 2) * batchSize, num_threads=1)
            pass
        global_gt_dict = {'object': object_inp, 'depth': depth_gt, 'num_objects': num_objects_gt, 'image_path': image_path, 'info': info}
        return image_inp, global_gt_dict, {}
