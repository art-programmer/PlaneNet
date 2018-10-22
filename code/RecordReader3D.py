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
NUM_PLANES = 20
NUM_THREADS = 4



class RecordReader3D():
    def __init__(self):
        return


    def getBatch(self, filename_queue, numOutputPlanes = 20, batchSize = 16, min_after_dequeue = 1000, random=True, getLocal=False, getSegmentation=False, test=True):
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
                'num_planes': tf.FixedLenFeature([], tf.int64),
                'plane': tf.FixedLenFeature([NUM_PLANES * 3], tf.float32),
                'plane_relation': tf.FixedLenFeature([NUM_PLANES * NUM_PLANES], tf.float32),
                'segmentation_raw': tf.FixedLenFeature([], tf.string),
                'depth': tf.FixedLenFeature([HEIGHT * WIDTH], tf.float32),
                'smooth_boundary_raw': tf.FixedLenFeature([], tf.string),
                'info': tf.FixedLenFeature([3 + 4*4], tf.float32),                
            })

        # Convert from a scalar string tensor (whose single string has
        # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
        # [mnist.IMAGE_PIXELS].
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
        image = tf.reshape(image, [HEIGHT, WIDTH, 3])
        
        depth = features['depth']
        depth = tf.reshape(depth, [HEIGHT, WIDTH, 1])

        numPlanes = tf.minimum(tf.cast(features['num_planes'], tf.int32), numOutputPlanes)
        
        planes = features['plane']
        planes = tf.reshape(planes, [NUM_PLANES, 3])
        
        # planes = tf.slice(planes, [0, 0], [numPlanes, 3])
        # #shuffle_inds = tf.one_hot(tf.random_shuffle(tf.range(numPlanes)), depth = numPlanes)
        # shuffle_inds = tf.one_hot(tf.range(numPlanes), depth = numPlanes)
        
        # planes = tf.transpose(tf.matmul(tf.transpose(planes), shuffle_inds))
        # planes = tf.reshape(planes, [numPlanes, 3])
        # planes = tf.concat([planes, tf.zeros([numOutputPlanes - numPlanes, 3])], axis=0)
        # planes = tf.reshape(planes, [numOutputPlanes, 3])

        
        smooth_boundary = tf.decode_raw(features['smooth_boundary_raw'], tf.uint8)
        smooth_boundary = tf.cast(smooth_boundary, tf.float32)
        smooth_boundary = tf.reshape(smooth_boundary, [HEIGHT, WIDTH, 1])

        segmentation = tf.decode_raw(features['segmentation_raw'], tf.uint8)
        segmentation = tf.cast(tf.reshape(segmentation, [HEIGHT, WIDTH, 1]), tf.int32)
        #segmentation = segmentation + tf.cast(tf.equal(segmentation, 19), tf.int32)
        plane_masks = segmentation

        # coef = tf.range(numPlanes)
        # coef = tf.reshape(tf.matmul(tf.reshape(coef, [-1, numPlanes]), tf.cast(shuffle_inds, tf.int32)), [1, 1, numPlanes])
        
        # plane_masks = tf.cast(tf.equal(segmentation, tf.cast(coef, tf.uint8)), tf.float32)
        # plane_masks = tf.concat([plane_masks, tf.zeros([HEIGHT, WIDTH, numOutputPlanes - numPlanes])], axis=2)
        # plane_masks = tf.reshape(plane_masks, [HEIGHT, WIDTH, numOutputPlanes])

        
        #non_plane_mask = tf.cast(tf.equal(segmentation, tf.cast(numOutputPlanes, tf.uint8)), tf.float32)
        #non_plane_mask = 1 - tf.reduce_max(plane_masks, axis=2, keep_dims=True)
        
        #tf.cast(tf.equal(segmentation, tf.cast(numOutputPlanes, tf.uint8)), tf.float32)
        
        if random:
            image_inp, plane_inp, depth_gt, plane_masks_gt, smooth_boundary_gt, num_planes_gt, image_path, info = tf.train.shuffle_batch([image, planes, depth, plane_masks, smooth_boundary, numPlanes, features['image_path'], features['info']], batch_size=batchSize, capacity=min_after_dequeue + (NUM_THREADS + 2) * batchSize, num_threads=NUM_THREADS, min_after_dequeue=min_after_dequeue)
        else:
            image_inp, plane_inp, depth_gt, plane_masks_gt, smooth_boundary_gt, num_planes_gt, image_path, info = tf.train.batch([image, planes, depth, plane_masks, smooth_boundary, numPlanes, features['image_path'], features['info']], batch_size=batchSize, capacity=(NUM_THREADS + 2) * batchSize, num_threads=1)
            pass
        global_gt_dict = {'plane': plane_inp, 'depth': depth_gt, 'segmentation': plane_masks_gt, 'smooth_boundary': smooth_boundary_gt, 'num_planes': num_planes_gt, 'image_path': image_path, 'info': info}
        return image_inp, global_gt_dict, {}
