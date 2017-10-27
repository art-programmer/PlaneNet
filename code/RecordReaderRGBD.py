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



class RecordReaderRGBD():
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
                'image_path': tf.FixedLenFeature([], tf.string),                
                'image_raw': tf.FixedLenFeature([], tf.string),
                'depth': tf.FixedLenFeature([HEIGHT * WIDTH], tf.float32),
                #'normal': tf.FixedLenFeature([HEIGHT * WIDTH * 3], tf.float32),
                'normal': tf.FixedLenFeature([427 * 561 * 3], tf.float32),                
                'plane': tf.FixedLenFeature([NUM_PLANES * 3], tf.float32),
                'num_planes': tf.FixedLenFeature([], tf.int64),                
                #'plane_relation': tf.FixedLenFeature([NUM_PLANES * NUM_PLANES], tf.float32),
                'segmentation_raw': tf.FixedLenFeature([], tf.string),
                #'smooth_boundary_raw': tf.FixedLenFeature([], tf.string),
                #'info': tf.FixedLenFeature([3 + 4*4], tf.float32),                
            })

        # Convert from a scalar string tensor (whose single string has
        # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
        # [mnist.IMAGE_PIXELS].
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
        image = tf.reshape(image, [HEIGHT, WIDTH, 3])
        
        depth = features['depth']
        depth = tf.reshape(depth, [HEIGHT, WIDTH, 1])

        normal = features['normal']
        #normal = tf.reshape(normal, [HEIGHT, WIDTH, 3])
        normal = tf.reshape(normal, [427, 561, 3])
        normal = tf.image.resize_images(normal, [HEIGHT, WIDTH])
        
        numPlanes = tf.minimum(tf.cast(features['num_planes'], tf.int32), numOutputPlanes)
        
        planes = features['plane']
        planes = tf.reshape(planes, [NUM_PLANES, 3])

        segmentation = tf.decode_raw(features['segmentation_raw'], tf.int32)
        segmentation = tf.cast(tf.reshape(segmentation, [HEIGHT, WIDTH, 1]), tf.int32)


        info = np.zeros(20)
        info[0] = 5.1885790117450188e+02
        info[2] = 3.2558244941119034e+02 - 40
        info[5] = 5.1946961112127485e+02
        info[6] = 2.5373616633400465e+02 - 44
        info[10] = 1
        info[15] = 1
        info[16] = 561
        info[17] = 427
        info[18] = 1000
        info[19] = 1
        
        if random:
            image_inp, plane_inp, depth_gt, normal_gt, segmentation_gt, num_planes_gt, image_path, info_gt = tf.train.shuffle_batch([image, planes, depth, normal, segmentation, numPlanes, features['image_path'], info], batch_size=batchSize, capacity=min_after_dequeue + (NUM_THREADS + 2) * batchSize, num_threads=NUM_THREADS, min_after_dequeue=min_after_dequeue)
        else:
            image_inp, plane_inp, depth_gt, normal_gt, segmentation_gt, num_planes_gt, image_path, info_gt = tf.train.batch([image, planes, depth, normal, segmentation, numPlanes, features['image_path'], info], batch_size=batchSize, capacity = (NUM_THREADS + 2) * batchSize, num_threads=1)
            pass
        
        global_gt_dict = {'plane': plane_inp, 'depth': depth_gt, 'normal': normal_gt, 'segmentation': segmentation_gt, 'num_planes': num_planes_gt, 'image_path': image_path, 'info': info_gt}
        return image_inp, global_gt_dict, {}
