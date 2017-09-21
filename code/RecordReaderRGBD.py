import tensorflow as tf
import numpy as np
import threading
import PIL.Image as Image
from functools import partial
from multiprocessing import Pool
import cv2

FETCH_BATCH_SIZE=32
BATCH_SIZE=32
HEIGHT=192
WIDTH=256
NUM_PLANES = 50
NUM_THREADS = 4


class RecordReaderRGBD():
    def __init__(self):
        return


    def getBatch(self, filename_queue, numOutputPlanes = 20, batchSize = BATCH_SIZE, min_after_dequeue = 1000, random=True, getLocal=False, getSegmentation=False):
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
            })

        # Convert from a scalar string tensor (whose single string has
        # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
        # [mnist.IMAGE_PIXELS].
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
        image = tf.reshape(image, [HEIGHT, WIDTH, 3])
        
        depth = features['depth']
        depth = tf.reshape(depth, [HEIGHT, WIDTH, 1])

        image_path = features['image_path']

        if random:
            image_inp, depth_gt, image_path_inp = tf.train.shuffle_batch([image, depth, image_path], batch_size=batchSize, capacity=min_after_dequeue + (NUM_THREADS + 2) * batchSize, num_threads=NUM_THREADS, min_after_dequeue=min_after_dequeue)
        else:
            image_inp, depth_gt, image_path_inp = tf.train.batch([image, depth, image_path], batch_size=batchSize, capacity=(NUM_THREADS + 2) * batchSize, num_threads=NUM_THREADS)

        global_gt_dict = {'depth': depth_gt, 'path': image_path_inp}
        local_gt_dict = {}
        return image_inp, global_gt_dict, local_gt_dict
