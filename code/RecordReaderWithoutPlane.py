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


def loadImagePaths():
    image_set_file = '../PythonScripts/SUNCG/image_list_500000.txt'
    with open(image_set_file) as f:
        filenames = [x.strip().replace('plane_global.npy', '') for x in f.readlines()]
        image_paths = [{'image': x + 'mlt.png', 'plane': x + 'plane_global.npy', 'normal': x + 'norm_camera.png', 'depth': x + 'depth.png', 'mask': x + 'valid.png', 'masks': x + 'masks.npy'} for x in filenames]
        pass
    return image_paths


class RecordReader():
    def __init__(self):
        self.imagePaths = loadImagePaths()
        self.imagePaths = self.imagePaths[:100000]
        self.numImages = len(self.imagePaths)
        self.trainingPercentage = 0.9
        self.numTrainingImages = int(self.numImages * self.trainingPercentage)
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
                'normal': tf.FixedLenFeature([HEIGHT * WIDTH * 3], tf.float32),
                'invalid_mask_raw': tf.FixedLenFeature([], tf.string),
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
        normal = tf.reshape(normal, [HEIGHT, WIDTH, 3])

        invalid_mask = tf.decode_raw(features['invalid_mask_raw'], tf.uint8)
        invalid_mask = tf.cast(invalid_mask > 128, tf.float32)
        invalid_mask = tf.reshape(invalid_mask, [HEIGHT, WIDTH, 1])
        
        image_path = features['image_path']
        
        image_inp, depth_gt, normal_gt, invalid_mask_gt, image_path_inp = tf.train.batch([image, depth, normal, invalid_mask, image_path], batch_size=batchSize, capacity=(NUM_THREADS + 2) * batchSize, num_threads=NUM_THREADS)
        return image_inp, depth_gt, normal_gt, invalid_mask_gt, image_path_inp
