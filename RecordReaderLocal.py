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

    def getBatch(self, filename_queue, options, min_after_dequeue = 1000, random=True):
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

        outputWidth = WIDTH / options.outputStride
        outputHeight = HEIGHT / options.outputStride

        if options.axisAligned or options.useAnchor:
            centers = objects[:, :3]
            sizes = objects[:, 3:6]
            towards = objects[:, 6:9]
            up = objects[:, 9:12]
            right = tf.cross(towards, up)
            right /= tf.maximum(tf.norm(right, axis=-1, keep_dims=True), 1e-4)
            directionsArray = np.array([[1, 1, 1],
                                        [1, 1, -1],
                                        [1, -1, 1],
                                        [1, -1, -1],
                                        [-1, 1, 1],
                                        [-1, 1, -1],
                                        [-1, -1, 1],
                                        [-1, -1, -1]])
            directions = tf.expand_dims(tf.constant(directionsArray.reshape(-1), shape=directionsArray.shape), 0)
            cornerPoints = tf.expand_dims(centers, 1) + tf.matmul(tf.expand_dims(sizes, 1) / 2 * directionsArray, tf.stack([towards, up, right], axis=1))
            maxs = tf.reduce_max(cornerPoints, axis=1)
            mins = tf.reduce_min(cornerPoints, axis=1)
            sizes = maxs - mins
            #centers = (mins + maxs) / 2
            objects = tf.concat([centers, sizes, objects[:, 6:]], axis=1)
            pass


        numObjects = tf.reduce_sum(tf.cast(tf.greater(objects[:, 12], 0), dtype=tf.int32))
        info = features['info']

        objects = objects[:numObjects]

        if options.useAnchor:
            centers = objects[:, :3] - objects[:, 3:6] / 2 * tf.expand_dims(tf.constant([1, 0, 0], dtype=tf.float32), 0)
        else:
            centers = objects[:, :3]
            pass

        U = (centers[:, 2] / centers[:, 0] * info[0] + info[2]) / info[16]
        V = tf.clip_by_value((-centers[:, 1] / centers[:, 0] * info[5] + info[6]) / info[17], 0, 1)

        if True:
            validMask = tf.logical_and(tf.logical_and(tf.logical_and(tf.greater_equal(U, 0), tf.less_equal(U, 1)), tf.logical_and(tf.greater_equal(V, 0), tf.less_equal(V, 1))), tf.greater(centers[:, 0], 0))
            objects = tf.boolean_mask(objects, validMask)
            U = tf.boolean_mask(U, validMask)
            V = tf.boolean_mask(V, validMask)
            numObjects = tf.reduce_sum(tf.cast(validMask, tf.int32))
            pass

        U = tf.clip_by_value(U, 0, 1)
        V = tf.clip_by_value(V, 0, 1)

        gridU = tf.clip_by_value(tf.cast((U * outputWidth), tf.int32), 0, outputWidth - 1)
        gridV = tf.clip_by_value(tf.cast((V * outputHeight), tf.int32), 0, outputHeight - 1)
        indices = gridV * outputWidth + gridU
        local_scores = tf.reshape(tf.maximum(tf.unsorted_segment_max(tf.ones([numObjects]), indices, num_segments=outputWidth * outputHeight), 0), (outputHeight, outputWidth, 1))
        class_prob = tf.one_hot(tf.cast(objects[:, 12], tf.int32), depth=40, axis=-1)
        local_classes = tf.reshape(tf.maximum(tf.unsorted_segment_max(class_prob, indices, num_segments=outputWidth * outputHeight), 0), (outputHeight, outputWidth, -1))

        if options.useAnchor:
            mins = objects[:, :3] - objects[:, 3:6] / 2 * tf.expand_dims(tf.constant([1, -1, 1], dtype=tf.float32), 0)
            maxs = objects[:, :3] - objects[:, 3:6] / 2 * tf.expand_dims(tf.constant([1, 1, -1], dtype=tf.float32), 0)
            minU = (mins[:, 2] / mins[:, 0] * info[0] + info[2]) / info[16]
            minV = (-mins[:, 1] / mins[:, 0] * info[5] + info[6]) / info[17]
            maxU = (maxs[:, 2] / maxs[:, 0] * info[0] + info[2]) / info[16]
            maxV = (-maxs[:, 1] / maxs[:, 0] * info[5] + info[6]) / info[17]
            boxes = tf.stack([U, V, maxU - minU, maxV - minV], axis=-1)
            boxes = tf.reshape(tf.maximum(tf.unsorted_segment_max(boxes, indices, num_segments=outputWidth * outputHeight), 0), (outputHeight, outputWidth, 4))

            anchorW = tf.fill((outputHeight, outputWidth, 1), 1.0 / outputWidth)
            anchorH = tf.fill((outputHeight, outputWidth, 1), 1.0 / outputHeight)
            anchors = tf.stack([tf.tile(tf.expand_dims(tf.range(outputWidth, dtype=tf.float32), 0), (outputHeight, 1)) / outputWidth, tf.tile(tf.expand_dims(tf.range(outputHeight, dtype=tf.float32), 1), (1, outputWidth)) / outputHeight], axis=-1)
            anchors = tf.concat([anchors, anchorW, anchorH], axis=-1)

            #local_parameters = tf.reshape(tf.unsorted_segment_sum(objects[:, :12], indices, num_segments=outputWidth * outputHeight), (outputHeight, outputWidth, 12))
            depths = tf.clip_by_value(tf.reshape(tf.maximum(tf.unsorted_segment_max(tf.stack([objects[:, 0] - objects[:, 3] / 2, objects[:, 0] + objects[:, 3] / 2], axis=-1), indices, num_segments=outputWidth * outputHeight), 0), (outputHeight, outputWidth, 2)), 0, 10)
            local_parameters = tf.concat([(boxes[:, :, :2] - anchors[:, :, :2]) / anchors[:, :, 2:4], tf.minimum(boxes[:, :, 2:4] / anchors[:, :, 2:4], options.outputStride * 2), depths], axis=-1)
            local_parameters *= local_scores
            local_parameters = tf.concat([local_parameters, tf.zeros(local_parameters.shape)], axis=-1)
            #local_parameters = tf.reshape(tf.unsorted_segment_sum(objects[:, :12], indices, num_segments=outputWidth * outputHeight), (outputHeight, outputWidth, 12))
        else:
            local_parameters = tf.reshape(tf.unsorted_segment_sum(objects[:, :12], indices, num_segments=outputWidth * outputHeight), (outputHeight, outputWidth, 12))
            pass

        objects = tf.reshape(tf.concat([objects, tf.zeros((NUM_OBJECTS - numObjects, 13))], axis=0), (NUM_OBJECTS, 13))

        if random:
            image_inp, object_inp, depth_gt, num_objects_gt, image_path, info, local_scores_gt, local_parameters_gt, local_classes_gt = tf.train.shuffle_batch([image, objects, depth, numObjects, features['image_path'], features['info'], local_scores, local_parameters, local_classes], batch_size=options.batchSize, capacity=min_after_dequeue + (NUM_THREADS + 2) * options.batchSize, num_threads=NUM_THREADS, min_after_dequeue=min_after_dequeue)
        else:
            image_inp, object_inp, depth_gt, num_objects_gt, image_path, info, local_scores_gt, local_parameters_gt, local_classes_gt = tf.train.batch([image, objects, depth, numObjects, features['image_path'], features['info'], local_scores, local_parameters, local_classes], batch_size=options.batchSize, capacity=min_after_dequeue + (NUM_THREADS + 2) * options.batchSize, num_threads=1)
            pass
        global_gt_dict = {'object': object_inp, 'depth': depth_gt, 'num_objects': num_objects_gt}
        local_gt_dict = {'score': local_scores_gt, 'object': local_parameters_gt, 'image_path': image_path, 'info': info, 'num_objects': num_objects_gt, 'class': local_classes_gt}
        return image_inp, global_gt_dict, local_gt_dict
