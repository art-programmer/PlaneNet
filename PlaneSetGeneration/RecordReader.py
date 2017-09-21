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


FETCH_BATCH_SIZE=32
BATCH_SIZE=32
HEIGHT=192
WIDTH=256
NUM_PLANES = 50
NUM_THREADS = 4


class RecordReader():
    def __init__(self):
        return


    def getBatch(self, filename_queue, numOutputPlanes = 20, batchSize = BATCH_SIZE, min_after_dequeue = 1000, random=True, getLocal=False, getSegmentation=False, test=True, suffix='forward'):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                #'height': tf.FixedLenFeature([], tf.int64),
                #'width': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string),
                'num_planes': tf.FixedLenFeature([], tf.int64),
                'plane': tf.FixedLenFeature([NUM_PLANES * 3], tf.float32),
                'plane_mask': tf.FixedLenFeature([HEIGHT * WIDTH], tf.int64),
                #'validating': tf.FixedLenFeature([], tf.int64)
                'depth': tf.FixedLenFeature([HEIGHT * WIDTH], tf.float32),
                'normal': tf.FixedLenFeature([HEIGHT * WIDTH * 3], tf.float32),
                'boundary_raw': tf.FixedLenFeature([], tf.string),
                'grid_s': tf.FixedLenFeature([HEIGHT / 8  * WIDTH / 8 * 1], tf.float32),
                'grid_p': tf.FixedLenFeature([HEIGHT / 8  * WIDTH / 8 * 3], tf.float32),
                'grid_m_raw': tf.FixedLenFeature([], tf.string),
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


        # planeAreaThreshold = 12 * 16
        # inds, _, counts = tf.unique_with_counts(plane_masks)
        # counts = counts * tf.cast(tf.greater(inds, 0), tf.int32)
        # numPlanes = tf.minimum(tf.reduce_sum(tf.cast(counts > planeAreaThreshold, tf.int32)), numOutputPlanes)


        if '_32' not in suffix and False:
            numPlanes = tf.minimum(tf.cast(features['num_planes'], tf.int32), numOutputPlanes)
            planes = features['plane']
            planes = tf.reshape(planes, [NUM_PLANES, 3])
            planes = tf.slice(planes, [0, 0], [numPlanes, 3])

            plane_masks = tf.cast(features['plane_mask'], tf.int64)  
            plane_masks = tf.reshape(plane_masks, [HEIGHT, WIDTH, 1])
            plane_mask_array = tf.tile(plane_masks, [1, 1, NUM_PLANES])
            coef = tf.range(tf.cast(NUM_PLANES, tf.int64), dtype=tf.int64)
            coef = tf.pow(tf.constant(2, tf.int64), coef)
            planeMasks = tf.reshape(tf.cast(tf.div(plane_mask_array, coef) % 2, tf.float32), [HEIGHT, WIDTH, NUM_PLANES])

            #planeMasks = tf.zeros([HEIGHT, WIDTH, numOutputPlanes])
        else:
            numPlanes = 30
            planes = features['plane']
            planes = tf.reshape(planes, [NUM_PLANES, 3])
            planes = tf.slice(planes, [0, 0], [numPlanes, 3])
        
            plane_masks = tf.cast(features['plane_mask'], tf.int64)  
        
            plane_masks = tf.reshape(plane_masks, [HEIGHT, WIDTH, 1])
            plane_mask_array = tf.tile(plane_masks, [1, 1, numPlanes])
            coef = tf.range(numPlanes, dtype=tf.int64)
            coef = tf.pow(tf.constant(2, tf.int64), coef)
            #coef = tf.reshape(tf.matmul(tf.reshape(coef, [-1, numPlanes]), tf.cast(shuffle_inds, tf.int32)), [numPlanes])
            #coef = tf.cast(coef, tf.int64)
            planeMasks = tf.cast(tf.div(plane_mask_array, coef) % 2, tf.float32)
        
            urange = tf.reshape(tf.range(WIDTH, dtype=tf.float32), [-1, 1])
            planeXs = tf.reduce_max(planeMasks, axis=0)
            planeMinX = WIDTH - tf.reduce_max(planeXs * (float(WIDTH) - urange), axis=0)
            planeMaxX = tf.reduce_max(planeXs * urange, axis=0)

            vrange = tf.reshape(tf.range(HEIGHT, dtype=tf.float32), [-1, 1])
            planeYs = tf.reduce_max(planeMasks, axis=1)
            planeMinY = HEIGHT - tf.reduce_max(planeYs * (float(HEIGHT) - vrange), axis=0)
            planeMaxY = tf.reduce_max(planeYs * vrange, axis=0)

            planeMaxX = tf.maximum(planeMinX, planeMaxX)
            planeMaxY = tf.maximum(planeMinY, planeMaxY)

            planeAreas = tf.reduce_sum(planeMasks, axis=[0, 1])
        
            localPlaneWidthThreshold = 32
            localPlaneHeightThreshold = 32
            globalPlaneAreaThreshold = 16 * 16
            globalPlaneWidthThreshold = 8


            globalPlaneMask = tf.logical_or(tf.greater(planeMaxX - planeMinX, localPlaneWidthThreshold), tf.greater(planeMaxY - planeMinY, localPlaneHeightThreshold))
            globalPlaneMask = tf.logical_and(globalPlaneMask, tf.greater((planeMaxX - planeMinX) * (planeMaxY - planeMinY), globalPlaneAreaThreshold))
            globalPlaneMask = tf.logical_and(globalPlaneMask, tf.greater(planeAreas / tf.sqrt(tf.pow(planeMaxX + 1 - planeMinX, 2) + tf.pow(planeMaxY + 1 - planeMinY, 2)), globalPlaneWidthThreshold))
            #globalPlaneMask = tf.logical_or(globalPlaneMask, tf.less(tf.range(numPlanes), tf.cast(features['num_planes'], tf.int32)))
            #globalPlaneMask = tf.cast(tf.squeeze(globalPlaneMask, axis=[2]), tf.float32)
            globalPlaneMask = tf.cast(globalPlaneMask, tf.float32)

            weightedPlaneAreas = globalPlaneMask * (planeAreas + HEIGHT * WIDTH) + (1 - globalPlaneMask) * planeAreas

            #test = tf.reshape(tf.stack([globalPlaneMask, planeAreas, weightedPlaneAreas, planeMinX, planeMaxX, planeMinY, planeMaxY], axis=0), [7, numPlanes])
            
            planeAreas, sortInds = tf.nn.top_k(weightedPlaneAreas, k=numPlanes)
            sortMap = tf.one_hot(sortInds, depth=numPlanes, axis=0)

            planeMasks = tf.reshape(tf.matmul(tf.reshape(planeMasks, [HEIGHT * WIDTH, numPlanes]), sortMap), [HEIGHT, WIDTH, numPlanes])
            planes = tf.transpose(tf.matmul(planes, sortMap, transpose_a=True), [1, 0])
        
            numPlanes = tf.minimum(tf.cast(tf.round(tf.reduce_sum(globalPlaneMask)), tf.int32), numOutputPlanes)
            
            planes = tf.slice(planes, [0, 0], [numPlanes, 3])
            planeMasks = tf.slice(planeMasks, [0, 0, 0], [HEIGHT, WIDTH, numPlanes])
            planeMasks = tf.reshape(tf.concat([planeMasks, tf.zeros([HEIGHT, WIDTH, numOutputPlanes - numPlanes])], axis=2), [HEIGHT, WIDTH, numOutputPlanes])
            pass

        # planeMasks_expanded = tf.expand_dims(planeMasks, 0)
        # boundary = tf.reduce_max(tf.nn.max_pool(planeMasks_expanded, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='max_pool') - planeMasks_expanded, axis=3, keep_dims=True)
        # max_depth_diff = 0.1
        # depth_expanded = tf.expand_dims(depth, 0)
        # kernel_size = 5
        # padding = (kernel_size - 1) / 2
        # neighbor_kernel_array = gaussian(kernel_size, kernel_size)
        # neighbor_kernel_array[(kernel_size - 1) / 2][(kernel_size - 1) / 2] = 0
        # neighbor_kernel_array /= neighbor_kernel_array.sum()
        # neighbor_kernel_array *= -1
        # neighbor_kernel_array[(kernel_size - 1) / 2][(kernel_size - 1) / 2] = 1
        # neighbor_kernel = tf.constant(neighbor_kernel_array.reshape(-1), shape=neighbor_kernel_array.shape, dtype=tf.float32)
        # neighbor_kernel = tf.reshape(neighbor_kernel, [kernel_size, kernel_size, 1, 1])
        
        # depth_diff = tf.abs(tf.nn.depthwise_conv2d(depth_expanded, neighbor_kernel, strides=[1, 1, 1, 1], padding='VALID'))
        # depth_diff = tf.pad(depth_diff, paddings = [[0, 0], [padding, padding], [padding, padding], [0, 0]])
        # smooth_boundary = boundary * tf.cast(tf.less(depth_diff, max_depth_diff), tf.float32)
        # occlusion_boundary = boundary - smooth_boundary
        # boundary = tf.squeeze(tf.concat([smooth_boundary, occlusion_boundary], axis=3), axis=0)

        
        #validating = tf.cast(features['validating'], tf.float32)        
        shuffle_inds = tf.one_hot(tf.random_shuffle(tf.range(numPlanes)), depth = numPlanes)
        #shuffle_inds = tf.one_hot(tf.range(numPlanes), depth = numPlanes)
        
        #shuffle_inds = tf.concat([shuffle_inds, tf.zeros((numPlanes, numOutputPlanes - numPlanes))], axis=1)
        #shuffle_inds = tf.concat([shuffle_inds, tf.concat([tf.zeros((numOutputPlanes - numPlanes, numPlanes)), tf.diag(tf.ones([numOutputPlanes - numPlanes]))], axis=1)], axis=0)
        planes = tf.transpose(tf.matmul(tf.transpose(planes), shuffle_inds))
        planes = tf.reshape(planes, [numPlanes, 3])
        planes = tf.concat([planes, tf.zeros([numOutputPlanes - numPlanes, 3])], axis=0)
        planes = tf.reshape(planes, [numOutputPlanes, 3])

        
        boundary = tf.decode_raw(features['boundary_raw'], tf.uint8)
        boundary = tf.cast(boundary > 128, tf.float32)
        boundary = tf.reshape(boundary, [HEIGHT, WIDTH, 2])
        #boundary = tf.slice(tf.reshape(boundary, [HEIGHT, WIDTH, 3]), [0, 0, 0], [HEIGHT, WIDTH, 2])

        grid_s = tf.reshape(features['grid_s'], [HEIGHT / 8, WIDTH / 8, 1])       
        grid_p = tf.reshape(features['grid_p'], [HEIGHT / 8, WIDTH / 8, 3])
        
        grid_m = tf.decode_raw(features['grid_m_raw'], tf.uint8)
        grid_m = tf.cast(tf.reshape(grid_m, [HEIGHT / 8, WIDTH / 8, 16 * 16]), tf.float32)

        if getLocal:
            if random:
                image_inp, plane_inp, depth_gt, normal_gt, plane_mask_inp, boundary_gt, grid_s_gt, grid_p_gt, grid_m_gt, num_planes_gt = tf.train.shuffle_batch([image, planes, depth, normal, planeMasks, boundary, grid_s, grid_p, grid_m, numPlanes], batch_size=batchSize, capacity=min_after_dequeue + (NUM_THREADS + 2) * batchSize, num_threads=NUM_THREADS, min_after_dequeue=min_after_dequeue)
            else:
                image_inp, plane_inp, depth_gt, normal_gt, plane_mask_inp, boundary_gt, grid_s_gt, grid_p_gt, grid_m_gt, num_planes_gt = tf.train.batch([image, planes, depth, normal, planeMasks, boundary, grid_s, grid_p, grid_m, numPlanes], batch_size=batchSize, capacity=(NUM_THREADS + 2) * batchSize, num_threads=1)
                pass
            return image_inp, plane_inp, depth_gt, normal_gt, plane_mask_inp, boundary_gt, grid_s_gt, grid_p_gt, grid_m_gt, num_planes_gt

        
        if not getSegmentation:
            if random:
                image_inp, plane_inp, depth_gt, normal_gt, plane_mask_inp = tf.train.shuffle_batch([image, planes, depth, normal, tf.zeros([HEIGHT, WIDTH, numOutputPlanes])], batch_size=batchSize, capacity=min_after_dequeue + (NUM_THREADS + 2) * batchSize, num_threads=NUM_THREADS, min_after_dequeue=min_after_dequeue)
            else:
                image_inp, plane_inp, depth_gt, normal_gt, plane_mask_inp = tf.train.batch([image, planes, depth, normal, tf.zeros([HEIGHT, WIDTH, numOutputPlanes])], batch_size=batchSize, capacity=(NUM_THREADS + 2) * batchSize, num_threads=NUM_THREADS)
                pass
            return image_inp, plane_inp, depth_gt, normal_gt, plane_mask_inp


        plane_masks = tf.cast(features['plane_mask'], tf.int64)  
        
        plane_masks = tf.reshape(plane_masks, [HEIGHT, WIDTH, 1])
        plane_mask_array = tf.tile(plane_masks, [1, 1, numPlanes])
        coef = tf.range(numPlanes, dtype=tf.int64)
        coef = tf.pow(2, coef)
        coef = tf.reshape(tf.matmul(tf.reshape(coef, [-1, numPlanes]), tf.cast(shuffle_inds, tf.int64)), [numPlanes])
        coef = tf.cast(coef, tf.int64)
        plane_mask_array = tf.cast(tf.div(plane_mask_array, coef) % 2, tf.float32)
        plane_mask_array = tf.concat([plane_mask_array, tf.zeros([HEIGHT, WIDTH, numOutputPlanes - numPlanes])], axis=2)
        plane_mask_array = tf.reshape(plane_mask_array, [HEIGHT, WIDTH, numOutputPlanes])
        #num_planes_array = tf.concat([tf.ones([numPlanes], dtype=np.float32) / tf.cast(numPlanes * BATCH_SIZE, np.float32), tf.zeros([numOutputPlanes - numPlanes])], axis=0)

        #image_inp, plane_inp, depth_gt, normal_gt, plane_mask_inp, num_planes, mask = tf.train.shuffle_batch([image, planes, depth, normal, plane_mask_array, numPlanes, plane_masks_test], batch_size=batchSize, capacity=min_after_dequeue + (NUM_THREADS + 2) * batchSize, num_threads=NUM_THREADS, min_after_dequeue=min_after_dequeue)

        # if True:
        #     image_inp, plane_inp, depth_gt, normal_gt, plane_mask_inp = tf.train.batch([image, planes, tf.ones((HEIGHT, WIDTH, 1)), tf.ones((HEIGHT, WIDTH, 3)), plane_mask_array], batch_size=batchSize, capacity=(NUM_THREADS + 2) * batchSize, num_threads=NUM_THREADS)
        #     return image_inp, plane_inp, depth_gt, normal_gt, plane_mask_inp

        if random:
            image_inp, plane_inp, depth_gt, normal_gt, plane_mask_inp, boundary_gt, grid_s_gt, grid_p_gt, grid_m_gt, num_planes_gt = tf.train.shuffle_batch([image, planes, depth, normal, plane_mask_array, boundary, grid_s, grid_p, grid_m, numPlanes], batch_size=batchSize, capacity=min_after_dequeue + (NUM_THREADS + 2) * batchSize, num_threads=NUM_THREADS, min_after_dequeue=min_after_dequeue)            
            image_inp, plane_inp, depth_gt, normal_gt, plane_mask_gt, boundary_gt, num_planes_gt = tf.train.shuffle_batch([image, planes, depth, normal, plane_mask_array, boundary, numPlanes], batch_size=batchSize, capacity=min_after_dequeue + (NUM_THREADS + 2) * batchSize, num_threads=NUM_THREADS, min_after_dequeue=min_after_dequeue)
        else:
            image_inp, plane_inp, depth_gt, normal_gt, plane_mask_inp, boundary_gt, grid_s_gt, grid_p_gt, grid_m_gt, num_planes_gt = tf.train.batch([image, planes, depth, normal, plane_mask_array, boundary, grid_s, grid_p, grid_m, numPlanes], batch_size=batchSize, capacity=(NUM_THREADS + 2) * batchSize, num_threads=1)            
            pass
        return image_inp, plane_inp, depth_gt, normal_gt, plane_mask_inp, boundary_gt, grid_s_gt, grid_p_gt, grid_m_gt, num_planes_gt
