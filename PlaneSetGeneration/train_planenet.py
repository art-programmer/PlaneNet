import sys
import tensorflow as tf
import numpy as np
import cv2
import os
import time
import sys
import tf_nndistance
import argparse
import glob
import PIL

#from SegmentationBatchFetcherV2 import *
from RecordReader import *


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
from layers import PlaneDepthLayer, PlaneNormalLayer
from modules import *

#from resnet import inference as resnet
#from resnet import fc as fc, conv as conv, block_transpose as block_transpose, conv_transpose as conv_transpose, bn as bn, UPDATE_OPS_COLLECTION
#from config import Config
from modules import *
import scipy.ndimage as ndimage
from planenet import PlaneNet
#from SegmentationRefinement import refineSegmentation

np.set_printoptions(precision=2, linewidth=200)

MOVING_AVERAGE_DECAY = 0.99
                          
deepSupervisionLayers=['res4b22_relu']
deepSupervisionLayers=[]

def build_graph(img_inp_train, img_inp_val, plane_gt_train, plane_gt_val, validating_inp, is_training=True, numOutputPlanes=20, gpu_id = 0, useCRF= 0, suffix='forward'):
    if suffix == '12_22':
        deepSupervisionLayers.append('res4b12_relu')
        pass
    
    with tf.device('/gpu:%d'%gpu_id):
        training_flag = tf.logical_not(validating_inp)
        #training_flag = tf.convert_to_tensor(True, dtype='bool', name='is_training')
        #training_flag = tf.convert_to_tensor(is_training, dtype='bool', name='is_training')
        
        img_inp = tf.cond(validating_inp, lambda: img_inp_val, lambda: img_inp_train)
        plane_gt = tf.cond(validating_inp, lambda: plane_gt_val, lambda: plane_gt_train)

        net = PlaneNet({'img_inp': img_inp}, is_training=training_flag, numGlobalPlanes=numOutputPlanes, deepSupervisionLayers=deepSupervisionLayers, networkType='planenet_' + suffix)
        segmentation_pred = net.layers['segmentation_pred']
        plane_pred = net.layers['plane_pred']
        boundary_pred = net.layers['boundary_pred']
        grid_s_pred = net.layers['s_8_pred']
        grid_p_pred = net.layers['p_8_pred']
        grid_m_pred = net.layers['m_8_pred']


        non_plane_mask_pred = net.layers['non_plane_mask_pred']
        non_plane_depth_pred = net.layers['non_plane_depth_pred']
        non_plane_normal_pred = net.layers['non_plane_normal_pred']

        if suffix == 'confidence' or suffix == 'deep':
            plane_confidence_pred = net.layers['plane_confidence_pred']
        else:
            plane_confidence_pred = tf.zeros((int(plane_pred.shape[0]), numOutputPlanes, 1))
            pass


        # dists_forward, map_forward, dists_backward, map_backward = tf_nndistance.nn_distance(plane_gt, plane_pred)
        # plane_map = tf.one_hot(map_forward, depth=numOutputPlanes, axis=-1)
        # plane_pred = tf.transpose(tf.matmul(plane_gt, plane_map, transpose_a=True), [0, 2, 1])
        #plane_pred = tf.tile(tf.slice(plane_gt, [0, 11, 0], [int(plane_gt.shape[0]), 1, 3]), [1, numOutputPlanes, 1])
        
        if not is_training and False:
            with tf.variable_scope('depth'):
                plane_parameters = tf.reshape(plane_pred, (-1, 3))
                plane_depths = planeDepthsModule(plane_parameters, WIDTH, HEIGHT)
                plane_depths = tf.transpose(tf.reshape(plane_depths, [HEIGHT, WIDTH, -1, numOutputPlanes]), [2, 0, 1, 3])

                segmentation = segmentation_pred
                # if useCRF > 0:
                #     segmentation = tf.nn.softmax(segmentation)
                #     with tf.variable_scope('crf'):
                #         segmentation = segmentationRefinementModule(segmentation, plane_depths, numIterations=useCRF)
                #         pass
                #     pass
                # else:
                #     #segmentation = tf.one_hot(tf.argmax(segmentation, 3), depth=numOutputPlanes)
                #     segmentation = tf.nn.softmax(segmentation)
                #     pass
                segmentation = tf.nn.softmax(segmentation)
                
                #segmentation = segmentationRefinementModuleBoundary(segmentation, plane_depths, numIterations=1)
                segmentation = tf.cond(training_flag, lambda: segmentation, lambda: tf.one_hot(tf.argmax(segmentation, 3), depth=numOutputPlanes))

                
                #plane_depths = tf.concat([plane_depths, non_plane_depth], 3)
                #segmentation = tf.concat([segmentation, tf.ones(non_plane_depth.shape) * 0.5], 3)            
                depth_pred = tf.reduce_sum(tf.multiply(plane_depths, segmentation), axis=3, keep_dims=True)
                pass

            with tf.variable_scope('normal'):
                plane_normals = planeNormalsModule(plane_parameters, WIDTH, HEIGHT)
                plane_normals = tf.reshape(plane_normals, [-1, 1, 1, numOutputPlanes, 3])
                normal_pred = tf.reduce_sum(tf.multiply(plane_normals, tf.expand_dims(segmentation, -1)), axis=3)
                pass
            pass
        else:
            depth_pred = tf.zeros((plane_pred.shape[0], HEIGHT, WIDTH, 1))
            normal_pred = tf.zeros((plane_pred.shape[0], HEIGHT, WIDTH, 1))
            segmentation = tf.zeros((plane_pred.shape[0], HEIGHT, WIDTH, numOutputPlanes))
            pass

    plane_preds = []
    segmentation_preds = []
    for layer in deepSupervisionLayers:
        plane_preds.append(net.layers[layer+'_plane_pred'])
        segmentation_preds.append(net.layers[layer+'_segmentation_pred'])
        continue
      
    return plane_pred, plane_confidence_pred, depth_pred, normal_pred, segmentation_pred, non_plane_mask_pred, non_plane_depth_pred, non_plane_normal_pred, boundary_pred, grid_s_pred, grid_p_pred, grid_m_pred, plane_preds, segmentation_preds, segmentation

def build_loss(plane_pred, plane_confidence_pred, depth_pred, normal_pred, segmentation_pred, non_plane_mask_pred, non_plane_depth_pred, non_plane_normal_pred, boundary_pred, grid_s_pred, grid_p_pred, grid_m_pred, plane_preds, segmentation_preds, plane_gt_train, depth_gt_train, normal_gt_train, segmentation_gt_train, boundary_gt_train, grid_s_gt_train, grid_p_gt_train, grid_m_gt_train, num_planes_gt_train, plane_gt_val, depth_gt_val, normal_gt_val, segmentation_gt_val, boundary_gt_val, grid_s_gt_val, grid_p_gt_val, grid_m_gt_val, num_planes_gt_val, validating_inp, numOutputPlanes = 20, gpu_id = 0, useCRF= 0, suffix='forward'):

    with tf.device('/gpu:%d'%gpu_id):    
        plane_gt = tf.cond(validating_inp, lambda: plane_gt_val, lambda: plane_gt_train)
        depth_gt = tf.cond(validating_inp, lambda: depth_gt_val, lambda: depth_gt_train)
        normal_gt = tf.cond(validating_inp, lambda: normal_gt_val, lambda: normal_gt_train)
        grid_s_gt = tf.cond(validating_inp, lambda: grid_s_gt_val, lambda: grid_s_gt_train)
        grid_p_gt = tf.cond(validating_inp, lambda: grid_p_gt_val, lambda: grid_p_gt_train)
        grid_m_gt = tf.cond(validating_inp, lambda: grid_m_gt_val, lambda: grid_m_gt_train)
        num_planes_gt = tf.cond(validating_inp, lambda: num_planes_gt_val, lambda: num_planes_gt_train)

        #if 'confidence' in suffix:
        #distr = tf.contrib.distributions.Bernoulli(logits = tf.ones(plane_confidence_pred.shape) * 100)
        #plane_pred *= tf.cast(tf.stop_gradient(tf.contrib.distributions.Bernoulli(probs = tf.sigmoid(plane_confidence_pred)).sample()), tf.float32)
        #sparsity_loss = tf.reduce_mean(1 - tf.sigmoid(plane_confidence_pred)) * 100
        #pass

        #segmentation_gt = tf.cond(validating_inp, lambda: segmentation_gt_val, lambda: segmentation_gt_train)
        normalDotThreshold = np.cos(np.deg2rad(5))
        distanceThreshold = 0.05

        focalLength = 517.97
        urange = (tf.range(WIDTH, dtype=tf.float32) / (WIDTH + 1) - 0.5) / focalLength * 641
        urange = tf.tile(tf.reshape(urange, [1, -1]), [HEIGHT, 1])
        vrange = (tf.range(HEIGHT, dtype=tf.float32) / (HEIGHT + 1) - 0.5) / focalLength * 481
        vrange = tf.tile(tf.reshape(vrange, [-1, 1]), [1, WIDTH])
        
        X = depth_gt * tf.expand_dims(urange, -1)
        Y = depth_gt
        Z = -depth_gt * tf.expand_dims(vrange, -1)
        XYZ = tf.concat([X, Y, Z], axis=3)
        XYZ = tf.reshape(XYZ, [-1, HEIGHT * WIDTH, 3])
        #ranges = tf.stack([urange, np.ones([height, width]), -vrange], axis=2)
        #ranges = tf.reshape(ranges, [-1, 3])
        #plane_parameters = tf.reshape(plane_gt, [-1, 3])
        plane_parameters = plane_gt
        planesD = tf.norm(plane_parameters, axis=2, keep_dims=True)
        planesD = tf.clip_by_value(planesD, 1e-5, 10)
        planesNormal = tf.div(tf.negative(plane_parameters), planesD)

        distance = tf.reshape(tf.abs(tf.matmul(XYZ, planesNormal, transpose_b=True) + tf.reshape(planesD, [-1, 1, numOutputPlanes])), [-1, HEIGHT, WIDTH, numOutputPlanes])
        angle = tf.reshape(np.abs(tf.matmul(tf.reshape(normal_gt, [-1, HEIGHT * WIDTH, 3]), planesNormal, transpose_b=True)), [-1, HEIGHT, WIDTH, numOutputPlanes])

        segmentation_gt = tf.cast(tf.logical_and(tf.greater(angle, normalDotThreshold), tf.less(distance, distanceThreshold)), tf.float32)        
        segmentation_gt = tf.nn.max_pool(segmentation_gt, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='max_pool')
        plane_mask = tf.reduce_max(segmentation_gt, axis=3, keep_dims=True)
        plane_mask = 1 - tf.nn.max_pool(1 - plane_mask, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='max_pool')
        segmentation_gt = tf.one_hot(tf.argmax(segmentation_gt * (distanceThreshold - distance), axis=3), depth=numOutputPlanes) * plane_mask

        validPlaneMask = tf.cast(tf.less(tf.tile(tf.expand_dims(tf.range(numOutputPlanes), 0), [int(plane_pred.shape[0]), 1]), tf.expand_dims(num_planes_gt, -1)), tf.float32)        

        
        if suffix == 'depth_sum':
            useBackward = 1
        else:
            useBackward = 0
            pass
        
        deep_supervision_loss = tf.constant(0.0)          
        if 'shallow' not in suffix:
            for layer, pred_p in enumerate(plane_preds):
                dists_forward_deep, map_forward_deep, dists_backward_deep, _ = tf_nndistance.nn_distance(plane_gt, pred_p)
                #plane_map = tf.one_hot(map_forward, depth=numOutputPlanes, axis=-1)
                #shuffled_planes = tf.transpose(tf.matmul(pred_p, plane_map, transpose_a=True, transpose_b=True), [0, 2, 1])
                #dists = tf.concat([dists, shuffled_planes, tf.expand_dims(dists_forward, -1)], axis=2)            

                dists_forward_deep *= validPlaneMask
                
                dists_forward_deep = tf.reduce_mean(dists_forward_deep)
                dists_backward_deep = tf.reduce_mean(dists_backward_deep)
                deep_supervision_loss += (dists_forward_deep + dists_backward_deep / 2.0 * useBackward) * 10000
            
                #loss_p_0 = (dists_forward + dists_backward / 2.0 * useBackward) * 10000

                pred_s = segmentation_preds[layer]
                forward_map_deep = tf.one_hot(map_forward_deep, depth=numOutputPlanes, axis=-1)

                segmentation_gt_shuffled_deep = tf.reshape(tf.matmul(tf.reshape(segmentation_gt, [-1, HEIGHT * WIDTH, numOutputPlanes]), forward_map_deep), [-1, HEIGHT, WIDTH, numOutputPlanes])
                segmentation_gt_shuffled_deep = tf.concat([segmentation_gt_shuffled_deep, 1 - plane_mask], axis=3)

                all_segmentations_deep = pred_s
                all_segmentations_deep = tf.concat([pred_s, non_plane_mask_pred], axis=3)
                deep_supervision_loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=all_segmentations_deep, labels=segmentation_gt_shuffled_deep)) * 1000
                
                continue
            pass


        if suffix == 'deep':
            forward_map = forward_map_deep
            valid_forward_map = forward_map * tf.expand_dims(validPlaneMask, -1)
            plane_confidence_gt = tf.transpose(tf.reduce_max(valid_forward_map, axis=1, keep_dims=True), [0, 2, 1])
            plane_confidence_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=plane_confidence_pred, labels=plane_confidence_gt)) * 1000
            
            plane_pred += plane_preds[-1]
            plane_gt_shuffled = tf.transpose(tf.matmul(plane_gt, forward_map_deep, transpose_a=True), [0, 2, 1])
            plane_loss = tf.reduce_mean(tf.squared_difference(plane_pred, plane_gt_shuffled) * plane_confidence_gt) * 1000

            all_segmentations = tf.concat([segmentation_pred, non_plane_mask_pred], axis=3)
            all_segmentations += segmentation_preds[-1]
            
            segmentation_gt_shuffled = segmentation_gt_shuffled_deep
            segmentation_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=all_segmentations, labels=segmentation_gt_shuffled)) * 1000
            dists = plane_confidence_gt
            pass
        else:
            dists_forward, map_forward, dists_backward, map_backward = tf_nndistance.nn_distance(plane_gt, plane_pred)
            forward_map = tf.one_hot(map_forward, depth=numOutputPlanes, axis=-1)
            
            if suffix == 'confidence':
                valid_forward_map = forward_map * tf.expand_dims(validPlaneMask, -1)
                plane_confidence_gt = tf.transpose(tf.reduce_max(valid_forward_map, axis=1, keep_dims=True), [0, 2, 1])
                plane_confidence_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=plane_confidence_pred, labels=plane_confidence_gt)) * 1000
                #plane_confidence_loss = tf.reduce_mean(tf.squared_difference(plane_confidence_pred, plane_confidence_gt)) * 1000
                #plane_confidence_loss = tf.reduce_mean(tf.squared_difference(plane_pred * plane_confidence_gt))) * 1000
                dists = plane_confidence_gt
            else:
                plane_confidence_loss = tf.constant(0.0)
                plane_confidence_gt = tf.ones(plane_confidence_pred.shape)
                pass


            dists_forward *= validPlaneMask        
            dists_forward = tf.reduce_mean(dists_forward)
            dists_backward = tf.reduce_mean(dists_backward)
            plane_loss = (dists_forward + dists_backward / 2.0 * useBackward) * 10000


            #plane_map = tf.one_hot(map_forward, depth=numOutputPlanes, axis=-1)
            #shuffled_planes = tf.transpose(tf.matmul(plane_pred, plane_map, transpose_a=True, transpose_b=True), [0, 2, 1])

            #dists = tf.concat([plane_gt, plane_pred, tf.expand_dims(dists_forward, -1), tf.expand_dims(dists_backward, -1), tf.expand_dims(tf.cast(map_forward, tf.float32), -1), tf.expand_dims(tf.cast(map_backward, tf.float32), -1)], axis=2)
            dists = tf.expand_dims(dists_forward, -1)
            
            if 'regression' in suffix:
                plane_mask = tf.reduce_max(segmentation_gt, axis=3, keep_dims=True)
            
                segmentation_map = tf.one_hot(map_forward, depth=numOutputPlanes, axis=-1)
                segmentation_gt_shuffled = tf.reshape(tf.matmul(tf.reshape(segmentation_gt, [-1, HEIGHT * WIDTH, numOutputPlanes]), segmentation_map), [-1, HEIGHT, WIDTH, numOutputPlanes])
                segmentation_gt_shuffled = tf.cast(segmentation_gt_shuffled > 0.5, tf.float32)
                segmentation_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=segmentation_pred, labels=segmentation_gt_shuffled)) * 1000
            else:
                segmentation_gt_shuffled = tf.reshape(tf.matmul(tf.reshape(segmentation_gt, [-1, HEIGHT * WIDTH, numOutputPlanes]), forward_map), [-1, HEIGHT, WIDTH, numOutputPlanes])
                #segmentation_gt_shuffled = tf.cast(segmentation_gt_shuffled > 0.5, tf.float32)
                segmentation_gt_shuffled = tf.concat([segmentation_gt_shuffled, 1 - plane_mask], axis=3)
                all_segmentations = tf.concat([segmentation_pred, non_plane_mask_pred], axis=3)
                segmentation_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=all_segmentations, labels=segmentation_gt_shuffled)) * 1000
                pass
          
            pass
        
        segmentation_test = segmentation_gt_shuffled
        

        errorMask = tf.zeros(depth_gt.shape)

        plane_parameters = tf.reshape(plane_pred, (-1, 3))
        plane_depths = planeDepthsModule(plane_parameters, WIDTH, HEIGHT)
        plane_depths = tf.transpose(tf.reshape(plane_depths, [HEIGHT, WIDTH, -1, numOutputPlanes]), [2, 0, 1, 3])

        all_segmentations_softmax = tf.nn.softmax(all_segmentations)
        all_depths = tf.concat([plane_depths, non_plane_depth_pred], axis=3)
        depth_loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(all_depths, depth_gt) * all_segmentations_softmax, axis=3, keep_dims=True) * tf.cast(tf.greater(depth_gt, 1e-4), tf.float32)) * 1000
        #depth_loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(plane_depths, depth_gt) * segmentation, axis=3, keep_dims=True) * plane_mask) * 1000


        #plane_normals = planeNormalsModule(plane_parameters, WIDTH, HEIGHT)
        #plane_normals = tf.reshape(plane_normals, [-1, 1, 1, numOutputPlanes, 3])
        #normal_pred = tf.reduce_sum(tf.multiply(plane_normals, tf.expand_dims(segmentation, -1)), axis=3)
            
        #normal_loss = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(plane_normals, tf.expand_dims(normal_gt, 3)) * tf.expand_dims(segmentation, -1), axis=3) * plane_mask) * 1000
        normal_loss = tf.reduce_mean(tf.squared_difference(non_plane_normal_pred, normal_gt) * (1 - plane_mask)) * 1000
        
    
        #s_8_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=s_8_pred, labels=s_8_gt)) * 1000
        grid_s_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=grid_s_pred, multi_class_labels=grid_s_gt, weights=tf.maximum(grid_s_gt * 10, 1))) * 1000
        grid_p_loss = tf.reduce_mean(tf.squared_difference(grid_p_pred, grid_p_gt) * grid_s_gt) * 10000
        grid_m_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=grid_m_pred, labels=grid_m_gt) * grid_s_gt) * 10000

        if suffix == 'boundary':
            boundary_gt = tf.cond(validating_inp, lambda: boundary_gt_val, lambda: boundary_gt_train)
            boundary_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=boundary_pred, multi_class_labels=boundary_gt, weights=tf.maximum(boundary_gt * 5, 1))) * 1000
        else:
            if True:
                kernel_size = 3
                padding = (kernel_size - 1) / 2
                neighbor_kernel_array = gaussian(kernel_size, kernel_size)
                neighbor_kernel_array[(kernel_size - 1) / 2][(kernel_size - 1) / 2] = 0
                neighbor_kernel_array /= neighbor_kernel_array.sum()
                neighbor_kernel_array *= -1
                neighbor_kernel_array[(kernel_size - 1) / 2][(kernel_size - 1) / 2] = 1
                neighbor_kernel = tf.constant(neighbor_kernel_array.reshape(-1), shape=neighbor_kernel_array.shape, dtype=tf.float32)
                neighbor_kernel = tf.reshape(neighbor_kernel, [kernel_size, kernel_size, 1, 1])
        
                depth_diff = tf.abs(tf.nn.depthwise_conv2d(depth_gt, neighbor_kernel, strides=[1, 1, 1, 1], padding='VALID'))
                depth_diff = tf.pad(depth_diff, paddings = [[0, 0], [padding, padding], [padding, padding], [0, 0]])
                max_depth_diff = 0.1
                depth_boundary = tf.greater(depth_diff, max_depth_diff)

                normal_diff = tf.norm(tf.nn.depthwise_conv2d(normal_gt, tf.tile(neighbor_kernel, [1, 1, 3, 1]), strides=[1, 1, 1, 1], padding='VALID'), axis=3, keep_dims=True)
                normal_diff = tf.pad(normal_diff, paddings = [[0, 0], [padding, padding], [padding, padding], [0, 0]])
                max_normal_diff = np.sqrt(2 * (1 - np.cos(np.deg2rad(20))))
                normal_boundary = tf.greater(normal_diff, max_normal_diff)

                #kernel_size = 7
                #segmentation_dilated = tf.nn.max_pool(segmentation_gt, ksize=[1, kernel_size, kernel_size, 1], strides=[1, 1, 1, 1], padding='SAME', name='max_pool')
                #segmentation_eroded = 1 - tf.nn.max_pool(1 - segmentation_gt, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='max_pool')
                #overlap_region = tf.greater_equal(tf.reduce_sum(segmentation_dilated - segmentation_eroded, axis=3, keep_dims=True), 2)
                plane_region = tf.nn.max_pool(plane_mask, ksize=[1, kernel_size, kernel_size, 1], strides=[1, 1, 1, 1], padding='SAME', name='max_pool')
                segmentation_eroded = 1 - tf.nn.max_pool(1 - segmentation_gt, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='max_pool')
                plane_region -= tf.reduce_max(segmentation_eroded, axis=3, keep_dims=True)
                boundary = tf.cast(tf.logical_or(depth_boundary, normal_boundary), tf.float32) * plane_region
                smooth_boundary = tf.cast(tf.logical_and(normal_boundary, tf.less_equal(depth_diff, max_depth_diff)), tf.float32) * plane_region
                boundary_gt = tf.concat([smooth_boundary, boundary - smooth_boundary], axis=3)
            else:
                segmentation_dilated = tf.nn.max_pool(segmentation_gt, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='max_pool')
                #segmentation_eroded = 1 - tf.nn.max_pool(1 - segmentation_gt, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='max_pool')
                boundary = tf.reduce_max(segmentation_dilated - segmentation_gt, axis=3, keep_dims=True)
                max_depth_diff = 0.1
                kernel_size = 5
                padding = (kernel_size - 1) / 2
                neighbor_kernel_array = gaussian(kernel_size, kernel_size)
                neighbor_kernel_array[(kernel_size - 1) / 2][(kernel_size - 1) / 2] = 0
                neighbor_kernel_array /= neighbor_kernel_array.sum()
                neighbor_kernel_array *= -1
                neighbor_kernel_array[(kernel_size - 1) / 2][(kernel_size - 1) / 2] = 1
                neighbor_kernel = tf.constant(neighbor_kernel_array.reshape(-1), shape=neighbor_kernel_array.shape, dtype=tf.float32)
                neighbor_kernel = tf.reshape(neighbor_kernel, [kernel_size, kernel_size, 1, 1])
        
                depth_diff = tf.abs(tf.nn.depthwise_conv2d(depth_gt, neighbor_kernel, strides=[1, 1, 1, 1], padding='VALID'))
                depth_diff = tf.pad(depth_diff, paddings = [[0, 0], [padding, padding], [padding, padding], [0, 0]])

                smooth_boundary = boundary * tf.cast(tf.less(depth_diff, max_depth_diff), tf.float32)
                # occlusion_boundary = boundary - smooth_boundary
                # boundary = tf.squeeze(tf.concat([smooth_boundary, occlusion_boundary], axis=3), axis=0)
                boundary_gt = tf.concat([smooth_boundary, boundary - smooth_boundary], axis=3) * plane_mask
                pass

            
            all_segmentations_one_hot = all_segmentations_softmax
                          
            all_segmentations_min = 1 - tf.nn.max_pool(1 - all_segmentations_one_hot, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
            segmentation_diff = tf.reduce_max(all_segmentations_softmax - all_segmentations_min, axis=3, keep_dims=True)
            #occlusion_boundary = tf.slice(boundary_gt, [0, 0, 0, 1], [int(boundary_gt.shape[0]), HEIGHT, WIDTH, 1])
            
            #boundary = tf.reduce_sum(boundary_gt, axis=3, keep_dims=True)
            #smooth_boundary = tf.slice(boundary_gt, [0, 0, 0, 0], [int(boundary_gt.shape[0]), HEIGHT, WIDTH, 1])
            
            #occlusion_boundary = occlusion_boundary * 2 - 1
            #segmentation_diff = segmentation_diff * 2 - 1

            depth_one_hot = tf.reduce_sum(tf.multiply(all_depths, all_segmentations_one_hot), axis=3, keep_dims=True)
            depth_neighbor = tf.nn.depthwise_conv2d(depth_one_hot, neighbor_kernel, strides=[1, 1, 1, 1], padding='SAME')
            #depth_neighbor = tf.pad(depth_neighbor, paddings = [[0, 0], [padding, padding], [padding, padding], [0, 0]])
            #maxDepthDiff = 0.1
            minDepthDiff = 0.02
            depth_diff = tf.clip_by_value(tf.squared_difference(depth_one_hot, depth_neighbor) - pow(minDepthDiff, 2), 0, 1)
            
            smooth_mask = segmentation_diff + boundary - 2 * segmentation_diff * boundary + depth_diff * smooth_boundary
            margin = 0.0
            smooth_mask = tf.nn.relu(smooth_mask - margin)
            errorMask = smooth_mask
            #errorMask = segmentation_diff
            
            boundary_loss = tf.reduce_mean(smooth_mask * plane_mask) * 1000
            
            #if suffix == 'boundary_pred':
            boundary_loss += tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=boundary_pred, multi_class_labels=boundary_gt, weights=tf.maximum(boundary_gt * 3, 1))) * 1000
            pass


        if 'diverse' in suffix:
            plane_diff = tf.reduce_sum(tf.pow(tf.expand_dims(plane_pred, 1) - tf.expand_dims(plane_pred, 2), 2), axis=3)
            plane_diff = tf.matrix_set_diag(plane_diff, tf.ones((int(plane_diff.shape[0]), numOutputPlanes)))
            minPlaneDiff = 0.1
            diverse_loss = tf.reduce_mean(tf.clip_by_value(1 - plane_diff / minPlaneDiff, 0, 1)) * 10000
        else:
            diverse_loss = tf.constant(0.0)
            pass          


        l2_losses = tf.add_n([5e-4 * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name])
        #loss = plane_loss + segmentation_loss + depth_loss + normal_loss + l2_losses
        loss = plane_loss + segmentation_loss + depth_loss + normal_loss + l2_losses + grid_s_loss + grid_p_loss + grid_m_loss + boundary_loss + plane_confidence_loss + diverse_loss + deep_supervision_loss
        #loss = plane_loss + segmentation_loss + depth_loss + normal_loss + l2_losses
      
        if suffix == 'pixelwise':
            normal_loss = tf.reduce_mean(tf.squared_difference(non_plane_normal_pred, normal_gt)) * 1000
            depth_loss = tf.reduce_mean(tf.squared_difference(non_plane_depth_pred, depth_gt) * tf.cast(tf.greater(depth_gt, 1e-4), tf.float32)) * 1000
            loss = normal_loss + depth_loss
            pass
        
    return loss, plane_loss, segmentation_loss + depth_loss + normal_loss + grid_s_loss + grid_p_loss + grid_m_loss + boundary_loss, deep_supervision_loss, diverse_loss + plane_confidence_loss, segmentation_test, boundary_gt, plane_mask, errorMask, dists
    #return loss, plane_loss, depth_loss + grid_s_loss + grid_p_loss + grid_m_loss + boundary_loss, forward_loss, backward_loss, segmentation_gt, plane_mask, errorMask, dists
    #return loss, plane_loss, segmentation_loss, loss_p_0, depth_loss, segmentation_test, plane_mask, errorMask, dists


def main(gpu_id, dumpdir, logdir, testdir, keyname, restore, numOutputPlanes=20, useCRF=0, batchSize=16, suffix='forward'):
    if not os.path.exists(dumpdir):
        os.system("mkdir -p %s"%dumpdir)
        pass
    if not os.path.exists(testdir):
        os.system("mkdir -p %s"%testdir)
        pass
    
    min_after_dequeue = 1000

    if False:
        reader_train = RecordReaderAll()
        filename_queue_train = tf.train.string_input_producer(['../planes_all_100000.tfrecords'], num_epochs=10000)    
        img_inp_train, plane_gt_train, depth_gt_train, normal_gt_train, num_planes_gt_train, _ = reader_train.getBatch(filename_queue_train, numOutputPlanes=numOutputPlanes, batchSize=batchSize, min_after_dequeue=min_after_dequeue)
        segmentation_gt_train = tf.ones((batchSize, HEIGHT, WIDTH, numOutputPlanes))

        reader_val = RecordReaderAll()
        filename_queue_val = tf.train.string_input_producer(['../planes_all_1000_100000.tfrecords'], num_epochs=10000)    
        img_inp_val, plane_gt_val, depth_gt_val, normal_gt_val, num_planes_gt_val, _ = reader_val.getBatch(filename_queue_val, numOutputPlanes=numOutputPlanes, batchSize=batchSize, min_after_dequeue=min_after_dequeue)
        segmentation_gt_val = tf.ones((batchSize, HEIGHT, WIDTH, numOutputPlanes))
    elif True:
        reader_train = RecordReader()
        filename_queue_train = tf.train.string_input_producer(['/mnt/vision/SUNCG_plane/planes_test_450000.tfrecords'], num_epochs=10000)
        img_inp_train, plane_gt_train, depth_gt_train, normal_gt_train, segmentation_gt_train, boundary_gt_train, grid_s_gt_train, grid_p_gt_train, grid_m_gt_train, num_planes_gt_train = reader_train.getBatch(filename_queue_train, numOutputPlanes=numOutputPlanes, batchSize=batchSize, min_after_dequeue=min_after_dequeue, getLocal=True, test=False, suffix=suffix, random=False)

        reader_val = RecordReader()
        filename_queue_val = tf.train.string_input_producer(['/mnt/vision/SUNCG_plane/planes_test_1000_450000.tfrecords'], num_epochs=10000)
        img_inp_val, plane_gt_val, depth_gt_val, normal_gt_val, segmentation_gt_val, boundary_gt_val, grid_s_gt_val, grid_p_gt_val, grid_m_gt_val, num_planes_gt_val = reader_val.getBatch(filename_queue_val, numOutputPlanes=numOutputPlanes, batchSize=batchSize, min_after_dequeue=min_after_dequeue, getLocal=True, suffix=suffix)
    else:
        reader_rgbd_train = RecordReaderRGBD()
        filename_queue_rgbd_train = tf.train.string_input_producer(['../planes_nyu_rgbd_train.tfrecords'], num_epochs=10000)
        img_inp_rgbd_train, global_gt_dict_rgbd_train, local_gt_dict_rgbd_train = reader_rgbd_train.getBatch(filename_queue_rgbd_train, numOutputPlanes=options.numOutputPlanes, batchSize=options.batchSize, min_after_dequeue=min_after_dequeue)
        #img_inp, global_gt_dict, local_gt_dict = reader.getBatch(filename_queue, numOutputPlanes=options.numOutputPlanes, batchSize=options.batchSize, min_after_dequeue=min_after_dequeue, getLocal=True, random=False)
    
        reader_rgbd_val = RecordReaderRGBD()
        filename_queue_rgbd_val = tf.train.string_input_producer(['../planes_nyu_rgbd_val.tfrecords'], num_epochs=10000)
        img_inp_rgbd_val, global_gt_dict_rgbd_val, local_gt_dict_rgbd_val = reader_rgbd_val.getBatch(filename_queue_rgbd_val, numOutputPlanes=options.numOutputPlanes, batchSize=options.batchSize, min_after_dequeue=min_after_dequeue)
        pass
    
    
    validating_inp = tf.placeholder(tf.bool, shape=[], name='validating_inp')
    plane_pred, plane_confidence_pred, depth_pred, normal_pred, segmentation_pred, non_plane_mask_pred, non_plane_depth_pred, non_plane_normal_pred, boundary_pred, grid_s_pred, grid_p_pred, grid_m_pred, plane_preds, segmentation_preds, _ = build_graph(img_inp_train, img_inp_val, plane_gt_train, plane_gt_val, validating_inp, numOutputPlanes=numOutputPlanes, useCRF=useCRF, suffix=suffix)

    #var_to_restore = [v for v in tf.global_variables() if 'planes' not in v.name and 'segmentation' not in v.name and 'moving_' not in v.name and 'connection' not in v.name]
    var_to_restore = [v for v in tf.global_variables()]
    #for op in tf.get_default_graph().get_operations():
    #print str(op)
    #continue

    loss, plane_loss, depth_loss, normal_loss, segmentation_loss, segmentation_gt, boundary_gt, plane_mask_gt, _, plane_confidence_gt = build_loss(plane_pred, plane_confidence_pred, depth_pred, normal_pred, segmentation_pred, non_plane_mask_pred, non_plane_depth_pred, non_plane_normal_pred, boundary_pred, grid_s_pred, grid_p_pred, grid_m_pred, plane_preds, segmentation_preds, plane_gt_train, depth_gt_train, normal_gt_train, segmentation_gt_train, boundary_gt_train, grid_s_gt_train, grid_p_gt_train, grid_m_gt_train, num_planes_gt_train, plane_gt_val, depth_gt_val, normal_gt_val, segmentation_gt_val, boundary_gt_val, grid_s_gt_val, grid_p_gt_val, grid_m_gt_val, num_planes_gt_val, validating_inp, numOutputPlanes = numOutputPlanes, gpu_id = gpu_id, useCRF=useCRF, suffix=suffix)

    train_writer = tf.summary.FileWriter(logdir + '/train', graph=tf.get_default_graph())
    val_writer = tf.summary.FileWriter(logdir + '/val')
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('plane_loss', plane_loss)
    #tf.summary.scalar('depth_loss', depth_loss)
    #tf.summary.scalar('normal_loss', normal_loss)
    tf.summary.scalar('segmentation_loss', segmentation_loss)    
    summary_op = tf.summary.merge_all()

    with tf.variable_scope('statistics'):
        batchno = tf.Variable(0, dtype=tf.int32, trainable=False, name='batchno')
        batchnoinc=batchno.assign(batchno+1)
        pass

    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #update_ops = tf.get_collection(UPDATE_OPS_COLLECTION)
    #with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(3e-5*BATCH_SIZE/FETCH_BATCH_SIZE)
    train_op = optimizer.minimize(loss, global_step=batchno)
    #var_to_train = [v for v in var_to_restore if 'plane' in v.name or 'res5d' in v.name]
    #train_op = optimizer.minimize(loss, global_step=batchno, var_list=var_to_train)
    
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True
    saver=tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if restore == 1:
            loader = tf.train.Saver(var_to_restore)
            loader.restore(sess,"%s/%s.ckpt"%(dumpdir, keyname))
            bno=sess.run(batchno)
            print(bno)
        elif restore == 0:
            var_to_restore = [v for v in var_to_restore if 'res5d' not in v.name and 'segmentation' not in v.name and 'plane' not in v.name and 'depth' not in v.name and 'normal' not in v.name]
            resnet_loader = tf.train.Saver(var_to_restore)
            resnet_loader.restore(sess,"../pretrained_models/deeplab_resnet.ckpt")
        elif restore == 2:
            var_to_restore = [v for v in tf.global_variables() if 'non_plane' not in v.name]
            resnet_loader = tf.train.Saver(var_to_restore)
            resnet_loader.restore(sess,"dump_supervision_deeplab_forward/train_supervision_deeplab_forward.ckpt")
            sess.run(batchno.assign(0))
        elif restore == 3:
            #var_to_restore = [v for v in tf.global_variables() if 'deep_supervision/res4b22_relu_segmentation_conv2' not in v.name]
            #var_to_restore = [v for v in tf.global_variables() if 'empty_mask' not in v.name]
            var_to_restore = [v for v in tf.global_variables() if 'plane_confidence' not in v.name]
            #var_to_restore = [v for v in tf.global_variables() if 'boundary' not in v.name]
            original_saver = tf.train.Saver(var_to_restore)
            original_saver.restore(sess,"dump_planenet_deep/train_planenet_deep.ckpt")
            #original_saver.restore(sess,"%s/%s.ckpt"%(dumpdir, keyname))
            sess.run(batchno.assign(1))
            pass
        elif restore == 4:
            var_to_restore_1 = [v for v in var_to_restore if 'deep_supervision' not in v.name]
            resnet_loader = tf.train.Saver(var_to_restore_1)
            resnet_loader.restore(sess,"dump_grid_deeplab_degridding/train_grid_deeplab_degridding.ckpt")
            sess.run(batchno.assign(1))
        elif restore == 5:
            #saver.restore(sess,"dump_all_resnet_v2/train_all_resnet_v2.ckpt")
            #var_to_restore = [v for v in var_to_restore if 'plane' not in v.name and 'segmentation' not in v.name]
            var_to_restore = [v for v in var_to_restore]
            loader = tf.train.Saver(var_to_restore)
            loader.restore(sess,"%s/%s.ckpt"%(dumpdir, keyname))
            sess.run(batchno.assign(1))
            pass
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        ema = [0., 0.]
        ema_acc = [1e-10, 1e-10]
        last_snapshot_time = time.time()
        bno=sess.run(batchno)
        #fetchworker.bno=bno//(FETCH_BATCH_SIZE/BATCH_SIZE)
        #fetchworker.start()
        try:
            while bno<300000:
                t0=time.time()
                #_, total_loss, train_loss, val_loss, summary_str, pred, gt, images, planes = sess.run([train_op, loss, training_loss, validation_loss, summary_op, segmentation_pred, segmentation_gt, img_inp, plane_inp])
                if bno % 100 > 0:
                    _, total_loss, loss_1, loss_2, loss_3, loss_4, summary_str = sess.run([train_op, loss, plane_loss, depth_loss, normal_loss, segmentation_loss, summary_op], feed_dict = {validating_inp:False})

                    #print(pred[0])
                    #print(gt[0])
                    #exit(1)
                    
                    # print(pred_map[0])
                    # print(pred_p[0])
                    # print(gt_p[0])
                    # print(num_planes[0])
                    # exit(1)
                    # print(pred_d.max())
                    # print(pred_d.min())
                    # print(gt_d.max())
                    # print(gt_d.min())
                    # exit(1)
                    train_writer.add_summary(summary_str, bno)
                    ema[0] = ema[0] * MOVING_AVERAGE_DECAY + total_loss
                    #ema[0] = ema[0] * MOVING_AVERAGE_DECAY + (loss_1 + loss_2 + loss_3 + loss_4)
                    ema_acc[0] = ema_acc[0] * MOVING_AVERAGE_DECAY + 1
                else:
                    _, total_loss, loss_1, loss_2, loss_3, loss_4, summary_str = sess.run([batchnoinc, loss, plane_loss, depth_loss, normal_loss, segmentation_loss, summary_op], feed_dict = {validating_inp:True})
                    val_writer.add_summary(summary_str, bno)
                    ema[1] = ema[1] * MOVING_AVERAGE_DECAY + total_loss
                    #ema[1] = ema[1] * MOVING_AVERAGE_DECAY + (loss_1 + loss_2 + loss_3 + loss_4)
                    ema_acc[1] = ema_acc[1] * MOVING_AVERAGE_DECAY + 1
                    pass
                #loss_3 = 0
                

                #images, planes, segmentation, normal, depth, num_planes, mask = sess.run([img_inp_train, plane_gt_train, segmentation_gt_train, normal_gt_train, depth_gt_train, num_planes_train, mask_train], feed_dict = {validating_inp:False})
                # images, planes, segmentation, normal, depth, num_planes = sess.run([img_inp_val, plane_gt_val, segmentation_gt_val, normal_gt_val, depth_gt_val, num_planes_gt_val], feed_dict = {validating_inp:False})

                # cv2.imwrite(testdir + '/segmentation_image.png', ((images[0] + 0.5) * 255).astype(np.uint8))
                # cv2.imwrite(testdir + '/segmentation_normal.png', np.minimum(np.maximum((normal[0] + 1) / 2 * 255, 0), 255).astype(np.uint8))
                # cv2.imwrite(testdir + '/segmentation_depth.png', np.minimum(np.maximum(depth[0, :, :, 0] / 10 * 255, 0), 255).astype(np.uint8))
                # print(planes[0])
                # exit(1)
                #cv2.imwrite(testdir + '/segmentation_background.png', (mask[0] == 0).astype(np.uint8) * 255)
                #unique_values = np.unique(mask[0].reshape([-1]))
                #print(unique_values)
                #for value in unique_values:
                #cv2.imwrite(testdir + '/segmentation_mask_' + str(value) + '.png', ((mask[0] == value) * 255).astype(np.uint8))
                #continue                  
                # for planeIndex in xrange(numOutputPlanes):
                #     print((planeIndex, planes[0, planeIndex]))
                #     cv2.imwrite(testdir + '/segmentation_' + str(planeIndex) + '_segmentation.png', (segmentation[0, :, :, planeIndex] * 255).astype(np.uint8))
                #     continue
                # exit(1)

                bno=sess.run(batchno)
                if time.time()-last_snapshot_time > 200:
                    print('save snapshot')
                    saver.save(sess,'%s/'%dumpdir+keyname+".ckpt")
                    last_snapshot_time = time.time()
                    pass
        
                print bno,'train', ema[0] / ema_acc[0], 'val', ema[1] / ema_acc[1], 'loss', total_loss, loss_1, loss_2, loss_3, loss_4, 'time', time.time()-t0
                continue

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
            saver.save(sess,'%s/'%dumpdir+keyname+".ckpt")
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            pass
        
        # Wait for threads to finish.
        coord.join(threads)
        sess.close()
        pass
    return


def test(gpu_id, dumpdir, logdir, testdir, keyname, restore, numOutputPlanes=20, useCRF=0, suffix='forward'):
    if not os.path.exists(testdir):
        os.system("mkdir -p %s"%testdir)
        pass


    # reader_train = RecordReader()
    # filename_queue_train = tf.train.string_input_producer(['../planes_all_1000_450000.tfrecords'], num_epochs=1)
    # img_inp, depth_gt, normal_gt, invalid_mask_gt, image_path_inp = reader_train.getBatch(filename_queue_train, numOutputPlanes=numOutputPlanes, batchSize=1, random=False)

    # init_op = tf.group(tf.global_variables_initializer(),
    #                    tf.local_variables_initializer())

    # with tf.Session() as sess:
    #     sess.run(init_op)
 
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
    #     try:
    #         for index in xrange(50):
    #             print(('image', index))
    #             t0=time.time()

    #             depth, invalid_mask = sess.run([depth_gt, invalid_mask_gt])
    #             depth = depth.squeeze()
    #             invalid_mask = invalid_mask.squeeze()


    #             if index < 10:
    #                 continue
    #             #print(depth[51][12])
    #             #print(depth[51][100])
    #             #print(depth[84][22])
    #             cv2.imwrite(testdir + '/' + str(index) + '_depth.png', drawDepthImage(depth))
    #             cv2.imwrite(testdir + '/' + str(index) + '_invalid_mask.png', drawMaskImage(invalid_mask))
    #             continue
        
    #     except tf.errors.OutOfRangeError:
    #         print('Done training -- epoch limit reached')
    #     finally:
    #         # When done, ask the threads to stop.
    #         coord.request_stop()
    #         pass
          
    #     # Wait for threads to finish.
    #     coord.join(threads)
    #     sess.close()
    #     pass
    # exit(1)
    
    batchSize = 1
    
    min_after_dequeue = 1000

    reader_val = RecordReader()
    filename_queue_val = tf.train.string_input_producer(['/mnt/vision/SUNCG_plane/planes_test_1000_450000.tfrecords'], num_epochs=10000)
    img_inp, plane_gt, depth_gt, normal_gt, segmentation_gt_original, boundary_gt, grid_s_gt, grid_p_gt, grid_m_gt, num_planes_gt = reader_val.getBatch(filename_queue_val, numOutputPlanes=numOutputPlanes, batchSize=batchSize, min_after_dequeue=min_after_dequeue, random=False, getLocal=True, getSegmentation=False, suffix=suffix)
    #img_inp, plane_gt, depth_gt, normal_gt, segmentation_gt, boundary_gt, s_8_gt = reader_val.getBatch(filename_queue_val, numOutputPlanes=numOutputPlanes, batchSize=batchSize, min_after_dequeue=min_after_dequeue, random=False, getLocal=True)

    validating_inp = tf.placeholder(tf.bool, shape=[], name='validating_inp')

    plane_pred, plane_confidence_pred, depth_pred, normal_pred, segmentation_pred, non_plane_mask_pred, non_plane_depth_pred, non_plane_normal_pred, boundary_pred, grid_s_pred, grid_p_pred, grid_m_pred, plane_preds, segmentation_preds, refined_segmentation = build_graph(img_inp, img_inp, plane_gt, plane_gt, validating_inp, numOutputPlanes=numOutputPlanes, useCRF=useCRF, is_training=False, suffix=suffix)

    var_to_restore = tf.global_variables()
    
    loss, plane_loss, depth_loss, normal_loss, segmentation_loss, segmentation_gt, boundary_gt, plane_mask_gt, error_mask, dists = build_loss(plane_pred, plane_confidence_pred, depth_pred, normal_pred, segmentation_pred, non_plane_mask_pred, non_plane_depth_pred, non_plane_normal_pred, boundary_pred, grid_s_pred, grid_p_pred, grid_m_pred, plane_preds, segmentation_preds, plane_gt, depth_gt, normal_gt, segmentation_gt_original, boundary_gt, grid_s_gt, grid_p_gt, grid_m_gt, num_planes_gt, plane_gt, depth_gt, normal_gt, segmentation_gt_original, boundary_gt, grid_s_gt, grid_p_gt, grid_m_gt, num_planes_gt, validating_inp, numOutputPlanes=numOutputPlanes, gpu_id=gpu_id, useCRF=useCRF, suffix=suffix)
        

    errorSum = np.zeros(4)
    
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        loader = tf.train.Saver(var_to_restore)
        loader.restore(sess,"%s/%s.ckpt"%(dumpdir, keyname))
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        randomColor = np.random.randint(255, size=(50 + 1, 3)).astype(np.uint8)
        randomColor[0] = 0
        
        try:
            gtDepths = []
            predDepths = []
            planeMasks = []
            #predMasks = []
            
            imageWidth = WIDTH
            imageHeight = HEIGHT
            focalLength = 517.97
            urange = np.arange(imageWidth).reshape(1, -1).repeat(imageHeight, 0) - imageWidth * 0.5
            vrange = np.arange(imageHeight).reshape(-1, 1).repeat(imageWidth, 1) - imageHeight * 0.5
            ranges = np.array([urange / imageWidth * 640 / focalLength, np.ones(urange.shape), -vrange / imageHeight * 480 / focalLength]).transpose([1, 2, 0])


            for index in xrange(100):
                print(('image', index))
                t0=time.time()
                #im, planes, depth, normal, boundary, s_8, p_8, b_8, s_16, p_16, b_16, s_32, p_32, b_32 = sess.run([img_inp, plane_gt, depth_gt, normal_gt, boundary_gt, s_8_gt, p_8_gt, b_8_gt, s_16_gt, p_16_gt, b_16_gt, s_32_gt, p_32_gt, b_32_gt])


                pred_p, pred_p_c, pred_d, pred_n, pred_s, loss_p, loss_d, loss_n, loss_s, im, gt_p, gt_d, gt_n, pred_np_m, pred_np_d, pred_np_n, gt_s, gt_plane_mask, pred_boundary, grid_s, grid_p, grid_m, gt_grid_s, gt_grid_m, refined_s, mask_e, preds_p, preds_s, distance, numPlanes, gt_s_ori, gt_boundary = sess.run([plane_pred, plane_confidence_pred, depth_pred, normal_pred, segmentation_pred, plane_loss, depth_loss, normal_loss, segmentation_loss, img_inp, plane_gt, depth_gt, normal_gt, non_plane_mask_pred, non_plane_depth_pred, non_plane_normal_pred, segmentation_gt, plane_mask_gt, boundary_pred, grid_s_pred, grid_p_pred, grid_m_pred, grid_s_gt, grid_m_gt, refined_segmentation, error_mask, plane_preds, segmentation_preds, dists, num_planes_gt, segmentation_gt_original, boundary_gt], feed_dict = {validating_inp:True})

                #if index != 8:
                #continue
                
                im = im[0]
                image = ((im + 0.5) * 255).astype(np.uint8)

                planes = gt_p[0]                
                gt_d = gt_d.squeeze()
                gt_n = gt_n[0]
                #grid_s = [s_8, s_16, s_32]
                #grid_p = [p_8, p_16, p_32]
                #grid_b = [b_8, b_16, b_32]

                # if index < 5:
                #     continue                
                # print(depth[51][12])
                # print(depth[51][100])
                # print(depth[84][22])
                # cv2.imwrite(testdir + '/depth.png', drawDepthImage(depth))
                # exit(1)
                
                grid_s = 1 / (1 + np.exp(-grid_s))
                grid_s = grid_s[0]
                grid_p = grid_p[0]
                grid_m = grid_m[0]
                gt_grid_s = gt_grid_s[0]
                gt_grid_m = gt_grid_m[0]                
                #grid_s = gt_grid_s
                #grid_m = gt_grid_m                

                
                gt_s = gt_s[0]
                pred_p = pred_p[0]
                pred_d = pred_d.squeeze()
                pred_n = pred_n[0]
                pred_s = pred_s[0]
                refined_s = refined_s[0]

                pred_np_m = pred_np_m[0]
                pred_np_d = pred_np_d[0]
                pred_np_n = pred_np_n[0]

                pred_p_c = pred_p_c[0]
 

                if False:
                    if index >= 10:
                        break
                    np.save(dumpdir + '/planes_' + str(index) + '.npy', pred_p)
                    np.save(dumpdir + '/segmentations_' + str(index) + '.npy', pred_s)
                    np.save(dumpdir + '/segmentations_gt_' + str(index) + '.npy', gt_s)
                    np.save(dumpdir + '/non_plane_depth_' + str(index) + '.npy', pred_np_d)
                    np.save(dumpdir + '/non_plane_segmentation_' + str(index) + '.npy', pred_np_m)
                    boundary = pred_boundary[0]
                    boundary = 1 / (1 + np.exp(-boundary))
                    boundary = np.concatenate([boundary, np.zeros((HEIGHT, WIDTH, 1))], axis=2)                    
                    cv2.imwrite(dumpdir + '/boundary_' + str(index) + '.png', drawMaskImage(boundary))
                    cv2.imwrite(dumpdir + '/image_' + str(index) + '.png', cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR))
                    np.save(dumpdir + '/depth_' + str(index) + '.npy', gt_d)
                    #continue

                  
                stride = 8
                boxSize = 64
                xs = np.arange(WIDTH / stride) * stride + stride / 2
                ys = np.arange(HEIGHT / stride) * stride + stride / 2
                padding = boxSize / 2 + 1
                maskImage = np.zeros((HEIGHT + padding * 2, WIDTH + padding * 2, 3), dtype=np.uint8)
                maskImage[padding:padding + HEIGHT, padding:padding + WIDTH, :] = image / 2
                for gridY, y in enumerate(ys):
                    for gridX, x in enumerate(xs):
                        score = grid_s[gridY][gridX]
                        if score < 0.5:
                            continue                              
                        mask = grid_m[gridY][gridX].reshape([16, 16])        
                        mask = cv2.resize(mask, (boxSize, boxSize))
                        maskImage[y - boxSize / 2 + padding:y + boxSize / 2 + padding, x - boxSize / 2 + padding:x + boxSize / 2 + padding, 0][mask > 0.5] = 255
                        continue
                    continue
                for gridY, y in enumerate(ys):
                    for gridX, x in enumerate(xs):
                        score = gt_grid_s[gridY][gridX]
                        if score < 0.5:
                            continue                              
                        mask = gt_grid_m[gridY][gridX].reshape([16, 16])        
                        mask = cv2.resize(mask, (boxSize, boxSize))
                        maskImage[y - boxSize / 2 + padding:y + boxSize / 2 + padding, x - boxSize / 2 + padding:x + boxSize / 2 + padding, 2][mask > 0.5] = 255
                        continue
                    continue

                #print((grid_s > 0.5).astype(np.int8).sum())
                #print(grid_s[gt_grid_s.astype(np.bool)])

                
                pred_p_c = 1 / (1 + np.exp(-pred_p_c))
                print((loss_p, loss_d, loss_n, loss_s, numPlanes[0], (pred_p_c > 0.5).sum()))
 
                errorSum += np.array([loss_p, loss_d, loss_n, loss_s])
                #planeMask = np.max(gt_s, 2) > 0.5
                planeMask = np.squeeze(gt_plane_mask)
                
                #predMasks.append(np.max(pred_s, 2) > 0.5)

                #cv2.imwrite(testdir + '/' + str(index) + '_image.png', ((im + 0.5) * 255).astype(np.uint8))
                #cv2.imwrite(testdir + '/' + str(index) + '_depth_inp.png', drawDepthImage(depth))
                #cv2.imwrite(testdir + '/' + str(index) + '_normal_inp.png', drawNormalImage(normal))
                
                #cv2.imwrite(testdir + '/' + str(index) + '_depth_gt.png', drawDepthImage(gt_d))
                #cv2.imwrite(testdir + '/' + str(index) + '_depth_gt_diff.png', drawDiffImage(gt_d, depth, 0.5))


                if index >= 10:
                    #cv2.imwrite(testdir + '/' + str(index) + '_depth.png', drawDepthImage(depth))
                    continue



                #segmentation = np.argmax(pred_s, 2)
                #pred_d = plane_depths.reshape(-1, numOutputPlanes)[np.arange(WIDTH * HEIGHT), segmentation.reshape(-1)].reshape(HEIGHT, WIDTH)


                
                #cv2.imwrite(testdir + '/' + str(index) + '_error_mask.png', drawMaskImage(np.clip(np.squeeze(mask_e) / 1, 0, 1)))
                #exit(1)
                cv2.imwrite(testdir + '/' + str(index) + '_image.png', image)
                cv2.imwrite(testdir + '/' + str(index) + '_depth.png', drawDepthImage(gt_d))
                cv2.imwrite(testdir + '/' + str(index) + '_normal.png', drawNormalImage(gt_n))

                if suffix == 'boundary_pred':
                    pred_boundary = pred_boundary[0]
                    pred_boundary = 1 / (1 + np.exp(-pred_boundary))
                    boundary = np.concatenate([pred_boundary, np.zeros((HEIGHT, WIDTH, 1))], axis=2)
                    cv2.imwrite(testdir + '/' + str(index) + '_boundary_pred.png', drawMaskImage(boundary))
                    pass
                
                gt_boundary = gt_boundary[0]
                boundary = np.concatenate([gt_boundary, np.zeros((HEIGHT, WIDTH, 1))], axis=2)                
                cv2.imwrite(testdir + '/' + str(index) + '_boundary_gt.png', drawMaskImage(boundary))
                                  
                #cv2.imwrite(testdir + '/' + str(index) + '_grid_mask.png', maskImage)
                

                #cv2.imwrite(testdir + '/' + str(index) + '_depth_pred_diff.png', drawDiffImage(pred_d * planeMask, depth * planeMask, 0.5))

                #cv2.imwrite(testdir + '/' + str(index) + '_normal_gt.png', drawNormalImage(gt_n))
                #cv2.imwrite(testdir + '/' + str(index) + '_normal_pred.png', drawNormalImage(np.clip(pred_np_n, 0, 1)))
                
                #cv2.imwrite(testdir + '/' + str(index) + '_segmentation_refined.png', drawSegmentationImage(refined_s))
                cv2.imwrite(testdir + '/' + str(index) + '_segmentation_gt.png', drawSegmentationImage(gt_s, planeMask=planeMask, numColors = 51))



                segmentation_deep = np.argmax(preds_s[0][0], 2)
                segmentation_deep[segmentation_deep == numOutputPlanes] = -1
                segmentation_deep += 1
                
                plane_depths_deep = calcPlaneDepths(preds_p[0][0], WIDTH, HEIGHT)
                all_depths_deep = np.concatenate([pred_np_d, plane_depths_deep], axis=2)
                pred_d_deep = all_depths_deep.reshape(-1, numOutputPlanes + 1)[np.arange(WIDTH * HEIGHT), segmentation_deep.reshape(-1)].reshape(HEIGHT, WIDTH)
                
                cv2.imwrite(testdir + '/' + str(index) + '_segmentation_pred_0.png', drawSegmentationImage(np.roll(preds_s[0][0], 1, axis=2), black=True))
                cv2.imwrite(testdir + '/' + str(index) + '_depth_pred_0.png', drawDepthImage(pred_d_deep))


                if suffix == 'confidence':
                    pred_s -= np.reshape(np.squeeze(pred_p_c < 0.5), [1, 1, -1]) * 100
                    pass

                #print(pred_np_m)
                #print(pred_s
                #print(planes)
                #print(pred_p)
                #exit(1)
                all_segmentations = np.concatenate([pred_np_m, pred_s], axis=2)
                if suffix == 'deep':
                    pred_p += preds_p[0][0]
                    #all_segmentations += preds_s[0][0]
                    all_segmentations[:, :, 0] += preds_s[0][0][:, :, numOutputPlanes]
                    all_segmentations[:, :, 1:numOutputPlanes + 1] += preds_s[0][0][:, :, :numOutputPlanes]
                    pass
                plane_depths = calcPlaneDepths(pred_p, WIDTH, HEIGHT)
                all_depths = np.concatenate([pred_np_d, plane_depths], axis=2)


                segmentation = np.argmax(all_segmentations, 2)
                if suffix != 'pixelwise':
                    pred_d = all_depths.reshape(-1, numOutputPlanes + 1)[np.arange(WIDTH * HEIGHT), segmentation.reshape(-1)].reshape(HEIGHT, WIDTH)
                else:
                    pred_d = np.squeeze(pred_np_d)
                    cv2.imwrite(testdir + '/' + str(index) + '_normal_pred.png', drawNormalImage(np.clip(pred_np_n, -1, 1)))
                    segmentation = np.zeros(segmentation.shape)
                    pass
                
                
                cv2.imwrite(testdir + '/' + str(index) + '_segmentation_pred.png', drawSegmentationImage(all_segmentations, black=True))
                cv2.imwrite(testdir + '/' + str(index) + '_depth_pred.png', drawDepthImage(pred_d))

                gtDepths.append(gt_d)
                planeMasks.append(planeMask)
                predDepths.append(pred_d)
                evaluateDepths(predDepths[-1], gtDepths[-1], np.ones(planeMasks[-1].shape), planeMasks[-1])
                
                #print(np.concatenate([np.arange(numOutputPlanes).reshape(-1, 1), planes, pred_p, preds_p[0][0]], axis=1))

                #print(np.concatenate([distance[0], preds_p[0][0], pred_p], axis=1))
                            
                #segmentation = np.argmax(pred_s, 2)
                writePLYFile(testdir, index, image, pred_d, segmentation, np.zeros(pred_boundary[0].shape))
                #writePLYFileParts(testdir, index, image, pred_d, segmentation)
                gt_s_ori = gt_s_ori[0]
                cv2.imwrite(testdir + '/' + str(index) + '_segmentation_gt_ori.png', drawSegmentationImage(gt_s_ori, planeMask=np.max(gt_s_ori, 2) > 0.5, numColors=51))

                if False:
                    #distance = distance[0]
                    print(distance)
                    diff = np.linalg.norm(np.expand_dims(planes, 1) - np.expand_dims(pred_p, 0), axis=2)
                    print(np.concatenate([planes, pred_p, pow(np.min(diff, axis=1, keepdims=True), 2), np.expand_dims(np.argmin(diff, axis=1), -1)], axis=1))
                    print(pred_p_c)
                    
                    #print(distance[:, 6:8].sum(0))
                    #print(pow(np.linalg.norm(distance[:, :3] - distance[:, 3:6], 2, 1), 2) * 100)
                    #print(test)
                    segmentation = np.argmax(all_segmentations, 2) - 1
                    for planeIndex in xrange(numOutputPlanes):
                        cv2.imwrite(testdir + '/segmentation_' + str(planeIndex) + '.png', drawMaskImage(segmentation == planeIndex))
                        cv2.imwrite(testdir + '/segmentation_' + str(planeIndex) + '_gt.png', drawMaskImage(gt_s[:, :, planeIndex]))
                        cv2.imwrite(testdir + '/segmentation_' + str(planeIndex) + '_gt_ori.png', drawMaskImage(gt_s_ori[:, :, planeIndex]))
                        continue
                    exit(1)
                continue

            predDepths = np.array(predDepths)
            gtDepths = np.array(gtDepths)
            planeMasks = np.array(planeMasks)
            #predMasks = np.array(predMasks)
            evaluateDepths(predDepths, gtDepths, np.ones(planeMasks.shape, dtype=np.bool), planeMasks)
            print(errorSum)
            exit(1)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            pass
        
        # Wait for threads to finish.
        coord.join(threads)
        sess.close()
        pass
    return


def findBadImages(gpu_id, dumpdir, logdir, testdir, keyname, restore, numOutputPlanes=20, useCRF=0, suffix='forward'):
    if not os.path.exists(testdir):
        os.system("mkdir -p %s"%testdir)
        pass
    
    batchSize = 1
    
    min_after_dequeue = 1000

    reader_val = RecordReader()
    filename_queue_val = tf.train.string_input_producer(['../planes_test_1000_450000.tfrecords'], num_epochs=10000)
    img_inp, plane_gt, depth_gt, normal_gt, segmentation_gt_original, boundary_gt, grid_s_gt, grid_p_gt, grid_m_gt, num_planes_gt = reader_val.getBatch(filename_queue_val, numOutputPlanes=numOutputPlanes, batchSize=batchSize, min_after_dequeue=min_after_dequeue, random=False, getLocal=False, getSegmentation=True, suffix=suffix)
    #img_inp, plane_gt, depth_gt, normal_gt, segmentation_gt, boundary_gt, s_8_gt = reader_val.getBatch(filename_queue_val, numOutputPlanes=numOutputPlanes, batchSize=batchSize, min_after_dequeue=min_after_dequeue, random=False, getLocal=True)

    validating_inp = tf.placeholder(tf.bool, shape=[], name='validating_inp')

    plane_pred, plane_confidence_pred, depth_pred, normal_pred, segmentation_pred, non_plane_mask_pred, non_plane_depth_pred, non_plane_normal_pred, boundary_pred, grid_s_pred, grid_p_pred, grid_m_pred, plane_preds, segmentation_preds, refined_segmentation = build_graph(img_inp, img_inp, plane_gt, plane_gt, validating_inp, numOutputPlanes=numOutputPlanes, useCRF=useCRF, is_training=False, suffix=suffix)

    var_to_restore = tf.global_variables()
    
    loss, plane_loss, depth_loss, normal_loss, segmentation_loss, segmentation_gt, boundary_gt, plane_mask_gt, error_mask, dists = build_loss(plane_pred, plane_confidence_pred, depth_pred, normal_pred, segmentation_pred, non_plane_mask_pred, non_plane_depth_pred, non_plane_normal_pred, boundary_pred, grid_s_pred, grid_p_pred, grid_m_pred, plane_preds, segmentation_preds, plane_gt, depth_gt, normal_gt, segmentation_gt_original, boundary_gt, grid_s_gt, grid_p_gt, grid_m_gt, plane_gt, depth_gt, normal_gt, segmentation_gt_original, boundary_gt, grid_s_gt, grid_p_gt, grid_m_gt, validating_inp, numOutputPlanes=numOutputPlanes, gpu_id=gpu_id, useCRF=useCRF, suffix=suffix)   

    
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    badImages = []
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        loader = tf.train.Saver(var_to_restore)
        loader.restore(sess,"%s/%s.ckpt"%(dumpdir, keyname))
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        randomColor = np.random.randint(255, size=(numOutputPlanes + 1, 3)).astype(np.uint8)
        randomColor[0] = 0
        
        try:
            gtDepths = []
            predDepths = []
            planeMasks = []
            #predMasks = []
            
            imageWidth = WIDTH
            imageHeight = HEIGHT
            focalLength = 517.97
            urange = np.arange(imageWidth).reshape(1, -1).repeat(imageHeight, 0) - imageWidth * 0.5
            vrange = np.arange(imageHeight).reshape(-1, 1).repeat(imageWidth, 1) - imageHeight * 0.5
            ranges = np.array([urange / imageWidth * 640 / focalLength, np.ones(urange.shape), -vrange / imageHeight * 480 / focalLength]).transpose([1, 2, 0])


            for index in xrange(1000):
                if index % 10 == 0:
                    print(('image', index))
                t0=time.time()
                #im, planes, depth, normal, boundary, s_8, p_8, b_8, s_16, p_16, b_16, s_32, p_32, b_32 = sess.run([img_inp, plane_gt, depth_gt, normal_gt, boundary_gt, s_8_gt, p_8_gt, b_8_gt, s_16_gt, p_16_gt, b_16_gt, s_32_gt, p_32_gt, b_32_gt])
                
                pred_d, im, depth, normal, gt_plane_mask = sess.run([depth_pred, img_inp, depth_gt, normal_gt, plane_mask_gt], feed_dict = {validating_inp:True})

                im = im[0]
                image = ((im + 0.5) * 255).astype(np.uint8)

                depth = depth.squeeze()
                normal = normal[0]
                
                
                pred_d = pred_d.squeeze()
                #pred_s = pred_s[0]
                #refined_s = refined_s[0]
                #pred_s = 1 / (1 + np.exp(-pred_s))

                planeMask = np.squeeze(gt_plane_mask)

                rms, accuracy = evaluateDepths(pred_d, depth, np.ones(planeMask.shape), planeMask, printInfo=False)
                if rms > 0.8 or accuracy < 0.7:
                    print((len(badImages), rms, accuracy))
                    cv2.imwrite(testdir + '/' + str(len(badImages)) + '_image.png', image)
                    cv2.imwrite(testdir + '/' + str(len(badImages)) + '_depth.png', drawDepthImage(depth))
                    cv2.imwrite(testdir + '/' + str(len(badImages)) + '_normal.png', drawNormalImage(normal))
                    cv2.imwrite(testdir + '/' + str(len(badImages)) + '_depth_pred.png', drawDepthImage(pred_d))
                    badImages.append(index)
                    pass
                continue
            print(badImages)

            exit(1)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            pass
        
        # Wait for threads to finish.
        coord.join(threads)
        sess.close()
        pass
    return

def predict(dumpdir, testdir, keyname, numOutputPlanes, useCRF=0, dataset='SUNCG', numImages=100, suffix='forward'):
    testdir += '_predict'
    if not os.path.exists(testdir):
        os.system("mkdir -p %s"%testdir)
        pass

    batchSize = 1
    img_inp = tf.placeholder(tf.float32,shape=(batchSize,HEIGHT,WIDTH,3),name='img_inp')
    plane_gt=tf.placeholder(tf.float32,shape=(batchSize,numOutputPlanes, 3),name='plane_inp')
    validating_inp = tf.constant(True, tf.bool)
 

    plane_pred, plane_confidence_pred, depth_pred, normal_pred, segmentation_pred, non_plane_mask_pred, non_plane_depth_pred, non_plane_normal_pred, boundary_pred, grid_s_pred, grid_p_pred, grid_m_pred, plane_preds, segmentation_preds, refined_segmentation = build_graph(img_inp, img_inp, plane_gt, plane_gt, validating_inp, numOutputPlanes=numOutputPlanes, useCRF=useCRF, is_training=False, suffix=suffix)

    var_to_restore = tf.global_variables()
    
 
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True


    if dataset == 'SUNCG':
        image_list_file = os.path.join('../PythonScripts/SUNCG/image_list_100_tail_500000.txt')
        with open(image_list_file) as f:
            im_names = [{'image': im_name.strip().replace('plane_global.npy', 'mlt.png'), 'depth': im_name.strip().replace('plane_global.npy', 'depth.png'), 'normal': im_name.strip().replace('plane_global.npy', 'norm_camera.png'), 'valid': im_name.strip().replace('plane_global.npy', 'valid.png'), 'plane': im_name.strip()} for im_name in f.readlines()]
            pass
    else:
        im_names = glob.glob('../../Data/NYU_RGBD/*_color.png')
        im_names = [{'image': im_name, 'depth': im_name.replace('color.png', 'depth.png'), 'normal': im_name.replace('color.png', 'norm_camera.png'), 'invalid_mask': im_name.replace('color.png', 'valid.png')} for im_name in im_names]
        pass
      
    if numImages > 0:
        im_names = im_names[:numImages]
        pass

    #if args.imageIndex > 0:
    #im_names = im_names[args.imageIndex:args.imageIndex + 1]
    #pass    

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())


    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        #sess.run(tf.global_variables_initializer())
        saver.restore(sess,"%s/%s.ckpt"%(dumpdir,keyname))


        randomColor = np.random.randint(255, size=(numOutputPlanes + 1, 3)).astype(np.uint8)
        randomColor[0] = 0
        gtDepths = []
        predDepths = []
        segmentationDepths = []
        predDepthsOneHot = []
        planeMasks = []
        predMasks = []

        imageWidth = WIDTH
        imageHeight = HEIGHT
        focalLength = 517.97
        urange = np.arange(imageWidth).reshape(1, -1).repeat(imageHeight, 0) - imageWidth * 0.5
        vrange = np.arange(imageHeight).reshape(-1, 1).repeat(imageWidth, 1) - imageHeight * 0.5
        ranges = np.array([urange / focalLength, np.ones(urange.shape), -vrange / focalLength]).transpose([1, 2, 0])
        
        cv2.imwrite(testdir + '/one.png', np.ones((HEIGHT, WIDTH), dtype=np.uint8) * 255)
        cv2.imwrite(testdir + '/zero.png', np.zeros((HEIGHT, WIDTH), dtype=np.uint8) * 255)
        for index, im_name in enumerate(im_names):
            if index <= -1:
                continue
            print(im_name['image'])
            im = cv2.imread(im_name['image'])
            image = im.astype(np.float32, copy=False)
            image = image / 255 - 0.5
            image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)

            #planes = np.load(im_name['plane'])
            # numPlanes = planes.shape[0]
            # if numPlanes > numOutputPlanes:
            #     planeAreas = planes[:, 3:].sum(1)
            #     sortInds = np.argsort(planeAreas)[::-1]
            #     planes = planes[sortInds[:numOutputPlanes]]
            #     pass
            # gt_p = np.zeros((1, numOutputPlanes, 3))
            # gt_p[0, :numPlanes] = planes[:numPlanes, :3]

            normal = np.array(PIL.Image.open(im_name['normal'])).astype(np.float) / 255 * 2 - 1
            norm = np.linalg.norm(normal, 2, 2)
            for c in xrange(3):
                normal[:, :, c] /= norm
                continue
            normal = cv2.resize(normal, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
            
            depth = np.array(PIL.Image.open(im_name['depth'])).astype(np.float) / 1000
            depth = cv2.resize(depth, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)

            invalid_mask = cv2.resize(cv2.imread(im_name['invalid_mask'], 0), (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR) > 128

            gtDepths.append(depth)

            
            pred_p, pred_d, pred_n, pred_s, pred_np_m, pred_np_d, pred_np_n, pred_boundary, pred_grid_s, pred_grid_p, pred_grid_m = sess.run([plane_pred, depth_pred, normal_pred, segmentation_pred, non_plane_mask_pred, non_plane_depth_pred, non_plane_normal_pred, boundary_pred, grid_s_pred, grid_p_pred, grid_m_pred], feed_dict = {img_inp:np.expand_dims(image, 0), plane_gt: np.zeros((batchSize, numOutputPlanes, 3))})

            if True:
                depth = global_pred['non_plane_depth'].squeeze()
                cv2.imwrite(options.test_dir + '/' + str(index) + '_image.png', image)
                cv2.imwrite(options.test_dir + '/' + str(index) + '_depth.png', drawDepthImage(depth))
                planes, planeSegmentation, depthPred = fitPlanes(depth, numPlanes=20)
                cv2.imwrite(options.test_dir + '/' + str(index) + '_segmentation_pred.png', drawSegmentationImage(planeSegmentation))
                cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_pred.png', drawDepthImage(depthPred))

                gtDepths.append(global_gt['depth'].squeeze())
                predDepths.append(depthPred)
                planeMasks.append((planeSegmentation < 20).astype(np.float32))
                continue
            

            pred_s = pred_s[0] 
            pred_p = pred_p[0]
            pred_np_m = pred_np_m[0]
            pred_np_d = pred_np_d[0]
            pred_np_n = pred_np_n[0]
            #pred_s = 1 / (1 + np.exp(-pred_s))

            plane_depths = calcPlaneDepths(pred_p, WIDTH, HEIGHT)
            all_depths = np.concatenate([pred_np_d, plane_depths], axis=2)

            all_segmentations = np.concatenate([pred_np_m, pred_s], axis=2)
            segmentation = np.argmax(all_segmentations, 2)
            if suffix != 'pixelwise':
                pred_d = all_depths.reshape(-1, numOutputPlanes + 1)[np.arange(WIDTH * HEIGHT), segmentation.reshape(-1)].reshape(HEIGHT, WIDTH)
            else:
                pred_d = np.squeeze(pred_np_d)
                pass
            predDepths.append(pred_d)
            predMasks.append(segmentation != 0)
            planeMasks.append(invalid_mask)

            #depthError, normalError, occupancy, segmentationTest, reconstructedDepth, occupancyMask = evaluatePlanes(pred_p, im_name['image'])
            #reconstructedDepth = cv2.resize(reconstructedDepth, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
            
            #evaluatePlanes(pred_p[0], im_name, testdir, index)
            #print(pred_p)
            #print(gt_p)
            #print((pow(pred_d[0, :, :, 0] - depth, 2) * (gt_s.max(2) > 0.5)).mean())
            #print((depthError, normalError, occupancy))
            
            evaluateDepths(predDepths[index], gtDepths[index], np.ones(planeMasks[index].shape), planeMasks[index])

            if index >= 10:
                continue
            cv2.imwrite(testdir + '/' + str(index) + '_image.png', cv2.resize(im, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR))
            #cv2.imwrite(testdir + '/' + str(index) + '_depth_gt.png', (minDepth / np.clip(depth, minDepth, 20) * 255).astype(np.uint8))
            #cv2.imwrite(testdir + '/' + str(index) + '_depth_pred.png', (minDepth / np.clip(pred_d[0, :, :, 0], minDepth, 20) * 255).astype(np.uint8))
            #cv2.imwrite(testdir + '/' + str(index) + '_depth_plane.png', (minDepth / np.clip(reconstructedDepth, minDepth, 20) * 255).astype(np.uint8))

            pred_boundary = pred_boundary[0]
            boundary = (1 / (1 + np.exp(-pred_boundary)) * 255).astype(np.uint8)
            boundary = np.concatenate([boundary, np.zeros((HEIGHT, WIDTH, 1))], axis=2)
            cv2.imwrite(testdir + '/' + str(index) + '_boundary_pred.png', boundary)
            
            cv2.imwrite(testdir + '/' + str(index) + '_depth_inp.png', drawDepthImage(depth))
            cv2.imwrite(testdir + '/' + str(index) + '_depth_pred.png', drawDepthImage(pred_d))
            #cv2.imwrite(testdir + '/' + str(index) + '_depth_plane.png', drawDepthImage(reconstructedDepth))
            cv2.imwrite(testdir + '/' + str(index) + '_depth_pred_diff.png', drawDiffImage(pred_d, depth, 0.5))
            #cv2.imwrite(testdir + '/' + str(index) + '_depth_plane_diff.png', np.minimum(np.abs(reconstructedDepth - depth) / 0.5 * 255, 255).astype(np.uint8))
            cv2.imwrite(testdir + '/' + str(index) + '_normal_inp.png', drawNormalImage(normal))
            cv2.imwrite(testdir + '/' + str(index) + '_normal_pred.png', drawNormalImage(pred_np_n))
            cv2.imwrite(testdir + '/' + str(index) + '_segmentation_pred.png', drawSegmentationImage(all_segmentations, black=True))

            segmentation = np.argmax(pred_s, 2)
            #writePLYFile(testdir, index, image, pred_p, segmentation)

            if index < 0:
                for planeIndex in xrange(numOutputPlanes):
                    cv2.imwrite(testdir + '/' + str(index) + '_segmentation_' + str(planeIndex) + '.png', drawMaskImage(pred_s[:, :, planeIndex]))
                    #cv2.imwrite(testdir + '/' + str(index) + '_segmentation_' + str(planeIndex) + '_gt.png', drawMaskImage(gt_s[:, :, planeIndex]))
                    continue
                pass
            continue
        predDepths = np.array(predDepths)
        gtDepths = np.array(gtDepths)
        planeMasks = np.array(planeMasks)
        predMasks = np.array(predMasks)
        #evaluateDepths(predDepths, gtDepths, planeMasks, predMasks)
        print(planeMasks.shape)
        print(np.ones(planeMasks.shape, dtype=np.bool).sum())
        evaluateDepths(predDepths, gtDepths, np.ones(planeMasks.shape, dtype=np.bool), planeMasks)
        #exit(1)
        pass
    return

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Planenet')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--task', dest='task',
                        help='task type',
                        default='train', type=str)
    parser.add_argument('--restore', dest='restore',
                        help='how to restore',
                        default=0, type=int)
    parser.add_argument('--numOutputPlanes', dest='numOutputPlanes',
                        help='the number of output planes',
                        default=20, type=int)
    parser.add_argument('--batchSize', dest='batchSize',
                        help='batch size',
                        default=16, type=int)
    parser.add_argument('--useCRF', dest='useCRF',
                        help='the number of CRF iterations',
                        default=0, type=int)
    parser.add_argument('--suffix', dest='suffix',
                        help='keyname suffix',
                        default='', type=str)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset name',
                        default='NYUV2', type=str)
    parser.add_argument('--numImages', dest='numImages',
                        help='the number of images to test',
                        default=100, type=int)
    #if len(sys.argv) == 1:
    #parser.print_help()     

    args = parser.parse_args()
    args.keyname = os.path.basename(__file__).rstrip('.py')
    if args.numOutputPlanes != 20:
        args.keyname += '_' + str(args.numOutputPlanes)
        pass
    if args.useCRF > 0:
        args.suffix = 'crf_' + str(args.useCRF)
        pass
    if args.suffix != '':
        args.keyname += '_' + args.suffix
        pass
    args.dumpdir = args.keyname.replace('train', 'dump')
    args.logdir = args.keyname.replace('train', 'log')
    args.testdir = args.keyname.replace('train', 'test')
    #if args.useCRF > 0 and args.task == 'predict':
    #args.testdir += '_crf_' + str(1)
    #pass
    return args


if __name__=='__main__':
    args = parse_args()

    print "dumpdir=%s task=%s started"%(args.dumpdir, args.task)
    #fetchworker=BatchFetcher()    
    try:
        if args.task == "train":
            main(args.gpu_id, args.dumpdir, args.logdir, args.testdir, args.keyname, args.restore, args.numOutputPlanes, useCRF=args.useCRF, batchSize=args.batchSize, suffix=args.suffix)
        elif args.task == "test":
            test(args.gpu_id, args.dumpdir, args.logdir, args.testdir, args.keyname, args.restore, args.numOutputPlanes, useCRF=args.useCRF, suffix=args.suffix)
        elif args.task == "predict":
            predict(args.dumpdir, args.testdir, args.keyname, args.numOutputPlanes, useCRF=args.useCRF, dataset=args.dataset, numImages=args.numImages, suffix=args.suffix)
        elif args.task == "testCRF":
            testCRF(args.dumpdir, args.testdir, args.keyname)
        elif args.task == "testNearestNeighbors":
            testNearestNeighbors(args.dumpdir, args.testdir, args.keyname)
        elif args.task == "filter":
            findBadImages(args.gpu_id, args.dumpdir, args.logdir, 'bad_images', args.keyname, args.restore, args.numOutputPlanes, useCRF=args.useCRF)
        else:
            assert False,"format wrong"
            pass
    finally:
        pass
        #stop_fetcher()

