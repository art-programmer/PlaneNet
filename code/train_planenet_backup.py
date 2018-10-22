import tensorflow as tf
import numpy as np
np.set_printoptions(precision=2, linewidth=200)
import cv2
import os
import time
import sys
import tf_nndistance
import argparse
import glob
import PIL
import scipy.ndimage as ndimage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
from plane_utils import *
from modules import *


from planenet import PlaneNet
from RecordReader import *
from RecordReaderRGBD import *
from RecordReaderScanNet import *

#training_flag: toggle dropout and batch normalization mode
#it's true for training and false for validation, testing, prediction
#it also controls which data batch to use (*_train or *_val)


def build_graph(img_inp_train, img_inp_val, img_inp_rgbd_train, img_inp_rgbd_val, training_flag, options):
    with tf.device('/gpu:%d'%options.gpu_id):
        img_inp_rgbd = tf.cond(tf.equal(training_flag % 2, 0), lambda: img_inp_rgbd_train, lambda: img_inp_rgbd_val)
        img_inp = tf.cond(tf.equal(training_flag % 2, 0), lambda: img_inp_train, lambda: img_inp_val)
        img_inp = tf.cond(tf.less(training_flag, 2), lambda: img_inp, lambda: img_inp_rgbd)
        
        net = PlaneNet({'img_inp': img_inp}, is_training=tf.equal(training_flag % 2, 0), options=options)

        #global predictions
        plane_pred = net.layers['plane_pred']
        
        segmentation_pred = net.layers['segmentation_pred']
        non_plane_mask_pred = net.layers['non_plane_mask_pred']
        non_plane_depth_pred = net.layers['non_plane_depth_pred']
        non_plane_normal_pred = net.layers['non_plane_normal_pred']
        
        global_pred_dict = {'plane': plane_pred, 'segmentation': segmentation_pred, 'non_plane_mask': non_plane_mask_pred, 'non_plane_depth': non_plane_depth_pred, 'non_plane_normal': non_plane_normal_pred}

        if options.predictBoundary:
            global_pred_dict['boundary'] = net.layers['boundary_pred']
            pass
        if options.predictConfidence:
            global_pred_dict['confidence'] = net.layers['plane_confidence_pred']
            pass
        
        #local predictions
        if options.predictLocal:
            local_pred_dict = {'score': net.layers['local_score_pred'], 'plane': net.layers['local_plane_pred'], 'mask': net.layers['local_mask_pred']}
        else:
            local_pred_dict = {}
            pass

        
        #deep supervision
        deep_pred_dicts = []
        for layer in options.deepSupervisionLayers:
            pred_dict = {'plane': net.layers[layer+'_plane_pred'], 'segmentation': net.layers[layer+'_segmentation_pred'], 'non_plane_mask': net.layers[layer+'_non_plane_mask_pred']}
            #if options.predictConfidence:
            #pred_dict['confidence'] = net.layers[layer+'_plane_confidence_pred']
            #pass
            deep_pred_dicts.append(pred_dict)
            continue
        pass
    
    return global_pred_dict, local_pred_dict, deep_pred_dicts


def build_loss_rgbd(global_pred_dict, local_pred_dict, deep_pred_dicts, global_gt_dict_train, local_gt_dict_train, global_gt_dict_val, local_gt_dict_val, training_flag, options):
    with tf.device('/gpu:%d'%options.gpu_id):
        global_gt_dict = {}
        for name in global_gt_dict_train.keys():
            global_gt_dict[name] = tf.cond(tf.equal(training_flag % 2, 0), lambda: global_gt_dict_train[name], lambda: global_gt_dict_val[name])
            continue
        local_gt_dict = {}
        for name in local_gt_dict_train.keys():
            local_gt_dict[name] = tf.cond(tf.equal(training_flag % 2, 0), lambda: local_gt_dict_train[name], lambda: local_gt_dict_val[name])
            continue

        #depth loss
        plane_parameters = tf.reshape(global_pred_dict['plane'], (-1, 3))
        plane_depths = planeDepthsModule(plane_parameters, WIDTH, HEIGHT)
        plane_depths = tf.transpose(tf.reshape(plane_depths, [HEIGHT, WIDTH, -1, options.numOutputPlanes]), [2, 0, 1, 3])

        all_segmentations = tf.concat([global_pred_dict['segmentation'], global_pred_dict['non_plane_mask']], axis=3)
        all_segmentations_softmax = tf.nn.softmax(all_segmentations)
        all_depths = tf.concat([plane_depths, global_pred_dict['non_plane_depth']], axis=3)
        validDepthMask = tf.cast(tf.greater(global_gt_dict['depth'], 1e-4), tf.float32)
        depth_loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(all_depths, global_gt_dict['depth']) * all_segmentations_softmax, axis=3, keep_dims=True) * validDepthMask) * 1000

        if options.predictPixelwise == 1:
            depth_loss += tf.reduce_mean(tf.squared_difference(global_pred_dict['non_plane_mask'], global_gt_dict['depth']) * validDepthMask) * 1000
            pass
        
        #non plane mask loss
        segmentation_loss = tf.reduce_mean(tf.slice(all_segmentations_softmax, [0, 0, 0, options.numOutputPlanes], [options.batchSize, HEIGHT, WIDTH, 1])) * 100
        
        l2_losses = tf.add_n([options.l2Weight * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name])
        loss = depth_loss + segmentation_loss + l2_losses
        loss_dict = {'depth': depth_loss, 'plane': tf.constant(0.0), 'segmentation': segmentation_loss}
        debug_dict = {}
        pass
    return loss, loss_dict, debug_dict


def build_loss(global_pred_dict, local_pred_dict, deep_pred_dicts, global_gt_dict_train, local_gt_dict_train, global_gt_dict_val, local_gt_dict_val, training_flag, options):
    
    with tf.device('/gpu:%d'%options.gpu_id):
        global_gt_dict = {}
        for name in global_gt_dict_train.keys():
            global_gt_dict[name] = tf.cond(tf.equal(training_flag % 2, 0), lambda: global_gt_dict_train[name], lambda: global_gt_dict_val[name])
            continue
        local_gt_dict = {}
        for name in local_gt_dict_train.keys():
            local_gt_dict[name] = tf.cond(tf.equal(training_flag % 2, 0), lambda: local_gt_dict_train[name], lambda: local_gt_dict_val[name])
            continue

        normalDotThreshold = np.cos(np.deg2rad(5))
        distanceThreshold = 0.05
        segmentation_gt, plane_mask = fitPlaneMasksModule(global_gt_dict['plane'], global_gt_dict['depth'], global_gt_dict['normal'], width=WIDTH, height=HEIGHT, normalDotThreshold=normalDotThreshold, distanceThreshold=distanceThreshold, closing=True, one_hot=True)
        
        validPlaneMask = tf.cast(tf.less(tf.tile(tf.expand_dims(tf.range(options.numOutputPlanes), 0), [options.batchSize, 1]), tf.expand_dims(global_gt_dict['num_planes'], -1)), tf.float32)        

        
        backwardLossWeight = options.backwardLossWeight
        

        #plane loss and segmentation loss (summation over deep supervisions and final supervision)
        all_pred_dicts = deep_pred_dicts + [global_pred_dict]
        plane_loss = tf.constant(0.0)
        segmentation_loss = tf.constant(0.0)
        plane_confidence_loss = tf.constant(0.0)
        diverse_loss = tf.constant(0.0)
        
        #keep forward map (segmentation gt) from previous supervision so that we can have same matching for all supervisions (options.sameMatching = 1)
        previous_forward_map = None
        previous_segmentation_gt = None
        for pred_index, pred_dict in enumerate(all_pred_dicts):
            if options.sameMatching and pred_index > 0:
                #use matching from previous supervision and map ground truth planes based on the mapping
                forward_map = previous_forward_map

                #number of ground truth mapped for each prediction
                num_matches = tf.transpose(tf.reduce_sum(forward_map, axis=1, keep_dims=True), [0, 2, 1])


                plane_gt_shuffled = tf.transpose(tf.matmul(global_gt_dict['plane'], forward_map, transpose_a=True), [0, 2, 1]) / tf.maximum(num_matches, 1e-4)
                plane_confidence_gt = tf.cast(num_matches > 0.5, tf.float32)
                plane_loss += tf.reduce_mean(tf.squared_difference(pred_dict['plane'], plane_gt_shuffled) * plane_confidence_gt) * 10000

                
                #all segmentations is the concatenation of plane segmentations and non plane mask
                all_segmentations = tf.concat([pred_dict['segmentation'], pred_dict['non_plane_mask']], axis=3)
                segmentation_gt_shuffled = previous_segmentation_gt
                segmentation_loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=all_segmentations, labels=segmentation_gt_shuffled)) * 1000
                
            else:
                #calculate new matching by finding nearest neighbors again
                dists_forward, map_forward, dists_backward, _ = tf_nndistance.nn_distance(global_gt_dict['plane'], pred_dict['plane'])
                dists_forward *= validPlaneMask
                
                dists_forward = tf.reduce_mean(dists_forward)
                dists_backward = tf.reduce_mean(dists_backward)
                plane_loss += (dists_forward + dists_backward * backwardLossWeight) * 10000

                forward_map = tf.one_hot(map_forward, depth=options.numOutputPlanes, axis=-1)
                forward_map *= tf.expand_dims(validPlaneMask, -1)

                previous_forward_map = forward_map
                
                segmentation_gt_shuffled = tf.reshape(tf.matmul(tf.reshape(segmentation_gt, [-1, HEIGHT * WIDTH, options.numOutputPlanes]), forward_map), [-1, HEIGHT, WIDTH, options.numOutputPlanes])
                segmentation_gt_shuffled = tf.concat([segmentation_gt_shuffled, 1 - plane_mask], axis=3)
                previous_segmentation_gt = segmentation_gt_shuffled
                
                all_segmentations = tf.concat([pred_dict['segmentation'], pred_dict['non_plane_mask']], axis=3)
                segmentation_loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=all_segmentations, labels=segmentation_gt_shuffled)) * 1000
                pass

            if options.diverseLoss:
                plane_diff = tf.reduce_sum(tf.pow(tf.expand_dims(pred_dict['plane'], 1) - tf.expand_dims(pred_dict['plane'], 2), 2), axis=3)
                plane_diff = tf.matrix_set_diag(plane_diff, tf.ones((options.batchSize, options.numOutputPlanes)))
                minPlaneDiff = 0.1
                diverse_loss += tf.reduce_mean(tf.clip_by_value(1 - plane_diff / minPlaneDiff, 0, 1)) * 10000
                pass
              
            continue



        #depth loss
        plane_parameters = tf.reshape(global_pred_dict['plane'], (-1, 3))
        plane_depths = planeDepthsModule(plane_parameters, WIDTH, HEIGHT)
        plane_depths = tf.transpose(tf.reshape(plane_depths, [HEIGHT, WIDTH, -1, options.numOutputPlanes]), [2, 0, 1, 3])

        all_segmentations_softmax = tf.nn.softmax(all_segmentations)
        non_plane_depth = global_pred_dict['non_plane_depth']
        all_depths = tf.concat([plane_depths, non_plane_depth], axis=3)
        validDepthMask = tf.cast(tf.greater(global_gt_dict['depth'], 1e-4), tf.float32)
        depth_loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(all_depths, global_gt_dict['depth']) * all_segmentations_softmax, axis=3, keep_dims=True) * validDepthMask) * 1000

        if options.predictPixelwise == 1:
            depth_loss += tf.reduce_mean(tf.squared_difference(global_pred_dict['non_plane_mask'], global_gt_dict['depth']) * validDepthMask) * 1000
            normal_loss = tf.reduce_mean(tf.squared_difference(global_pred_dict['non_plane_normal'], global_gt_dict['normal']) * validDepthMask) * 1000
        else:
            #normal loss for non-plane region
            normal_loss = tf.reduce_mean(tf.squared_difference(global_pred_dict['non_plane_normal'], global_gt_dict['normal']) * (1 - plane_mask)) * 1000
            pass
        

        #local loss
        if options.predictLocal:
            local_score_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=local_pred_dict['score'], multi_class_labels=local_gt_dict['score'], weights=tf.maximum(local_gt_dict['score'] * 10, 1))) * 1000
            local_plane_loss = tf.reduce_mean(tf.squared_difference(local_pred_dict['plane'], local_gt_dict['plane']) * local_gt_dict['score']) * 10000
            local_mask_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=local_pred_dict['mask'], labels=local_gt_dict['mask']) * local_gt_dict['score']) * 10000
        else:
            local_score_loss = tf.constant(0.0)
            local_plane_loss = tf.constant(0.0)
            local_mask_loss = tf.constant(0.0)
            pass
        
        
        #boundary loss
        boundary_loss = tf.constant(0.0)
        #load or calculate boundary ground truth
        if False:
            #load from dataset
            boundary_gt = tf.cond(training_flag, lambda: boundary_gt_train, lambda: boundary_gt_val)
        else:
            #calculate boundary ground truth on-the-fly as the calculation is subject to change
            kernel_size = 3
            padding = (kernel_size - 1) / 2
            neighbor_kernel_array = gaussian(kernel_size, kernel_size)
            neighbor_kernel_array[(kernel_size - 1) / 2][(kernel_size - 1) / 2] = 0
            neighbor_kernel_array /= neighbor_kernel_array.sum()
            neighbor_kernel_array *= -1
            neighbor_kernel_array[(kernel_size - 1) / 2][(kernel_size - 1) / 2] = 1
            neighbor_kernel = tf.constant(neighbor_kernel_array.reshape(-1), shape=neighbor_kernel_array.shape, dtype=tf.float32)
            neighbor_kernel = tf.reshape(neighbor_kernel, [kernel_size, kernel_size, 1, 1])
        
            depth_diff = tf.abs(tf.nn.depthwise_conv2d(global_gt_dict['depth'], neighbor_kernel, strides=[1, 1, 1, 1], padding='VALID'))
            depth_diff = tf.pad(depth_diff, paddings = [[0, 0], [padding, padding], [padding, padding], [0, 0]])
            max_depth_diff = 0.1
            depth_boundary = tf.greater(depth_diff, max_depth_diff)

            normal_diff = tf.norm(tf.nn.depthwise_conv2d(global_gt_dict['normal'], tf.tile(neighbor_kernel, [1, 1, 3, 1]), strides=[1, 1, 1, 1], padding='VALID'), axis=3, keep_dims=True)
            normal_diff = tf.pad(normal_diff, paddings = [[0, 0], [padding, padding], [padding, padding], [0, 0]])
            max_normal_diff = np.sqrt(2 * (1 - np.cos(np.deg2rad(20))))
            normal_boundary = tf.greater(normal_diff, max_normal_diff)

            plane_region = tf.nn.max_pool(plane_mask, ksize=[1, kernel_size, kernel_size, 1], strides=[1, 1, 1, 1], padding='SAME', name='max_pool')
            segmentation_eroded = 1 - tf.nn.max_pool(1 - segmentation_gt, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='max_pool')
            plane_region -= tf.reduce_max(segmentation_eroded, axis=3, keep_dims=True)
            boundary = tf.cast(tf.logical_or(depth_boundary, normal_boundary), tf.float32) * plane_region
            smooth_boundary = tf.cast(tf.logical_and(normal_boundary, tf.less_equal(depth_diff, max_depth_diff)), tf.float32) * plane_region
            boundary_gt = tf.concat([smooth_boundary, boundary - smooth_boundary], axis=3)
            pass


        if options.boundaryLoss == 1:
            all_segmentations_pred = all_segmentations_softmax
            all_segmentations_min = 1 - tf.nn.max_pool(1 - all_segmentations_pred, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
            segmentation_diff = tf.reduce_max(all_segmentations_pred - all_segmentations_min, axis=3, keep_dims=True)

            depth_pred = tf.reduce_sum(tf.multiply(all_depths, all_segmentations_pred), axis=3, keep_dims=True)
            depth_neighbor = tf.nn.depthwise_conv2d(depth_pred, neighbor_kernel, strides=[1, 1, 1, 1], padding='SAME')
            minDepthDiff = 0.02
            depth_diff = tf.clip_by_value(tf.squared_difference(depth_pred, depth_neighbor) - pow(minDepthDiff, 2), 0, 1)

            boundary = tf.reduce_max(boundary_gt, axis=3, keep_dims=True)
            smooth_boundary = tf.slice(boundary_gt, [0, 0, 0, 0], [options.batchSize, HEIGHT, WIDTH, 1])
            smooth_mask = segmentation_diff + boundary - 2 * segmentation_diff * boundary + depth_diff * smooth_boundary
            margin = 0.0
            smooth_mask = tf.nn.relu(smooth_mask - margin)
            boundary_loss += tf.reduce_mean(smooth_mask * plane_mask) * 1000            
            pass
        elif options.boundaryLoss == 2:
            all_segmentations_pred = all_segmentations_softmax
            all_segmentations_min = 1 - tf.nn.max_pool(1 - all_segmentations_pred, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
            segmentation_diff = tf.reduce_max(all_segmentations_pred - all_segmentations_min, axis=3, keep_dims=True)

            depth_pred = tf.reduce_sum(tf.multiply(all_depths, all_segmentations_pred), axis=3, keep_dims=True)
            depth_neighbor = tf.nn.depthwise_conv2d(depth_pred, neighbor_kernel, strides=[1, 1, 1, 1], padding='SAME')
            minDepthDiff = 0.02
            depth_diff = tf.clip_by_value(tf.squared_difference(depth_pred, depth_neighbor) - pow(minDepthDiff, 2), 0, 1)

            boundary = tf.reduce_max(boundary_gt, axis=3, keep_dims=True)
            #smooth_boundary = tf.slice(boundary_gt, [0, 0, 0, 0], [options.batchSize, HEIGHT, WIDTH, 1])
            occlusion_boundary = tf.slice(boundary_gt, [0, 0, 0, 1], [options.batchSize, HEIGHT, WIDTH, 1])
            smooth_mask = segmentation_diff + boundary - 2 * segmentation_diff * boundary + depth_diff * (boundary - occlusion_boundary)
            margin = 0.0
            smooth_mask = tf.nn.relu(smooth_mask - margin)
            boundary_loss += tf.reduce_mean(smooth_mask) * 1000            
            pass
          
        if options.predictBoundary:
            #we predict boundaries directly for post-processing purpose
            boundary_loss += tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=global_pred_dict['boundary'], multi_class_labels=boundary_gt, weights=tf.maximum(global_gt_dict['boundary'] * 3, 1))) * 1000

          
        #regularization
        l2_losses = tf.add_n([options.l2Weight * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name])


        loss = plane_loss + segmentation_loss + depth_loss + normal_loss + plane_confidence_loss + diverse_loss + boundary_loss + local_score_loss + local_plane_loss + local_mask_loss + l2_losses

        #if options.pixelwiseLoss:
        #normal_loss = tf.reduce_mean(tf.squared_difference(global_pred_dict['non_plane_normal'], global_gt_dict['normal'])) * 1000
        #depth_loss = tf.reduce_mean(tf.squared_difference(global_pred_dict['non_plane_depth'], global_gt_dict['depth']) * validDepthMask) * 1000
        #pass

        loss_dict = {'plane': plane_loss, 'segmentation': segmentation_loss, 'depth': depth_loss, 'normal': normal_loss, 'boundary': boundary_loss, 'diverse': diverse_loss, 'confidence': plane_confidence_loss, 'local_score': local_score_loss, 'local_plane': local_plane_loss, 'local_mask': local_mask_loss}
        debug_dict = {'segmentation': segmentation_gt, 'boundary': boundary_gt}
        pass
    return loss, loss_dict, debug_dict
    #return loss, plane_loss, depth_loss + local_score_loss + local_p_loss + local_mask_loss + boundary_loss, forward_loss, backward_loss, segmentation_gt, plane_mask, errorMask, dists
    #return loss, plane_loss, segmentation_loss, loss_p_0, depth_loss, segmentation_test, plane_mask, errorMask, dists


def main(options):
    if not os.path.exists(options.checkpoint_dir):
        os.system("mkdir -p %s"%options.checkpoint_dir)
        pass
    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass
    
    min_after_dequeue = 1000

    reader_train = RecordReader()
    filename_queue_train = tf.train.string_input_producer(['/mnt/vision/SUNCG_plane/planes_test_450000.tfrecords'], num_epochs=10000)
    img_inp_train, global_gt_dict_train, local_gt_dict_train = reader_train.getBatch(filename_queue_train, numOutputPlanes=options.numOutputPlanes, batchSize=options.batchSize, min_after_dequeue=min_after_dequeue, getLocal=True, random=False)

    reader_val = RecordReader()
    filename_queue_val = tf.train.string_input_producer(['/home/chenliu/Projects/Data/SUNCG_plane/planes_test_1000_450000.tfrecords'], num_epochs=10000)
    img_inp_val, global_gt_dict_val, local_gt_dict_val = reader_val.getBatch(filename_queue_val, numOutputPlanes=options.numOutputPlanes, batchSize=options.batchSize, min_after_dequeue=min_after_dequeue, getLocal=True)

    reader_rgbd_train = RecordReaderRGBD()
    filename_queue_rgbd_train = tf.train.string_input_producer(['../planes_nyu_rgbd_train.tfrecords'], num_epochs=10000)
    img_inp_rgbd_train, global_gt_dict_rgbd_train, local_gt_dict_rgbd_train = reader_rgbd_train.getBatch(filename_queue_rgbd_train, numOutputPlanes=options.numOutputPlanes, batchSize=options.batchSize, min_after_dequeue=min_after_dequeue, getLocal=True)

    reader_rgbd_val = RecordReaderRGBD()
    filename_queue_rgbd_val = tf.train.string_input_producer(['../planes_nyu_rgbd_val.tfrecords'], num_epochs=10000)
    img_inp_rgbd_val, global_gt_dict_rgbd_val, local_gt_dict_rgbd_val = reader_rgbd_val.getBatch(filename_queue_rgbd_val, numOutputPlanes=options.numOutputPlanes, batchSize=options.batchSize, min_after_dequeue=min_after_dequeue, getLocal=True)

    
    training_flag = tf.placeholder(tf.int32, shape=[], name='training_flag')
    
    global_pred_dict, local_pred_dict, deep_pred_dicts = build_graph(img_inp_train, img_inp_val, img_inp_rgbd_train, img_inp_rgbd_val, training_flag, options)

    var_to_restore = [v for v in tf.global_variables()]

    
    loss, loss_dict, _ = build_loss(global_pred_dict, local_pred_dict, deep_pred_dicts, global_gt_dict_train, local_gt_dict_train, global_gt_dict_val, local_gt_dict_val, training_flag, options)
    loss_rgbd, loss_dict_rgbd, _ = build_loss_rgbd(global_pred_dict, local_pred_dict, deep_pred_dicts, global_gt_dict_rgbd_train, local_gt_dict_rgbd_train, global_gt_dict_rgbd_val, local_gt_dict_rgbd_val, training_flag, options)
    
    loss = tf.cond(tf.less(training_flag, 2), lambda: loss, lambda: loss_rgbd)

    
    train_writer = tf.summary.FileWriter(options.log_dir + '/train')
    val_writer = tf.summary.FileWriter(options.log_dir + '/val')
    train_writer_rgbd = tf.summary.FileWriter(options.log_dir + '/train_rgbd')
    val_writer_rgbd = tf.summary.FileWriter(options.log_dir + '/val_rgbd')
    writers = [train_writer, val_writer, train_writer_rgbd, val_writer_rgbd]
    
    tf.summary.scalar('loss', loss)
    summary_op = tf.summary.merge_all()

    with tf.variable_scope('statistics'):
        batchno = tf.Variable(0, dtype=tf.int32, trainable=False, name='batchno')
        batchnoinc=batchno.assign(batchno+1)
        pass


    optimizer = tf.train.AdamOptimizer(options.LR)
    train_op = optimizer.minimize(loss, global_step=batchno)

    
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True
    saver=tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if options.restore == 0:
            #fine-tune from DeepLab model
            var_to_restore = [v for v in var_to_restore if 'res5d' not in v.name and 'segmentation' not in v.name and 'plane' not in v.name and 'deep_supervision' not in v.name and 'local' not in v.name and 'boundary' not in v.name and 'degridding' not in v.name and 'res2a_branch2a' not in v.name and 'res2a_branch1' not in v.name]
            pretrained_model_loader = tf.train.Saver(var_to_restore)
            pretrained_model_loader.restore(sess,"../pretrained_models/deeplab_resnet.ckpt")
        elif options.restore == 1:
            #restore the same model from checkpoint
            loader = tf.train.Saver(var_to_restore)
            loader.restore(sess,"%s/checkpoint.ckpt"%(options.checkpoint_dir))
            bno=sess.run(batchno)
            print(bno)
        elif options.restore == 2:            
            #restore the same model from checkpoint but reset batchno to 1
            loader = tf.train.Saver(var_to_restore)
            loader.restore(sess,"%s/checkpoint.ckpt"%(options.checkpoint_dir))
            sess.run(batchno.assign(1))
        elif options.restore == 3:            
            #restore the same model from standard training
            if options.predictBoundary == 1:
                var_to_restore = [v for v in var_to_restore if 'boundary' not in v.name]
                pass            
            if options.predictConfidence == 1:
                var_to_restore = [v for v in var_to_restore if 'confidence' not in v.name]
                pass
            loader = tf.train.Saver(var_to_restore)
            loader.restore(sess,"checkpoint/planenet/checkpoint.ckpt")
            sess.run(batchno.assign(1))
        elif options.restore == 4:
            #fine-tune another model
            var_to_restore = [v for v in var_to_restore if 'res4b22_relu_non_plane' not in v.name]
            loader = tf.train.Saver(var_to_restore)
            loader.restore(sess, options.fineTuningCheckpoint)
            sess.run(batchno.assign(1))
            pass
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        MOVING_AVERAGE_DECAY = 0.99
        ema = [0., 0., 0., 0.]
        ema_acc = [1e-10, 1e-10, 1e-10, 1e-10]
        last_snapshot_time = time.time()
        bno=sess.run(batchno)

        try:
            while bno<300000:
                t0 = time.time()

                batchIndexPeriod = bno % 100
                if batchIndexPeriod == 0:
                    batchType = 1
                elif batchIndexPeriod == 1:
                    if options.hybrid == 0:
                        batchType = 1
                    else:
                        batchType = 3
                        pass
                elif batchIndexPeriod % 10 == 0:
                    if options.hybrid == 0:
                        batchType = 0
                    else:
                        batchType = 2
                        pass
                else:
                    batchType = 0
                    pass


                _, total_loss, losses, losses_rgbd, summary_str = sess.run([train_op, loss, loss_dict, loss_dict_rgbd, summary_op], feed_dict = {training_flag: batchType})
                writers[batchType].add_summary(summary_str, bno)
                ema[batchType] = ema[batchType] * MOVING_AVERAGE_DECAY + total_loss
                ema_acc[batchType] = ema_acc[batchType] * MOVING_AVERAGE_DECAY + 1

                bno = sess.run(batchno)
                if time.time()-last_snapshot_time > 900:
                    print('save snapshot')
                    saver.save(sess,'%s/checkpoint.ckpt'%options.checkpoint_dir)
                    last_snapshot_time = time.time()
                    pass
        
                print bno,'train', ema[0] / ema_acc[0], 'val', ema[1] / ema_acc[1], 'train rgbd', ema[2] / ema_acc[2], 'val rgbd', ema[3] / ema_acc[3], 'loss', total_loss, 'time', time.time()-t0

                if np.random.random() < 0.01:
                    if batchType < 2:
                        print(losses)
                    else:
                        print(losses_rgbd)
                        pass
                    pass                
                continue

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


def test(options):
    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass
    
    options.batchSize = 1
    min_after_dequeue = 1000

    if options.dataset == 'SUNCG':
        reader = RecordReader()
        filename_queue = tf.train.string_input_producer(['/home/chenliu/Projects/Data/SUNCG_plane/planes_test_1000_450000.tfrecords'], num_epochs=10000)
        img_inp, global_gt_dict, local_gt_dict = reader.getBatch(filename_queue, numOutputPlanes=options.numOutputPlanes, batchSize=options.batchSize, min_after_dequeue=min_after_dequeue, getLocal=True, random=False)
    else:
        reader = RecordReaderRGBD()
        filename_queue = tf.train.string_input_producer(['../planes_nyu_rgbd_val.tfrecords'], num_epochs=1)
        img_inp, global_gt_dict, local_gt_dict = reader.getBatch(filename_queue, numOutputPlanes=options.numOutputPlanes, batchSize=options.batchSize, min_after_dequeue=min_after_dequeue, getLocal=True, random=False)

        options.deepSupervision = 0
        options.predictLocal = 0
        pass

    training_flag = tf.constant(1, tf.int32)

    global_pred_dict, local_pred_dict, deep_pred_dicts = build_graph(img_inp, img_inp, img_inp, img_inp, training_flag, options)
    var_to_restore = tf.global_variables()

    if options.dataset == 'SUNCG':
        loss, loss_dict, debug_dict = build_loss(global_pred_dict, local_pred_dict, deep_pred_dicts, global_gt_dict, local_gt_dict, global_gt_dict, local_gt_dict, training_flag, options)
    else:
        loss, loss_dict, debug_dict = build_loss_rgbd(global_pred_dict, local_pred_dict, deep_pred_dicts, global_gt_dict, local_gt_dict, global_gt_dict, local_gt_dict, training_flag, options)
        pass

    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        #var_to_restore = [v for v in var_to_restore if 'res4b22_relu_non_plane' not in v.name]
        loader = tf.train.Saver(var_to_restore)
        loader.restore(sess, "%s/checkpoint.ckpt"%(options.checkpoint_dir))
        #loader.restore(sess, options.fineTuningCheckpoint)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        
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


            for index in xrange(10):
                print(('image', index))
                t0=time.time()

                img, global_gt, local_gt, global_pred, local_pred, deep_preds, total_loss, losses, debug = sess.run([img_inp, global_gt_dict, local_gt_dict, global_pred_dict, local_pred_dict, deep_pred_dicts, loss, loss_dict, debug_dict])
                if False:
                    image = ((img[0] + 0.5) * 255).astype(np.uint8)
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
                print((losses['plane'], losses['segmentation'], losses['depth']))
                if index >= 10:
                    continue
                  
                im = img[0]
                image = ((im + 0.5) * 255).astype(np.uint8)                

                gt_d = global_gt['depth'].squeeze()
                
                if 'normal' in global_gt:
                    gt_n = global_gt['normal'][0]
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_normal.png', drawNormalImage(gt_n))
                    pass
                  

                if options.predictLocal:
                    pred_local_s = 1 / (1 + np.exp(-local_pred['score'][0]))
                    pred_local_p = local_pred['plane'][0]
                    pred_local_m = local_pred['mask'][0]
                  
                    gt_local_s = local_gt['score'][0]
                    gt_local_m = local_gt['mask'][0]

                    #visualize local plane prediction
                    stride = 8
                    boxSize = 64
                    xs = np.arange(WIDTH / stride) * stride + stride / 2
                    ys = np.arange(HEIGHT / stride) * stride + stride / 2
                    padding = boxSize / 2 + 1
                    maskImage = np.zeros((HEIGHT + padding * 2, WIDTH + padding * 2, 3), dtype=np.uint8)
                    maskImage[padding:padding + HEIGHT, padding:padding + WIDTH, :] = image / 2
                    for gridY, y in enumerate(ys):
                        for gridX, x in enumerate(xs):
                            score = pred_local_s[gridY][gridX]
                            if score < 0.5:
                                continue                              
                            mask = pred_local_m[gridY][gridX].reshape([16, 16])        
                            mask = cv2.resize(mask, (boxSize, boxSize))
                            maskImage[y - boxSize / 2 + padding:y + boxSize / 2 + padding, x - boxSize / 2 + padding:x + boxSize / 2 + padding, 0][mask > 0.5] = 255
                            continue
                        continue
                    for gridY, y in enumerate(ys):
                        for gridX, x in enumerate(xs):
                            score = gt_local_s[gridY][gridX]
                            if score < 0.5:
                                continue                              
                            mask = gt_local_m[gridY][gridX].reshape([16, 16])        
                            mask = cv2.resize(mask, (boxSize, boxSize))
                            maskImage[y - boxSize / 2 + padding:y + boxSize / 2 + padding, x - boxSize / 2 + padding:x + boxSize / 2 + padding, 2][mask > 0.5] = 255
                            continue
                        continue
                    pass

                pred_p = global_pred['plane'][0]
                pred_s = global_pred['segmentation'][0]

                pred_np_m = global_pred['non_plane_mask'][0]
                pred_np_d = global_pred['non_plane_depth'][0]
                pred_np_n = global_pred['non_plane_normal'][0]
                
                if 'plane_mask' in debug:
                    planeMask = np.squeeze(debug['plane_mask'])
                else:
                    #planeMask = np.ones((HEIGHT, WIDTH))
                    planeMask = (np.max(pred_s, axis=2) > pred_np_m.squeeze()).astype(np.float32)
                    pass
                
                if 'segmentation' in global_gt:
                    gt_s = global_gt['segmentation'][0]
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_segmentation_gt.png', drawSegmentationImage(gt_s, planeMask=planeMask, numColors = 51))
                    pass
                  


                if options.predictConfidence:
                    pred_p_c = global_pred['confidence'][0]
                    pred_p_c = 1 / (1 + np.exp(-pred_p_c))
                    numPlanes = global_gt['num_planes'][0]
                    print((numPlanes, (pred_p_c > 0.5).sum()))
                    pass
                  
                if False:
                    #dump results for post processing
                    if index >= 10:
                        break
                    np.save(options.dump_dir + '/planes_' + str(index) + '.npy', pred_p)
                    np.save(options.dump_dir + '/segmentations_' + str(index) + '.npy', pred_s)
                    np.save(options.dump_dir + '/segmentations_gt_' + str(index) + '.npy', gt_s)
                    np.save(options.dump_dir + '/non_plane_depth_' + str(index) + '.npy', pred_np_d)
                    np.save(options.dump_dir + '/non_plane_segmentation_' + str(index) + '.npy', pred_np_m)
                    boundary = pred_boundary[0]
                    boundary = 1 / (1 + np.exp(-boundary))
                    boundary = np.concatenate([boundary, np.zeros((HEIGHT, WIDTH, 1))], axis=2)                    
                    cv2.imwrite(options.dump_dir + '/boundary_' + str(index) + '.png', drawMaskImage(boundary))
                    cv2.imwrite(options.dump_dir + '/image_' + str(index) + '.png', cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR))
                    np.save(options.dump_dir + '/depth_' + str(index) + '.npy', gt_d)
                    continue                

                  
                cv2.imwrite(options.test_dir + '/' + str(index) + '_image.png', image)
                cv2.imwrite(options.test_dir + '/' + str(index) + '_depth.png', drawDepthImage(gt_d))

                if options.predictBoundary:
                    pred_boundary = global_pred['boundary'][0]
                    pred_boundary = 1 / (1 + np.exp(-pred_boundary))
                    boundary = np.concatenate([pred_boundary, np.zeros((HEIGHT, WIDTH, 1))], axis=2)
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_boundary_pred.png', drawMaskImage(boundary))
                    pass

                if 'boundary' in debug:
                    gt_boundary = debug['boundary'][0]
                    boundary = np.concatenate([gt_boundary, np.zeros((HEIGHT, WIDTH, 1))], axis=2)                
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_boundary_gt.png', drawMaskImage(boundary))
                    pass


                if options.deepSupervision >= 1:
                    segmentation_deep = np.argmax(deep_preds[0]['segmentation'][0], 2)
                    segmentation_deep[segmentation_deep == options.numOutputPlanes] = -1
                    segmentation_deep += 1
                
                    plane_depths_deep = calcPlaneDepths(deep_preds[0]['plane'][0], WIDTH, HEIGHT)
                    all_depths_deep = np.concatenate([pred_np_d, plane_depths_deep], axis=2)
                    pred_d_deep = all_depths_deep.reshape(-1, options.numOutputPlanes + 1)[np.arange(WIDTH * HEIGHT), segmentation_deep.reshape(-1)].reshape(HEIGHT, WIDTH)
                
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_segmentation_pred_0.png', drawSegmentationImage(deep_preds[0]['segmentation'][0]))
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_pred_0.png', drawDepthImage(pred_d_deep))
                    pass


                #print(pred_np_m)
                #print(pred_s)
                #print(global_gt['plane'][0])
                #print(pred_p)
                #exit(1)                
                all_segmentations = np.concatenate([pred_np_m, pred_s], axis=2)
                plane_depths = calcPlaneDepths(pred_p, WIDTH, HEIGHT)
                all_depths = np.concatenate([pred_np_d, plane_depths], axis=2)


                segmentation = np.argmax(all_segmentations, 2)
                pred_d = all_depths.reshape(-1, options.numOutputPlanes + 1)[np.arange(WIDTH * HEIGHT), segmentation.reshape(-1)].reshape(HEIGHT, WIDTH)
 
                
                cv2.imwrite(options.test_dir + '/' + str(index) + '_segmentation_pred.png', drawSegmentationImage(all_segmentations, black=True))
                cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_pred.png', drawDepthImage(pred_d))                

                gtDepths.append(gt_d)
                planeMasks.append(planeMask)
                predDepths.append(pred_d)
                evaluateDepths(predDepths[-1], gtDepths[-1], np.ones(planeMasks[-1].shape), planeMasks[-1])
                
                #print(np.concatenate([np.arange(options.numOutputPlanes).reshape(-1, 1), planes, pred_p, preds_p[0][0]], axis=1))

                #print(np.concatenate([distance[0], preds_p[0][0], pred_p], axis=1))
                            
                #segmentation = np.argmax(pred_s, 2)
                #writePLYFile(options.test_dir, index, image, pred_d, segmentation, np.zeros(pred_boundary[0].shape))
                #writePLYFileParts(options.test_dir, index, image, pred_d, segmentation)
                #gt_s_ori = gt_s_ori[0]
                #cv2.imwrite(options.test_dir + '/' + str(index) + '_segmentation_gt_ori.png', drawSegmentationImage(gt_s_ori, planeMask=np.max(gt_s_ori, 2) > 0.5, numColors=51))
                
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
                    for planeIndex in xrange(options.numOutputPlanes):
                        cv2.imwrite(options.test_dir + '/segmentation_' + str(planeIndex) + '.png', drawMaskImage(segmentation == planeIndex))
                        cv2.imwrite(options.test_dir + '/segmentation_' + str(planeIndex) + '_gt.png', drawMaskImage(gt_s[:, :, planeIndex]))
                        cv2.imwrite(options.test_dir + '/segmentation_' + str(planeIndex) + '_gt_ori.png', drawMaskImage(gt_s_ori[:, :, planeIndex]))
                        continue
                    exit(1)
                continue

            predDepths = np.array(predDepths)
            gtDepths = np.array(gtDepths)
            planeMasks = np.array(planeMasks)
            #predMasks = np.array(predMasks)
            evaluateDepths(predDepths, gtDepths, np.ones(planeMasks.shape, dtype=np.bool), planeMasks)
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


def predict(options):
    options.test_dir += '_predict'
    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass

    batchSize = 1
    img_inp = tf.placeholder(tf.float32,shape=(batchSize,HEIGHT,WIDTH,3),name='img_inp')
    plane_gt=tf.placeholder(tf.float32,shape=(batchSize,options.numOutputPlanes, 3),name='plane_inp')
    validating_inp = tf.constant(0, tf.int32)

    global_pred_dict, local_pred_dict, deep_pred_dicts = build_graph(img_inp_train, img_inp_val, img_inp_rgbd_train, img_inp_rgbd_val, training_flag, options)

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
        saver.restore(sess,"%s/%s.ckpt"%(options.checkpoint_dir,keyname))

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
        
        cv2.imwrite(options.test_dir + '/one.png', np.ones((HEIGHT, WIDTH), dtype=np.uint8) * 255)
        cv2.imwrite(options.test_dir + '/zero.png', np.zeros((HEIGHT, WIDTH), dtype=np.uint8) * 255)
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
            # if numPlanes > options.numOutputPlanes:
            #     planeAreas = planes[:, 3:].sum(1)
            #     sortInds = np.argsort(planeAreas)[::-1]
            #     planes = planes[sortInds[:options.numOutputPlanes]]
            #     pass
            # gt_p = np.zeros((1, options.numOutputPlanes, 3))
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

            
            pred_p, pred_d, pred_n, pred_s, pred_np_m, pred_np_d, pred_np_n, pred_boundary, pred_local_score, pred_local_p, pred_local_mask = sess.run([plane_pred, depth_pred, normal_pred, segmentation_pred, non_plane_mask_pred, non_plane_depth_pred, non_plane_normal_pred, boundary_pred, local_score_pred, local_p_pred, local_mask_pred], feed_dict = {img_inp:np.expand_dims(image, 0), plane_gt: np.zeros((batchSize, options.numOutputPlanes, 3))})


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
                pred_d = all_depths.reshape(-1, options.numOutputPlanes + 1)[np.arange(WIDTH * HEIGHT), segmentation.reshape(-1)].reshape(HEIGHT, WIDTH)
            else:
                pred_d = np.squeeze(pred_np_d)
                pass
            predDepths.append(pred_d)
            predMasks.append(segmentation != 0)
            planeMasks.append(invalid_mask)

            #depthError, normalError, occupancy, segmentationTest, reconstructedDepth, occupancyMask = evaluatePlanes(pred_p, im_name['image'])
            #reconstructedDepth = cv2.resize(reconstructedDepth, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
            
            #evaluatePlanes(pred_p[0], im_name, options.test_dir, index)
            #print(pred_p)
            #print(gt_p)
            #print((pow(pred_d[0, :, :, 0] - depth, 2) * (gt_s.max(2) > 0.5)).mean())
            #print((depthError, normalError, occupancy))
            
            evaluateDepths(predDepths[index], gtDepths[index], np.ones(planeMasks[index].shape), planeMasks[index])

            if index >= 10:
                continue
            cv2.imwrite(options.test_dir + '/' + str(index) + '_image.png', cv2.resize(im, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR))
            #cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_gt.png', (minDepth / np.clip(depth, minDepth, 20) * 255).astype(np.uint8))
            #cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_pred.png', (minDepth / np.clip(pred_d[0, :, :, 0], minDepth, 20) * 255).astype(np.uint8))
            #cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_plane.png', (minDepth / np.clip(reconstructedDepth, minDepth, 20) * 255).astype(np.uint8))

            pred_boundary = pred_boundary[0]
            boundary = (1 / (1 + np.exp(-pred_boundary)) * 255).astype(np.uint8)
            boundary = np.concatenate([boundary, np.zeros((HEIGHT, WIDTH, 1))], axis=2)
            cv2.imwrite(options.test_dir + '/' + str(index) + '_boundary_pred.png', boundary)
            
            cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_inp.png', drawDepthImage(depth))
            cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_pred.png', drawDepthImage(pred_d))
            #cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_plane.png', drawDepthImage(reconstructedDepth))
            cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_pred_diff.png', drawDiffImage(pred_d, depth, 0.5))
            #cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_plane_diff.png', np.minimum(np.abs(reconstructedDepth - depth) / 0.5 * 255, 255).astype(np.uint8))
            cv2.imwrite(options.test_dir + '/' + str(index) + '_normal_inp.png', drawNormalImage(normal))
            cv2.imwrite(options.test_dir + '/' + str(index) + '_normal_pred.png', drawNormalImage(pred_np_n))
            cv2.imwrite(options.test_dir + '/' + str(index) + '_segmentation_pred.png', drawSegmentationImage(all_segmentations, black=True))

            segmentation = np.argmax(pred_s, 2)
            #writePLYFile(options.test_dir, index, image, pred_p, segmentation)

            if index < 0:
                for planeIndex in xrange(options.numOutputPlanes):
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_segmentation_' + str(planeIndex) + '.png', drawMaskImage(pred_s[:, :, planeIndex]))
                    #cv2.imwrite(options.test_dir + '/' + str(index) + '_segmentation_' + str(planeIndex) + '_gt.png', drawMaskImage(gt_s[:, :, planeIndex]))
                    continue
                pass
            continue
        predDepths = np.array(predDepths)
        gtDepths = np.array(gtDepths)
        planeMasks = np.array(planeMasks)
        predMasks = np.array(predMasks)
        #evaluateDepths(predDepths, gtDepths, planeMasks, predMasks)
        evaluateDepths(predDepths, gtDepths, planeMasks, planeMasks)
        #exit(1)
        pass
    return


def fitPlanesRGBD(options):
    writeHTMLRGBD('../results/RANSAC_RGBD/index.html', 10)
    exit(1)
    if not os.path.exists(options.checkpoint_dir):
        os.system("mkdir -p %s"%options.checkpoint_dir)
        pass
    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass
    
    min_after_dequeue = 1000

    reader_rgbd = RecordReaderRGBD()
    filename_queue_rgbd = tf.train.string_input_producer(['../planes_nyu_rgbd_train.tfrecords'], num_epochs=1)
    img_inp_rgbd, global_gt_dict_rgbd, local_gt_dict_rgbd = reader_rgbd.getBatch(filename_queue_rgbd, numOutputPlanes=options.numOutputPlanes, batchSize=1, min_after_dequeue=min_after_dequeue, getLocal=True)

    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session(config=config) as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            gtDepths = []
            predDepths = []
            planeMasks = []
            for index in xrange(10):
                image, depth, path = sess.run([img_inp_rgbd, global_gt_dict_rgbd['depth'], global_gt_dict_rgbd['path']])
                image = ((image[0] + 0.5) * 255).astype(np.uint8)
                depth = depth.squeeze()
                
                cv2.imwrite(options.test_dir + '/' + str(index) + '_image.png', image)
                cv2.imwrite(options.test_dir + '/' + str(index) + '_depth.png', drawDepthImage(depth))
                #cv2.imwrite(options.test_dir + '/' + str(index) + '_mask.png', drawMaskImage(depth == 0))
                planes, planeSegmentation, depthPred = fitPlanes(depth, numPlanes=20)                
                cv2.imwrite(options.test_dir + '/' + str(index) + '_segmentation_pred.png', drawSegmentationImage(planeSegmentation))
                cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_pred.png', drawDepthImage(depthPred))

                gtDepths.append(depth)
                predDepths.append(depthPred)
                planeMasks.append((planeSegmentation < 20).astype(np.float32))
                continue
            predDepths = np.array(predDepths)
            gtDepths = np.array(gtDepths)
            planeMasks = np.array(planeMasks)
            evaluateDepths(predDepths, gtDepths, np.ones(planeMasks.shape, dtype=np.bool), planeMasks)            
        except tf.errors.OutOfRangeError:
            print('done fitting')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
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
    #task: [train, test, predict]
    parser.add_argument('--task', dest='task',
                        help='task type: [train, test, predict]',
                        default='train', type=str)
    parser.add_argument('--restore', dest='restore',
                        help='how to restore the model',
                        default=0, type=int)
    parser.add_argument('--numOutputPlanes', dest='numOutputPlanes',
                        help='the number of output planes',
                        default=20, type=int)
    parser.add_argument('--batchSize', dest='batchSize',
                        help='batch size',
                        default=16, type=int)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset name for test/predict',
                        default='SUNCG', type=str)
    parser.add_argument('--numImages', dest='numImages',
                        help='the number of images to test/predict',
                        default=100, type=int)
    parser.add_argument('--boundaryLoss', dest='boundaryLoss',
                        help='use boundary loss: [0, 1]',
                        default=1, type=int)
    parser.add_argument('--diverseLoss', dest='diverseLoss',
                        help='use diverse loss: [0, 1]',
                        default=1, type=int)
    parser.add_argument('--deepSupervision', dest='deepSupervision',
                        help='deep supervision level: [0, 1, 2]',
                        default=1, type=int)
    parser.add_argument('--sameMatching', dest='sameMatching',
                        help='use the matching for all deep supervision layers and the final prediction: [0, 1]',
                        default=1, type=int)    
    parser.add_argument('--crf', dest='crf',
                        help='the number of CRF iterations',
                        default=0, type=int)
    parser.add_argument('--backwardLossWeight', dest='backwardLossWeight',
                        help='backward matching loss',
                        default=0, type=float)
    parser.add_argument('--predictBoundary', dest='predictBoundary',
                        help='whether predict boundary or not: [0, 1]',
                        default=0, type=int)
    parser.add_argument('--predictLocal', dest='predictLocal',
                        help='whether predict local planes or not: [0, 1]',
                        default=0, type=int)
    parser.add_argument('--predictConfidence', dest='predictConfidence',
                        help='whether predict plane confidence or not: [0, 1]',
                        default=0, type=int)
    parser.add_argument('--predictPixelwise', dest='predictPixelwise',
                        help='whether predict pixelwise depth or not: [0, 1]',
                        default=0, type=int)    
    parser.add_argument('--fineTuningCheckpoint', dest='fineTuningCheckpoint',
                        help='specify the model for fine-tuning',
                        default='../PlaneSetGeneration/dump_planenet_diverse/train_planenet_diverse.ckpt', type=str)
    parser.add_argument('--suffix', dest='suffix',
                        help='add a suffix to keyname to distinguish experiments',
                        default='', type=str)
    parser.add_argument('--l2Weight', dest='l2Weight',
                        help='L2 regulation weight',
                        default=5e-4, type=float)
    parser.add_argument('--LR', dest='LR',
                        help='learning rate',
                        default=3e-5, type=float)
    parser.add_argument('--hybrid', dest='hybrid',
                        help='hybrid training',
                        default=0, type=int)
    

    args = parser.parse_args()
    args.keyname = os.path.basename(__file__).rstrip('.py')
    args.keyname = args.keyname.replace('train_', '')

    if args.numOutputPlanes != 20:
        args.keyname += '_np' + str(args.numOutputPlanes)
        pass
    if args.boundaryLoss != 1:
        args.keyname += '_bl' + str(args.boundaryLoss)
        pass
    if args.diverseLoss == 0:
        args.keyname += '_dl0'
        pass
    if args.deepSupervision != 1:
        args.keyname += '_ds' + str(args.deepSupervision)
        pass
    if args.crf > 0:
        args.keyname += '_crf' + str(args.crf)
        pass
    if args.backwardLossWeight > 0:
        args.keyname += '_bw'
        pass    
    if args.predictBoundary == 1:
        args.keyname += '_pb'
        pass
    if args.predictLocal == 1:
        args.keyname += '_pl'
        pass
    if args.predictConfidence == 1:
        args.keyname += '_pc'
        pass    
    if args.predictPixelwise == 1:
        args.keyname += '_pp'
        pass    
    if args.sameMatching == 0:
        args.keyname += '_sm0'
        pass
    if args.hybrid > 0:
        args.keyname += '_hybrid' + str(args.hybrid)
        pass
    
    args.checkpoint_dir = 'checkpoint/' + args.keyname
    args.log_dir = 'log/' + args.keyname
    args.test_dir = 'test/' + args.keyname + '_' + args.dataset
    args.predict_dir = 'predict/' + args.keyname + '_' + args.dataset
    args.dump_dir = 'dump/' + args.keyname
    
    #layers where deep supervision happens
    args.deepSupervisionLayers = []
    if args.deepSupervision >= 1:
        args.deepSupervisionLayers.append('res4b22_relu')
        pass
    if args.deepSupervision >= 2:
        args.deepSupervisionLayers.append('res4b12_relu')
        pass
    return args


if __name__=='__main__':
    args = parse_args()

    print "keyname=%s task=%s started"%(args.keyname, args.task)
    try:
        if args.task == "train":
            main(args)
        elif args.task == "test":
            test(args)
        elif args.task == "predict":
            predict(args)
        elif args.task == "fit":
            fitPlanesRGBD(args)
        else:
            assert False,"format wrong"
            pass
    finally:
        pass

