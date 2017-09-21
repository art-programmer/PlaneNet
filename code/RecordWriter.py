import tensorflow as tf
import numpy as np
import cv2
import random
import PIL.Image
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules import *
from utils import *
from RecordReaderV5 import *

from planenet import PlaneNet
import tf_nndistance

np.set_printoptions(precision=2, linewidth=200)


HEIGHT=192
WIDTH=256
NUM_PLANES = 20

MOVING_AVERAGE_DECAY = 0.99
                          
deepSupervisionLayers=['res4b22_relu']

def build_graph(img_inp_train, img_inp_val, plane_gt_train, plane_gt_val, validating_inp, is_training=True, numOutputPlanes=20, gpu_id = 0, without_segmentation=False, without_plane=False, without_depth=False, useCRF=0):
    
    with tf.device('/gpu:%d'%gpu_id):
        training_flag = tf.logical_not(validating_inp)
        #training_flag = tf.convert_to_tensor(True, dtype='bool', name='is_training')
        #training_flag = tf.convert_to_tensor(is_training, dtype='bool', name='is_training')
        
        img_inp = tf.cond(validating_inp, lambda: img_inp_val, lambda: img_inp_train)
        plane_gt = tf.cond(validating_inp, lambda: plane_gt_val, lambda: plane_gt_train)

        net = PlaneNet({'img_inp': img_inp}, is_training=training_flag, numGlobalPlanes=numOutputPlanes, deepSupervisionLayers=deepSupervisionLayers)
        segmentation_pred = net.layers['segmentation_pred']
        plane_pred = net.layers['plane_pred']
        boundary_pred = net.layers['boundary_pred']
        grid_s_pred = net.layers['s_8_pred']
        grid_p_pred = net.layers['p_8_pred']
        grid_m_pred = net.layers['m_8_pred']


        # dists_forward, map_forward, dists_backward, map_backward = tf_nndistance.nn_distance(plane_gt, plane_pred)
        # plane_map = tf.one_hot(map_forward, depth=numOutputPlanes, axis=-1)
        # plane_pred = tf.transpose(tf.matmul(plane_gt, plane_map, transpose_a=True), [0, 2, 1])
        #plane_pred = tf.tile(tf.slice(plane_gt, [0, 11, 0], [int(plane_gt.shape[0]), 1, 3]), [1, numOutputPlanes, 1])
        
        if not is_training:
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
      
    return plane_pred, depth_pred, normal_pred, segmentation_pred, boundary_pred, grid_s_pred, grid_p_pred, grid_m_pred, plane_preds, segmentation_preds, segmentation

def build_loss(plane_pred, depth_pred, normal_pred, segmentation_pred, boundary_pred, grid_s_pred, grid_p_pred, grid_m_pred, plane_preds, segmentation_preds, plane_gt_train, depth_gt_train, normal_gt_train, segmentation_gt_train, boundary_gt_train, grid_s_gt_train, grid_p_gt_train, grid_m_gt_train, plane_gt_val, depth_gt_val, normal_gt_val, segmentation_gt_val, boundary_gt_val, grid_s_gt_val, grid_p_gt_val, grid_m_gt_val, validating_inp, numOutputPlanes = 20, gpu_id = 0, without_segmentation=False, without_depth=False, useCRF=False):
    with tf.device('/gpu:%d'%gpu_id):
        plane_gt = tf.cond(validating_inp, lambda: plane_gt_val, lambda: plane_gt_train)
        depth_gt = tf.cond(validating_inp, lambda: depth_gt_val, lambda: depth_gt_train)
        normal_gt = tf.cond(validating_inp, lambda: normal_gt_val, lambda: normal_gt_train)
        boundary_gt = tf.cond(validating_inp, lambda: boundary_gt_val, lambda: boundary_gt_train)
        grid_s_gt = tf.cond(validating_inp, lambda: grid_s_gt_val, lambda: grid_s_gt_train)
        grid_p_gt = tf.cond(validating_inp, lambda: grid_p_gt_val, lambda: grid_p_gt_train)
        grid_m_gt = tf.cond(validating_inp, lambda: grid_m_gt_val, lambda: grid_m_gt_train)
        
        dists_forward, map_forward, dists_backward, map_backward = tf_nndistance.nn_distance(plane_gt, plane_pred)
        plane_map = tf.one_hot(map_forward, depth=numOutputPlanes, axis=-1)
        shuffled_planes = tf.transpose(tf.matmul(plane_pred, plane_map, transpose_a=True, transpose_b=True), [0, 2, 1])
        dists = tf.concat([plane_gt, shuffled_planes, tf.expand_dims(dists_forward, -1), tf.expand_dims(dists_backward, -1), tf.expand_dims(tf.cast(map_forward, tf.float32), -1), tf.expand_dims(tf.cast(map_backward, tf.float32), -1)], axis=2)

        
        useBackward = 0
        
        dists_forward = tf.reduce_mean(dists_forward)
        dists_backward = tf.reduce_mean(dists_backward)
        plane_loss = (dists_forward + dists_backward / 2.0 * useBackward) * 10000

        forward_loss = dists_forward * 10000
        backward_loss = dists_backward / 2.0 * 10000

        
        if False:
            segmentation_gt = tf.cond(validating_inp, lambda: segmentation_gt_val, lambda: segmentation_gt_train)
        else:
            normalDotThreshold = np.cos(np.deg2rad(1))
            distanceThreshold = 0.1

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
            pass
        

        segmentation_map = tf.one_hot(map_forward, depth=numOutputPlanes, axis=-1)
        segmentation_gt_shuffled = tf.reshape(tf.matmul(tf.reshape(segmentation_gt, [-1, HEIGHT * WIDTH, numOutputPlanes]), segmentation_map), [-1, HEIGHT, WIDTH, numOutputPlanes])
        segmentation_gt_shuffled = tf.cast(segmentation_gt_shuffled > 0.5, tf.float32)
        segmentation_test = segmentation_gt_shuffled
        
        segmentation_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=segmentation_pred, labels=segmentation_gt_shuffled)) * 1000

        plane_mask = tf.reduce_max(segmentation_gt, axis=3)
        plane_mask = tf.expand_dims(plane_mask, -1)

        errorMask = tf.zeros(depth_gt.shape)
        if not without_depth and False:
            depth_loss = tf.reduce_mean(tf.squared_difference(depth_pred, depth_gt) * plane_mask) * 1000
            normal_loss = tf.reduce_mean(tf.squared_difference(normal_pred, normal_gt) * plane_mask) * 1000
        else:
            if False:
                depth_loss = 0
            else:
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
                depth_loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(plane_depths, depth_gt) * segmentation, axis=3, keep_dims=True) * plane_mask) * 1000
                pass

            plane_normals = planeNormalsModule(plane_parameters, WIDTH, HEIGHT)
            plane_normals = tf.reshape(plane_normals, [-1, 1, 1, numOutputPlanes, 3])
            #normal_pred = tf.reduce_sum(tf.multiply(plane_normals, tf.expand_dims(segmentation, -1)), axis=3)
            
            #normal_loss = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(plane_normals, tf.expand_dims(normal_gt, 3)) * tf.expand_dims(segmentation, -1), axis=3) * plane_mask) * 1000
            normal_loss = 0
            
            if useCRF:
                kernel_size = 5
                padding = (kernel_size - 1) / 2
                neighbor_kernel_array = gaussian(kernel_size, kernel_size)
                neighbor_kernel_array[(kernel_size - 1) / 2][(kernel_size - 1) / 2] = 0
                neighbor_kernel_array /= neighbor_kernel_array.sum()
                #neighbor_kernel_array *= -1
                #neighbor_kernel_array[(kernel_size - 1) / 2][(kernel_size - 1) / 2] = 1
                neighbor_kernel = tf.constant(neighbor_kernel_array.reshape(-1), shape=neighbor_kernel_array.shape, dtype=tf.float32)
                neighbor_kernel = tf.reshape(neighbor_kernel, [kernel_size, kernel_size, 1, 1])

                #DS_diff = tf.clip_by_value(tf.abs(plane_depths - depth_pred), 0, 1)
                #DS_diff = tf.clip_by_value(tf.abs(plane_depths - depth_pred), 0, 1) * tf.cast(tf.greater(tf.sigmoid(tf.slice(boundary_pred, [0, 0, 0, 0], [int(boundary_pred.shape[0]), HEIGHT, WIDTH, 1])),  0.5), tf.float32)
                #DS = tf.nn.depthwise_conv2d(DS_diff, tf.tile(neighbor_kernel, [1, 1, numOutputPlanes, 1]), strides=[1, 1, 1, 1], padding='VALID')
                #DS = tf.pad(DS, paddings = [[0, 0], [padding, padding], [padding, padding], [0, 0]])


                max_depth_diff = 0.2
                #depth_diff_cost = 0.2

                depth_neighbor = tf.nn.depthwise_conv2d(depth_gt, neighbor_kernel, strides=[1, 1, 1, 1], padding='VALID')
                depth_neighbor = tf.pad(depth_neighbor, paddings = [[0, 0], [padding, padding], [padding, padding], [0, 0]])
                #depth_diff = tf.abs(plane_depths - depth_neighbor)
                #depth_diff = tf.clip_by_value(tf.pow(depth_diff / max_depth_diff, 2), 0, 1)
                depth_diff = tf.reduce_sum(tf.squared_difference(plane_depths, depth_neighbor) * segmentation, axis=3, keep_dims=True)

                neighbor_kernel_array *= -1
                neighbor_kernel_array[(kernel_size - 1) / 2][(kernel_size - 1) / 2] = 1
                neighbor_kernel = tf.constant(neighbor_kernel_array.reshape(-1), shape=neighbor_kernel_array.shape, dtype=tf.float32)
                neighbor_kernel = tf.reshape(neighbor_kernel, [kernel_size, kernel_size, 1, 1])                
                #segmentation = tf.one_hot(tf.argmax(segmentation, 3), depth=numOutputPlanes)
                segmentation_diff = tf.nn.depthwise_conv2d(segmentation, tf.tile(neighbor_kernel, [1, 1, numOutputPlanes, 1]), strides=[1, 1, 1, 1], padding='VALID')
                segmentation_diff = tf.pad(segmentation_diff, paddings = [[0, 0], [padding, padding], [padding, padding], [0, 0]])
                segmentation_diff = tf.clip_by_value(tf.reduce_sum(tf.abs(segmentation_diff), axis=3, keep_dims=True), 0, 1)
                
                #smooth_boundary = tf.sigmoid(tf.slice(boundary_pred, [0, 0, 0, 0], [int(boundary_pred.shape[0]), HEIGHT, WIDTH, 1]))
                #occlusion_boundary = tf.sigmoid(tf.slice(boundary_pred, [0, 0, 0, 1], [int(boundary_pred.shape[0]), HEIGHT, WIDTH, 1]))
                smooth_boundary = tf.slice(boundary_gt, [0, 0, 0, 0], [int(boundary_gt.shape[0]), HEIGHT, WIDTH, 1])
                occlusion_boundary = tf.slice(boundary_gt, [0, 0, 0, 1], [int(boundary_gt.shape[0]), HEIGHT, WIDTH, 1])
                smooth_mask = depth_diff * smooth_boundary + segmentation_diff * tf.clip_by_value(1 - smooth_boundary - occlusion_boundary, 0, 1)
                #smooth_mask = segmentation_diff * tf.clip_by_value(1 - smooth_boundary - occlusion_boundary, 0, 1)
                #smooth_mask = tf.clip_by_value(1 - smooth_boundary - occlusion_boundary, 0, 1)
                #smooth_mask = depth_diff * smooth_boundary
                smooth_loss = tf.reduce_mean(smooth_mask) * 10000
                
                errorMask = smooth_mask
                #errorMask = (1 - tf.sigmoid(tf.slice(boundary_pred, [0, 0, 0, 1], [int(boundary_pred.shape[0]), HEIGHT, WIDTH, 1])))
            else:
                smooth_loss = tf.constant(0.0)
            pass
        
        #s_8_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=s_8_pred, labels=s_8_gt)) * 1000
        grid_s_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=grid_s_pred, multi_class_labels=grid_s_gt, weights=tf.maximum(grid_s_gt * 10, 1))) * 1000
        grid_p_loss = tf.reduce_mean(tf.squared_difference(grid_p_pred, grid_p_gt) * grid_s_gt) * 10000
        grid_m_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=grid_m_pred, labels=grid_m_gt) * grid_s_gt) * 10000
        boundary_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=boundary_pred, multi_class_labels=boundary_gt, weights=tf.maximum(boundary_gt * 5, 1))) * 1000
        
        l2_losses = tf.add_n([5e-4 * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name])
        #loss = plane_loss + segmentation_loss + depth_loss + normal_loss + l2_losses
        loss = plane_loss + segmentation_loss + depth_loss + normal_loss + l2_losses + grid_s_loss + grid_p_loss + grid_m_loss + boundary_loss + smooth_loss
        #loss = plane_loss + segmentation_loss + depth_loss + normal_loss + l2_losses
        pass


    if True:
        for layer, pred_p in enumerate(plane_preds):
            dists_forward, map_forward, dists_backward, _ = tf_nndistance.nn_distance(plane_gt, pred_p)
            plane_map = tf.one_hot(map_forward, depth=numOutputPlanes, axis=-1)
            shuffled_planes = tf.transpose(tf.matmul(pred_p, plane_map, transpose_a=True, transpose_b=True), [0, 2, 1])
            #dists = tf.concat([dists, shuffled_planes, tf.expand_dims(dists_forward, -1)], axis=2)            

            dists_forward = tf.reduce_mean(dists_forward)
            dists_backward = tf.reduce_mean(dists_backward)
            loss += (dists_forward + dists_backward / 2.0 * useBackward) * 10000
            
            loss_p_0 = (dists_forward + dists_backward / 2.0 * useBackward) * 10000

            segmentation_map = tf.one_hot(map_forward, depth=numOutputPlanes, axis=-1)
            segmentation_gt_shuffled = tf.reshape(tf.matmul(tf.reshape(segmentation_gt, [-1, HEIGHT * WIDTH, numOutputPlanes]), segmentation_map), [-1, HEIGHT, WIDTH, numOutputPlanes])
            segmentation_gt_shuffled = tf.cast(segmentation_gt_shuffled > 0.5, tf.float32)
            
            pred_s = segmentation_preds[layer]
            loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_s, labels=segmentation_gt_shuffled)) * 1000
            loss_s_0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_s, labels=segmentation_gt_shuffled)) * 1000
            pass
        pass

    return loss, plane_loss, segmentation_loss + depth_loss + normal_loss + grid_s_loss + grid_p_loss + grid_m_loss, boundary_loss, smooth_loss, segmentation_gt, plane_mask, errorMask, dists


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def writeRecordFile(split):
    
    batchSize = 8
    numOutputPlanes = 20
    if split == 'train':
        reader_train = RecordReader()
        filename_queue_train = tf.train.string_input_producer(['../planes_new_450000.tfrecords'], num_epochs=1)
        img_inp, plane_gt, depth_gt, normal_gt, plane_mask_gt, boundary_gt, grid_s_gt, grid_p_gt, grid_m_gt, num_planes_gt, image_path_inp = reader_train.getBatch(filename_queue_train, batchSize=batchSize, random=False)
        writer = tf.python_io.TFRecordWriter('../planes_temp_450000.tfrecords')
    else:
        reader_val = RecordReader()
        filename_queue_val = tf.train.string_input_producer(['../planes_new_1000_450000.tfrecords'], num_epochs=1)
        img_inp, plane_gt, depth_gt, normal_gt, plane_mask_gt, boundary_gt, grid_s_gt, grid_p_gt, grid_m_gt, num_planes_gt, image_path_inp = reader_val.getBatch(filename_queue_val, batchSize=batchSize, random=False)
        writer = tf.python_io.TFRecordWriter('../planes_temp_1000_450000.tfrecords')
        pass


    validating_inp = tf.placeholder(tf.bool, shape=[], name='validating_inp')
    plane_pred, depth_pred, normal_pred, segmentation_pred, boundary_pred, grid_s_pred, grid_p_pred, grid_m_pred, plane_preds, segmentation_preds, refined_segmentation = build_graph(img_inp, img_inp, plane_gt, plane_gt, validating_inp, without_segmentation=False, without_plane=False, useCRF=False, is_training=False)
    var_to_restore = tf.global_variables()    
    loss, plane_loss, depth_loss, normal_loss, segmentation_loss, segmentation_gt, mask_gt, error_mask, dists = build_loss(plane_pred, depth_pred, normal_pred, segmentation_pred, boundary_pred, grid_s_pred, grid_p_pred, grid_m_pred, plane_preds, segmentation_preds, tf.slice(plane_gt, [0, 0, 0], [batchSize, numOutputPlanes, 3]), depth_gt, normal_gt, plane_mask_gt, boundary_gt, grid_s_gt, grid_p_gt, grid_m_gt, tf.slice(plane_gt, [0, 0, 0], [batchSize, numOutputPlanes, 3]), depth_gt, normal_gt, plane_mask_gt, boundary_gt, grid_s_gt, grid_p_gt, grid_m_gt, validating_inp, gpu_id=0, without_segmentation=False, useCRF=False, without_depth=True)   
    
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    testdir = 'test_all_resnet_v2/'
    
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        loader = tf.train.Saver(var_to_restore)
        loader.restore(sess,"dump_supervision_deeplab_forward/train_supervision_deeplab_forward.ckpt")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for _ in xrange(100000):
                print(_)
                images, global_p, depths, normals, global_m, boundaries, grid_s, grid_p, grid_m, num_planes, image_paths, pred_d, gt_mask = sess.run([img_inp, plane_gt, depth_gt, normal_gt, plane_mask_gt, boundary_gt, grid_s_gt, grid_p_gt, grid_m_gt, num_planes_gt, image_path_inp, depth_pred, mask_gt], feed_dict = {validating_inp:True})
                for batchIndex in xrange(batchSize):
                    # print(global_p[batchIndex])
                    # print(original_p[batchIndex])
                    # exit(1)

                    image = ((images[batchIndex] + 0.5) * 255).astype(np.uint8)
                    img_raw = image.tostring()
                    depth = depths[batchIndex]
                    normal = normals[batchIndex]
                    numPlanes = int(num_planes[batchIndex])
                    planes = global_p[batchIndex]
                    planeMask = global_m[batchIndex]
                    boundary = boundaries[batchIndex]
                    boundary_raw = (boundary * 255).astype(np.uint8).tostring()
                    
                    grid_scores = grid_s[batchIndex]
                    grid_planes = grid_p[batchIndex]
                    grid_masks = grid_m[batchIndex]
                    
                    grid_masks = (grid_masks > 0.5).astype(np.uint8)
                    grid_masks_raw = grid_masks.reshape(-1).tostring()
                    image_path = image_paths[batchIndex]

                    rms, accuracy = evaluateDepths(pred_d[batchIndex], depth, np.ones(gt_mask[batchIndex].shape), gt_mask[batchIndex], printInfo=False)
                    if (rms < 0.8 and accuracy > 0.7) or (rms < 1 and accuracy > 0.85):
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'num_planes': _int64_feature([numPlanes]),
                            'image_raw': _bytes_feature(img_raw),
                            'image_path': _bytes_feature(image_path),
                            'normal': _float_feature(normal.reshape(-1)),
                            'depth': _float_feature(depth.reshape(-1)),
                            #'invalid_mask_raw': _bytes_feature(invalid_mask_raw),
                            'plane': _float_feature(planes.reshape(-1)),
                            'plane_mask': _int64_feature(planeMask.reshape(-1)),
                            'boundary_raw': _bytes_feature(boundary_raw),
                            #'local_box': _float_feature(boxes.reshape(-1)),
                            'grid_s': _float_feature(grid_scores.astype(np.float).reshape(-1)),
                            'grid_p': _float_feature(grid_planes.astype(np.float).reshape(-1)),
                            'grid_m_raw': _bytes_feature(grid_masks_raw),
                        }))
                        writer.write(example.SerializeToString())
                        pass
                    continue
                continue
            pass
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            pass
        
        # Wait for threads to finish.
        coord.join(threads)
        sess.close()    
    return

    
if __name__=='__main__':
    writeRecordFile('train')
