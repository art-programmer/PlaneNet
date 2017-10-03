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

from train_planenet import *
from planenet import PlaneNet
from RecordReader import *
from RecordReaderRGBD import *
from RecordReader3D import *


def evaluatePlanePrediction(options):
    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass

    if options.useCache == 1 and os.path.exists(options.test_dir + '/results.npy'):
        results = np.load(options.test_dir + '/results.npy')
        results = results[()]
    else:
        results = getResults(options)
        if options.useCache != -1:
            np.save(options.test_dir + '/results.npy', results)
            pass
        pass
    
    gt_dict = results['gt']
    predictions = results['pred']

    #print(results)


    # depth = gt_dict['depth'][4]
    # cv2.imwrite(options.test_dir + '/test_depth_gt.png', drawDepthImage(depth))
    # pred_p, pred_s, pred_d = fitPlanes(depth, getSUNCGCamera(), numPlanes=20, planeAreaThreshold=3*4, numIterations=100, distanceThreshold=0.05, local=0.2)
    # cv2.imwrite(options.test_dir + '/test_depth.png', drawDepthImage(pred_d))
    # cv2.imwrite(options.test_dir + '/test_segmentation.png', drawSegmentationImage(pred_s))
    # exit(1)
   
    for image_index in xrange(options.visualizeImages):
        cv2.imwrite(options.test_dir + '/' + str(image_index) + '_image.png', gt_dict['image'][image_index])
        cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_gt.png', drawDepthImage(gt_dict['depth'][image_index]))
        cv2.imwrite(options.test_dir + '/' + str(image_index) + '_segmentation_gt.png', drawSegmentationImage(gt_dict['segmentation'][image_index]))

        for method_index, pred_dict in enumerate(predictions):
            cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_dict['depth'][image_index]))
            cv2.imwrite(options.test_dir + '/' + str(image_index) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage(pred_dict['segmentation'][image_index]))
            continue
        continue
    
    metric_curves = [[], [], [], []]
    for method_index, pred_dict in enumerate(predictions):
        if method_index == 1:
            continue
        curves = evaluatePlaneSegmentation(pred_dict['plane'], pred_dict['segmentation'], gt_dict['plane'], gt_dict['segmentation'], gt_dict['num_planes'], planeDistanceThreshold = 0.3, IOUThreshold = 0.5, numOutputPlanes = options.numOutputPlanes)
        for metric_index, curve in enumerate(curves):
            continue
        for metric_index, curve in enumerate(curves):
            metric_curves[metric_index].append(curve)
            continue
        continue

    xs = []
    xs.append((np.arange(11) * 0.1).tolist())
    xs.append((np.arange(11) * 0.1).tolist())
    xs.append((0.5 - np.arange(11) * 0.05).tolist())
    xs.append((0.5 - np.arange(11) * 0.05).tolist())
    xlabels = ['IOU', 'IOU', 'plane diff', 'plane diff']
    ylabels = ['pixel coverage', 'pixel coverage', 'pixel coverage', 'pixel coverage']
    labels = ['planenet', 'pixelwise+RANSAC', 'GT+RANSAC']
    titles = ['plane diff 0.1', 'plane diff 0.3', 'IOU 0.3', 'IOU 0.5']
    for metric_index, curves in enumerate(metric_curves):
        filename = options.test_dir + '/curve_' + titles[metric_index].replace(' ', '_') + '.png'
        plotCurves(xs[metric_index], curves, filename = filename, xlabel=xlabels[metric_index], ylabel=ylabels[metric_index], title=titles[metric_index], labels=labels)
        continue
    return
    
def getResults(options):
    checkpoint_prefix = 'checkpoint/planenet_'

    methods = [('pb_pp_hybrid1', ''), ('pb_pp_hybrid1', 'pixelwise_1'), ('pb_pp_hybrid1', 'pixelwise_2'), ('pb_pp_hybrid1', 'pixelwise_3')]
    gt_dict = getGroundTruth(options)

    
    options.deepSupervisionLayers = ['res4b22_relu', ]
    options.predictConfidence = 0
    options.predictLocal = 0
    options.predictPixelwise = 1
    options.predictBoundary = 1

    predictions = []
    for method_index, method in enumerate(methods):
        options.checkpoint_dir = checkpoint_prefix + method[0]
        options.suffix = method[1]

        pred_dict = getPrediction(options)

        for image_index in xrange(options.visualizeImages):
            cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_dict['depth'][image_index]))
            cv2.imwrite(options.test_dir + '/' + str(image_index) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage(pred_dict['segmentation'][image_index]))
            continue
        
        predictions.append(pred_dict)
        continue
    #np.save(options.test_dir + '/curves.npy', curves)
    results = {'gt': gt_dict, 'pred': predictions}
    return results

def getPrediction(options):
    tf.reset_default_graph()
    
    options.batchSize = 1
    min_after_dequeue = 1000

    if options.dataset == 'SUNCG':
        reader = RecordReader()
        filename_queue = tf.train.string_input_producer(['/home/chenliu/Projects/Data/SUNCG_plane/planes_test_1000_450000.tfrecords'], num_epochs=10000)
        img_inp, global_gt_dict, local_gt_dict = reader.getBatch(filename_queue, numOutputPlanes=options.numOutputPlanes, batchSize=options.batchSize, min_after_dequeue=min_after_dequeue, getLocal=True, random=False)
        camera = getSUNCGCamera()
    elif options.dataset == 'NYU_RGBD':
        reader = RecordReaderRGBD()
        filename_queue = tf.train.string_input_producer(['../planes_nyu_rgbd_val.tfrecords'], num_epochs=1)
        img_inp, global_gt_dict, local_gt_dict = reader.getBatch(filename_queue, numOutputPlanes=options.numOutputPlanes, batchSize=options.batchSize, min_after_dequeue=min_after_dequeue, getLocal=True, random=False)

        options.deepSupervision = 0
        options.predictLocal = 0
        camera = getNYURGBDCamera()
    else:
        reader = RecordReader3D()
        filename_queue = tf.train.string_input_producer(['../planes_scannet_val.tfrecords'], num_epochs=1)
        img_inp, global_gt_dict, local_gt_dict = reader.getBatch(filename_queue, numOutputPlanes=options.numOutputPlanes, batchSize=options.batchSize, min_after_dequeue=min_after_dequeue, getLocal=True, random=False)
        
        options.deepSupervision = 0
        options.predictLocal = 0
        camera = get3DCamera()
        pass

    
    training_flag = tf.constant(1, tf.int32)

    options.gpu_id = 0
    global_pred_dict, local_pred_dict, deep_pred_dicts = build_graph(img_inp, img_inp, img_inp, img_inp, img_inp, img_inp, training_flag, options)

    var_to_restore = tf.global_variables()


    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    pred_dict = {}
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        #var_to_restore = [v for v in var_to_restore if 'res4b22_relu_non_plane' not in v.name]
        loader = tf.train.Saver(var_to_restore)
        loader.restore(sess, "%s/checkpoint.ckpt"%(options.checkpoint_dir))
        #loader.restore(sess, options.fineTuningCheckpoint)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        
        try:
            predDepths = []
            planeMasks = []
            predPlanes = []
            predSegmentations = []

            for index in xrange(options.numImages):
                if index % 10 == 0:
                    print(('image', index))
                    pass
                t0=time.time()

                img, global_gt, global_pred = sess.run([img_inp, global_gt_dict, global_pred_dict])

                if 'pixelwise' in options.suffix:
                    pred_d = global_pred['non_plane_depth'].squeeze()
                    #depth = global_gt['depth'].squeeze()
                    if '_2' in options.suffix:
                        pred_p, pred_s, pred_d = fitPlanes(pred_d, camera, numPlanes=20, planeAreaThreshold=3*4, numIterations=100, distanceThreshold=0.05, local=0.2)
                    elif '_3' in options.suffix:
                        pred_p, pred_s, pred_d = fitPlanes(global_gt['depth'].squeeze(), camera, numPlanes=20, planeAreaThreshold=3*4, numIterations=100, distanceThreshold=0.05, local=0.2)
                    else:
                        pred_p = np.zeros((options.numOutputPlanes, 3))
                        pred_s = np.zeros((HEIGHT, WIDTH, options.numOutputPlanes))
                        pass
                    pass
                else:
                    pred_p = global_pred['plane'][0]
                    pred_s = global_pred['segmentation'][0]
                        
                    pred_np_m = global_pred['non_plane_mask'][0]
                    pred_np_d = global_pred['non_plane_depth'][0]
                    pred_np_n = global_pred['non_plane_normal'][0]

                    all_segmentations = np.concatenate([pred_np_m, pred_s], axis=2)
                    plane_depths = calcPlaneDepths(pred_p, WIDTH, HEIGHT)
                    all_depths = np.concatenate([pred_np_d, plane_depths], axis=2)

                    segmentation = np.argmax(all_segmentations, 2)
                    pred_d = all_depths.reshape(-1, options.numOutputPlanes + 1)[np.arange(WIDTH * HEIGHT), segmentation.reshape(-1)].reshape(HEIGHT, WIDTH)
                    pass
                
                predDepths.append(pred_d)

                if options.dataset != 'NYU_RGBD' or ('_2' in options.suffix or '_3' in options.suffix):
                    predPlanes.append(pred_p)
                    predSegmentations.append(pred_s)
                    pass
                    
                continue
            pred_dict['plane'] = np.array(predPlanes)
            pred_dict['segmentation'] = np.array(predSegmentations)
            pred_dict['depth'] = np.array(predDepths)
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
        pass
    return pred_dict

def getGroundTruth(options):    
    options.batchSize = 1
    min_after_dequeue = 1000

    if options.dataset == 'SUNCG':
        reader = RecordReader()
        filename_queue = tf.train.string_input_producer(['/home/chenliu/Projects/Data/SUNCG_plane/planes_test_1000_450000.tfrecords'], num_epochs=10000)
        img_inp, global_gt_dict, local_gt_dict = reader.getBatch(filename_queue, numOutputPlanes=options.numOutputPlanes, batchSize=options.batchSize, min_after_dequeue=min_after_dequeue, getLocal=True, random=False)
    elif options.dataset == 'NYU_RGBD':
        reader = RecordReaderRGBD()
        filename_queue = tf.train.string_input_producer(['../planes_nyu_rgbd_val.tfrecords'], num_epochs=1)
        img_inp, global_gt_dict, local_gt_dict = reader.getBatch(filename_queue, numOutputPlanes=options.numOutputPlanes, batchSize=options.batchSize, min_after_dequeue=min_after_dequeue, getLocal=True, random=False)

        options.deepSupervision = 0
        options.predictLocal = 0
    else:
        reader = RecordReader3D()
        filename_queue = tf.train.string_input_producer(['../planes_scannet_val.tfrecords'], num_epochs=1)
        img_inp, global_gt_dict, local_gt_dict = reader.getBatch(filename_queue, numOutputPlanes=options.numOutputPlanes, batchSize=options.batchSize, min_after_dequeue=min_after_dequeue, getLocal=True, random=False)
        
        options.deepSupervision = 0
        options.predictLocal = 0        
        pass

    training_flag = tf.constant(1, tf.int32)

    if options.dataset == 'NYU_RGBD':
        global_gt_dict['segmentation'], global_gt_dict['plane_mask'] = tf.ones((options.batchSize, HEIGHT, WIDTH, options.numOutputPlanes)), tf.ones((options.batchSize, HEIGHT, WIDTH, 1))
    else:
        normalDotThreshold = np.cos(np.deg2rad(5))
        distanceThreshold = 0.05        
        global_gt_dict['segmentation'], global_gt_dict['plane_mask'] = fitPlaneMasksModule(global_gt_dict['plane'], global_gt_dict['depth'], global_gt_dict['normal'], width=WIDTH, height=HEIGHT, normalDotThreshold=normalDotThreshold, distanceThreshold=distanceThreshold, closing=True, one_hot=True)
        pass

    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    gt_dict = {}
    
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)        
        
        try:
            gtDepths = []
            planeMasks = []
            #predMasks = []
            gtPlanes = []
            gtSegmentations = []
            gtNumPlanes = []
            images = []

            for index in xrange(options.numImages):
                print(('image', index))
                t0=time.time()

                img, global_gt = sess.run([img_inp, global_gt_dict])

                image = ((img[0] + 0.5) * 255).astype(np.uint8)
                images.append(image)

                gt_d = global_gt['depth'].squeeze()
                gtDepths.append(gt_d)

                planeMask = np.squeeze(global_gt['plane_mask'])                    
                planeMasks.append(planeMask)
                
                if options.dataset != 'NYU_RGBD':
                    gt_p = global_gt['plane'][0]
                    gtPlanes.append(gt_p)
                    
                    gt_s = global_gt['segmentation'][0]
                    gtSegmentations.append(gt_s)
                
                    gt_num_p = global_gt['num_planes'][0]
                    gtNumPlanes.append(gt_num_p)
                else:
                    gtPlanes = []
                    gtSegmentations = []
                    gtNumPlanes = []
                    pass
                                
                continue

            gt_dict['image'] = np.array(images)
            gt_dict['depth'] = np.array(gtDepths)
            gt_dict['plane_mask'] = np.array(planeMasks)
            gt_dict['plane'] = np.array(gtPlanes)
            gt_dict['segmentation'] = np.array(gtSegmentations)
            gt_dict['num_planes'] = np.array(gtNumPlanes)

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
    return gt_dict


if __name__=='__main__':
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Planenet')
    parser.add_argument('--task', dest='task',
                        help='task type',
                        default='plane', type=str)
    parser.add_argument('--numOutputPlanes', dest='numOutputPlanes',
                        help='the number of output planes',
                        default=20, type=int)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset name',
                        default='SUNCG', type=str)
    parser.add_argument('--visualizeImages', dest='visualizeImages',
                        help='visualize image',
                        default=10, type=int)    
    parser.add_argument('--numImages', dest='numImages',
                        help='the number of images',
                        default=100, type=int)
    parser.add_argument('--useCache', dest='useCache',
                        help='use cache',
                        default=0, type=int)
    parser.add_argument('--useCRF', dest='useCRF',
                        help='use crf',
                        default=0, type=int)
    parser.add_argument('--useNonPlaneDepth', dest='useNonPlaneDepth',
                        help='use non-plane depth',
                        default=0, type=int)
    parser.add_argument('--imageIndex', dest='imageIndex',
                        help='image index',
                        default=-1, type=int)
    
    args = parser.parse_args()
    args.test_dir = 'evaluate/' + args.task + '/' + args.dataset + '/'

    if args.task == 'plane':
        evaluatePlanePrediction(args)
        pass
