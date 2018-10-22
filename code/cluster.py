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
from RecordReaderAll import *
#from SegmentationRefinement import refineSegmentation


def clusterPlanes(options):
    tf.reset_default_graph()
    
    options.batchSize = 1
    min_after_dequeue = 1000

    reader = RecordReaderAll()
    if options.dataset == 'SUNCG':
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_SUNCG_val.tfrecords'], num_epochs=10000)
    elif options.dataset == 'NYU_RGBD':
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_nyu_rgbd_val.tfrecords'], num_epochs=1)
        options.deepSupervision = 0
        options.predictLocal = 0
    elif options.dataset == 'matterport':
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_matterport_val.tfrecords'], num_epochs=1)
    else:
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_scannet_val.tfrecords'], num_epochs=1)
        pass
    
    img_inp, global_gt_dict, local_gt_dict = reader.getBatch(filename_queue, numOutputPlanes=options.numOutputPlanes, batchSize=options.batchSize, min_after_dequeue=min_after_dequeue, getLocal=True, random=False)
        

    
    training_flag = tf.constant(False, tf.bool)

    options.gpu_id = 0
    global_pred_dict, local_pred_dict, deep_pred_dicts = build_graph(img_inp, img_inp, training_flag, options)

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
            predPlanes = []
            for index in xrange(options.numImages):
                if index % 10 == 0:
                    print(('image', index))
                    pass
                t0=time.time()

                img, global_gt, global_pred = sess.run([img_inp, global_gt_dict, global_pred_dict])

                predPlanes.append(global_pred['plane'][0])
                continue
            predPlanes = np.array(predPlanes)
            print(predPlanes.shape)
            predPlanes = np.mean(predPlanes, axis=0)
            np.save('dump/plane.npy', predPlanes)
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
    parser.add_argument('--hybrid', dest='hybrid',
                        help='hybrid',
                        default='0', type=str)
    parser.add_argument('--visualizeImages', dest='visualizeImages',
                        help='visualize image',
                        default=10, type=int)    
    parser.add_argument('--numImages', dest='numImages',
                        help='the number of images',
                        default=10, type=int)
    parser.add_argument('--useCache', dest='useCache',
                        help='use cache',
                        default=1, type=int)
    parser.add_argument('--useCRF', dest='useCRF',
                        help='use crf',
                        default=0, type=int)
    parser.add_argument('--useSemantics', dest='useSemantics',
                        help='use semantics',
                        default=0, type=int)
    parser.add_argument('--useNonPlaneDepth', dest='useNonPlaneDepth',
                        help='use non-plane depth',
                        default=0, type=int)
    parser.add_argument('--imageIndex', dest='imageIndex',
                        help='image index',
                        default=-1, type=int)
    parser.add_argument('--methods', dest='methods',
                        help='methods',
                        default='012345', type=str)
    parser.add_argument('--rootFolder', dest='rootFolder',
                        help='root folder',
                        default='/mnt/vision/PlaneNet/', type=str)
    
    args = parser.parse_args()
    args.hybrid = 'hybrid' + args.hybrid
    args.test_dir = 'evaluate/' + args.task + '/' + args.dataset + '/' + args.hybrid + '/'
    args.visualizeImages = min(args.visualizeImages, args.numImages)
    if args.imageIndex >= 0:
        args.visualizeImages = 1
        args.numImages = 1            
        pass

    if args.dataset == 'SUNCG':
        args.camera = getSUNCGCamera()
    elif args.dataset == 'NYU_RGBD':
        args.camera = getNYURGBDCamera()
    else:
        args.camera = get3DCamera()
        pass


    args.deepSupervisionLayers = ['res4b22_relu', ]
    args.predictConfidence = 0
    args.predictLocal = 0
    args.predictPixelwise = 1
    args.predictBoundary = 1

    checkpoint_prefix = args.rootFolder + '/checkpoint/planenet_'
    args.checkpoint_dir = checkpoint_prefix + args.hybrid + '_pb_pp'
    
    clusterPlanes(args)
    
