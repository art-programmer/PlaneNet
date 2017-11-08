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


def getResults(options):
    checkpoint_prefix = options.rootFolder + '/checkpoint/'

    
    method = ('hybrid_hybrid1_bl0_dl0_ll1_sm0', '')
    
    if 'ds0' not in method[0]:
        options.deepSupervisionLayers = ['res4b22_relu', ]
    else:
        options.deepSupervisionLayers = []
        pass
    options.predictConfidence = 0
    options.predictLocal = 0
    options.predictPixelwise = 1
    options.predictBoundary = int('pb' in method[0])
    options.anchorPlanes = 0
    options.predictSemantics = 0
    options.batchSize = 1

    if 'crfrnn' in method[0]:
        options.crfrnn = 10
    else:
        options.crfrnn = 0
        pass    
    if 'ap1' in method[0]:
        options.anchorPlanes = 1
        pass
        
    options.checkpoint_dir = checkpoint_prefix + method[0]
    print(options.checkpoint_dir)
        
    options.suffix = method[1]


    pred_dict = getPrediction(options)
    #np.save(options.test_dir + '/curves.npy', curves)
    return

def getPrediction(options):
    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass

    
    tf.reset_default_graph()
    

    #image_list = glob.glob('/home/chenliu/Projects/Data/LSUN/images/*.jpg')
    image_list = glob.glob('/mnt/vision/NYU_RGBD/images/*.png')
    
    training_flag = tf.constant(False, tf.bool)

    img_inp = tf.placeholder(tf.float32, shape=[1, HEIGHT, WIDTH, 3], name='image')
    
    options.gpu_id = 0
    global_pred_dict, local_pred_dict, deep_pred_dicts = build_graph(img_inp, img_inp, training_flag, options)

    var_to_restore = tf.global_variables()


    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())


    info = np.zeros(20)
    info[0] = 5.1885790117450188e+02
    info[2] = 3.2558244941119034e+02
    info[5] = 5.1946961112127485e+02
    info[6] = 2.5373616633400465e+02
    info[10] = 1
    info[15] = 1
    info[16] = 640
    info[17] = 480
    info[18] = 1000
    info[19] = 1
    
    pred_dict = {}
    left_walls = [0, 5, 11]
    right_walls = [4, 10]    
    floors = [14]

    layout_planes = [left_walls, right_walls, floors]
    
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        #var_to_restore = [v for v in var_to_restore if 'res4b22_relu_non_plane' not in v.name]
        loader = tf.train.Saver(var_to_restore)
        loader.restore(sess, "%s/checkpoint.ckpt"%(options.checkpoint_dir))
        #loader.restore(sess, options.fineTuningCheckpoint)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        
        try:
            for index in xrange(options.startIndex + options.numImages):
                if index < options.startIndex:
                    continue                                
                if index % 10 == 0:
                    print(('image', index))
                    pass

                # print(image_list[index])
                # import PIL.Image
                # img = PIL.Image.open(image_list[index])
                # print(img._getexif())
                # print(img.shape)
                # exit(1)
                
                img_ori = cv2.imread(image_list[index])

                img = cv2.resize(img_ori, (WIDTH, HEIGHT))
                img = img.astype(np.float32) / 255 - 0.5
                
                t0=time.time()

                global_pred = sess.run(global_pred_dict, feed_dict={img_inp: np.expand_dims(img, 0)})



                pred_p = global_pred['plane'][0]
                pred_s = global_pred['segmentation'][0]
                
                pred_np_m = global_pred['non_plane_mask'][0]
                pred_np_d = global_pred['non_plane_depth'][0]
                pred_np_n = global_pred['non_plane_normal'][0]

                pred_b = global_pred['boundary'][0]

                
                #all_segmentations = np.concatenate([pred_s, pred_np_m], axis=2)
                all_segmentations = np.concatenate([pred_s], axis=2)
                plane_depths = calcPlaneDepths(pred_p, WIDTH, HEIGHT, info)
                # all_depths = np.concatenate([pred_np_d, plane_depths], axis=2)

                segmentation = np.argmax(all_segmentations, 2)
                # pred_d = all_depths.reshape(-1, options.numOutputPlanes + 1)[np.arange(WIDTH * HEIGHT), segmentation.reshape(-1)].reshape(HEIGHT, WIDTH)
                        

                #print(pred_p)
                if index - options.startIndex < options.visualizeImages:
                    #cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_pred.png', drawDepthImage(pred_d))
                    #cv2.imwrite(options.test_dir + '/' + str(index) + '_segmentation_pred.png', drawSegmentationImage(segmentation, blackIndex=options.numOutputPlanes))
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_image.png', img_ori)

                    layout_plane_depths = []
                    for planeInds in layout_planes:
                        maxArea = 0
                        for planeIndex in planeInds:
                            area = (segmentation == planeIndex).sum()
                            if area > maxArea:
                                layout_plane_index = planeIndex
                                maxArea = area
                                pass
                            continue
                        
                        if maxArea > WIDTH * HEIGHT / 400:
                            layout_plane_depths.append(plane_depths[:, :, layout_plane_index])
                        else:
                            layout_plane_depths.append(np.ones((HEIGHT, WIDTH)) * 10)
                            pass
                        continue

                    layout_plane_depths = np.stack(layout_plane_depths, axis=2)
                    #print(layout_plane_depths.shape)
                    #print(np.argmin(layout_plane_depths, axis=-1).shape)
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_layout_pred.png', drawSegmentationImage(np.argmin(layout_plane_depths, axis=-1)))
                    # for planeIndex in xrange(options.numOutputPlanes):
                    #     cv2.imwrite(options.test_dir + '/mask_' + str(planeIndex) + '.png', drawMaskImage(segmentation == planeIndex))
                    #     continue
                    pass
                #exit(1)
                pass    
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
        pass
    return pred_dict



if __name__=='__main__':
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Planenet')
    parser.add_argument('--task', dest='task',
                        help='task type',
                        default='layout', type=str)
    parser.add_argument('--numOutputPlanes', dest='numOutputPlanes',
                        help='the number of output planes',
                        default=20, type=int)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset name',
                        default='ScanNet', type=str)
    parser.add_argument('--hybrid', dest='hybrid',
                        help='hybrid',
                        default='3', type=str)
    parser.add_argument('--visualizeImages', dest='visualizeImages',
                        help='visualize image',
                        default=30, type=int)    
    parser.add_argument('--numImages', dest='numImages',
                        help='the number of images',
                        default=30, type=int)
    parser.add_argument('--startIndex', dest='startIndex',
                        help='start index',
                        default=0, type=int)    
    parser.add_argument('--useCache', dest='useCache',
                        help='use cache',
                        default=0, type=int)
    # parser.add_argument('--useCRF', dest='useCRF',
    #                     help='use crf',
    #                     default=0, type=int)
    # parser.add_argument('--useSemantics', dest='useSemantics',
    #                     help='use semantics',
    #                     default=0, type=int)
    parser.add_argument('--useNonPlaneDepth', dest='useNonPlaneDepth',
                        help='use non-plane depth',
                        default=0, type=int)
    parser.add_argument('--imageIndex', dest='imageIndex',
                        help='image index',
                        default=-1, type=int)
    parser.add_argument('--methods', dest='methods',
                        help='methods',
                        default='0123', type=str)
    parser.add_argument('--rootFolder', dest='rootFolder',
                        help='root folder',
                        default='/mnt/vision/PlaneNet/', type=str)
    
    args = parser.parse_args()
    #args.hybrid = 'hybrid' + args.hybrid
    args.test_dir = 'evaluate/' + args.task + '/' + args.dataset + '/hybrid' + args.hybrid + '/'
    args.visualizeImages = min(args.visualizeImages, args.numImages)
    if args.imageIndex >= 0:
        args.visualizeImages = 1
        args.numImages = 1            
        pass


    getResults(args)
