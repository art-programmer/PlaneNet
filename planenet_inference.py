import tensorflow as tf
import numpy as np
np.set_printoptions(precision=2, linewidth=200)
import cv2
import os
import time
import sys
import argparse
import glob

#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from planenet_utils import calcPlaneDepths, drawSegmentationImage, drawDepthImage
from PlaneNet.utils import calcPlaneDepths, drawSegmentationImage, drawDepthImage

from train_planenet import build_graph, parse_args

WIDTH = 256
HEIGHT = 192

ALL_TITLES = ['PlaneNet']
ALL_METHODS = [('sample_np10_hybrid3_bl0_dl0_ds0_crfrnn5_sm0', '', 0, 2)]

class PlaneNetDetector():
    def __init__(self, batchSize=1):
        tf.reset_default_graph()

        self.img_inp = tf.placeholder(tf.float32, shape=[batchSize, HEIGHT, WIDTH, 3], name='image')
        training_flag = tf.constant(False, tf.bool)

        self.options = parse_args()
        self.global_pred_dict, _, _ = build_graph(self.img_inp, self.img_inp, training_flag, self.options)

        var_to_restore = tf.global_variables()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())


        self.sess = tf.Session(config=config)
        self.sess.run(init_op)
        loader = tf.train.Saver(var_to_restore)
        path = os.path.dirname(os.path.realpath(__file__))
        checkpoint_dir = path + '/checkpoint/sample_np10_hybrid3_bl0_dl0_ds0_crfrnn5_sm0'
        loader.restore(self.sess, "%s/checkpoint.ckpt"%(checkpoint_dir))
        return

    def detect(self, image, estimateFocalLength=False):

        pred_dict = {}
        if True:
            t0 = time.time()

            #image_inp = np.array([cv2.resize(image, (WIDTH, HEIGHT)) for image in images])
            image_inp = np.expand_dims(cv2.resize(image, (WIDTH, HEIGHT)), 0)
            image_inp = image_inp.astype(np.float32) / 255 - 0.5
            global_pred = self.sess.run(self.global_pred_dict, feed_dict={self.img_inp: image_inp})

            pred_p = global_pred['plane'][0]
            pred_s = global_pred['segmentation'][0]    
            pred_np_m = global_pred['non_plane_mask'][0]
            pred_np_d = global_pred['non_plane_depth'][0]
            
            all_segmentations = np.concatenate([pred_s, pred_np_m], axis=2)

            info = np.zeros(20)
            if estimateFocalLength:
                focalLength = estimateFocalLength(img_ori)
                info[0] = focalLength
                info[5] = focalLength
                info[2] = img_ori.shape[1] / 2
                info[6] = img_ori.shape[0] / 2
                info[16] = img_ori.shape[1]
                info[17] = img_ori.shape[0]
                info[10] = 1
                info[15] = 1
                info[18] = 1000
                info[19] = 5
            else:
                info[0] = 571.87
                info[2] = 320
                info[5] = 571.87
                info[6] = 240
                info[16] = 640
                info[17] = 480
                info[10] = 1
                info[15] = 1
                info[18] = 1000
                info[19] = 5
                pass

            #width_high_res = images[0].shape[1]
            #height_high_res = images[0].shape[0]
            width_high_res = 640
            height_high_res = 480
            
            plane_depths = calcPlaneDepths(pred_p, width_high_res, height_high_res, info)

            pred_np_d = np.expand_dims(cv2.resize(pred_np_d.squeeze(), (width_high_res, height_high_res)), -1)
            all_depths = np.concatenate([plane_depths, pred_np_d], axis=2)

            all_segmentations = np.stack([cv2.resize(all_segmentations[:, :, planeIndex], (width_high_res, height_high_res)) for planeIndex in xrange(all_segmentations.shape[-1])], axis=2)
                
            segmentation = np.argmax(all_segmentations, 2)
            pred_d = all_depths.reshape(-1, self.options.numOutputPlanes + 1)[np.arange(height_high_res * width_high_res), segmentation.reshape(-1)].reshape(height_high_res, width_high_res)

            #print(pred_p)
            # for segmentIndex in range(segmentation.max() + 1):
            #     cv2.imwrite('test/mask_' + str(segmentIndex) + '.png', (segmentation == segmentIndex).astype(np.uint8) * 255)
            #     print(all_depths[:, :, segmentIndex].min(), all_depths[:, :, segmentIndex].max())
            #     cv2.imwrite('test/depth_' + str(segmentIndex) + '.png', drawDepthImage(all_depths[:, :, segmentIndex]))
            #     continue            
            pred_dict['plane'] = pred_p
            pred_dict['segmentation'] = segmentation
            pred_dict['depth'] = pred_d
            pred_dict['info'] = info
        else:
            print('prediction failed')
            pass        
        return pred_dict
