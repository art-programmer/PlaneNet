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

from train_sample import build_graph
from planenet import PlaneNet
from RecordReaderAll import *
from SegmentationRefinement import *
from crfasrnn_layer import CrfRnnLayer

np.set_printoptions(precision=2, linewidth=200)


def findFloorPlane(planes, segmentation):
    minZ = 0
    minZPlaneIndex = -1
    minFloorArea = 32 * 24
    for planeIndex, plane in enumerate(planes):
        if plane[2] < 0 and abs(plane[2]) > max(abs(plane[0]), abs(plane[1])) and plane[2] < minZ and (segmentation == planeIndex).sum() > minFloorArea:
            minZPlaneIndex = planeIndex
            minZ = plane[2]
            pass
        continue
    return minZPlaneIndex

def findCornerPoints(plane, depth, mask, axis=2, rectangle=True):
    focalLength = 517.97
    width = depth.shape[1]
    height = depth.shape[0]
    urange = (np.arange(width).reshape(1, -1).repeat(height, 0) - width * 0.5) / width * 640
    vrange = (np.arange(height).reshape(-1, 1).repeat(width, 1) - height * 0.5) / height * 480
    ranges = np.stack([urange / focalLength, np.ones(urange.shape), -vrange / focalLength], axis=2)

    XYZ = ranges * np.expand_dims(depth, -1)
    XYZ = XYZ[mask].reshape(-1, 3)

    maxs = XYZ.max(0)
    mins = XYZ.min(0)

    planeD = np.linalg.norm(plane)
    planeNormal = plane / planeD
    if axis == 2:
        points = np.array([[mins[0], mins[1]], [mins[0], maxs[1]], [maxs[0], mins[1]], [maxs[0], maxs[1]]])
        pointsZ = (planeD - planeNormal[0] * points[:, 0] - planeNormal[1] * points[:, 1]) / planeNormal[2]
        points = np.concatenate([points, np.expand_dims(pointsZ, -1)], axis=1)
        pass
    u = points[:, 0] / points[:, 1] * focalLength / 640 * width + width / 2
    v = -points[:, 2] / points[:, 1] * focalLength / 480 * height + height / 2

    if rectangle:
        minU = u.min()
        maxU = u.max()
        minV = v.min()
        maxV = v.max()
        uv = np.array([[minU, minV], [minU, maxV], [maxU, minV], [maxU, maxV]])
    else:
        uv = np.stack([u, v], axis=1)
        pass
    return uv

def copyTextureTest(options):
    testdir = 'texture_test/'
    for index in xrange(1):
        planes = np.load(testdir + '/planes_' + str(index) + '.npy')
        image = cv2.imread(testdir + '/image_' + str(index) + '.png')
        segmentations = np.load(testdir + '/segmentations_' + str(index) + '.npy')
        segmentation = np.argmax(segmentations, axis=2)
        plane_depths = calcPlaneDepths(planes, WIDTH, HEIGHT)
        
        textureImage = cv2.imread('../textures/texture_0.jpg')
        textureImage = cv2.resize(textureImage, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        floorPlaneIndex = findFloorPlane(planes, segmentation)
        if floorPlaneIndex == -1:
            continue
        mask = segmentation == floorPlaneIndex
        uv = findCornerPoints(planes[floorPlaneIndex], plane_depths[:, :, floorPlaneIndex], mask)
        source_uv = np.array([[0, 0], [0, HEIGHT], [WIDTH, 0], [WIDTH, HEIGHT]])

        h, status = cv2.findHomography(source_uv, uv)
        textureImageWarped = cv2.warpPerspective(textureImage, h, (WIDTH, HEIGHT))
        image[mask] = textureImageWarped[mask]
        cv2.imwrite(testdir + '/' + str(index) + '_texture.png', textureImageWarped)
        cv2.imwrite(testdir + '/' + str(index) + '_result.png', image)
        continue
    return


def copyTexture(options):

    if os.path.exists(options.result_filename) and options.useCache == 1:
        pred_dict = np.load(options.result_filename)
        pred_dict = pred_dict[()]
    else:
        pred_dict = getResults(options)
        np.save(options.result_filename, pred_dict)        
        pass


    texture_image_names = glob.glob('../textures/*.png') + glob.glob('../textures/*.jpg')
    
    for image_index in xrange(options.numImages):
        planes = pred_dict['plane'][image_index]
        segmentation = pred_dict['segmentation'][image_index]
        image = pred_dict['image'][image_index]
        plane_depths = pred_dict['plane_depth'][image_index]

        #writePLYFile(options.test_dir, index, image, pred_d, segmentation, np.zeros(pred_boundary.shape))
        oriWidth = image.shape[1]
        oriHeight = image.shape[0]
        
        for texture_index, texture_image_name in enumerate(texture_image_names):
            textureImage = cv2.imread(texture_image_name)
            #textureImage = cv2.imread('../textures/texture_2.jpg')
            textureImage = cv2.resize(textureImage, (oriWidth, oriHeight), interpolation=cv2.INTER_LINEAR)
            floorPlaneIndex = findFloorPlane(planes, segmentation)
            mask = segmentation == floorPlaneIndex
            #mask = cv2.resize(mask.astype(np.float32), (oriWidth, oriHeight), interpolation=cv2.INTER_LINEAR) > 0.5
            #plane_depths = calcPlaneDepths(pred_p, oriWidth, oriHeight)
            depth = plane_depths[:, :, floorPlaneIndex]
            #depth = cv2.resize(depth, (oriWidth, oriHeight), interpolation=cv2.INTER_LINEAR) > 0.5
            uv = findCornerPoints(planes[floorPlaneIndex], depth, mask)
            print(uv)
            source_uv = np.array([[0, 0], [0, oriHeight], [oriWidth, 0], [oriWidth, oriHeight]])
                
            h, status = cv2.findHomography(source_uv, uv)
            #textureImageWarped = cv2.warpPerspective(textureImage, h, (WIDTH, HEIGHT))
            textureImageWarped = cv2.warpPerspective(textureImage, h, (oriWidth, oriHeight))
            resultImage = image.copy()

            resultImage[mask] = textureImageWarped[mask]
            #cv2.imwrite(options.test_dir + '/' + str(index) + '_texture.png', textureImageWarped)
            cv2.imwrite(options.test_dir + '/' + str(index) + '_result_' + str(texture_index) + '.png', resultImage)
            continue
        continue
    return

    
def getResults(options):

    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass

    checkpoint_prefix = options.rootFolder + '/checkpoint/'


    image_list = glob.glob('testing_images/*.png') + glob.glob('testing_images/*.jpg')
    #print(image_list)
    #exit(1)
    
    method = ('hybrid_hybrid1_bl0_dl0_ll1_sm0', '')
    #method = ('finetuning_hybrid1_ps', '')
    #method = ('planenet_hybrid1_bl0_ll1_ds0_pp_ps', '')
    # left_walls = [0, 5, 6, 11, 18]
    # right_walls = [4, 10, 7, 19]
    # floors = [14]
    # ceilings = []    
    # layout_planes = [ceilings, floors, left_walls + right_walls]

    #method = ('sample_np10_hybrid3_bl0_dl0_hl2_ds0_crfrnn5_sm0', '')
    #method = ('planenet_np10_hybrid3_bl0_dl0_crfrnn-10_sm0', '')
    # left_walls = [0, 5, 6, 11, 18]
    # right_walls = [4, 10]
    # floors = [14]
    # ceilings = []    
    # layout_planes = [ceilings, floors, left_walls + right_walls]
    
    
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

    
    batchSize = 1
    img_inp = tf.placeholder(tf.float32,shape=(batchSize, HEIGHT, WIDTH, 3),name='img_inp')
    training_flag = tf.constant(True, tf.bool)
 

    options.gpu_id = 0
    if 'sample' in options.checkpoint_dir:
        global_pred_dict, _, _ = build_graph_sample(img_inp, img_inp, training_flag, options)
    else:
        global_pred_dict, _, _ = build_graph(img_inp, img_inp, training_flag, options)
        pass

    var_to_restore = tf.global_variables()
    
 
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True


    #im_names = glob.glob('../AdobeImages/*.png') + glob.glob('../AdobeImages/*.jpg')

      
    if options.numImages > 0:
        image_list = image_list[:options.numImages]
        pass

    if options.imageIndex >= 0:
        image_list = [image_list[args.imageIndex:args.imageIndex]]
    pass    


    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())


    pred_dict = {}    
    with tf.Session(config=config) as sess:
        loader = tf.train.Saver()
        #sess.run(tf.global_variables_initializer())
        loader.restore(sess, "%s/checkpoint.ckpt"%(options.checkpoint_dir))
        #saver.restore(sess,"dump_planenet/train_planenet.ckpt")

        images = []
        predPlanes = []
        predSegmentations = []
        predDepths = []        
        predPlaneDepths = []

        # imageWidth = WIDTH
        # imageHeight = HEIGHT
        # focalLength = 517.97
        # urange = np.arange(imageWidth).reshape(1, -1).repeat(imageHeight, 0) - imageWidth * 0.5
        # vrange = np.arange(imageHeight).reshape(-1, 1).repeat(imageWidth, 1) - imageHeight * 0.5
        # ranges = np.array([urange / focalLength, np.ones(urange.shape), -vrange / focalLength]).transpose([1, 2, 0])
        
        for index, image_filename in enumerate(image_list):
            if index <= -1:
                continue
            print(image_filename)
            im = cv2.imread(image_filename)
            im_resized = cv2.resize(im, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(options.test_dir + '/' + str(index) + '_image.png', im)
            #continue
            width_high_res = im.shape[1]
            height_high_res = im.shape[0]
            image = im.astype(np.float32, copy=False)
            image = image / 255 - 0.5
            image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)

            global_pred = sess.run(global_pred_dict, feed_dict={img_inp: np.expand_dims(image, 0)})

            pred_p = global_pred['plane'][0]
            pred_s = global_pred['segmentation'][0]
                
            pred_np_m = global_pred['non_plane_mask'][0]
            pred_np_d = global_pred['non_plane_depth'][0]
            pred_np_n = global_pred['non_plane_normal'][0]
            

            info = np.zeros(info.shape)
            focalLength = estimateFocalLength(im)
            info[0] = focalLength
            info[5] = focalLength
            info[2] = im.shape[1] / 2
            info[6] = im.shape[0] / 2
            info[16] = im.shape[1]
            info[17] = im.shape[0]
            info[10] = 1
            info[15] = 1
            info[18] = 1000
            info[19] = 5
            
            all_segmentations = np.concatenate([pred_s, pred_np_m], axis=2)
            plane_depths = calcPlaneDepths(pred_p, width_high_res, height_high_res, info)

            pred_np_d = np.expand_dims(cv2.resize(pred_np_d.squeeze(), (width_high_res, height_high_res)), -1)
            all_depths = np.concatenate([plane_depths, pred_np_d], axis=2)

            all_segmentations = np.stack([cv2.resize(all_segmentations[:, :, planeIndex], (width_high_res, height_high_res)) for planeIndex in xrange(all_segmentations.shape[-1])], axis=2)
                
            segmentation = np.argmax(all_segmentations, 2)
            pred_d = all_depths.reshape(-1, options.numOutputPlanes + 1)[np.arange(height_high_res * width_high_res), segmentation.reshape(-1)].reshape(height_high_res, width_high_res)
            
            cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_pred.png', drawDepthImage(pred_d))
            cv2.imwrite(options.test_dir + '/' + str(index) + '_segmentation_pred.png', drawSegmentationImage(all_segmentations, black=True))


            images.append(im)
            predDepths.append(pred_d)
            predPlanes.append(pred_p)
            predSegmentations.append(segmentation)
            predPlaneDepths.append(plane_depths)
            continue

        pred_dict['image'] = np.array(images)
        pred_dict['plane'] = np.array(predPlanes)
        pred_dict['segmentation'] = np.array(predSegmentations)
        pred_dict['depth'] = np.array(predDepths)
        pred_dict['plane_depth'] = np.array(predPlaneDepths)
        pass
    return pred_dict



if __name__=='__main__':
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Planenet')
    parser.add_argument('--task', dest='task',
                        help='task type',
                        default='texture', type=str)
    parser.add_argument('--numOutputPlanes', dest='numOutputPlanes',
                        help='the number of output planes',
                        default=20, type=int)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset name',
                        default='NYU_RGBD', type=str)
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
                        default=1, type=int)
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
    args.test_dir = 'evaluate/' + args.task + '/'
    args.visualizeImages = args.numImages
    args.result_filename = args.test_dir + '/results.npy'
    
    # image = cv2.imread('evaluate/layout/ScanNet/hybrid3/22_image.png')
    # focal_length = estimateFocalLength(image)
    # print(focal_length)
    # exit(1)
    copyTexture(args)
