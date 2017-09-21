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
from RecordReaderAll import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
from layers import PlaneDepthLayer, PlaneNormalLayer
from modules import *

#from resnet import inference as resnet
#from resnet import fc as fc, conv as conv, block_transpose as block_transpose, conv_transpose as conv_transpose, bn as bn, UPDATE_OPS_COLLECTION
from config import Config
from modules import *
import scipy.ndimage as ndimage
from planenet import PlaneNet
#from SegmentationRefinement import refineSegmentation
from train_planenet import build_graph as build_graph
from SegmentationRefinement import refineSegmentation

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

def copyTextureTest():
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
        
def copyTexture(numOutputPlanes=20, useCRF=0, dataset='SUNCG', numImages=100):
    testdir = 'texture_test/'
    dumpdir = 'texture_dump/'
    if not os.path.exists(testdir):
        os.system("mkdir -p %s"%testdir)
        pass
    if not os.path.exists(dumpdir):
        os.system("mkdir -p %s"%dumpdir)
        pass

    batchSize = 1
    img_inp = tf.placeholder(tf.float32,shape=(batchSize,HEIGHT,WIDTH,3),name='img_inp')
    plane_gt=tf.placeholder(tf.float32,shape=(batchSize,numOutputPlanes, 3),name='plane_inp')
    validating_inp = tf.constant(True, tf.bool)
 

    plane_pred, plane_confidence_pred, depth_pred, normal_pred, segmentation_pred, non_plane_mask_pred, non_plane_depth_pred, non_plane_normal_pred, boundary_pred, grid_s_pred, grid_p_pred, grid_m_pred, plane_preds, segmentation_preds, refined_segmentation = build_graph(img_inp, img_inp, plane_gt, plane_gt, validating_inp, numOutputPlanes=numOutputPlanes, useCRF=useCRF, is_training=False)

    var_to_restore = tf.global_variables()
    
 
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True


    #im_names = glob.glob('../test_images/*.png') + glob.glob('../test_images/*.jpg')
    im_names = glob.glob('../AdobeImages/*.png') + glob.glob('../AdobeImages/*.jpg')
    im_names = [{'image': im_name} for im_name in im_names]
    
    texture_image_names = glob.glob('../textures/*.png') + glob.glob('../textures/*.jpg')
      
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
        saver.restore(sess,"dump_planenet/train_planenet.ckpt")


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
        
        for index, im_name in enumerate(im_names):
            if index <= -1:
                continue
            print(im_name['image'])
            im = cv2.imread(im_name['image'])
            im_resized = cv2.resize(im, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(testdir + '/' + str(index) + '_image.png', im)
            continue
            oriWidth = im.shape[1]
            oriHeight = im.shape[0]
            image = im.astype(np.float32, copy=False)
            image = image / 255 - 0.5
            image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)

            
            pred_p, pred_d, pred_n, pred_s, pred_np_m, pred_np_d, pred_np_n, pred_boundary, pred_grid_s, pred_grid_p, pred_grid_m = sess.run([plane_pred, depth_pred, normal_pred, segmentation_pred, non_plane_mask_pred, non_plane_depth_pred, non_plane_normal_pred, boundary_pred, grid_s_pred, grid_p_pred, grid_m_pred], feed_dict = {img_inp:np.expand_dims(image, 0), plane_gt: np.zeros((batchSize, numOutputPlanes, 3))})


            #if index != 13:
            #continue
              
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
            pred_d = all_depths.reshape(-1, numOutputPlanes + 1)[np.arange(WIDTH * HEIGHT), segmentation.reshape(-1)].reshape(HEIGHT, WIDTH)

            pred_boundary = pred_boundary[0]
            pred_boundary = 1 / (1 + np.exp(-pred_boundary))

            #refined_s = refineSegmentation(pred_s, plane_depths, pred_boundary[:, :, 0], pred_boundary[:, :, 1])
            #cv2.imwrite(testdir + '/' + str(index) + '_segmentation_refined.png', drawSegmentationImage(refined_s))
            #exit(1)
            

            #cv2.imwrite(testdir + '/' + str(index) + '_depth_gt.png', (minDepth / np.clip(depth, minDepth, 20) * 255).astype(np.uint8))
            #cv2.imwrite(testdir + '/' + str(index) + '_depth_pred.png', (minDepth / np.clip(pred_d[0, :, :, 0], minDepth, 20) * 255).astype(np.uint8))
            #cv2.imwrite(testdir + '/' + str(index) + '_depth_plane.png', (minDepth / np.clip(reconstructedDepth, minDepth, 20) * 255).astype(np.uint8))


            cv2.imwrite(testdir + '/' + str(index) + '_depth_pred.png', drawDepthImage(pred_d))
            #cv2.imwrite(testdir + '/' + str(index) + '_normal_pred.png', drawNormalImage(pred_n[0]))
            cv2.imwrite(testdir + '/' + str(index) + '_segmentation_pred.png', drawSegmentationImage(all_segmentations, black=True))
            
            segmentation = np.argmax(all_segmentations, axis=2)
            writePLYFile(testdir, index, image, pred_d, segmentation, np.zeros(pred_boundary.shape))
            
            if index == 12 and False:
                #np.save(testdir + '/planes_' + str(index) + '.npy', pred_p)
                #cv2.imwrite(testdir + '/image_' + str(index) + '.png', im_resized)
                #np.save(testdir + '/segmentations_' + str(index) + '.npy', pred_s)
                
                np.save(testdir + '/' + str(index) + '_segmentation_pred.npy', segmentation)
                np.save(testdir + '/' + str(index) + '_planes.npy', pred_p)
                exit(1)
                np.save(dumpdir + '/planes_' + str(index) + '.npy', pred_p)
                np.save(dumpdir + '/segmentations_' + str(index) + '.npy', pred_s)
                np.save(dumpdir + '/non_plane_depth_' + str(index) + '.npy', pred_np_d)
                np.save(dumpdir + '/non_plane_segmentation_' + str(index) + '.npy', pred_np_m)
                boundary = np.concatenate([pred_boundary, np.zeros((HEIGHT, WIDTH, 1))], axis=2)                    
                cv2.imwrite(dumpdir + '/boundary_' + str(index) + '.png', drawMaskImage(boundary))
                cv2.imwrite(dumpdir + '/image_' + str(index) + '.png', im_resized)
                exit(1)
                continue
                pass

            continue

            planes = pred_p
            segmentations = pred_s
            segmentation = np.argmax(segmentations, axis=2)
            #textureImage = cv2.imread('../textures/texture_0.jpg')
            #textureImage = cv2.imread('../textures/texture_2.jpg')
            for texture_index, texture_image_name in enumerate(texture_image_names):
                textureImage = cv2.imread(texture_image_name)
                #textureImage = cv2.imread('../textures/texture_2.jpg')
                textureImage = cv2.resize(textureImage, (oriWidth, oriHeight), interpolation=cv2.INTER_LINEAR)
                floorPlaneIndex = findFloorPlane(planes, segmentation)
                mask = segmentation == floorPlaneIndex
                mask = cv2.resize(mask.astype(np.float32), (oriWidth, oriHeight), interpolation=cv2.INTER_LINEAR) > 0.5
                plane_depths = calcPlaneDepths(pred_p, oriWidth, oriHeight)
                depth = plane_depths[:, :, floorPlaneIndex]
                #depth = cv2.resize(depth, (oriWidth, oriHeight), interpolation=cv2.INTER_LINEAR) > 0.5
                uv = findCornerPoints(planes[floorPlaneIndex], depth, mask)
                print(uv)
                source_uv = np.array([[0, 0], [0, oriHeight], [oriWidth, 0], [oriWidth, oriHeight]])
                
                h, status = cv2.findHomography(source_uv, uv)
                #textureImageWarped = cv2.warpPerspective(textureImage, h, (WIDTH, HEIGHT))
                textureImageWarped = cv2.warpPerspective(textureImage, h, (oriWidth, oriHeight))
                image = im

                image[mask] = textureImageWarped[mask]
                cv2.imwrite(testdir + '/' + str(index) + '_texture.png', textureImageWarped)
                cv2.imwrite(testdir + '/' + str(index) + '_result_' + str(texture_index) + '.png', image)


            # if index < 0:
            #     segmentation = np.argmax(pred_s, 2)
            #     for planeIndex in xrange(numOutputPlanes):
            #         cv2.imwrite(testdir + '/' + str(index) + '_segmentation_' + str(planeIndex) + '.png', drawMaskImage(pred_s[:, :, planeIndex]))
            #         #cv2.imwrite(testdir + '/' + str(index) + '_segmentation_' + str(planeIndex) + '_gt.png', drawMaskImage(gt_s[:, :, planeIndex]))
            #         continue
            #     pass
            continue
        #exit(1)
        pass
    return

copyTexture()
