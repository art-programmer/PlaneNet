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
import itertools

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
from plane_utils import *
from modules import *

from train_planenet import build_graph
from train_sample import build_graph as build_graph_sample
from planenet import PlaneNet
HEIGHT=192
WIDTH=256

def getGroundTruth(options):
    if options.useCache == 1 and os.path.exists(options.test_dir + '/room_layout_gt.npy'):
        results = np.load(options.test_dir + '/room_layout_gt.npy')
        results = results[()]
        return results['index'], results['layout']

    import scipy.io as sio
    if options.dataset == 'NYU_RGBD':
        image_indices = sio.loadmat('../../Data/NYU_RGBD/room_layout/index_303.mat')['index'].squeeze()
        print(image_indices.shape)
        test_indices = sio.loadmat('../../Data/NYU_RGBD/room_layout/data_split.mat')['test'].squeeze()
        train_indices = sio.loadmat('../../Data/NYU_RGBD/room_layout/data_split.mat')['train'].squeeze()
        indices = image_indices.nonzero()[0][test_indices - 1]
        room_layouts = sio.loadmat('../../Data/NYU_RGBD/room_layout/layout_GT.mat')['layout_GT'].squeeze()
        room_layouts = room_layouts[:, :, test_indices - 1]
        room_layouts = np.transpose(room_layouts, [2, 0, 1])
    else:
        filenames = glob.glob('/mnt/vision/RoomLayout_Hedau/*.mat')
        room_layouts = []
        indices = []
        for filename in filenames:
            room_layout = sio.loadmat(filename)
            if 'fields' not in room_layout:
                continue
            room_layout = room_layout['fields'].squeeze()
            new_room_layout = np.zeros(room_layout.shape)
            new_room_layout[room_layout == 5] = 1
            new_room_layout[room_layout == 1] = 2
            new_room_layout[room_layout == 4] = 3
            new_room_layout[room_layout == 2] = 4
            new_room_layout[room_layout == 3] = 5
            
            room_layouts.append(new_room_layout)
            indices.append(filename)
            continue
        pass
        
    np.save(options.test_dir + '/room_layout_gt.npy', {'index': indices, 'layout': room_layouts})
    #print(room_layouts.shape)
    #print(room_layouts.max())
    #print(room_layouts.min())        
    return indices, room_layouts
    

def getResults(options):

    
    checkpoint_prefix = options.rootFolder + '/checkpoint/'

    
    #method = ('hybrid_hybrid1_bl0_dl0_ll1_sm0', '')
    #method = ('finetuning_hybrid1_ps', '')
    #method = ('planenet_hybrid1_bl0_ll1_ds0_pp_ps', '')
    left_walls = [0, 5, 6, 11, 18]
    right_walls = [4, 10, 7, 19]
    floors = [14]
    ceilings = []    

    #method = ('hybrid_np10_hybrid1_bl0_dl0_ds0_crfrnn5_sm0', '')
    method = ('finetuning_np10_hybrid1_ds0_ps', '')
    #method = ('sample_np10_hybrid3_bl0_dl0_ds0_crfrnn5_sm0', '')
    left_walls = [2]
    right_walls = [3, 7]
    floors = [5]
    ceilings = []    
    #method = ('sample_np10_hybrid3_bl0_dl0_hl2_ds0_crfrnn5_sm0', '')
    #method = ('planenet_np10_hybrid3_bl0_dl0_crfrnn-10_sm0', '')
    # left_walls = [0, 5, 6, 11, 18]
    # right_walls = [4, 10]
    # floors = [14]
    # ceilings = []    
    # layout_planes = [ceilings, floors, left_walls + right_walls]    
    layout_planes = [ceilings, floors, left_walls + right_walls]
    
    
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


    pred_dict = getPrediction(options, layout_planes)
    #np.save(options.test_dir + '/curves.npy', curves)
    return

def getPrediction(options, layout_planes):
    print(options.test_dir)
    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass

    indices, room_layouts = getGroundTruth(options)
    
    
    #image_list = glob.glob('/home/chenliu/Projects/Data/LSUN/images/*.jpg')
    #image_list = glob.glob('/mnt/vision/NYU_RGBD/images/*.png')


    if options.dataset == 'NYU_RGBD':
        image_list = ['/mnt/vision/NYU_RGBD/images/' + ('%08d' % (image_index + 1)) + '.png' for image_index in indices]
    else:
        image_list = [filename.replace('RoomLayout_Hedau', 'RoomLayout_Hedau/Images').replace('_labels.mat', '.jpg') for filename in indices]
        #image_list = glob.glob('/mnt/vision/RoomLayout_Hedau/Images/*.png') + glob.glob('/mnt/vision/RoomLayout_Hedau/Images/*.jpg')
        pass
    #print(len(image_list))
    #exit(1)
    options.numImages = min(options.numImages, len(image_list))

    
    tf.reset_default_graph()    

    
    training_flag = tf.constant(False, tf.bool)

    img_inp = tf.placeholder(tf.float32, shape=[1, HEIGHT, WIDTH, 3], name='image')
    
    options.gpu_id = 0
    if 'sample' or 'hybrid_' in options.checkpoint_dir:
        global_pred_dict, local_pred_dict, deep_pred_dicts = build_graph_sample(img_inp, img_inp, training_flag, options)
    else:
        global_pred_dict, local_pred_dict, deep_pred_dicts = build_graph(img_inp, img_inp, training_flag, options)
        pass

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

    print(np.concatenate([np.expand_dims(np.arange(22), 1), ColorPalette(22).getColorMap()], axis=1))

    planeAreaThresholds = [WIDTH * HEIGHT / 400, WIDTH * HEIGHT / 400, WIDTH * HEIGHT / 400]
    dotThreshold = np.cos(np.deg2rad(60))
    width_high_res = 640
    height_high_res = 480
    
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        #var_to_restore = [v for v in var_to_restore if 'res4b22_relu_non_plane' not in v.name]
        loader = tf.train.Saver(var_to_restore)
        loader.restore(sess, "%s/checkpoint.ckpt"%(options.checkpoint_dir))
        #loader.restore(sess, options.fineTuningCheckpoint)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        
        try:
            total_accuracy = 0
            predSegmentations = []
            predPlaneDepths = []
            predAllSegmentations = []
            predNormals = []
            for index in xrange(options.startIndex + options.numImages):
                if index < options.startIndex:
                    continue
                if options.imageIndex >= 0 and index != options.imageIndex:
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

                #print(image_list[index])
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


                if options.dataset != 'NYU_RGBD':
                    info = np.zeros(info.shape)
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
                    width_high_res = img_ori.shape[1]
                    height_high_res = img_ori.shape[0]
                    pass
                    
                #all_segmentations = np.concatenate([pred_s, pred_np_m], axis=2)
                all_segmentations = pred_s
                plane_depths = calcPlaneDepths(pred_p, width_high_res, height_high_res, info)

                all_segmentations = softmax(all_segmentations)
                #segmentation = np.argmax(all_segmentations[:, :, :pred_s.shape[-1]], 2)
                segmentation = np.argmax(all_segmentations, 2)

                
                planeNormals = pred_p / np.maximum(np.linalg.norm(pred_p, axis=-1, keepdims=True), 1e-4)
                predSegmentations.append(segmentation)
                predPlaneDepths.append(plane_depths)
                predAllSegmentations.append(all_segmentations)
                predNormals.append(planeNormals)
                continue

                #print(pred_p)
                if True:

                    #all_depths = np.concatenate([plane_depths, np.expand_dims(cv2.resize(pred_np_d.squeeze(), (width_high_res, height_high_res)), -1)], axis=2)
                    #pred_d = all_depths.reshape(-1, options.numOutputPlanes + 1)[np.arange(WIDTH * HEIGHT), segmentation.reshape(-1)].reshape(HEIGHT, WIDTH)                    
                    #cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_pred.png', drawDepthImage(pred_d))
                    if options.imageIndex >= 0:
                        for planeIndex in xrange(options.numOutputPlanes):
                            cv2.imwrite(options.test_dir + '/mask_' + str(planeIndex) + '.png', drawMaskImage(all_segmentations[:, :, planeIndex]))
                            #cv2.imwrite(options.test_dir + '/mask_' + str(planeIndex) + '_depth.png', drawDepthImage(plane_depths[:, :, planeIndex]))                        
                            continue
                        pass
                    
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_segmentation_pred.png', drawSegmentationImage(all_segmentations[:, :, :options.numOutputPlanes], blackIndex=options.numOutputPlanes))
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_image.png', img_ori)

                    layout_plane_inds = []
                    for layoutIndex, planeInds in enumerate(layout_planes[:2]):
                        maxArea = 0
                        for planeIndex in planeInds:
                            area = (all_segmentations[:, :, planeIndex]).sum()
                            #area = (segmentation == planeIndex).sum()
                            if area > maxArea:
                                layout_plane_index = planeIndex
                                maxArea = area
                                pass
                            continue
                        if maxArea > planeAreaThresholds[layoutIndex]:
                            layout_plane_inds.append(layout_plane_index)
                        else:
                            layout_plane_inds.append(-1)
                            pass
                        continue

                    # wallPlanes = []
                    # for planeIndex in layout_planes[2]:
                    #     area = (all_segmentations[:, :, planeIndex]).sum()                        
                    #     #area = (segmentation == planeIndex).sum()
                    #     if area > planeAreaThresholds[2]:
                    #         wallPlanes.append([planeIndex, area])
                    #         pass
                    #     #print(planeIndex, area)
                    #     continue


                    # if options.imageIndex >= 0:
                    #     print(wallPlanes)
                    #     pass
                    
                    # wallPlanes = sorted(wallPlanes, key=lambda x: -x[1])
                    # while True:
                    #     hasChange = False
                    #     for wallPlaneIndex, wallPlane in enumerate(wallPlanes):
                    #         newWallPlanes = []
                    #         for otherWallPlane in wallPlanes[wallPlaneIndex + 1:]:
                    #             if np.dot(planeNormals[otherWallPlane[0]], planeNormals[wallPlane[0]]) > dotThreshold:
                    #                 if options.imageIndex >= 0:
                    #                     print('invalid', wallPlane, otherWallPlane)
                    #                     pass
                    #                 hasChange = True
                    #                 continue
                    #             newWallPlanes.append(otherWallPlane)
                    #             continue
                    #         if hasChange:
                    #             wallPlanes = wallPlanes[:wallPlaneIndex + 1] + newWallPlanes
                    #             break
                    #         continue
                    #     if not hasChange:
                    #         break
                    #     continue

                    # if options.imageIndex >= 0:
                    #     print(wallPlanes)                        
                    #     print(all_segmentations.sum(axis=(0, 1)))
                    #     print(wallPlanes)
                    #     print(planeNormals)
                    #     pass
                    
                    # if len(wallPlanes) > 3:
                    #     wallPlanes = wallPlanes[:3]
                    #     pass
                    # angleWallPlanes = []
                    # for wallPlane in wallPlanes:
                    #     planeNormal = planeNormals[wallPlane[0]]
                    #     angle = np.rad2deg(np.arctan2(planeNormal[1], planeNormal[0]))
                    #     angleWallPlanes.append((angle, wallPlane))
                    #     #direction = min(max(int(angle / 45), 0), 3)
                    #     #directionPlaneMask[direction] = wallPlane[0]
                    #     continue

                    # walls = [-1, -1, -1]
                    # minAngleDiff = 90
                    # for angle, wallPlane in angleWallPlanes:
                    #     if abs(angle - 90) < minAngleDiff:
                    #         walls[1] = wallPlane[0]
                    #         minAngleDiff = abs(angle - 90)
                    #         middleAngle = angle
                    #         pass
                    #     continue
                    # if walls[1] >= 0:
                    #     maxScore = 0
                    #     for angle, wallPlane in angleWallPlanes:
                    #         if angle > middleAngle + 1e-4:
                    #             if wallPlane[1] > maxScore:
                    #                 walls[0] = wallPlane[0]
                    #                 maxScore = wallPlane[1]
                    #                 pass
                    #             pass
                    #         continue
                    #     maxScore = 0
                    #     for angle, wallPlane in angleWallPlanes:
                    #         if angle < middleAngle - 1e-4:
                    #             if wallPlane[1] > maxScore:
                    #                 walls[2] = wallPlane[0]
                    #                 maxScore = wallPlane[1]
                    #                 pass
                    #             pass
                    #         continue
                    #     pass

                    walls = []
                    for planeIndex in layout_planes[2]:
                        area = (all_segmentations[:, :, planeIndex]).sum()                        
                        #area = (segmentation == planeIndex).sum()
                        if area > planeAreaThresholds[2]:
                            walls.append(planeIndex)
                            pass
                        #print(planeIndex, area)
                        continue
                    
                    best_layout_plane_inds = layout_plane_inds + walls
                    bestScore = 0
                    for numWalls in xrange(1, min(len(walls), 3) + 1):
                        for selectedWalls in itertools.combinations(walls, numWalls):
                            selected_plane_inds = np.array(layout_plane_inds + list(selectedWalls))
                            depths = []
                            for wall in selected_plane_inds:
                                depths.append(plane_depths[:, :, wall])
                                continue
                            depths.append(np.full((height, width), 10))
                            depths = np.stack(depths, axis=2)
                            selected_plane_segmentation = np.argmin(depths, 2)
                            emptyMask = selected_plane_segmentation == depths.shape[-1] - 1
                            selected_plane_segmentation = selected_plane_inds[np.minimum(selected_plane_segmentation.reshape(-1), selected_plane_inds.shape[0] - 1)].reshape(selected_plane_segmentation.shape)
                            selected_plane_segmentation[emptyMask] = -1
                            #overlap = (selected_plane_segmentation == segmentation).sum()
                            overlap = 0
                            for planeIndex in xrange(options.numOutputPlanes):
                                overlap += segmentations[:, :, planeIndex][selected_plane_segmentation == planeIndex].sum()
                                continue
                            if overlap > bestScore:
                                best_layout_plane_inds = selected_plane_inds
                                bestScore = overlap
                                pass
                            continue
                        continue
                    layout_plane_inds = best_layout_plane_inds
                    layout_plane_depths = []
                    for planeIndex in layout_plane_inds:
                        if planeIndex >= 0:
                            layout_plane_depths.append(plane_depths[:, :, planeIndex])
                        else:
                            layout_plane_depths.append(np.ones((height_high_res, width_high_res)) * 10)
                            pass
                        continue

                    # walls = [-1, -1, -1]
                    # if directionPlaneMask[0] >= 0:
                    #     if directionPlaneMask[1] >= 0:
                    #         if directionPlaneMask[2] >= 0:
                    #             walls = [directionPlaneMask[0], directionPlaneMask[1], directionPlaneMask[2]]
                    #         elif directionPlaneMask[3] >= 0:
                    #             walls = [directionPlaneMask[0], directionPlaneMask[1], directionPlaneMask[3]]
                    #         else:
                    #             walls = [directionPlaneMask[0], directionPlaneMask[1], -1]
                    #             pass
                    #     else:
                    #         if directionPlaneMask[2] >= 0:
                    #             if directionPlaneMask[3] >= 0:
                    #                 walls = [directionPlaneMask[0], directionPlaneMask[2], directionPlaneMask[3]]
                    #             else:
                    #                 walls = [directionPlaneMask[0], directionPlaneMask[2], -1]
                    #                 pass
                    #         else:
                    #             if directionPlaneMask[3] >= 0:
                    #                 walls = [directionPlaneMask[0], -1, directionPlaneMask[3]]
                    #             else:
                    #                 walls = [directionPlaneMask[0], -1, -1]
                    #                 pass
                        
                        
                    layout_plane_depths = np.stack(layout_plane_depths, axis=2)
                    #print(layout_plane_depths.shape)
                    #print(np.argmin(layout_plane_depths, axis=-1).shape)
                    layout_pred = np.argmin(layout_plane_depths, axis=-1) + 1
                    layout_gt = room_layouts[index]

                    layout_pred_img = drawSegmentationImage(layout_pred)
                    #cv2.imwrite(options.test_dir + '/' + str(index) + '_layout_pred.png', layout_pred_img)
                    #cv2.imwrite(options.test_dir + '/' + str(index) + '_layout_pred.png', img_ori / 2 + layout_pred_img / 2)
                    layout_plane_inds = np.array(layout_plane_inds)
                    
                    layout_segmentation_img = layout_plane_inds[layout_pred.reshape(-1) - 1].reshape(layout_pred.shape)
                    layout_segmentation_img[layout_segmentation_img == -1] = options.numOutputPlanes
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_layout_pred.png', drawSegmentationImage(layout_segmentation_img, blackIndex=options.numOutputPlanes))
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_layout_gt.png', drawSegmentationImage(layout_gt, blackIndex=0))

                    pred_d = plane_depths.reshape(-1, options.numOutputPlanes)[np.arange(width_high_res * height_high_res), cv2.resize(segmentation, (width_high_res, height_high_res), interpolation=cv2.INTER_NEAREST).reshape(-1)].reshape(height_high_res, width_high_res)
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_pred.png', drawDepthImage(pred_d))                    
                    #continue
                
                    # numWalls = 0
                    # for wall in walls:
                    #     if wall >= 0:
                    #         numWalls += 1
                    #         pass
                    #     continue
                    numWalls = layout_plane_inds.shape[0] - 2
                    if numWalls == 2:
                        gtMiddleWallMask = layout_gt == 4
                        leftWallScore = np.logical_and(layout_pred == 3, gtMiddleWallMask).sum()
                        middleWallScore = np.logical_and(layout_pred == 4, gtMiddleWallMask).sum()                        
                        rightWallScore = np.logical_and(layout_pred == 5, gtMiddleWallMask).sum()

                        if leftWallScore > middleWallScore:
                            layout_pred[layout_pred >= 3] += 1
                            pass
                        if rightWallScore > middleWallScore:                        
                            layout_pred[layout_pred >= 3] -= 1
                            pass
                        pass
                    if numWalls == 1:
                        layout_pred[layout_pred == 3] += 1
                        pass

                    # leftWallMask = layout_gt == 3
                    # middleWallMask = layout_gt == 4
                    # rightWallMask = layout_gt == 5
                    # if leftWallMask.sum() > middleWallMask.sum() and rightWallMask.sum() == 0:
                    #     layout_gt[np.logical_or(leftWallMask, middleWallMask)] += 1
                    #     pass
                    # if rightWallMask.sum() > middleWallMask.sum() and leftWallMask.sum() == 0:
                    #     layout_gt[np.logical_or(rightWallMask, middleWallMask)] -= 1
                    #     pass
                    # pass

                    accuracy = float((layout_pred == layout_gt).sum()) / (width_high_res * height_high_res)
                    print((index, accuracy))
                    total_accuracy += accuracy                    
                    pass
                if options.imageIndex >= 0:                    
                    exit(1)
                    pass
                continue
            segmentations = np.array(predSegmentations)
            np.save('test/segmentation.npy', segmentations)
            planeDepths = np.array(predPlaneDepths)
            np.save('test/plane_depths.npy', planeDepths)
            predAllSegmentations = np.array(predAllSegmentations)
            np.save('test/all_segmentations.npy', predAllSegmentations)
            predNormals = np.array(predNormals)
            np.save('test/normals.npy', predNormals) 
            print('accuracy', total_accuracy / options.numImages)
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


def testRoomLayout(options):
    left_walls = [2, 9]
    right_walls = [3, 7, 8]
    floors = [5]
    ceilings = [1]
    layout_planes = [ceilings, floors, left_walls + right_walls]
    
    #planeAreaThresholds = [WIDTH * HEIGHT / 400, WIDTH * HEIGHT / 400, WIDTH * HEIGHT / 400]
    planeAreaThresholds = [0, 0, 0]

    indices, room_layouts = getGroundTruth(options)    
    if options.dataset == 'NYU_RGBD':
        image_list = ['/mnt/vision/NYU_RGBD/images/' + ('%08d' % (image_index + 1)) + '.png' for image_index in indices]
    else:
        image_list = [filename.replace('RoomLayout_Hedau', 'RoomLayout_Hedau/Images').replace('_labels.mat', '.jpg') for filename in indices]
        #image_list = glob.glob('/mnt/vision/RoomLayout_Hedau/Images/*.png') + glob.glob('/mnt/vision/RoomLayout_Hedau/Images/*.jpg')
        pass
    options.numImages = min(options.numImages, len(image_list))
    dotThreshold = np.cos(np.deg2rad(60))
    width_high_res = 640
    height_high_res = 480
    
    predSegmentations = np.load('test/segmentation.npy')
    predAllSegmentations = np.load('test/all_segmentations.npy')    
    predPlaneDepths = np.load('test/plane_depths.npy')
    predNormals = np.load('test/normals.npy')
    
    total_accuracy = 0
    
    for index in xrange(predSegmentations.shape[0]):
        if index < options.startIndex:
            continue
        if options.imageIndex >= 0 and index != options.imageIndex:
            continue        
        segmentation = predSegmentations[index]
        plane_depths = predPlaneDepths[index]
        img_ori = cv2.imread(image_list[index])
        all_segmentations = predAllSegmentations[index]
        planeNormals = predNormals[index]
        if options.imageIndex >= 0:
            for planeIndex in xrange(options.numOutputPlanes):
                #cv2.imwrite(options.test_dir + '/mask_' + str(planeIndex) + '.png', drawMaskImage(all_segmentations[:, :, planeIndex]))
                cv2.imwrite(options.test_dir + '/mask_' + str(planeIndex) + '.png', drawMaskImage(segmentation == planeIndex))
                #cv2.imwrite(options.test_dir + '/mask_' + str(planeIndex) + '_depth.png', drawDepthImage(plane_depths[:, :, planeIndex]))
                continue
            pass
                    
        #cv2.imwrite(options.test_dir + '/' + str(index) + '_segmentation_pred.png', drawSegmentationImage(all_segmentations[:, :, :options.numOutputPlanes], blackIndex=options.numOutputPlanes))
        cv2.imwrite(options.test_dir + '/' + str(index) + '_segmentation_pred.png', drawSegmentationImage(segmentation, blackIndex=options.numOutputPlanes))
        cv2.imwrite(options.test_dir + '/' + str(index) + '_image.png', img_ori)

        layout_plane_inds = []
        for layoutIndex, planeInds in enumerate(layout_planes[:2]):
            maxArea = 0
            for planeIndex in planeInds:
                #area = (all_segmentations[:, :, planeIndex]).sum()
                area = (segmentation == planeIndex).sum()
                if area > maxArea:
                    layout_plane_index = planeIndex
                    maxArea = area
                    pass
                continue
            if maxArea > planeAreaThresholds[layoutIndex]:
                layout_plane_inds.append(layout_plane_index)
            else:
                layout_plane_inds.append(-1)
                pass
            continue


        walls = []
        for planeIndex in layout_planes[2]:
            #area = (all_segmentations[:, :, planeIndex]).sum()                        
            area = (segmentation == planeIndex).sum()
            if area > planeAreaThresholds[2]:
                walls.append(planeIndex)
                pass
            #print(planeIndex, area)
            continue

        best_layout_plane_inds = layout_plane_inds + walls
        bestScore = 0
        layout_plane_inds_array = [layout_plane_inds]
        if layout_plane_inds[0] != -1:
            layout_plane_inds_array.append([-1, layout_plane_inds[1]])
            pass
        for layout_plane_inds in layout_plane_inds_array:
            for numWalls in xrange(1, min(len(walls), 3) + 1):
                for combinationIndex, selectedWalls in enumerate(itertools.combinations(walls, numWalls)):
                    invalidCombination = False
                    for wallPlaneIndex, wallPlane in enumerate(selectedWalls):
                        for otherWallPlane in selectedWalls[wallPlaneIndex + 1:]:
                            if np.dot(planeNormals[otherWallPlane], planeNormals[wallPlane]) > dotThreshold:
                                invalidCombination = True
                                break
                                pass
                            continue
                        if invalidCombination:
                            break
                        continue
                    if invalidCombination:
                        continue
                    selected_plane_inds = np.array(layout_plane_inds + list(selectedWalls))
                    depths = []
                    for wall in selected_plane_inds:
                        if wall >= 0:
                            depths.append(plane_depths[:, :, wall])
                        else:
                            depths.append(np.full((height_high_res, width_high_res), 10))
                        continue
                    depths.append(np.full((height_high_res, width_high_res), 10))
                    depths = np.stack(depths, axis=2)
                    selected_plane_segmentation = np.argmin(depths, 2)
                    emptyMask = selected_plane_segmentation == depths.shape[-1] - 1
                    selected_plane_segmentation = selected_plane_inds[np.minimum(selected_plane_segmentation.reshape(-1), selected_plane_inds.shape[0] - 1)].reshape(selected_plane_segmentation.shape)
                    selected_plane_segmentation[emptyMask] = -1

                    selected_plane_segmentation = cv2.resize(selected_plane_segmentation, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)

                    overlap = (selected_plane_segmentation == segmentation).sum()
                    # overlap = 0
                    # for planeIndex in xrange(options.numOutputPlanes):
                    #     overlap += all_segmentations[:, :, planeIndex][selected_plane_segmentation == planeIndex].sum()
                    #     continue

                    if options.imageIndex >= 0:
                        if layout_plane_inds[0] == -1:
                            cv2.imwrite('test/segmentation_' + str(numWalls) + '_' + str(combinationIndex) + '.png', drawSegmentationImage(selected_plane_segmentation, blackIndex=options.numOutputPlanes))
                        else:
                            cv2.imwrite('test/segmentation_' + str(numWalls) + '_' + str(combinationIndex) + '_1.png', drawSegmentationImage(selected_plane_segmentation, blackIndex=options.numOutputPlanes))
                            pass
                        print(combinationIndex, selectedWalls, selected_plane_inds, depths.shape, overlap)
                        pass
                    if overlap > bestScore:
                        best_layout_plane_inds = selected_plane_inds
                        bestScore = overlap
                        pass
                    continue
                continue
            continue
        layout_plane_inds = best_layout_plane_inds

        if options.imageIndex >= 0:
            print(walls)
            print(layout_plane_inds)
            #exit(1)
            pass
        
        layout_plane_depths = []
        for planeIndex in layout_plane_inds:
            if planeIndex >= 0:
                layout_plane_depths.append(plane_depths[:, :, planeIndex])
            else:
                layout_plane_depths.append(np.ones((height_high_res, width_high_res)) * 10)
                pass
            continue

        layout_plane_depths = np.stack(layout_plane_depths, axis=2)
        #print(layout_plane_depths.shape)
        #print(np.argmin(layout_plane_depths, axis=-1).shape)
        layout_pred = np.argmin(layout_plane_depths, axis=-1) + 1
        layout_gt = room_layouts[index]

        layout_pred_img = drawSegmentationImage(layout_pred)
        #cv2.imwrite(options.test_dir + '/' + str(index) + '_layout_pred.png', layout_pred_img)
        #cv2.imwrite(options.test_dir + '/' + str(index) + '_layout_pred.png', img_ori / 2 + layout_pred_img / 2)
        layout_plane_inds = np.array(layout_plane_inds)

        layout_segmentation_img = layout_plane_inds[layout_pred.reshape(-1) - 1].reshape(layout_pred.shape)
        layout_segmentation_img[layout_segmentation_img == -1] = options.numOutputPlanes
        layout_segmentation_img = drawSegmentationImage(layout_segmentation_img, blackIndex=options.numOutputPlanes)        
        cv2.imwrite(options.test_dir + '/' + str(index) + '_layout_pred.png', img_ori * 0.7 + layout_segmentation_img * 0.3)
        cv2.imwrite(options.test_dir + '/' + str(index) + '_layout_gt.png', drawSegmentationImage(layout_gt, blackIndex=0))

        pred_d = plane_depths.reshape(-1, options.numOutputPlanes)[np.arange(width_high_res * height_high_res), cv2.resize(segmentation, (width_high_res, height_high_res), interpolation=cv2.INTER_NEAREST).reshape(-1)].reshape(height_high_res, width_high_res)
        cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_pred.png', drawDepthImage(pred_d))                    
        #continue

        # numWalls = 0
        # for wall in walls:
        #     if wall >= 0:
        #         numWalls += 1
        #         pass
        #     continue
        numWalls = layout_plane_inds.shape[0] - 2
        if numWalls == 2:
            gtMiddleWallMask = layout_gt == 4
            leftWallScore = np.logical_and(layout_pred == 3, gtMiddleWallMask).sum()
            middleWallScore = np.logical_and(layout_pred == 4, gtMiddleWallMask).sum()                        
            rightWallScore = np.logical_and(layout_pred == 5, gtMiddleWallMask).sum()

            if leftWallScore > middleWallScore:
                layout_pred[layout_pred >= 3] += 1
                pass
            if rightWallScore > middleWallScore:                        
                layout_pred[layout_pred >= 3] -= 1
                pass
            pass
        if numWalls == 1:
            layout_pred[layout_pred == 3] += 1
            pass

        # leftWallMask = layout_gt == 3
        # middleWallMask = layout_gt == 4
        # rightWallMask = layout_gt == 5
        # if leftWallMask.sum() > middleWallMask.sum() and rightWallMask.sum() == 0:
        #     layout_gt[np.logical_or(leftWallMask, middleWallMask)] += 1
        #     pass
        # if rightWallMask.sum() > middleWallMask.sum() and leftWallMask.sum() == 0:
        #     layout_gt[np.logical_or(rightWallMask, middleWallMask)] -= 1
        #     pass
        # pass

        accuracy = float((layout_pred == layout_gt).sum()) / (width_high_res * height_high_res)
        print((index, accuracy))
        total_accuracy += accuracy                    
        if options.imageIndex >= 0:                    
            exit(1)
            pass
        continue
    print('accuracy', total_accuracy / options.numImages)


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
    parser.add_argument('--suffix', dest='suffix',
                        help='suffix',
                        default='', type=str)
    
    args = parser.parse_args()
    #args.hybrid = 'hybrid' + args.hybrid
    args.test_dir = 'evaluate/' + args.task + '/' + args.dataset + '/hybrid' + args.hybrid + '/'
    args.visualizeImages = args.numImages

    # image = cv2.imread('evaluate/layout/ScanNet/hybrid3/22_image.png')
    # focal_length = estimateFocalLength(image)
    # print(focal_length)
    # exit(1)

    if args.suffix == '':
        getResults(args)
    else:
        testRoomLayout(args)
        pass
