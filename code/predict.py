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

WIDTH = 256
HEIGHT = 192

ALL_TITLES = ['PlaneNet']
ALL_METHODS = [('sample_np10_hybrid3_bl0_dl0_ds0_crfrnn5_sm0', '', 0, 2), ('planenet_hybrid3_crf1_pb_pp', '', 1, 2), ('sample_np10_hybrid3_bl0_dl0_hl2_ds0_crfrnn5_sm0', '', 1, 2), ('', '', 1, 2), ('', '', 1, 2), ('', '', 1, 2), ('', '', 1, 2)]


def writeHTML(options):
    from html import HTML

    titles = options.titles

    h = HTML('html')
    h.p('Results')
    h.br()
    path = '.'
    #methods = ['planenet', 'pixelwise', 'pixelwise+RANSAC', 'GT+RANSAC', 'planenet+crf', 'pixelwise+semantics+RANSAC']
    #methods = ['planenet', 'pixelwise', 'pixelwise+RANSAC', 'GT+RANSAC']

    for image_index in xrange(options.numImages):

        t = h.table(border='1')
        r_inp = t.tr()
        r_inp.td('input ' + str(image_index + options.startIndex))
        r_inp.td().img(src=path + '/' + str(image_index + options.startIndex) + '_image.png')
        r_inp.td().img(src=path + '/' + str(image_index + options.startIndex) + '_depth_gt.png')
        r_inp.td().img(src=path + '/' + str(image_index + options.startIndex) + '_segmentation_gt.png')
        r_inp.td().img(src=path + '/' + str(image_index + options.startIndex) + '_semantics_gt.png')
        r_inp.td().img(src=path + '/' + str(image_index + options.startIndex) + '_depth_gt_plane.png')
        r_inp.td().img(src=path + '/' + str(image_index + options.startIndex) + '_depth_gt_diff.png')
        # r = t.tr()
        # r.td('PlaneNet prediction')
        # r.td().img(src=firstFolder + '/' + str(index) + '_segmentation_pred.png')
        # r.td().img(src=firstFolder + '/' + str(index) + '_depth_pred.png')

        r = t.tr()
        r.td('methods')
        for method_index, method in enumerate(titles):
            r.td(method)
            continue

        r = t.tr()
        r.td('segmentation')
        for method_index, method in enumerate(titles):
            r.td().img(src=path + '/' + str(image_index + options.startIndex) + '_segmentation_pred_' + str(method_index) + '.png')
            r.td().img(src=path + '/' + str(image_index + options.startIndex) + '_segmentation_pred_blended_' + str(method_index) + '.png')
            continue

        r = t.tr()
        r.td('depth')
        for method_index, method in enumerate(titles):
            r.td().img(src=path + '/' + str(image_index + options.startIndex) + '_depth_pred_' + str(method_index) + '.png')
            continue
        h.br()
        continue

    metric_titles = ['depth error 0.1', 'depth error 0.2', 'depth error 0.3', 'IOU 0.3', 'IOU 0.5', 'IOU 0.7']

    h.p('Curves on plane accuracy')
    for title in metric_titles:
        h.img(src='curve_plane_' + title.replace(' ', '_') + '.png')
        continue

    h.p('Curves on pixel coverage')
    for title in metric_titles:
        h.img(src='curve_pixel_' + title.replace(' ', '_') + '.png')
        continue


    html_file = open(options.test_dir + '/index.html', 'w')
    html_file.write(str(h))
    html_file.close()
    return

def evaluatePlanes(options):
    #writeHTML(options)
    #exit(1)
    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass

    results = getResults(options)

    gt_dict = results['gt']
    predictions = results['pred']


    saving = True
    if gt_dict['image'].shape[0] != options.numImages:
        saving = False
        pass


    for key, value in gt_dict.iteritems():
        if value.shape[0] > options.numImages:
            gt_dict[key] = value[:options.numImages]
            pass
        continue
    for pred_dict in predictions:
        for key, value in pred_dict.iteritems():
            if value.shape[0] > options.numImages:
                pred_dict[key] = value[:options.numImages]
                pass
            continue
        continue

    #methods = ['planenet', 'pixelwise+RANSAC', 'GT+RANSAC']



    #predictions[2] = predictions[3]

    if options.suffix == 'grids':
        image_list = glob.glob(options.test_dir + '/*_image.png')
        print(len(image_list))
        gridImage = writeGridImage(image_list[80:336], 3200, 1800, (16, 16))
        cv2.imwrite(options.test_dir + '/grid_images/grid_1616.png', gridImage)
        exit(1)

    for image_index in xrange(options.visualizeImages):
        if options.imageIndex >= 0 and image_index + options.startIndex != options.imageIndex:
            continue
        if options.suffix == 'grids':
            cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_image.png', gt_dict['image'][image_index])
            segmentation = predictions[0]['segmentation'][image_index]
            #segmentation = np.argmax(np.concatenate([segmentation, pred_dict['np_mask'][image_index]], axis=2), -1)
            segmentationImage = drawSegmentationImage(segmentation, blackIndex=options.numOutputPlanes)
            #cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_segmentation_pred_' + str(0) + '.png', segmentationImage)
            segmentationImageBlended = (segmentationImage * 0.7 + gt_dict['image'][image_index] * 0.3).astype(np.uint8)
            cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_segmentation_pred_blended_' + str(0) + '.png', segmentationImageBlended)
            continue


        cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_image.png', gt_dict['image'][image_index])
        cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_depth_gt.png', drawDepthImage(gt_dict['depth'][image_index]))
        #cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_normal_gt.png', drawNormalImage(gt_dict['normal'][image_index]))
        cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_segmentation_gt.png', drawSegmentationImage(np.concatenate([gt_dict['segmentation'][image_index], 1 - np.expand_dims(gt_dict['plane_mask'][image_index], -1)], axis=2), blackIndex=options.numOutputPlanes))
        #cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_semantics_gt.png', drawSegmentationImage(gt_dict['semantics'][image_index], blackIndex=0))


        #plane_depths = calcPlaneDepths(gt_dict['plane'][image_index], WIDTH, HEIGHT, gt_dict['info'][image_index])
        #all_depths = np.concatenate([plane_depths, np.expand_dims(gt_dict['depth'][image_index], -1)], axis=2)
        #depth = np.sum(all_depths * np.concatenate([gt_dict['segmentation'][image_index], 1 - np.expand_dims(gt_dict['plane_mask'][image_index], -1)], axis=2), axis=2)
        #cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_depth_gt_plane.png', drawDepthImage(depth))
        #cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_depth_gt_diff.png', drawMaskImage((depth - gt_dict['depth'][image_index]) * 5 + 0.5))

        info = gt_dict['info'][image_index]
        #print(info)
        #print(np.rad2deg(np.arctan(info[16] / 2 / info[0])) * 2)
        for method_index, pred_dict in enumerate(predictions):
            cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_dict['depth'][image_index]))

            if 'pixelwise' in options.methods[method_index][1]:
                continue
            segmentation = pred_dict['segmentation'][image_index]
            #segmentation = np.argmax(np.concatenate([segmentation, pred_dict['np_mask'][image_index]], axis=2), -1)
            segmentationImage = drawSegmentationImage(segmentation, blackIndex=options.numOutputPlanes)
            cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_segmentation_pred_' + str(method_index) + '.png', segmentationImage)
            segmentationImageBlended = (segmentationImage * 0.7 + gt_dict['image'][image_index] * 0.3).astype(np.uint8)
            cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_segmentation_pred_blended_' + str(method_index) + '.png', segmentationImageBlended)

            segmentationImageBlended = np.minimum(segmentationImage * 0.3 + gt_dict['image'][image_index] * 0.7, 255).astype(np.uint8)
            if options.imageIndex >= 0:
                if options.suffix == 'video':
                    copyLogoVideo(options.test_dir, image_index + options.startIndex, gt_dict['image'][image_index], pred_dict['depth'][image_index], pred_dict['plane'][image_index], segmentation, gt_dict['info'][image_index], wallTexture=False)
                elif options.suffix == 'wall_video':
                    copyLogoVideo(options.test_dir, image_index + options.startIndex, gt_dict['image'][image_index], pred_dict['depth'][image_index], pred_dict['plane'][image_index], segmentation, gt_dict['info'][image_index], wallTexture=True, wallInds=[7, 9])
                elif options.suffix == 'ruler':
                    addRulerComplete(options.test_dir, image_index + options.startIndex, gt_dict['image'][image_index], pred_dict['depth'][image_index], pred_dict['plane'][image_index], segmentation, gt_dict['info'][image_index], startPixel=(280, 190), endPixel=(380, 390), fixedEndPoint=True, numFrames=1000)
                elif options.suffix == 'texture':
                    for planeIndex in xrange(options.numOutputPlanes):
                        cv2.imwrite('test/mask_' + str(planeIndex) + '.png', drawMaskImage(segmentation == planeIndex))
                        continue
                    #resultImages = copyTexture(gt_dict['image'][image_index], pred_dict['plane'][image_index], segmentation, gt_dict['info'][image_index], 6)
                    #for resultIndex, resultImage in enumerate(resultImages):
                    #cv2.imwrite('test/texture_' + str(image_index + options.startIndex) + '_' + str(resultIndex) + '.png', resultImage)
                    #continue


                    resultImage = copyLogo(options.test_dir, image_index + options.startIndex, gt_dict['image'][image_index], pred_dict['depth'][image_index], pred_dict['plane'][image_index], segmentation, gt_dict['info'][image_index])
                    #resultImage = copyWallTexture(options.test_dir, image_index + options.startIndex, gt_dict['image'][image_index], pred_dict['depth'][image_index], pred_dict['plane'][image_index], segmentation, gt_dict['info'][image_index], [7, 9])
                    cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_result.png', resultImage)
                    writePLYFile(options.test_dir, image_index + options.startIndex, gt_dict['image'][image_index], pred_dict['depth'][image_index], segmentation, pred_dict['plane'][image_index], gt_dict['info'][image_index])
                elif options.suffix == 'dump':
                    planes = pred_dict['plane']
                    planes /= np.linalg.norm(planes, axis=-1, keepdims=True)
                    print([(planeIndex, plane) for planeIndex, plane in enumerate(planes[0])])
                    for planeIndex in xrange(options.numOutputPlanes):
                        cv2.imwrite('test/mask_' + str(planeIndex) + '.png', drawMaskImage(segmentation == planeIndex))
                        continue
                    print('dump')
                    newPlanes = []
                    newSegmentation = np.full(segmentation.shape, -1)
                    newPlaneIndex = 0
                    planes = pred_dict['plane'][image_index]
                    for planeIndex in xrange(options.numOutputPlanes):
                        mask = segmentation == planeIndex
                        if mask.sum() > 0:
                            newPlanes.append(planes[planeIndex])
                            newSegmentation[mask] = newPlaneIndex
                            newPlaneIndex += 1
                            pass
                        continue

                    np.save('test/' + str(image_index + options.startIndex) + '_planes.npy', np.stack(newPlanes, axis=0))
                    #print(global_gt['non_plane_mask'].shape)
                    np.save('test/' + str(image_index + options.startIndex) + '_segmentation.npy', newSegmentation)
                    print(newSegmentation.max(), newSegmentation.min())
                    cv2.imwrite('test/' + str(image_index + options.startIndex) + '_image.png', gt_dict['image'][image_index])
                    depth = pred_dict['depth'][image_index]
                    np.save('test/' + str(image_index + options.startIndex) + '_depth.npy', depth)
                    info = gt_dict['info'][image_index]
                    #normal = calcNormal(depth, info)
                    #np.save('test/' + str(image_index + options.startIndex) + '_normal.npy', normal)
                    np.save('test/' + str(image_index + options.startIndex) + '_info.npy', info)
                    exit(1)

                else:
                    np_mask = (segmentation == options.numOutputPlanes).astype(np.float32)
                    np_depth = pred_dict['np_depth'][image_index].squeeze()
                    np_depth = cv2.resize(np_depth, (np_mask.shape[1], np_mask.shape[0]))
                    cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_np_depth_pred_' + str(method_index) + '.png', drawDepthImage(np_depth * np_mask))
                    writePLYFile(options.test_dir, image_index + options.startIndex, segmentationImageBlended, pred_dict['depth'][image_index], segmentation, pred_dict['plane'][image_index], gt_dict['info'][image_index])
                    pass
                exit(1)
                pass
            continue
        continue

    writeHTML(options)
    exit(1)

    #post processing
    for method_index, method in enumerate(options.methods):
        if method[1] == '':
            continue
        if len(method) < 4 or method[3] == 0:
            continue
        if len(method) >= 3 and method[2] >= 0:
            pred_dict = predictions[method[2]]
        else:
            pred_dict = predictions[method_index]
            pass

        if method[1] == 'graphcut':
            #pred_dict = gt_dict
            predSegmentations = []
            predDepths = []
            for image_index in xrange(options.numImages):
                #if image_index != 3:
                #continue
                print('graph cut ' + str(image_index + options.startIndex))

                segmentation = np.argmax(np.concatenate([pred_dict['segmentation'][image_index], 1 - np.expand_dims(pred_dict['plane_mask'][image_index], -1)], axis=2), axis=2)
                #pred_s = getSegmentationsGraphCut(pred_dict['plane'][image_index], gt_dict['image'][image_index], pred_dict['depth'][image_index], pred_dict['normal'][image_index], segmentation, pred_dict['semantics'][image_index], pred_dict['info'][image_index], gt_dict['num_planes'][image_index])

                pred_p, pred_s, numPlanes = removeSmallSegments(pred_dict['plane'][image_index], gt_dict['image'][image_index], pred_dict['depth'][image_index], pred_dict['normal'][image_index], segmentation, pred_dict['semantics'][image_index], pred_dict['info'][image_index], gt_dict['num_planes'][image_index])
                #pred_p, pred_s, numPlanes = pred_dict['plane'][image_index], segmentation, gt_dict['num_planes'][image_index]
                print((gt_dict['num_planes'][image_index], numPlanes))
                planeDepths = calcPlaneDepths(pred_p, WIDTH, HEIGHT, gt_dict['info'][image_index])
                allDepths = np.concatenate([planeDepths, np.expand_dims(pred_dict['depth'][image_index], -1)], axis=2)
                pred_d = allDepths.reshape(-1, options.numOutputPlanes + 1)[np.arange(WIDTH * HEIGHT), pred_s.reshape(-1)].reshape(HEIGHT, WIDTH)

                predSegmentations.append(pred_s)
                predDepths.append(pred_d)

                cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_d))
                cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage(pred_s, blackIndex=options.numOutputPlanes))
                continue
            new_pred_dict = {}
            for key, value in pred_dict.iteritems():
                new_pred_dict[key] = value
                continue
            new_pred_dict['segmentation'] = np.array(predSegmentations)
            if method_index < len(predictions):
                predictions[method_index] = new_pred_dict
            else:
                predictions.append(new_pred_dict)
                pass
        if method[1] == 'crf_tf':
            predSegmentations = []
            predDepths = []

            image_inp = tf.placeholder(tf.float32, shape=[1, HEIGHT, WIDTH, 3], name='image')
            segmentation_inp = tf.placeholder(tf.float32, shape=[1, HEIGHT, WIDTH, options.numOutputPlanes + 1], name='segmentation')
            plane_inp = tf.placeholder(tf.float32, shape=[1, options.numOutputPlanes, 3], name='plane')
            non_plane_depth_inp = tf.placeholder(tf.float32, shape=[1, HEIGHT, WIDTH, 1], name='non_plane_depth')
            info_inp = tf.placeholder(tf.float32, shape=[20], name='info')


            plane_parameters = tf.reshape(plane_inp, (-1, 3))
            plane_depths = planeDepthsModule(plane_parameters, WIDTH, HEIGHT, info_inp)
            plane_depths = tf.transpose(tf.reshape(plane_depths, [HEIGHT, WIDTH, -1, options.numOutputPlanes]), [2, 0, 1, 3])
            all_depths = tf.concat([plane_depths, non_plane_depth_inp], axis=3)

            planesY = plane_inp[:, :, 1]
            planesD = tf.maximum(tf.norm(plane_inp, axis=-1), 1e-4)
            planesY /= planesD
            planesY = tf.concat([planesY, tf.ones((1, 1))], axis=1)

            #refined_segmentation = crfModule(segmentation_inp, plane_inp, non_plane_depth_inp, info_inp, numOutputPlanes = options.numOutputPlanes, numIterations=5)

            imageDiff = calcImageDiff(image_inp)
            #refined_segmentation, debug_dict = segmentationRefinementModule(segmentation_inp, all_depths, planesY, imageDiff, numOutputPlanes = options.numOutputPlanes + 1, numIterations=5)
            refined_segmentation, debug_dict = meanfieldModule(segmentation_inp, all_depths, planesY, imageDiff, numOutputPlanes = options.numOutputPlanes + 1, maxDepthDiff=0.2, varDepthDiff=pow(0.2, 2))

            config=tf.ConfigProto()
            config.gpu_options.allow_growth=True
            config.allow_soft_placement=True

            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            with tf.Session(config=config) as sess:
                sess.run(init_op)
                for image_index in xrange(options.numImages):
                    #if image_index != 1:
                    #continue
                    print('crf tf ' + str(image_index + options.startIndex))
                    allSegmentations = np.concatenate([pred_dict['segmentation'][image_index], pred_dict['np_mask'][image_index]], axis=2)
                    allSegmentations = softmax(allSegmentations)
                    pred_s, debug = sess.run([refined_segmentation, debug_dict], feed_dict={segmentation_inp: np.expand_dims(allSegmentations, 0), plane_inp: np.expand_dims(pred_dict['plane'][image_index], 0), non_plane_depth_inp: np.expand_dims(pred_dict['np_depth'][image_index], 0), info_inp: gt_dict['info'][image_index], image_inp: gt_dict['image'][image_index:image_index + 1]})

                    pred_s = pred_s[0]
                    planeDepths = calcPlaneDepths(pred_dict['plane'][image_index], WIDTH, HEIGHT, gt_dict['info'][image_index])
                    allDepths = np.concatenate([planeDepths, pred_dict['np_depth'][image_index]], axis=2)
                    pred_d = np.sum(allDepths * pred_s, axis=-1)

                    predSegmentations.append(pred_s)
                    predDepths.append(pred_d)

                    cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_d))
                    cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage(pred_s, blackIndex=options.numOutputPlanes))

                    if 'diff' in debug:
                        segmentation = np.argmax(allSegmentations, axis=-1)
                        for planeIndex in xrange(options.numOutputPlanes + 1):
                            cv2.imwrite('test/mask_' + str(planeIndex) + '.png', drawMaskImage(allSegmentations[:, :, planeIndex]))
                            continue

                        for planeIndex in xrange(debug['diff'].shape[-1]):
                            cv2.imwrite('test/cost_mask_' + str(planeIndex) + '.png', drawMaskImage(debug['diff'][0, :, :, planeIndex] / 2))
                            continue
                        exit(1)
                        pass
                    continue
                pass
            new_pred_dict = {}
            for key, value in pred_dict.iteritems():
                new_pred_dict[key] = value
                continue
            segmentations = np.array(predSegmentations)
            new_pred_dict['segmentation'] = segmentations[:, :, :, :options.numOutputPlanes]
            new_pred_dict['non_plane_mask'] = segmentations[:, :, :, options.numOutputPlanes:options.numOutputPlanes + 1]
            #new_pred_dict['non_plane_mask'] = segmentations[:, :, :, :options.numOutputPlanes]
            new_pred_dict['depth'] = np.array(predDepths)
            if method_index < len(predictions):
                predictions[method_index] = new_pred_dict
            else:
                predictions.append(new_pred_dict)
                pass
            pass

        if method[1] == 'crf':
            predSegmentations = []
            predDepths = []
            for image_index in xrange(options.numImages):
                print('crf ' + str(image_index + options.startIndex))
                boundaries = pred_dict['boundary'][image_index]
                boundaries = sigmoid(boundaries)
                cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_boundary.png', drawMaskImage(np.concatenate([boundaries, np.zeros((HEIGHT, WIDTH, 1))], axis=2)))

                allSegmentations = np.concatenate([pred_dict['segmentation'][image_index], pred_dict['np_mask'][image_index]], axis=2)
                allSegmentations = softmax(allSegmentations)
                planeDepths = calcPlaneDepths(pred_dict['plane'][image_index], WIDTH, HEIGHT, gt_dict['info'][image_index])
                allDepths = np.concatenate([planeDepths, pred_dict['np_depth'][image_index]], axis=2)
                #boundaries = np.concatenate([np.ones((allSegmentations.shape[0], allSegmentations.shape[1], 1)), -np.ones((allSegmentations.shape[0], allSegmentations.shape[1], 1))], axis=2)
                #if options.imageIndex >= 0:
                #boundaries = cv2.imread(options.test_dir + '/' + str(options.imageIndex) + '_boundary.png')
                #else:
                #boundaries = cv2.imread(options.test_dir + '/' + str(image_index + options.startIndex) + '_boundary.png')
                #pass
                #boundaries = (boundaries > 128).astype(np.float32)[:, :, :2]

                allDepths[:, :, options.numOutputPlanes] = 0
                pred_s = refineSegmentation(gt_dict['image'][image_index], allSegmentations, allDepths, boundaries, numOutputPlanes = 20, numIterations=20, numProposals=5)
                pred_d = allDepths.reshape(-1, options.numOutputPlanes + 1)[np.arange(WIDTH * HEIGHT), pred_s.reshape(-1)].reshape(HEIGHT, WIDTH)

                predSegmentations.append(pred_s)
                predDepths.append(pred_d)

                cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_d))
                cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage(pred_s, blackIndex=options.numOutputPlanes))

                #segmentation = np.argmax(allSegmentations, axis=-1)
                for planeIndex in xrange(options.numOutputPlanes + 1):
                    cv2.imwrite('test/mask_' + str(planeIndex) + '.png', drawMaskImage(allSegmentations[:, :, planeIndex]))
                    continue
                #cv2.imwrite(options.test_dir + '/mask_' + str(21) + '.png', drawDepthImage(pred_dict['np_depth'][0]))
                #for plane_index in xrange(options.numOutputPlanes + 1):
                #cv2.imwrite(options.test_dir + '/mask_' + str(plane_index) + '.png', drawMaskImage(pred_s == plane_index))
                #continue
                #exit(1)
                continue

            new_pred_dict = {}
            for key, value in pred_dict.iteritems():
                new_pred_dict[key] = value
                continue
            new_pred_dict['segmentation'] = np.array(predSegmentations)
            new_pred_dict['depth'] = np.array(predDepths)
            if method_index < len(predictions):
                predictions[method_index] = new_pred_dict
            else:
                predictions.append(new_pred_dict)
                pass
            pass


        if 'pixelwise' in method[1]:
            predPlanes = []
            predSegmentations = []
            predDepths = []
            predNumPlanes = []
            for image_index in xrange(options.numImages):
                pred_d = pred_dict['np_depth'][image_index].squeeze()
                pred_n = pred_dict['np_normal'][image_index].squeeze()
                if '_1' in method[1]:
                    pred_s = np.zeros(pred_dict['segmentation'][image_index].shape)
                    pred_p = np.zeros(pred_dict['plane'][image_index].shape)
                elif '_2' in method[1]:
                    parameters = {'distanceCostThreshold': 0.1, 'smoothnessWeight': 0.05, 'semantics': True}
                    pred_p, pred_s = fitPlanesNYU(gt_dict['image'], gt_dict['depth'][image_index].squeeze(), gt_dict['normal'][image_index], gt_dict['semantics'][image_index], gt_dict['info'][image_index], numOutputPlanes=20, parameters=parameters)
                elif '_3' in method[1]:
                    parameters = {'distanceCostThreshold': 0.1, 'smoothnessWeight': 0.03, 'semantics': True, 'distanceThreshold': 0.2}
                    pred_p, pred_s = fitPlanesNYU(gt_dict['image'], pred_d, pred_n, pred_dict['semantics'][image_index], gt_dict['info'][image_index], numOutputPlanes=20, parameters=parameters)
                elif '_4' in method[1]:
                    parameters = {'numProposals': 5, 'distanceCostThreshold': 0.1, 'smoothnessWeight': 30, 'dominantLineThreshold': 3, 'offsetGap': 0.1}
                    pred_p, pred_s = fitPlanesManhattan(gt_dict['image'][image_index], gt_dict['depth'][image_index].squeeze(), gt_dict['normal'][image_index], gt_dict['info'][image_index], numOutputPlanes=20, parameters=parameters)
                    pred_d = np.zeros((HEIGHT, WIDTH))
                elif '_5' in method[1]:
                    parameters = {'numProposals': 5, 'distanceCostThreshold': 0.1, 'smoothnessWeight': 100, 'dominantLineThreshold': 3, 'offsetGap': 0.6}
                    pred_p, pred_s = fitPlanesManhattan(gt_dict['image'][image_index], pred_d, pred_n, gt_dict['info'][image_index], numOutputPlanes=20, parameters=parameters)
                    pred_d = np.zeros((HEIGHT, WIDTH))
                elif '_6' in method[1]:
                    parameters = {'distanceCostThreshold': 0.1, 'smoothnessWeight': 300, 'numProposals': 5, 'normalWeight': 1, 'meanshift': 0.2}
                    pred_p, pred_s = fitPlanesPiecewise(gt_dict['image'][image_index], gt_dict['depth'][image_index].squeeze(), gt_dict['normal'][image_index], gt_dict['info'][image_index], numOutputPlanes=20, parameters=parameters)
                    pred_d = np.zeros((HEIGHT, WIDTH))
                elif '_7' in method[1]:
                    parameters = {'numProposals': 5, 'distanceCostThreshold': 0.1, 'smoothnessWeight': 300, 'normalWeight': 1, 'meanshift': 0.2}
                    pred_p, pred_s = fitPlanesPiecewise(gt_dict['image'][image_index], pred_d, pred_n, gt_dict['info'][image_index], numOutputPlanes=20, parameters=parameters)
                    pred_d = np.zeros((HEIGHT, WIDTH))
                    pass
                predPlanes.append(pred_p)
                predSegmentations.append(pred_s)
                predDepths.append(pred_d)
                predNumPlanes.append(pred_p.shape[0])

                cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_d))
                cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage(pred_s))
                #exit(1)
                continue
            new_pred_dict = {}
            for key, value in pred_dict.iteritems():
                new_pred_dict[key] = value
                continue
            new_pred_dict['plane'] = np.array(predPlanes)
            new_pred_dict['segmentation'] = np.array(predSegmentations)
            new_pred_dict['depth'] = np.array(predDepths)
            new_pred_dict['num_planes'] = np.array(predNumPlanes)
            if method_index < len(predictions):
                predictions[method_index] = new_pred_dict
            else:
                predictions.append(new_pred_dict)
                pass
            #titles.append('pixelwise+semantics+RANSAC')
            pass

        if method[1] == 'crfrnn':
            predSegmentations = []
            predDepths = []

            image_inp = tf.placeholder(tf.float32, shape=[1, HEIGHT, WIDTH, 3], name='image')
            segmentation_inp = tf.placeholder(tf.float32, shape=[1, HEIGHT, WIDTH, options.numOutputPlanes + 1], name='segmentation')

            refined_segmentation = CrfRnnLayer(image_dims=(HEIGHT, WIDTH), num_classes=21, theta_alpha=120., theta_beta=3., theta_gamma=3., num_iterations=10, name='crfrnn')([segmentation_inp, image_inp])

            config=tf.ConfigProto()
            config.gpu_options.allow_growth=True
            config.allow_soft_placement=True

            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            with tf.Session(config=config) as sess:
                sess.run(init_op)
                for image_index in xrange(options.numImages):
                    #if image_index != 1:
                    #continue
                    print('crf rnn ' + str(image_index + options.startIndex))
                    allSegmentations = np.concatenate([pred_dict['segmentation'][image_index], pred_dict['np_mask'][image_index]], axis=2)
                    img = gt_dict['image'][image_index:image_index + 1].astype(np.float32) - 128


                    pred_s = sess.run(refined_segmentation, feed_dict={segmentation_inp: np.expand_dims(allSegmentations, 0), image_inp: img})

                    # print(pred_s.shape)
                    # print(pred_s[0].max())
                    # print(pred_s.sum(-1).max())
                    # exit(1)
                    pred_s = pred_s[0]
                    # print(allSegmentations.max())
                    # print(pred_s.max())
                    # print(img.max())
                    # print(img.min())
                    # print(np.abs(pred_s - allSegmentations).max())
                    # print(np.abs(np.argmax(pred_s, axis=-1) - np.argmax(allSegmentations, axis=-1)).max())
                    pred_s = one_hot(np.argmax(pred_s, axis=-1), options.numOutputPlanes + 1)


                    planeDepths = calcPlaneDepths(pred_dict['plane'][image_index], WIDTH, HEIGHT, gt_dict['info'][image_index])
                    allDepths = np.concatenate([planeDepths, pred_dict['np_depth'][image_index]], axis=2)
                    pred_d = np.sum(allDepths * pred_s, axis=-1)

                    predSegmentations.append(pred_s)
                    predDepths.append(pred_d)

                    cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_d))
                    cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage(pred_s, blackIndex=options.numOutputPlanes))

                    continue
                pass
            new_pred_dict = {}
            for key, value in pred_dict.iteritems():
                new_pred_dict[key] = value
                continue
            segmentations = np.array(predSegmentations)
            new_pred_dict['segmentation'] = segmentations[:, :, :, :options.numOutputPlanes]
            new_pred_dict['non_plane_mask'] = segmentations[:, :, :, options.numOutputPlanes:options.numOutputPlanes + 1]
            #new_pred_dict['non_plane_mask'] = segmentations[:, :, :, :options.numOutputPlanes]
            new_pred_dict['depth'] = np.array(predDepths)
            if method_index < len(predictions):
                predictions[method_index] = new_pred_dict
            else:
                predictions.append(new_pred_dict)
                pass
            pass
        if saving:
            np.save(options.result_filename, {'gt': gt_dict, 'pred': predictions})
            pass
        continue


    #plotResults(gt_dict, predictions, options)
    writeHTML(options)
    return

def plotAll():
    result_filenames = glob.glob(options.test_dir + '/results_*.npy')
    assert(len(result_filenames) > 0)
    results = np.load(result_filenames[0])
    results = results[()]
    gt_dict = results['gt']
    predictions = results['pred']

    for index in xrange(1, len(result_filenames)):
        other_results = np.load(result_filenames[index])
        other_results = other_results[()]
        other_gt_dict = other_results['gt']
        other_predictions = other_results['pred']

        for k, v in other_gt_dict.iteritems():
            gt_dict[k] = np.concatenate([gt_dict[k], v], axis=0)
            continue
        for methodIndex, other_pred_dict in enumerate(other_predictions):
            for k, v in other_pred_dict.iteritems():
                predictions[methodIndex][k] = np.concatenate([predictions[methodIndex][k], v], axis=0)
                continue
            continue
        continue

    plotResults(gt_dict, predictions, options)
    return


def plotResults(gt_dict, predictions, options):
    titles = options.titles

    pixel_metric_curves = []
    plane_metric_curves = []
    for method_index, pred_dict in enumerate(predictions):
        if titles[method_index] == 'pixelwise':
            continue
        segmentations = pred_dict['segmentation']
        #if method_index == 0:
        #segmentations = softmax(segmentations)
        #pass
        #pixel_curves, plane_curves = evaluatePlaneSegmentation(pred_dict['plane'], segmentations, gt_dict['plane'], gt_dict['segmentation'], gt_dict['num_planes'], numOutputPlanes = options.numOutputPlanes)

        pixel_curves = np.zeros((6, 11))
        plane_curves = np.zeros((6, 11, 3))
        numImages = segmentations.shape[0]
        for image_index in xrange(numImages):
            gtDepths = calcPlaneDepths(gt_dict['plane'][image_index], WIDTH, HEIGHT, gt_dict['info'][image_index])
            predDepths = calcPlaneDepths(pred_dict['plane'][image_index], WIDTH, HEIGHT, gt_dict['info'][image_index])
            if 'num_planes' in pred_dict:
                predNumPlanes = pred_dict['num_planes'][image_index]
            else:
                predNumPlanes = options.numOutputPlanes
                pass
            pixelStatistics, planeStatistics = evaluatePlanePrediction(predDepths, segmentations[image_index], predNumPlanes, gtDepths, gt_dict['segmentation'][image_index], gt_dict['num_planes'][image_index])

            pixel_curves += np.array(pixelStatistics)
            plane_curves += np.array(planeStatistics)
            continue

        if len(pixel_metric_curves) == 0:
            for metric_index, pixel_curve in enumerate(pixel_curves):
                pixel_metric_curves.append([])
                plane_metric_curves.append([])
                continue
            pass

        for metric_index, pixel_curve in enumerate(pixel_curves):
            pixel_metric_curves[metric_index].append(pixel_curve / numImages)
            continue
        for metric_index, plane_curve in enumerate(plane_curves):
            #planeScore = plane_curve[:, 0] / plane_curve[:, 1]
            plane_metric_curves[metric_index].append(plane_curve)
            continue
        continue


    np.save(options.test_dir + '/pixel_curves.npy', np.array(pixel_curves))
    np.save(options.test_dir + '/plane_curves.npy', np.array(plane_curves))


    xs = []
    xs.append((np.arange(11) * 0.1).tolist())
    xs.append((np.arange(11) * 0.1).tolist())
    xs.append((np.arange(11) * 0.1).tolist())
    xs.append((np.arange(11) * 0.05).tolist())
    xs.append((np.arange(11) * 0.05).tolist())
    xs.append((np.arange(11) * 0.05).tolist())
    xlabels = ['IOU', 'IOU', 'IOU', 'plane diff', 'plane diff', 'plane diff']
    curve_titles = ['depth error 0.1', 'depth error 0.2', 'depth error 0.3', 'IOU 0.3', 'IOU 0.5', 'IOU 0.7']
    curve_labels = [title for title in titles if title != 'pixelwise']
    for metric_index, curves in enumerate(pixel_metric_curves):
        filename = options.test_dir + '/curve_pixel_' + curve_titles[metric_index].replace(' ', '_') + '.png'
        plotCurves(xs[metric_index], curves, filename = filename, xlabel=xlabels[metric_index], ylabel='pixel coverage', title=curve_titles[metric_index], labels=curve_labels)
        continue
    for metric_index, curves in enumerate(plane_metric_curves):
        filename = options.test_dir + '/curve_plane_' + curve_titles[metric_index].replace(' ', '_') + '.png'
        curves = [curve[:, 0] / curve[:, 1] for curve in curves]
        plotCurves(xs[metric_index], curves, filename = filename, xlabel=xlabels[metric_index], ylabel='plane accuracy', title=curve_titles[metric_index], labels=curve_labels)
        continue


def evaluateDepthPrediction(options):

    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass

    if options.useCache == 1 and os.path.exists(options.result_filename):
        results = np.load(options.result_filename)
        results = results[()]
    else:
        results = getResults(options)
        if options.useCache != -2:
            np.save(options.result_filename, results)
            pass
        pass

    gt_dict = results['gt']
    predictions = results['pred']

    for key, value in gt_dict.iteritems():
        if value.shape[0] > options.numImages:
            gt_dict[key] = value[:options.numImages]
            pass
        continue
    for pred_dict in predictions:
        for key, value in pred_dict.iteritems():
            if value.shape[0] > options.numImages:
                pred_dict[key] = value[:options.numImages]
                pass
            continue
        continue

    titles = options.titles


    for image_index in xrange(options.visualizeImages):
        cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_image.png', gt_dict['image'][image_index])
        cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_depth_gt.png', drawDepthImage(gt_dict['depth'][image_index]))
        cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_segmentation_gt.png', drawSegmentationImage(np.concatenate([gt_dict['segmentation'][image_index], 1 - np.expand_dims(gt_dict['plane_mask'][image_index], -1)], axis=2), blackIndex=options.numOutputPlanes))
        cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_semantics_gt.png', drawSegmentationImage(gt_dict['semantics'][image_index], blackIndex=0))



        # plane_depths = calcPlaneDepths(gt_dict['plane'][image_index], WIDTH, HEIGHT, gt_dict['info'][image_index])
        # all_depths = np.concatenate([plane_depths, np.expand_dims(gt_dict['depth'][image_index], -1)], axis=2)
        # depth = np.sum(all_depths * np.concatenate([gt_dict['segmentation'][image_index], 1 - np.expand_dims(gt_dict['plane_mask'][image_index], -1)], axis=2), axis=2)
        # cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_depth_gt_plane.png', drawDepthImage(depth))

        for method_index, pred_dict in enumerate(predictions):
            cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_dict['depth'][image_index]))

            if titles[method_index] == 'pixelwise':
                continue
            segmentation = pred_dict['segmentation'][image_index]
            segmentation = np.concatenate([segmentation, pred_dict['np_mask'][image_index]], axis=2)
            cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage(segmentation, blackIndex=options.numOutputPlanes))
            continue
        continue

    #post processing
    for method_index, method in enumerate(options.methods):
        if method[1] == 'graphcut':
            pred_dict = gt_dict
            predSegmentations = []
            predDepths = []
            for image_index in xrange(options.numImages):
                #if image_index != 3:
                #continue
                print('graph cut ' + str(image_index + options.startIndex))

                segmentation = np.argmax(np.concatenate([pred_dict['segmentation'][image_index], 1 - np.expand_dims(pred_dict['plane_mask'][image_index], -1)], axis=2), axis=2)
                #pred_s = getSegmentationsGraphCut(pred_dict['plane'][image_index], gt_dict['image'][image_index], pred_dict['depth'][image_index], pred_dict['normal'][image_index], segmentation, pred_dict['semantics'][image_index], pred_dict['info'][image_index], gt_dict['num_planes'][image_index])

                pred_p, pred_s, numPlanes = removeSmallSegments(pred_dict['plane'][image_index], gt_dict['image'][image_index], pred_dict['depth'][image_index], pred_dict['normal'][image_index], segmentation, pred_dict['semantics'][image_index], pred_dict['info'][image_index], gt_dict['num_planes'][image_index])
                #pred_p, pred_s, numPlanes = pred_dict['plane'][image_index], segmentation, gt_dict['num_planes'][image_index]
                print((gt_dict['num_planes'][image_index], numPlanes))
                planeDepths = calcPlaneDepths(pred_p, WIDTH, HEIGHT, gt_dict['info'][image_index])
                allDepths = np.concatenate([planeDepths, np.expand_dims(pred_dict['depth'][image_index], -1)], axis=2)
                pred_d = allDepths.reshape(-1, options.numOutputPlanes + 1)[np.arange(WIDTH * HEIGHT), pred_s.reshape(-1)].reshape(HEIGHT, WIDTH)

                predSegmentations.append(pred_s)
                predDepths.append(pred_d)

                cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_d))
                cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage(pred_s, blackIndex=options.numOutputPlanes))
                continue
            new_pred_dict = {}
            for key, value in pred_dict.iteritems():
                new_pred_dict[key] = value
                continue
            new_pred_dict['segmentation'] = np.array(predSegmentations)
            predictions[method_index] = new_pred_dict
        if method[1] == 'crf_tf':
            pred_dict = predictions[method_index]
            predSegmentations = []
            predDepths = []

            image_inp = tf.placeholder(tf.float32, shape=[1, HEIGHT, WIDTH, 3], name='image')
            segmentation_inp = tf.placeholder(tf.float32, shape=[1, HEIGHT, WIDTH, options.numOutputPlanes + 1], name='segmentation')
            plane_inp = tf.placeholder(tf.float32, shape=[1, options.numOutputPlanes, 3], name='plane')
            non_plane_depth_inp = tf.placeholder(tf.float32, shape=[1, HEIGHT, WIDTH, 1], name='non_plane_depth')
            info_inp = tf.placeholder(tf.float32, shape=[20], name='info')


            plane_parameters = tf.reshape(plane_inp, (-1, 3))
            plane_depths = planeDepthsModule(plane_parameters, WIDTH, HEIGHT, info_inp)
            plane_depths = tf.transpose(tf.reshape(plane_depths, [HEIGHT, WIDTH, -1, options.numOutputPlanes]), [2, 0, 1, 3])
            all_depths = tf.concat([plane_depths, non_plane_depth_inp], axis=3)

            planesY = plane_inp[:, :, 1]
            planesD = tf.maximum(tf.norm(plane_inp, axis=-1), 1e-4)
            planesY /= planesD
            planesY = tf.concat([planesY, tf.ones((1, 1))], axis=1)

            #refined_segmentation = crfModule(segmentation_inp, plane_inp, non_plane_depth_inp, info_inp, numOutputPlanes = options.numOutputPlanes, numIterations=5)

            imageDiff = calcImageDiff(image_inp)
            #refined_segmentation, debug_dict = segmentationRefinementModule(segmentation_inp, all_depths, planesY, imageDiff, numOutputPlanes = options.numOutputPlanes + 1, numIterations=5)
            refined_segmentation, debug_dict = meanfieldModule(segmentation_inp, all_depths, planesY, imageDiff, numOutputPlanes = options.numOutputPlanes + 1, maxDepthDiff=0.2, varDepthDiff=pow(0.2, 2))

            config=tf.ConfigProto()
            config.gpu_options.allow_growth=True
            config.allow_soft_placement=True

            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            with tf.Session(config=config) as sess:
                sess.run(init_op)
                for image_index in xrange(options.numImages):
                    #if image_index != 1:
                    #continue
                    print('crf tf ' + str(image_index + options.startIndex))
                    allSegmentations = np.concatenate([pred_dict['segmentation'][image_index], pred_dict['np_mask'][image_index]], axis=2)
                    allSegmentations = softmax(allSegmentations)
                    pred_s, debug = sess.run([refined_segmentation, debug_dict], feed_dict={segmentation_inp: np.expand_dims(allSegmentations, 0), plane_inp: np.expand_dims(pred_dict['plane'][image_index], 0), non_plane_depth_inp: np.expand_dims(pred_dict['np_depth'][image_index], 0), info_inp: gt_dict['info'][image_index], image_inp: gt_dict['image'][image_index:image_index + 1]})

                    pred_s = pred_s[0]
                    planeDepths = calcPlaneDepths(pred_dict['plane'][image_index], WIDTH, HEIGHT, gt_dict['info'][image_index])
                    allDepths = np.concatenate([planeDepths, pred_dict['np_depth'][image_index]], axis=2)
                    pred_d = np.sum(allDepths * pred_s, axis=-1)

                    predSegmentations.append(pred_s)
                    predDepths.append(pred_d)

                    cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_d))
                    cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage(pred_s, blackIndex=options.numOutputPlanes))

                    if 'diff' in debug:
                        segmentation = np.argmax(allSegmentations, axis=-1)
                        for planeIndex in xrange(options.numOutputPlanes + 1):
                            cv2.imwrite('test/mask_' + str(planeIndex) + '.png', drawMaskImage(allSegmentations[:, :, planeIndex]))
                            continue

                        for planeIndex in xrange(debug['diff'].shape[-1]):
                            cv2.imwrite('test/cost_mask_' + str(planeIndex) + '.png', drawMaskImage(debug['diff'][0, :, :, planeIndex] / 2))
                            continue
                        exit(1)
                        pass
                    continue
                pass
            new_pred_dict = {}
            for key, value in pred_dict.iteritems():
                new_pred_dict[key] = value
                continue
            segmentations = np.array(predSegmentations)
            new_pred_dict['segmentation'] = segmentations[:, :, :, :options.numOutputPlanes]
            new_pred_dict['non_plane_mask'] = segmentations[:, :, :, options.numOutputPlanes:options.numOutputPlanes + 1]
            #new_pred_dict['non_plane_mask'] = segmentations[:, :, :, :options.numOutputPlanes]
            new_pred_dict['depth'] = np.array(predDepths)
            predictions[method_index] = new_pred_dict
            pass

        if method[1] == 'crf':
            pred_dict = predictions[method_index]
            predSegmentations = []
            predDepths = []
            for image_index in xrange(options.numImages):
                print('crf ' + str(image_index + options.startIndex))
                boundaries = pred_dict['boundary'][image_index]
                boundaries = sigmoid(boundaries)
                cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_boundary.png', drawMaskImage(np.concatenate([boundaries, np.zeros((HEIGHT, WIDTH, 1))], axis=2)))

                allSegmentations = np.concatenate([pred_dict['segmentation'][image_index], pred_dict['np_mask'][image_index]], axis=2)
                allSegmentations = softmax(allSegmentations)
                planeDepths = calcPlaneDepths(pred_dict['plane'][image_index], WIDTH, HEIGHT, gt_dict['info'][image_index])
                allDepths = np.concatenate([planeDepths, pred_dict['np_depth'][image_index]], axis=2)
                #boundaries = np.concatenate([np.ones((allSegmentations.shape[0], allSegmentations.shape[1], 1)), -np.ones((allSegmentations.shape[0], allSegmentations.shape[1], 1))], axis=2)
                #if options.imageIndex >= 0:
                #boundaries = cv2.imread(options.test_dir + '/' + str(options.imageIndex) + '_boundary.png')
                #else:
                #boundaries = cv2.imread(options.test_dir + '/' + str(image_index + options.startIndex) + '_boundary.png')
                #pass
                #boundaries = (boundaries > 128).astype(np.float32)[:, :, :2]

                allDepths[:, :, options.numOutputPlanes] = 0
                pred_s = refineSegmentation(gt_dict['image'][image_index], allSegmentations, allDepths, boundaries, numOutputPlanes = 20, numIterations=20, numProposals=5)
                pred_d = allDepths.reshape(-1, options.numOutputPlanes + 1)[np.arange(WIDTH * HEIGHT), pred_s.reshape(-1)].reshape(HEIGHT, WIDTH)

                predSegmentations.append(pred_s)
                predDepths.append(pred_d)

                cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_d))
                cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage(pred_s, blackIndex=options.numOutputPlanes))

                #segmentation = np.argmax(allSegmentations, axis=-1)
                for planeIndex in xrange(options.numOutputPlanes + 1):
                    cv2.imwrite('test/mask_' + str(planeIndex) + '.png', drawMaskImage(allSegmentations[:, :, planeIndex]))
                    continue
                #cv2.imwrite(options.test_dir + '/mask_' + str(21) + '.png', drawDepthImage(pred_dict['np_depth'][0]))
                #for plane_index in xrange(options.numOutputPlanes + 1):
                #cv2.imwrite(options.test_dir + '/mask_' + str(plane_index) + '.png', drawMaskImage(pred_s == plane_index))
                #continue
                #exit(1)
                continue

            new_pred_dict = {}
            for key, value in pred_dict.iteritems():
                new_pred_dict[key] = value
                continue
            new_pred_dict['segmentation'] = np.array(predSegmentations)
            new_pred_dict['depth'] = np.array(predDepths)
            predictions[method_index] = new_pred_dict
            pass


        if 'pixelwise' in method[1]:
            pred_dict = predictions[method_index]
            predPlanes = []
            predSegmentations = []
            predDepths = []
            for image_index in xrange(options.numImages):
                pred_d = pred_dict['np_depth'][image_index].squeeze()
                if '_1' in method[1]:
                    pred_s = np.zeros(pred_dict['segmentation'][image_index].shape)
                    pred_p = np.zeros(pred_dict['plane'][image_index].shape)
                elif '_2' in methods[1]:
                    pred_p, pred_s, pred_d = fitPlanes(pred_d, gt_dict['info'][image_index], numPlanes=20, planeAreaThreshold=3*4, numIterations=100, distanceThreshold=0.05, local=0.2)
                elif '_3' in methods[1]:
                    pred_p, pred_s, pred_d = fitPlanes(pred_d, gt_dict['info'][image_index], numPlanes=20, planeAreaThreshold=3*4, numIterations=100, distanceThreshold=0.05, local=0.2)
                elif '_4' in methods[1]:
                    pred_p, pred_s, pred_d = fitPlanesSegmentation(pred_d, pred_dict['semantics'][image_index], gt_dict['info'][image_index], numPlanes=20, planeAreaThreshold=3*4, numIterations=100, distanceThreshold=0.05, local=0.2)
                    pass
                predPlanes.append(pred_p)
                predSegmentations.append(pred_s)
                predDepths.append(pred_d)

                cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_d))
                cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage(pred_s))
                continue
            new_pred_dict = {}
            for key, value in pred_dict.iteritems():
                new_pred_dict[key] = value
                continue
            new_pred_dict['plane'] = np.array(predPlanes)
            new_pred_dict['segmentation'] = np.array(predSegmentations)
            new_pred_dict['depth'] = np.array(predDepths)
            predictions[method_index] = new_pred_dict
            #titles.append('pixelwise+semantics+RANSAC')
            pass
        continue



    for method_index, pred_dict in enumerate(predictions):
        print(titles[method_index])
        evaluateDepths(pred_dict['depth'], gt_dict['depth'], np.ones(gt_dict['depth'].shape))
        continue
    return

def getResults(options):
    checkpoint_prefix = options.rootFolder + '/checkpoint/'

    methods = options.methods
    predictions = []

    if os.path.exists(options.result_filename):
        if options.useCache == 1:
            results = np.load(options.result_filename)
            results = results[()]
            return results
        elif options.useCache == 2:
            results = np.load(options.result_filename)
            results = results[()]
            gt_dict = results['gt']
            predictions = results['pred']
        else:
            gt_dict = getGroundTruth(options)
            pass
    else:
        gt_dict = getGroundTruth(options)
        pass



    for method_index, method in enumerate(methods):
        if len(method) < 4 or method[3] < 2:
            continue
        if method[0] == '':
            continue

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
        if 'ps' in method[0]:
            options.predictSemantics = 1
        else:
            options.predictSemantics = 0
            pass
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

        #options.suffix = method[1]

        method_names = [previous_method[0] for previous_method in methods[:method_index]]

        if method[0] in method_names:
            pred_dict = predictions[method_names.index(method[0])]
        elif method[0] == 'gt':
            pred_dict = gt_dict
        else:
            pred_dict = getPrediction(options)
            pass

        # for image_index in xrange(options.visualizeImages):
        #     cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_dict['depth'][image_index]))
        #     cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage())
        #     continue

        if len(method) >= 4 and method[3] == 3:
            predictions.insert(0, pred_dict)
        else:
            if method_index < len(predictions):
                predictions[method_index] = pred_dict
            else:
                predictions.append(pred_dict)
                pass
            pass
        continue
    #np.save(options.test_dir + '/curves.npy', curves)
    results = {'gt': gt_dict, 'pred': predictions}

    if options.useCache != -1:
        np.save(options.result_filename, results)
        pass
    pass

    return results

def getPrediction(options):
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


    width_high_res = 640
    height_high_res = 480


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
            predPlanes = []
            predSegmentations = []
            predSemantics = []
            predNonPlaneDepths = []
            predNonPlaneNormals = []
            predNonPlaneMasks = []
            predBoundaries = []
            for index in xrange(options.startIndex + options.numImages):
                if index % 10 == 0:
                    print(('image', index))
                    pass
                t0=time.time()

                img, global_gt, global_pred = sess.run([img_inp, global_gt_dict, global_pred_dict])

                if index < options.startIndex:
                    continue


                pred_p = global_pred['plane'][0]
                pred_s = global_pred['segmentation'][0]

                pred_np_m = global_pred['non_plane_mask'][0]
                pred_np_d = global_pred['non_plane_depth'][0]
                pred_np_n = global_pred['non_plane_normal'][0]

                if global_gt['info'][0][19] > 1 and global_gt['info'][0][19] < 4 and False:
                    pred_np_n = calcNormal(pred_np_d.squeeze(), global_gt['info'][0])
                    pass


                #pred_b = global_pred['boundary'][0]
                predNonPlaneMasks.append(pred_np_m)
                predNonPlaneDepths.append(pred_np_d)
                predNonPlaneNormals.append(pred_np_n)
                #predBoundaries.append(pred_b)

                all_segmentations = np.concatenate([pred_s, pred_np_m], axis=2)

                plane_depths = calcPlaneDepths(pred_p, width_high_res, height_high_res, global_gt['info'][0])

                pred_np_d = np.expand_dims(cv2.resize(pred_np_d.squeeze(), (width_high_res, height_high_res)), -1)
                all_depths = np.concatenate([plane_depths, pred_np_d], axis=2)

                all_segmentations = np.stack([cv2.resize(all_segmentations[:, :, planeIndex], (width_high_res, height_high_res)) for planeIndex in xrange(all_segmentations.shape[-1])], axis=2)

                segmentation = np.argmax(all_segmentations, 2)
                pred_d = all_depths.reshape(-1, options.numOutputPlanes + 1)[np.arange(height_high_res * width_high_res), segmentation.reshape(-1)].reshape(height_high_res, width_high_res)

                if 'semantics' in global_pred:
                    #cv2.imwrite('test/semantics.png', drawSegmentationImage(np.argmax(global_pred['semantics'][0], axis=-1)))
                    #exit(1)
                    predSemantics.append(np.argmax(global_pred['semantics'][0], axis=-1))
                else:
                    predSemantics.append(np.zeros((HEIGHT, WIDTH)))
                    pass

                predDepths.append(pred_d)
                predPlanes.append(pred_p)
                predSegmentations.append(segmentation)
                continue
            pred_dict['plane'] = np.array(predPlanes)
            pred_dict['segmentation'] = np.array(predSegmentations)
            pred_dict['depth'] = np.array(predDepths)
            #pred_dict['semantics'] = np.array(predSemantics)
            pred_dict['np_depth'] = np.array(predNonPlaneDepths)
            #pred_dict['np_normal'] = np.array(predNonPlaneNormals)
            pred_dict['np_mask'] = np.array(predNonPlaneMasks)
            #pred_dict['boundary'] = np.array(predBoundaries)
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

    # if options.dataset == 'NYU_RGBD':
    #     global_gt_dict['segmentation'], global_gt_dict['plane_mask'] = tf.ones((options.batchSize, HEIGHT, WIDTH, options.numOutputPlanes)), tf.ones((options.batchSize, HEIGHT, WIDTH, 1))
    # elif options.dataset == 'SUNCG':
    #     normalDotThreshold = np.cos(np.deg2rad(5))
    #     distanceThreshold = 0.05
    #     global_gt_dict['segmentation'], global_gt_dict['plane_mask'] = fitPlaneMasksModule(global_gt_dict['plane'], global_gt_dict['depth'], global_gt_dict['normal'], width=WIDTH, height=HEIGHT, normalDotThreshold=normalDotThreshold, distanceThreshold=distanceThreshold, closing=True, one_hot=True)
    # else:
    #     global_gt_dict['plane_mask'] = 1 - global_gt_dict['non_plane_mask']
    #     pass

    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    gt_dict = {}

    width_high_res = 640
    height_high_res = 480

    with tf.Session(config=config) as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            gtDepths = []
            gtNormals = []
            planeMasks = []
            #predMasks = []
            gtPlanes = []
            gtSegmentations = []
            gtSemantics = []
            gtInfo = []
            gtNumPlanes = []
            images = []

            for index in xrange(options.startIndex + options.numImages):
                print(('image', index))
                t0=time.time()

                img, global_gt = sess.run([img_inp, global_gt_dict])

                if index < options.startIndex:
                    continue


                imagePath = global_gt['image_path'][0]
                #exit(1)
                # if index == 11:
                #     cv2.imwrite('test/mask.png', drawMaskImage(global_gt['non_plane_mask'].squeeze()))
                #     exit(1)

                #image = ((img[0] + 0.5) * 255).astype(np.uint8)
                image = cv2.imread(imagePath)
                image = cv2.resize(image, (width_high_res, height_high_res))
                images.append(image)

                #cv2.imwrite(options.test_dir + '/' + str(index) + '_boundary.png', drawMaskImage(np.concatenate([global_gt['boundary'][0], np.zeros((HEIGHT, WIDTH, 1))], axis=2)))

                #gt_d = global_gt['depth'].squeeze()
                gt_d = np.array(PIL.Image.open(imagePath.replace('color.jpg', 'depth.pgm'))).astype(np.float32) / global_gt['info'][0][18]
                gt_d = cv2.resize(gt_d, (width_high_res, height_high_res), interpolation=cv2.INTER_LINEAR)
                gtDepths.append(gt_d)

                if global_gt['info'][0][19] == 3 and False:
                    gt_n = calcNormal(gt_d, global_gt['info'][0])
                    #cv2.imwrite('test/normal.png', drawNormalImage(gt_n))
                    #exit(1)
                else:
                    gt_n = global_gt['normal'][0]
                    pass
                gtNormals.append(gt_n)

                planeMask = np.squeeze(1 - global_gt['non_plane_mask'])
                planeMasks.append(planeMask)

                gt_p = global_gt['plane'][0]
                gtPlanes.append(gt_p)
                gt_s = global_gt['segmentation'][0]
                gtSegmentations.append(gt_s)
                gt_semantics = global_gt['semantics'][0]
                gtSemantics.append(gt_semantics)
                gt_num_p = global_gt['num_planes'][0]
                gtNumPlanes.append(gt_num_p)

                gtInfo.append(global_gt['info'][0])
                continue

            gt_dict['image'] = np.array(images)
            gt_dict['depth'] = np.array(gtDepths)
            gt_dict['normal'] = np.array(gtNormals)
            gt_dict['plane_mask'] = np.array(planeMasks)
            gt_dict['plane'] = np.array(gtPlanes)
            gt_dict['segmentation'] = np.array(gtSegmentations)
            gt_dict['semantics'] = np.array(gtSemantics)
            gt_dict['num_planes'] = np.array(gtNumPlanes)
            gt_dict['info'] = np.array(gtInfo)

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
                        default='predict', type=str)
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
                        default='0', type=str)
    parser.add_argument('--suffix', dest='suffix',
                        help='suffix',
                        default='', type=str)
    parser.add_argument('--rootFolder', dest='rootFolder',
                        help='root folder',
                        default='/mnt/vision/PlaneNet/', type=str)

    args = parser.parse_args()
    #args.hybrid = 'hybrid' + args.hybrid
    args.test_dir = 'evaluate/' + args.task + '/' + args.dataset + '/hybrid' + args.hybrid + '/'
    args.visualizeImages = min(args.visualizeImages, args.numImages)

    #args.titles = [ALL_TITLES[int(method)] for method in args.methods]
    #args.methods = [ALL_METHODS[int(method)] for method in args.methods]
    args.titles = ALL_TITLES
    args.methods = [ALL_METHODS[int(args.methods[0])]]

    args.result_filename = args.test_dir + '/results_' + str(args.startIndex) + '.npy'

    #if args.imageIndex >= 0 and args.suffix != '':
    if args.suffix != '':
        args.test_dir += '/' + args.suffix + '/'
        pass

    print(args.titles)

    if args.task == 'predict':
        evaluatePlanes(args)
    elif args.task == 'depth':
        evaluateDepthPrediction(args)
    elif args.task == 'search':
        gridSearch(args)
        pass
