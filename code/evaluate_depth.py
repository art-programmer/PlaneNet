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

from train_planenet import build_graph
from train_sample import build_graph as build_graph_sample
from planenet import PlaneNet
from RecordReaderAll import *
from RecordReaderMake3D import *
#from RecordReaderRGBD import *
from SegmentationRefinement import *
import scipy.io as sio
import csv

#ALL_TITLES = ['planenet', 'pixelwise', 'pixelwise+RANSAC', 'depth observation+RANSAC', 'pixelwise+semantics+RANSAC', 'gt']
#ALL_METHODS = [('bl2_ll1_bw0.5_pb_pp_sm0', ''), ('pb_pp', 'pixelwise_1'), ('pb_pp', 'pixelwise_2'), ('pb_pp', 'pixelwise_3'), ('pb_pp', 'semantics'), ('pb_pp', 'gt')]

#ALL_TITLES = ['planenet label loss', 'planenet crf', 'planenet label backward', 'planenet different matching']
#ALL_METHODS = [('ll1_pb_pp', ''), ('crf1_pb_pp', ''), ('bl0_ll1_bw0.5_pb_pp_ps_sm0', ''), ('ll1_bw0.5_pb_pp_sm0', '')]

#ALL_METHODS = [('ll1_pb_pp', 'pixelwise_1'), ('crf1_pb_pp', 'pixelwise_2'), ('bl0_ll1_bw0.5_pb_pp_ps_sm0', 'pixelwise_3'), ('ll1_bw0.5_pb_pp_sm0', 'pixelwise_4')]


ALL_TITLES = ['planenet', 'pixelwise', 'fine-tuning', 'ScanNet', 'Make3D']
#ALL_METHODS = [('bl0_ll1_bw0.5_pp_ps_sm0', ''), ('bl0_ll1_bw0.5_pp_ps_sm0', 'pixelwise_1')]
#ALL_METHODS = [('planenet_hybrid1_bl0_ll1_ds0_pp_ps', ''), ('pixelwise_hybrid1_ps', 'pixelwise_1')]
#ALL_TITLES = ['crf', 'different matching']
#ALL_METHODS = [('pb_pp_sm0', 'crf'), ('pb_pp_sm0', '')]

#ALL_METHODS = [('planenet_hybrid1_bl0_ll1_ds0_pp_ps', ''), ('pixelwise_hybrid1_ps', '')]
ALL_METHODS = [('pixelwise_np10_hybrid1_ds0', ''), ('finetuning_hybrid1_ps', ''), ('finetuning_np10_hybrid1_ds0_ps', ''), ('sample_np10_hybrid3_bl0_dl0_ds0_crfrnn5_sm0', ''), ('finetuning_np10_hybrid4_ds0', '')]

def writeHTML(options):
    from html import HTML

    titles = options.titles

    h = HTML('html')
    h.p('Results')
    h.br()
    path = '.'
    #methods = ['planenet', 'pixelwise', 'pixelwise+RANSAC', 'GT+RANSAC', 'planenet+crf', 'pixelwise+semantics+RANSAC']
    #methods = ['planenet', 'pixelwise', 'pixelwise+RANSAC', 'GT+RANSAC']

    for index in xrange(options.numImages):

        t = h.table(border='1')
        r_inp = t.tr()
        r_inp.td('input')
        r_inp.td().img(src=path + '/' + str(index) + '_image.png')
        r_inp.td().img(src=path + '/' + str(index) + '_depth_gt.png')
        #r_inp.td().img(src=path + '/' + str(index) + '_depth_gt_plane.png')
        r_inp.td().img(src=path + '/' + str(index) + '_normal_gt.png')
        #r_inp.td().img(src=path + '/' + str(index) + '_segmentation_gt.png')
        r_inp.td().img(src='/home/chenliu/Projects/PlaneNet/code/test/' + str(index) + '_dominant_lines.png')
        #r_inp.td().img(src=path + '/' + str(index) + '_semantics_gt.png')

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
            r.td().img(src=path + '/' + str(index) + '_segmentation_pred_' + str(method_index) + '.png')
            continue

        continue

        r = t.tr()
        r.td('depth')
        for method_index, method in enumerate(titles):
            r.td().img(src=path + '/' + str(index) + '_depth_pred_' + str(method_index) + '.png')
            continue

        r = t.tr()
        r.td('depth_diff')
        for method_index, method in enumerate(titles):
            r.td().img(src=path + '/' + str(index) + '_depth_diff_' + str(method_index) + '.png')
            continue

        r = t.tr()
        r.td('normal')
        for method_index, method in enumerate(titles):
            r.td().img(src=path + '/' + str(index) + '_normal_pred_' + str(method_index) + '.png')
            continue

        # r = t.tr()
        # r.td('depth_normal')
        # for method_index, method in enumerate(titles):
        #     r.td().img(src=path + '/' + str(index) + '_depth_normal_pred_' + str(method_index) + '.png')
        #     continue

        h.br()
        continue

    metric_titles = ['plane diff 0.1', 'plane diff 0.3', 'plane diff 0.5', 'IOU 0.3', 'IOU 0.5', 'IOU 0.7']

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

    for key, value in gt_dict.iteritems():
        if options.imageIndex >= 0:
            gt_dict[key] = value[options.imageIndex:options.imageIndex + 1]
        elif value.shape[0] > options.numImages:
            gt_dict[key] = value[:options.numImages]
            pass
        continue
    for pred_dict in predictions:
        for key, value in pred_dict.iteritems():
            if options.imageIndex >= 0:
                pred_dict[key] = value[options.imageIndex:options.imageIndex + 1]
            elif value.shape[0] > options.numImages:
                pred_dict[key] = value[:options.numImages]
                pass
            continue
        continue

    #methods = ['planenet', 'pixelwise+RANSAC', 'GT+RANSAC']
    titles = options.titles




    #predictions[2] = predictions[3]




    for image_index in xrange(options.visualizeImages):
        cv2.imwrite(options.test_dir + '/' + str(image_index) + '_image.png', gt_dict['image'][image_index])
        cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_gt.png', drawDepthImage(gt_dict['depth'][image_index]))
        cv2.imwrite(options.test_dir + '/' + str(image_index) + '_normal_gt.png', drawNormalImage(gt_dict['normal'][image_index]))
        cv2.imwrite(options.test_dir + '/' + str(image_index) + '_segmentation_gt.png', drawSegmentationImage(gt_dict['segmentation'][image_index], blackIndex=options.numOutputPlanes))
        cv2.imwrite(options.test_dir + '/' + str(image_index) + '_semantics_gt.png', drawSegmentationImage(gt_dict['semantics'][image_index], blackIndex=0))


        # plane_depths = calcPlaneDepths(gt_dict['plane'][image_index], options.width, options.height, gt_dict['info'][image_index])
        # all_depths = np.concatenate([plane_depths, np.expand_dims(gt_dict['depth'][image_index], -1)], axis=2)
        # depth = np.sum(all_depths * np.concatenate([gt_dict['segmentation'][image_index], 1 - np.expand_dims(gt_dict['plane_mask'][image_index], -1)], axis=2), axis=2)
        # cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_gt_plane.png', drawDepthImage(depth))

        for method_index, pred_dict in enumerate(predictions):
            cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_dict['depth'][image_index]))

            if titles[method_index] == 'pixelwise':
                continue
            segmentation = pred_dict['segmentation'][image_index]
            cv2.imwrite(options.test_dir + '/' + str(image_index) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage(segmentation, blackIndex=options.numOutputPlanes))
            cv2.imwrite(options.test_dir + '/' + str(image_index) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage(pred_dict['semantics'][image_index], blackIndex=0))
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
                print('graph cut ' + str(image_index))

                segmentation = pred_dict['segmentation'][image_index]
                #pred_s = getSegmentationsGraphCut(pred_dict['plane'][image_index], gt_dict['image'][image_index], pred_dict['depth'][image_index], pred_dict['normal'][image_index], segmentation, pred_dict['semantics'][image_index], pred_dict['info'][image_index], gt_dict['num_planes'][image_index])

                pred_p, pred_s, numPlanes = removeSmallSegments(pred_dict['plane'][image_index], gt_dict['image'][image_index], pred_dict['depth'][image_index], pred_dict['normal'][image_index], segmentation, pred_dict['semantics'][image_index], pred_dict['info'][image_index], gt_dict['num_planes'][image_index])
                #pred_p, pred_s, numPlanes = pred_dict['plane'][image_index], segmentation, gt_dict['num_planes'][image_index]
                print((gt_dict['num_planes'][image_index], numPlanes))
                planeDepths = calcPlaneDepths(pred_p, options.width, options.height, gt_dict['info'][image_index])
                allDepths = np.concatenate([planeDepths, np.expand_dims(pred_dict['depth'][image_index], -1)], axis=2)
                pred_d = allDepths.reshape(-1, options.numOutputPlanes + 1)[np.arange(options.width * options.height), pred_s.reshape(-1)].reshape(options.height, options.width)

                predSegmentations.append(pred_s)
                predDepths.append(pred_d)

                cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_d))
                cv2.imwrite(options.test_dir + '/' + str(image_index) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage(pred_s, blackIndex=options.numOutputPlanes))
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

            image_inp = tf.placeholder(tf.float32, shape=[1, options.height, options.width, 3], name='image')
            segmentation_inp = tf.placeholder(tf.float32, shape=[1, options.height, options.width, options.numOutputPlanes + 1], name='segmentation')
            plane_inp = tf.placeholder(tf.float32, shape=[1, options.numOutputPlanes, 3], name='plane')
            non_plane_depth_inp = tf.placeholder(tf.float32, shape=[1, options.height, options.width, 1], name='non_plane_depth')
            info_inp = tf.placeholder(tf.float32, shape=[20], name='info')


            plane_parameters = tf.reshape(plane_inp, (-1, 3))
            plane_depths = planeDepthsModule(plane_parameters, options.width, options.height, info_inp)
            plane_depths = tf.transpose(tf.reshape(plane_depths, [options.height, options.width, -1, options.numOutputPlanes]), [2, 0, 1, 3])
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
                    print('crf tf ' + str(image_index))
                    allSegmentations = np.concatenate([pred_dict['segmentation'][image_index], pred_dict['np_mask'][image_index]], axis=2)
                    allSegmentations = softmax(allSegmentations)
                    pred_s, debug = sess.run([refined_segmentation, debug_dict], feed_dict={segmentation_inp: np.expand_dims(allSegmentations, 0), plane_inp: np.expand_dims(pred_dict['plane'][image_index], 0), non_plane_depth_inp: np.expand_dims(pred_dict['np_depth'][image_index], 0), info_inp: gt_dict['info'][image_index], image_inp: gt_dict['image'][image_index:image_index + 1]})

                    pred_s = pred_s[0]
                    planeDepths = calcPlaneDepths(pred_dict['plane'][image_index], options.width, options.height, gt_dict['info'][image_index])
                    allDepths = np.concatenate([planeDepths, pred_dict['np_depth'][image_index]], axis=2)
                    pred_d = np.sum(allDepths * pred_s, axis=-1)

                    predSegmentations.append(pred_s)
                    predDepths.append(pred_d)

                    cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_d))
                    cv2.imwrite(options.test_dir + '/' + str(image_index) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage(pred_s, blackIndex=options.numOutputPlanes))

                    if 'diff' in debug:
                        segmentation = np.argmax(allSegmentations, axis=-1)
                        for planeIndex in xrange(options.numOutputPlanes + 1):
                            cv2.imwrite('test/mask_' + str(planeIndex) + '.png', drawMaskImage(allSegmentations[:, :, planeIndex]))
                            continue

                        for planeIndex in xrange(debug['diff'].shape[-1]):
                            cv2.imwrite('test/cost_mask_' + str(planeIndex) + '.png', drawMaskImage(debug['diff'][0, :, :, planeIndex] / 2))
                            continue
                        #exit(1)
                        pass
                    continue
                pass
            new_pred_dict = {}
            for key, value in pred_dict.iteritems():
                new_pred_dict[key] = value
                continue
            segmentations = np.array(predSegmentations)
            new_pred_dict['segmentation'] = segmentations[:, :, :, :options.numOutputPlanes]
            #new_pred_dict['non_plane_mask'] = segmentations[:, :, :, options.numOutputPlanes:options.numOutputPlanes + 1]
            #new_pred_dict['non_plane_mask'] = segmentations[:, :, :, :options.numOutputPlanes]
            new_pred_dict['depth'] = np.array(predDepths)
            predictions[method_index] = new_pred_dict
            pass

        if method[1] == 'crf':
            pred_dict = predictions[method_index]
            predSegmentations = []
            predDepths = []
            for image_index in xrange(options.numImages):
                print('crf ' + str(image_index))
                boundaries = pred_dict['boundary'][image_index]
                boundaries = sigmoid(boundaries)
                cv2.imwrite(options.test_dir + '/' + str(image_index) + '_boundary.png', drawMaskImage(np.concatenate([boundaries, np.zeros((options.height, options.width, 1))], axis=2)))

                allSegmentations = np.concatenate([pred_dict['segmentation'][image_index], pred_dict['np_mask'][image_index]], axis=2)
                allSegmentations = softmax(allSegmentations)
                planeDepths = calcPlaneDepths(pred_dict['plane'][image_index], options.width, options.height, gt_dict['info'][image_index])
                allDepths = np.concatenate([planeDepths, pred_dict['np_depth'][image_index]], axis=2)
                #boundaries = np.concatenate([np.ones((allSegmentations.shape[0], allSegmentations.shape[1], 1)), -np.ones((allSegmentations.shape[0], allSegmentations.shape[1], 1))], axis=2)
                #if options.imageIndex >= 0:
                #boundaries = cv2.imread(options.test_dir + '/' + str(options.imageIndex) + '_boundary.png')
                #else:
                #boundaries = cv2.imread(options.test_dir + '/' + str(image_index) + '_boundary.png')
                #pass
                #boundaries = (boundaries > 128).astype(np.float32)[:, :, :2]

                allDepths[:, :, options.numOutputPlanes] = 0
                pred_s = refineSegmentation(gt_dict['image'][image_index], allSegmentations, allDepths, boundaries, numOutputPlanes = 20, numIterations=20, numProposals=5)
                pred_d = allDepths.reshape(-1, options.numOutputPlanes + 1)[np.arange(options.width * options.height), pred_s.reshape(-1)].reshape(options.height, options.width)

                predSegmentations.append(pred_s)
                predDepths.append(pred_d)

                cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_d))
                cv2.imwrite(options.test_dir + '/' + str(image_index) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage(pred_s, blackIndex=options.numOutputPlanes))

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
                print('pixelwise ', image_index)
                pred_d = pred_dict['np_depth'][image_index].squeeze()
                if '_1' in method[1]:
                    pred_s = np.zeros(pred_dict['segmentation'][image_index].shape)
                    pred_p = np.zeros(pred_dict['plane'][image_index].shape)
                elif '_2' in methods[1]:
                    pred_p, pred_s, pred_d = fitPlanes(pred_d, gt_dict['info'][image_index], numPlanes=20, planeAreaThreshold=3*4, numIterations=100, distanceThreshold=0.05, local=0.2)
                elif '_3' in methods[1]:
                    pred_p, pred_s, pred_d = fitPlanes(pred_d, gt_dict['info'][image_index], numPlanes=20, planeAreaThreshold=3*4, numIterations=100, distanceThreshold=0.05, local=0.2)
                elif '_4' in methods[1]:
                    pred_p, pred_s, pred_d = fitPlanesSegmentation(pred_d, pred_dict['semantics'][image_index], gt_dict['info'][image_index], numOutputPlanes=20, planeAreaThreshold=3*4, numIterations=100, distanceThreshold=0.05, local=0.2)
                    pass
                predPlanes.append(pred_p)
                predSegmentations.append(pred_s)
                predDepths.append(pred_d)

                cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_d))
                cv2.imwrite(options.test_dir + '/' + str(image_index) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage(pred_s))
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


    #exit(1)

    #print(results)

    # depth = gt_dict['depth'][4]
    # cv2.imwrite(options.test_dir + '/test_depth_gt.png', drawDepthImage(depth))
    # pred_p, pred_s, pred_d = fitPlanes(depth, getSUNCGCamera(), numPlanes=20, planeAreaThreshold=3*4, numIterations=100, distanceThreshold=0.05, local=0.2)
    # cv2.imwrite(options.test_dir + '/test_depth.png', drawDepthImage(pred_d))
    # cv2.imwrite(options.test_dir + '/test_segmentation.png', drawSegmentationImage(pred_s))
    # exit(1)




    pixel_metric_curves = [[], [], [], [], [], []]
    plane_metric_curves = [[], [], [], [], [], []]
    for method_index, pred_dict in enumerate(predictions):
        if titles[method_index] == 'pixelwise':
            continue
        segmentations = pred_dict['segmentation']
        if method_index == 0:
            segmentations = softmax(segmentations)
            pass
        pixel_curves, plane_curves = evaluatePlaneSegmentation(pred_dict['plane'], segmentations, gt_dict['plane'], gt_dict['segmentation'], gt_dict['num_planes'], numOutputPlanes = options.numOutputPlanes)

        for metric_index, pixel_curve in enumerate(pixel_curves):
            pixel_metric_curves[metric_index].append(pixel_curve)
            continue
        for metric_index, plane_curve in enumerate(plane_curves):
            plane_metric_curves[metric_index].append(plane_curve)
            continue
        continue

    xs = []
    xs.append((np.arange(11) * 0.1).tolist())
    xs.append((np.arange(11) * 0.1).tolist())
    xs.append((np.arange(11) * 0.1).tolist())
    xs.append((np.arange(11) * 0.05).tolist())
    xs.append((np.arange(11) * 0.05).tolist())
    xs.append((np.arange(11) * 0.05).tolist())
    xlabels = ['IOU', 'IOU', 'IOU', 'plane diff', 'plane diff', 'plane diff']
    curve_titles = ['plane diff 0.1', 'plane diff 0.3', 'plane diff 0.5', 'IOU 0.3', 'IOU 0.5', 'IOU 0.7']
    curve_labels = [title for title in titles if title != 'pixelwise']
    for metric_index, curves in enumerate(pixel_metric_curves):
        filename = options.test_dir + '/curve_pixel_' + curve_titles[metric_index].replace(' ', '_') + '.png'
        plotCurves(xs[metric_index], curves, filename = filename, xlabel=xlabels[metric_index], ylabel='pixel coverage', title=curve_titles[metric_index], labels=curve_labels)
        continue
    for metric_index, curves in enumerate(plane_metric_curves):
        filename = options.test_dir + '/curve_plane_' + curve_titles[metric_index].replace(' ', '_') + '.png'
        plotCurves(xs[metric_index], curves, filename = filename, xlabel=xlabels[metric_index], ylabel='plane accuracy', title=curve_titles[metric_index], labels=curve_labels)
        continue

    writeHTML(options)
    return

def evaluateDepthPrediction(options):


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

    for key, value in gt_dict.iteritems():
        if options.imageIndex >= 0:
            gt_dict[key] = value[options.imageIndex:options.imageIndex + 1]
        elif value.shape[0] > options.numImages:
            gt_dict[key] = value[:options.numImages]
            pass
        continue
    for pred_dict in predictions:
        for key, value in pred_dict.iteritems():
            if options.imageIndex >= 0:
                pred_dict[key] = value[options.imageIndex:options.imageIndex + 1]
            elif value.shape[0] > options.numImages:
                pred_dict[key] = value[:options.numImages]
                pass
            continue
        continue

    titles = options.titles


    for image_index in xrange(options.visualizeImages):
        cv2.imwrite(options.test_dir + '/' + str(image_index) + '_image.png', gt_dict['image'][image_index])
        cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_gt.png', drawDepthImage(gt_dict['depth'][image_index]))
        cv2.imwrite(options.test_dir + '/' + str(image_index) + '_normal_gt.png', drawNormalImage(gt_dict['normal'][image_index]))
        cv2.imwrite(options.test_dir + '/' + str(image_index) + '_segmentation_gt.png', drawSegmentationImage(gt_dict['segmentation'][image_index], blackIndex=options.numOutputPlanes))
        cv2.imwrite(options.test_dir + '/' + str(image_index) + '_semantics_gt.png', drawSegmentationImage(gt_dict['semantics'][image_index], blackIndex=0))


        #plane_depths = calcPlaneDepths(gt_dict['plane'][image_index], options.width, options.height, gt_dict['info'][image_index])
        #all_depths = np.concatenate([plane_depths, np.expand_dims(gt_dict['depth'][image_index], -1)], axis=2)
        #depth = all_depths.reshape(-1, options.numOutputPlanes + 1)[np.arange(options.width * options.height), gt_dict['segmentation'][image_index].reshape(-1)].reshape(options.height, options.width)
        #cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_gt_plane.png', drawDepthImage(depth))

        #cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_gt_plane.png', drawMaskImage((depth - gt_dict['depth'][image_index]) * 10 + 0.5))

        # plane_depths = calcPlaneDepths(gt_dict['plane'][image_index], options.width, options.height, gt_dict['info'][image_index])
        # all_depths = np.concatenate([plane_depths, np.expand_dims(gt_dict['depth'][image_index], -1)], axis=2)
        # depth = np.sum(all_depths * np.concatenate([gt_dict['segmentation'][image_index], 1 - np.expand_dims(gt_dict['plane_mask'][image_index], -1)], axis=2), axis=2)
        # cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_gt_plane.png', drawDepthImage(depth))

        for method_index, pred_dict in enumerate(predictions):
            cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_dict['depth'][image_index]))

            cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_diff_' + str(method_index) + '.png', drawMaskImage(np.abs(pred_dict['depth'][image_index] - gt_dict['depth'][image_index]) * 1))

            cv2.imwrite(options.test_dir + '/' + str(image_index) + '_normal_pred_' + str(method_index) + '.png', drawNormalImage(pred_dict['normal'][image_index]))

            #normal = calcNormal(pred_dict['depth'][image_index], gt_dict['info'][image_index])
            #cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_normal_pred_' + str(method_index) + '.png', drawNormalImage(pred_dict['depth_normal'][image_index]))
            #exit(1)
            if titles[method_index] == 'pixelwise':
                continue
            segmentation = pred_dict['segmentation'][image_index]
            cv2.imwrite(options.test_dir + '/' + str(image_index) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage(segmentation, blackIndex=options.numOutputPlanes))
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
                print('graph cut ' + str(image_index))

                segmentation = np.argmax(np.concatenate([pred_dict['segmentation'][image_index], 1 - np.expand_dims(pred_dict['plane_mask'][image_index], -1)], axis=2), axis=2)
                #pred_s = getSegmentationsGraphCut(pred_dict['plane'][image_index], gt_dict['image'][image_index], pred_dict['depth'][image_index], pred_dict['normal'][image_index], segmentation, pred_dict['semantics'][image_index], pred_dict['info'][image_index], gt_dict['num_planes'][image_index])

                pred_p, pred_s, numPlanes = removeSmallSegments(pred_dict['plane'][image_index], gt_dict['image'][image_index], pred_dict['depth'][image_index], pred_dict['normal'][image_index], segmentation, pred_dict['semantics'][image_index], pred_dict['info'][image_index], gt_dict['num_planes'][image_index])
                #pred_p, pred_s, numPlanes = pred_dict['plane'][image_index], segmentation, gt_dict['num_planes'][image_index]
                print((gt_dict['num_planes'][image_index], numPlanes))
                planeDepths = calcPlaneDepths(pred_p, options.width, options.height, gt_dict['info'][image_index])
                allDepths = np.concatenate([planeDepths, np.expand_dims(pred_dict['depth'][image_index], -1)], axis=2)
                pred_d = allDepths.reshape(-1, options.numOutputPlanes + 1)[np.arange(options.width * options.height), pred_s.reshape(-1)].reshape(options.height, options.width)

                predSegmentations.append(pred_s)
                predDepths.append(pred_d)

                cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_d))
                cv2.imwrite(options.test_dir + '/' + str(image_index) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage(pred_s, blackIndex=options.numOutputPlanes))
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

            image_inp = tf.placeholder(tf.float32, shape=[1, options.height, options.width, 3], name='image')
            segmentation_inp = tf.placeholder(tf.float32, shape=[1, options.height, options.width, options.numOutputPlanes + 1], name='segmentation')
            plane_inp = tf.placeholder(tf.float32, shape=[1, options.numOutputPlanes, 3], name='plane')
            non_plane_depth_inp = tf.placeholder(tf.float32, shape=[1, options.height, options.width, 1], name='non_plane_depth')
            info_inp = tf.placeholder(tf.float32, shape=[20], name='info')


            plane_parameters = tf.reshape(plane_inp, (-1, 3))
            plane_depths = planeDepthsModule(plane_parameters, options.width, options.height, info_inp)
            plane_depths = tf.transpose(tf.reshape(plane_depths, [options.height, options.width, -1, options.numOutputPlanes]), [2, 0, 1, 3])
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
                    print('crf tf ' + str(image_index))
                    allSegmentations = np.concatenate([pred_dict['segmentation'][image_index], pred_dict['np_mask'][image_index]], axis=2)
                    allSegmentations = softmax(allSegmentations)
                    pred_s, debug = sess.run([refined_segmentation, debug_dict], feed_dict={segmentation_inp: np.expand_dims(allSegmentations, 0), plane_inp: np.expand_dims(pred_dict['plane'][image_index], 0), non_plane_depth_inp: np.expand_dims(pred_dict['np_depth'][image_index], 0), info_inp: gt_dict['info'][image_index], image_inp: gt_dict['image'][image_index:image_index + 1]})

                    pred_s = pred_s[0]
                    planeDepths = calcPlaneDepths(pred_dict['plane'][image_index], options.width, options.height, gt_dict['info'][image_index])
                    allDepths = np.concatenate([planeDepths, pred_dict['np_depth'][image_index]], axis=2)
                    pred_d = np.sum(allDepths * pred_s, axis=-1)

                    predSegmentations.append(pred_s)
                    predDepths.append(pred_d)

                    cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_d))
                    cv2.imwrite(options.test_dir + '/' + str(image_index) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage(pred_s, blackIndex=options.numOutputPlanes))

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
            #new_pred_dict['non_plane_mask'] = segmentations[:, :, :, options.numOutputPlanes:options.numOutputPlanes + 1]
            #new_pred_dict['non_plane_mask'] = segmentations[:, :, :, :options.numOutputPlanes]
            new_pred_dict['depth'] = np.array(predDepths)
            predictions[method_index] = new_pred_dict
            pass

        if method[1] == 'crf':
            pred_dict = predictions[method_index]
            predSegmentations = []
            predDepths = []
            for image_index in xrange(options.numImages):
                print('crf ' + str(image_index))
                boundaries = pred_dict['boundary'][image_index]
                boundaries = sigmoid(boundaries)
                cv2.imwrite(options.test_dir + '/' + str(image_index) + '_boundary.png', drawMaskImage(np.concatenate([boundaries, np.zeros((options.height, options.width, 1))], axis=2)))

                allSegmentations = np.concatenate([pred_dict['segmentation'][image_index], pred_dict['np_mask'][image_index]], axis=2)
                allSegmentations = softmax(allSegmentations)
                planeDepths = calcPlaneDepths(pred_dict['plane'][image_index], options.width, options.height, gt_dict['info'][image_index])
                allDepths = np.concatenate([planeDepths, pred_dict['np_depth'][image_index]], axis=2)
                #boundaries = np.concatenate([np.ones((allSegmentations.shape[0], allSegmentations.shape[1], 1)), -np.ones((allSegmentations.shape[0], allSegmentations.shape[1], 1))], axis=2)
                #if options.imageIndex >= 0:
                #boundaries = cv2.imread(options.test_dir + '/' + str(options.imageIndex) + '_boundary.png')
                #else:
                #boundaries = cv2.imread(options.test_dir + '/' + str(image_index) + '_boundary.png')
                #pass
                #boundaries = (boundaries > 128).astype(np.float32)[:, :, :2]

                allDepths[:, :, options.numOutputPlanes] = 0
                pred_s = refineSegmentation(gt_dict['image'][image_index], allSegmentations, allDepths, boundaries, numOutputPlanes = 20, numIterations=20, numProposals=5)
                pred_d = allDepths.reshape(-1, options.numOutputPlanes + 1)[np.arange(options.width * options.height), pred_s.reshape(-1)].reshape(options.height, options.width)

                predSegmentations.append(pred_s)
                predDepths.append(pred_d)

                cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_d))
                cv2.imwrite(options.test_dir + '/' + str(image_index) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage(pred_s, blackIndex=options.numOutputPlanes))

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
            predNormals = []
            for image_index in xrange(options.numImages):
                pred_d = pred_dict['np_depth'][image_index].squeeze()
                pred_n = pred_dict['np_normal'][image_index]
                if '_1' in method[1]:
                    pred_s = np.zeros(pred_dict['segmentation'][image_index].shape)
                    pred_p = np.zeros((options.numOutputPlanes, 3))
                elif '_2' in method[1]:
                    #pred_p, pred_s, pred_d, pred_n = fitPlanes(pred_d, gt_dict['info'][image_index], numPlanes=20, planeAreaThreshold=3*4, numIterations=100, distanceThreshold=0.05, local=0.2)
                    pred_p, pred_s, pred_d, pred_n = fitPlanesNYU(gt_dict['image'], gt_dict['depth'][image_index].squeeze(), gt_dict['normal'][image_index], gt_dict['semantics'], gt_dict['info'][image_index], numOutputPlanes=20, planeAreaThreshold=3*4, distanceThreshold=0.05, local=0.2)
                elif '_3' in method[1]:
                    #pred_p, pred_s, pred_d, pred_n = fitPlanes(pred_d, gt_dict['info'][image_index], numPlanes=20, planeAreaThreshold=3*4, numIterations=100, distanceThreshold=0.05, local=0.2)
                    pred_p, pred_s, pred_d, pred_n = fitPlanesNYU(gt_dict['image'], pred_d, pred_n, gt_dict['semantics'], gt_dict['info'][image_index], numOutputPlanes=20, planeAreaThreshold=3*4, distanceThreshold=0.05, local=0.2)
                elif '_4' in method[1]:
                    pred_p, pred_s, pred_d, pred_n = fitPlanesSegmentation(pred_d, pred_dict['semantics'][image_index], gt_dict['info'][image_index], numPlanes=20, planeAreaThreshold=3*4, numIterations=100, distanceThreshold=0.05, local=0.2)
                elif '_5' in method[1]:

                    # print(gt_dict['plane'][image_index] / np.linalg.norm(gt_dict['plane'][image_index], axis=-1, keepdims=True))
                    # cv2.imwrite('test/image.png', gt_dict['image'][image_index])
                    # for planeIndex in xrange(20):
                    #     cv2.imwrite('test/mask_' + str(planeIndex) + '.png', drawMaskImage(gt_dict['segmentation'][image_index][:, :, planeIndex]))
                    #     continue

                    pred_p, pred_s, pred_d, pred_n = fitPlanesManhattan(gt_dict['image'][image_index], pred_d, pred_n, gt_dict['info'][image_index], numOutputPlanes=20)
                elif '_6' in method[1]:

                    # print(gt_dict['plane'][image_index] / np.linalg.norm(gt_dict['plane'][image_index], axis=-1, keepdims=True))
                    # cv2.imwrite('test/depth.png', drawDepthImage(gt_dict['depth'][image_index]))
                    # cv2.imwrite('test/image.png', gt_dict['image'][image_index])
                    # for planeIndex in xrange(20):
                    #     cv2.imwrite('test/mask_' + str(planeIndex) + '.png', drawMaskImage(gt_dict['segmentation'][image_index][:, :, planeIndex]))
                    #     continue

                    pred_p, pred_s, pred_d, pred_n = fitPlanesManhattan(gt_dict['image'][image_index], gt_dict['depth'][image_index].squeeze(), gt_dict['normal'][image_index], gt_dict['info'][image_index], numOutputPlanes=20, imageIndex=image_index)
                    exit(1)

                    pass
                predPlanes.append(pred_p)
                predSegmentations.append(pred_s)
                predDepths.append(pred_d)
                predNormals.append(pred_n)

                cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_d))
                cv2.imwrite(options.test_dir + '/' + str(image_index) + '_normal_pred_' + str(method_index) + '.png', drawNormalImage(pred_n))
                cv2.imwrite(options.test_dir + '/' + str(image_index) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage(pred_s))
                #exit(1)
                continue
            new_pred_dict = {}
            for key, value in pred_dict.iteritems():
                new_pred_dict[key] = value
                continue
            new_pred_dict['plane'] = np.array(predPlanes)
            new_pred_dict['segmentation'] = np.array(predSegmentations)
            new_pred_dict['depth'] = np.array(predDepths)
            new_pred_dict['normal'] = np.array(predNormals)
            predictions[method_index] = new_pred_dict
            #titles.append('pixelwise+semantics+RANSAC')
            pass

        if 'gt' in method[1]:
            pred_dict = predictions[method_index]
            predPlanes = []
            predSegmentations = []
            predDepths = []
            predNormals = []
            for image_index in xrange(options.numImages):
                if '_s' in method[1]:
                    pred_s = np.argmax(np.concatenate([gt_dict['segmentation'][image_index], 1 - np.expand_dims(gt_dict['plane_mask'][image_index], -1)], axis=-1), axis=-1)
                else:
                    pred_s = np.argmax(np.concatenate([pred_dict['segmentation'][image_index], pred_dict['np_mask'][image_index]], axis=-1), axis=-1)
                    pass
                if '_p' in method[1]:
                    pred_p = gt_dict['plane'][image_index]
                else:
                    pred_p = pred_dict['plane'][image_index]
                    pass

                planeDepths = calcPlaneDepths(pred_p, options.width, options.height, gt_dict['info'][image_index])
                allDepths = np.concatenate([planeDepths, np.expand_dims(pred_dict['depth'][image_index], -1)], axis=2)
                pred_d = allDepths.reshape(-1, options.numOutputPlanes + 1)[np.arange(options.width * options.height), pred_s.reshape(-1)].reshape(options.height, options.width)

                plane_normals = calcPlaneNormals(pred_p, options.width, options.height)
                all_normals = np.concatenate([plane_normals, np.expand_dims(pred_dict['np_normal'][image_index], 2)], axis=2)
                pred_n = all_normals.reshape(-1, options.numOutputPlanes + 1, 3)[np.arange(options.width * options.height), pred_s.reshape(-1)].reshape((options.height, options.width, 3))

                predPlanes.append(pred_p)
                predSegmentations.append(pred_s)
                predDepths.append(pred_d)
                predNormals.append(pred_n)


                cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_d))
                cv2.imwrite(options.test_dir + '/' + str(image_index) + '_normal_pred_' + str(method_index) + '.png', drawNormalImage(pred_n))
                cv2.imwrite(options.test_dir + '/' + str(image_index) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage(pred_s, blackIndex=options.numOutputPlanes))
                continue

            new_pred_dict = {}
            for key, value in pred_dict.iteritems():
                new_pred_dict[key] = value
                continue
            new_pred_dict['segmentation'] = np.array(predSegmentations)
            new_pred_dict['plane'] = np.array(predPlanes)
            new_pred_dict['depth'] = np.array(predDepths)
            new_pred_dict['normal'] = np.array(predNormals)
            predictions[method_index] = new_pred_dict
        continue



    for method_index, pred_dict in enumerate(predictions):
        print(titles[method_index])
        evaluateDepths(pred_dict['depth'], gt_dict['depth'], np.ones(gt_dict['depth'].shape))
        evaluateDepths(pred_dict['depth'], gt_dict['depth'], np.ones(gt_dict['depth'].shape), planeMasks=gt_dict['segmentation'] < options.numOutputPlanes)

        #boundaries = gt_dict['semantics']
        #evaluateDepths(pred_dict['depth'], gt_dict['depth'], np.ones(gt_dict['depth'].shape), planeMasks=)

        #evaluateNormal(pred_dict['normal'], gt_dict['segmentation'], gt_dict['num_planes'], gt_dict['plane'])
        #evaluateNormal(pred_dict['depth_normal'], gt_dict['segmentation'])
        #evaluateNormal(pred_dict['np_normal'], gt_dict['segmentation'], gt_dict['num_planes'], gt_dict['plane'])
        #evaluateNormal(gt_dict['normal'], gt_dict['segmentation'], gt_dict['num_planes'], gt_dict['plane'])
        continue

    writeHTML(options)
    return

def getResults(options):
    checkpoint_prefix = options.rootFolder + '/checkpoint/'

    methods = options.methods

    if options.highRes == 1:
        gt_dict = getGroundTruthHighRes(options)
    else:
        gt_dict = getGroundTruth(options)
        pass

    predictions = []


    for method_index, method in enumerate(methods):
        if 'ds0' not in method[0]:
            options.deepSupervisionLayers = ['res4b22_relu', ]
        else:
            options.deepSupervisionLayers = []
            pass
        options.predictConfidence = 0
        options.predictLocal = 0
        options.predictPixelwise = 1
        options.predictBoundary = 0
        options.anchorPlanes = 0
        if 'ps' in method[0] and 'hybrid_' not in method[0]:
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
        if options.hybrid != '1':
            options.checkpoint_dir = options.checkpoint_dir.replace('hybrid1', 'hybrid' + str(options.hybrid))
            pass

        print(options.checkpoint_dir)

        options.suffix = method[1]

        method_names = [previous_method[0] for previous_method in methods[:method_index]]

        if method[0] in method_names:
            pred_dict = predictions[method_names.index(method[0])]
        elif method[0] == 'gt':
            pred_dict = gt_dict
        else:
            if options.highRes == 1:
                pred_dict = getPredictionHighRes(options)
            else:
                pred_dict = getPrediction(options)
                pass
            pass

        # for image_index in xrange(options.visualizeImages):
        #     cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_dict['depth'][image_index]))
        #     cv2.imwrite(options.test_dir + '/' + str(image_index) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage())
        #     continue

        predictions.append(pred_dict)
        continue
    #np.save(options.test_dir + '/curves.npy', curves)
    results = {'gt': gt_dict, 'pred': predictions}
    return results

def getPrediction(options):
    tf.reset_default_graph()

    options.batchSize = 1
    min_after_dequeue = 1000

    reader = RecordReaderAll()
    if options.dataset == 'SUNCG':
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_SUNCG_val.tfrecords'], num_epochs=10000)
    elif options.dataset == 'NYU_RGBD_raw':
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_nyu_rgbd_val.tfrecords'], num_epochs=1)
        options.deepSupervision = 0
        options.predictLocal = 0
    elif options.dataset == 'matterport':
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_matterport_val.tfrecords'], num_epochs=1)
    elif options.dataset == 'ScanNet':
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_scannet_val.tfrecords'], num_epochs=1)
    elif options.dataset == 'NYU_RGBD':
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_nyu_rgbd_labeled_val.tfrecords'], num_epochs=1)
        pass

    img_inp, global_gt_dict, local_gt_dict = reader.getBatch(filename_queue, numOutputPlanes=options.numOutputPlanes, batchSize=options.batchSize, min_after_dequeue=min_after_dequeue, getLocal=True, random=False)



    training_flag = tf.constant(False, tf.bool)

    options.gpu_id = 0
    if 'sample' not in options.checkpoint_dir:
        global_pred_dict, local_pred_dict, deep_pred_dicts = build_graph(img_inp, img_inp, training_flag, options)
    else:
        print('sample')
        global_pred_dict, local_pred_dict, deep_pred_dicts = build_graph_sample(img_inp, img_inp, training_flag, options)
        pass

    var_to_restore = tf.global_variables()


    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    print(options)

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
            predNormals = []
            predPlanes = []
            predSegmentations = []
            predNonPlaneDepths = []
            predNonPlaneMasks = []
            predNonPlaneNormals = []
            predBoundaries = []
            predDepthNormals = []
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

                #pred_b = global_pred['boundary'][0]
                predNonPlaneMasks.append(pred_np_m)
                predNonPlaneDepths.append(pred_np_d)
                predNonPlaneNormals.append(pred_np_n)
                #predBoundaries.append(pred_b)

                all_segmentations = np.concatenate([pred_np_m, pred_s], axis=2)
                plane_depths = calcPlaneDepths(pred_p, options.width, options.height, global_gt['info'][0])
                all_depths = np.concatenate([pred_np_d, plane_depths], axis=2)

                segmentation = np.argmax(all_segmentations, 2)
                pred_d = all_depths.reshape(-1, options.numOutputPlanes + 1)[np.arange(options.width * options.height), segmentation.reshape(-1)].reshape(options.height, options.width)

                plane_normals = calcPlaneNormals(pred_p, options.width, options.height)
                all_normals = np.concatenate([np.expand_dims(pred_np_n, 2), plane_normals], axis=2)
                pred_n = all_normals.reshape(-1, options.numOutputPlanes + 1, 3)[np.arange(options.width * options.height), segmentation.reshape(-1)].reshape((options.height, options.width, 3))

                predDepths.append(pred_d)
                predNormals.append(pred_n)
                predPlanes.append(pred_p)
                predSegmentations.append(pred_s)
                predDepthNormals.append(calcNormal(pred_d, global_gt['info'][0]))
                pass

                continue
            pred_dict['plane'] = np.array(predPlanes)
            pred_dict['segmentation'] = np.array(predSegmentations)
            pred_dict['depth'] = np.array(predDepths)
            pred_dict['normal'] = np.array(predNormals)
            pred_dict['np_depth'] = np.array(predNonPlaneDepths)
            pred_dict['np_mask'] = np.array(predNonPlaneMasks)
            pred_dict['np_normal'] = np.array(predNonPlaneNormals)
            pred_dict['depth_normal'] = np.array(predDepthNormals)
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

def getPredictionHighRes(options):
    tf.reset_default_graph()

    options.batchSize = 1
    min_after_dequeue = 1000

    if options.dataset == 'SUNCG':
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_SUNCG_val.tfrecords'], num_epochs=10000)
    elif options.dataset == 'NYU_RGBD_raw':
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_nyu_rgbd_val.tfrecords'], num_epochs=1)
        options.deepSupervision = 0
        options.predictLocal = 0
    elif options.dataset == 'matterport':
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_matterport_val.tfrecords'], num_epochs=1)
    elif options.dataset == 'ScanNet':
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_scannet_val.tfrecords'], num_epochs=1)
    elif options.dataset == 'NYU_RGBD':
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_nyu_rgbd_labeled_val.tfrecords'], num_epochs=1)
    elif options.dataset == 'Make3D':
        filename_queue = tf.train.string_input_producer(['../planes_make3d_val.tfrecords'], num_epochs=1)
        pass

    if options.dataset != 'Make3D':
        reader = RecordReaderAll()
    else:
        reader = RecordReaderMake3D()
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

    print(options)

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
            predNormals = []
            predPlanes = []
            predSegmentations = []
            predNonPlaneDepths = []
            predNonPlaneMasks = []
            predNonPlaneNormals = []
            predBoundaries = []
            predDepthNormals = []
            for index in xrange(options.startIndex + options.numImages):
                if index % 10 == 0:
                    print(('image', index))
                    pass
                t0=time.time()

                img, global_gt, global_pred = sess.run([img_inp, global_gt_dict, global_pred_dict])

                if index < options.startIndex:
                    continue

                info = global_gt['info'][0]

                width_high_res = int(info[16])
                height_high_res = int(info[17])

                pred_p = global_pred['plane'][0]
                pred_s = global_pred['segmentation'][0]

                pred_np_m = global_pred['non_plane_mask'].squeeze()
                pred_np_d = global_pred['non_plane_depth'].squeeze()
                pred_np_n = global_pred['non_plane_normal'][0]

                pred_np_m = cv2.resize(pred_np_m, (width_high_res, height_high_res), interpolation=cv2.INTER_LINEAR)
                pred_np_m = np.expand_dims(pred_np_m, -1)

                pred_np_d = cv2.resize(pred_np_d, (width_high_res, height_high_res), interpolation=cv2.INTER_LINEAR)
                pred_np_n = cv2.resize(pred_np_n, (width_high_res, height_high_res), interpolation=cv2.INTER_LINEAR)
                pred_np_d = np.expand_dims(pred_np_d, -1)

                pred_s_high_res = []
                for planeIndex in xrange(pred_s.shape[-1]):
                    pred_s_high_res.append(cv2.resize(pred_s[:, :, planeIndex], (width_high_res, height_high_res), interpolation=cv2.INTER_LINEAR))
                    continue
                pred_s = np.stack(pred_s_high_res, axis=2)
                #pred_b = global_pred['boundary'][0]

                predNonPlaneDepths.append(pred_np_d)
                predNonPlaneNormals.append(pred_np_n)
                #predBoundaries.append(pred_b)

                all_segmentations = np.concatenate([pred_np_m, pred_s], axis=2)
                plane_depths = calcPlaneDepths(pred_p, width_high_res, height_high_res, info)
                all_depths = np.concatenate([pred_np_d, plane_depths], axis=2)

                segmentation = np.argmax(all_segmentations, 2)
                pred_d = all_depths.reshape(-1, options.numOutputPlanes + 1)[np.arange(width_high_res * height_high_res), segmentation.reshape(-1)].reshape((height_high_res, width_high_res))

                plane_normals = calcPlaneNormals(pred_p, width_high_res, height_high_res)
                all_normals = np.concatenate([np.expand_dims(pred_np_n, 2), plane_normals], axis=2)
                pred_n = all_normals.reshape(-1, options.numOutputPlanes + 1, 3)[np.arange(width_high_res * height_high_res), segmentation.reshape(-1)].reshape((height_high_res, width_high_res, 3))

                predDepths.append(pred_d)
                predNormals.append(pred_n)
                predPlanes.append(pred_p)
                predSegmentations.append(segmentation)
                #predDepthNormals.append(calcNormal(pred_d, info))
                continue

            #pred_dict['plane'] = np.array(predPlanes)
            pred_dict['segmentation'] = np.array(predSegmentations)
            pred_dict['depth'] = np.array(predDepths)
            pred_dict['normal'] = np.array(predNormals)
            pred_dict['np_depth'] = np.array(predNonPlaneDepths)
            #pred_dict['np_mask'] = np.array(predNonPlaneMasks)
            pred_dict['np_normal'] = np.array(predNonPlaneNormals)
            #pred_dict['depth_normal'] = np.array(predDepthNormals)
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
    elif options.dataset == 'NYU_RGBD_raw':
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_nyu_rgbd_val.tfrecords'], num_epochs=1)
        options.deepSupervision = 0
        options.predictLocal = 0
    elif options.dataset == 'matterport':
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_matterport_val.tfrecords'], num_epochs=1)
    elif options.dataset == 'ScanNet':
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_scannet_val.tfrecords'], num_epochs=1)
    elif options.dataset == 'NYU_RGBD':
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_nyu_rgbd_labeled_val.tfrecords'], num_epochs=1)
        pass


    img_inp, global_gt_dict, local_gt_dict = reader.getBatch(filename_queue, numOutputPlanes=options.numOutputPlanes, batchSize=options.batchSize, min_after_dequeue=min_after_dequeue, getLocal=True, random=False)


    training_flag = tf.constant(False, tf.bool)

    # if options.dataset == 'NYU_RGBD':
    #     global_gt_dict['segmentation'], global_gt_dict['plane_mask'] = tf.ones((options.batchSize, options.height, options.width, options.numOutputPlanes)), tf.ones((options.batchSize, options.height, options.width, 1))
    # elif options.dataset == 'SUNCG':
    #     normalDotThreshold = np.cos(np.deg2rad(5))
    #     distanceThreshold = 0.05
    #     global_gt_dict['segmentation'], global_gt_dict['plane_mask'] = fitPlaneMasksModule(global_gt_dict['plane'], global_gt_dict['depth'], global_gt_dict['normal'], width=options.width, height=options.height, normalDotThreshold=normalDotThreshold, distanceThreshold=distanceThreshold, closing=True, one_hot=True)
    # else:
    #     global_gt_dict['plane_mask'] = 1 - global_gt_dict['non_plane_mask']
    #     pass

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


                # if index == 11:
                #     cv2.imwrite('test/mask.png', drawMaskImage(global_gt['non_plane_mask'].squeeze()))
                #     exit(1)
                image = ((img[0] + 0.5) * 255).astype(np.uint8)
                images.append(image)

                #cv2.imwrite(options.test_dir + '/' + str(index) + '_boundary.png', drawMaskImage(np.concatenate([global_gt['boundary'][0], np.zeros((options.height, options.width, 1))], axis=2)))

                gt_d = global_gt['depth'].squeeze()
                gtDepths.append(gt_d)

                gt_n = global_gt['normal'][0]
                gtNormals.append(gt_n)

                planeMask = np.squeeze(1 - global_gt['non_plane_mask'])
                planeMasks.append(planeMask)

                gt_p = global_gt['plane'][0]
                gtPlanes.append(gt_p)

                gt_s = global_gt['segmentation'][0]
                gtSegmentations.append(gt_s)


                # if options.dataset != 'NYU_RGBD':
                #gt_s = global_gt['segmentation'][0] == np.arange(options.numOutputPlanes).reshape([1, 1, -1]).astype(np.float32)
                #planeMask = 1 - np.squeeze(global_gt['segmentation'] == options.numOutputPlanes).astype(np.float32)
                #     gt_semantics = global_gt['semantics'][0]
                # else:
                #     semantics_path = global_gt['image_path'][0].replace('images_rgb', 'labels_objects').replace('rgb', 'labels')
                #     semantics_data = sio.loadmat(semantics_path)
                #     gt_semantics = semantics_data['imgObjectLabels']
                #     gt_semantics = cv2.resize(gt_semantics, (options.height, options.width))
                #     pass

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


def getGroundTruthHighRes(options):
    options.batchSize = 1
    min_after_dequeue = 1000

    if options.dataset == 'SUNCG':
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_SUNCG_val.tfrecords'], num_epochs=10000)
    elif options.dataset == 'NYU_RGBD_raw':
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_nyu_rgbd_val.tfrecords'], num_epochs=1)
        options.deepSupervision = 0
        options.predictLocal = 0
    elif options.dataset == 'matterport':
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_matterport_val.tfrecords'], num_epochs=1)
    elif options.dataset == 'ScanNet':
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_scannet_val.tfrecords'], num_epochs=1)
    elif options.dataset == 'NYU_RGBD':
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_nyu_rgbd_labeled_val.tfrecords'], num_epochs=1)
    elif options.dataset == 'Make3D':
        filename_queue = tf.train.string_input_producer(['../planes_make3d_val.tfrecords'], num_epochs=1)
        pass

    if options.dataset != 'Make3D':
        reader = RecordReaderAll()
    else:
        reader = RecordReaderMake3D()
        pass


    img_inp, global_gt_dict, local_gt_dict = reader.getBatch(filename_queue, numOutputPlanes=options.numOutputPlanes, batchSize=options.batchSize, min_after_dequeue=min_after_dequeue, getLocal=True, random=False)


    training_flag = tf.constant(False, tf.bool)

    # if options.dataset == 'NYU_RGBD':
    #     global_gt_dict['segmentation'], global_gt_dict['plane_mask'] = tf.ones((options.batchSize, options.height, options.width, options.numOutputPlanes)), tf.ones((options.batchSize, options.height, options.width, 1))
    # elif options.dataset == 'SUNCG':
    #     normalDotThreshold = np.cos(np.deg2rad(5))
    #     distanceThreshold = 0.05
    #     global_gt_dict['segmentation'], global_gt_dict['plane_mask'] = fitPlaneMasksModule(global_gt_dict['plane'], global_gt_dict['depth'], global_gt_dict['normal'], width=options.width, height=options.height, normalDotThreshold=normalDotThreshold, distanceThreshold=distanceThreshold, closing=True, one_hot=True)
    # else:
    #     global_gt_dict['plane_mask'] = 1 - global_gt_dict['non_plane_mask']
    #     pass

    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    gt_dict = {}


    id_map = np.zeros(1300, dtype=np.uint8)
    with open('../../Data/ScanNet/tasks/scannet-labels.combined.tsv') as label_file:
        label_reader = csv.reader(label_file, delimiter='\t')
        for line_index, line in enumerate(label_reader):
            if line_index > 0 and line[3] != '' and line[4] != '':
                id_map[int(line[3])] = int(line[4])
                pass
            continue
        pass

    with tf.Session(config=config) as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            gtDepths = []
            gtNormals = []
            planeMasks = []
            #predMasks = []
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


                if options.dataset == 'NYU_RGBD':
                    image = sio.loadmat(imagePath)['imgRgb']
                    gt_d = sio.loadmat(imagePath.replace('rgb', 'depth'))['imgDepth']
                    gt_n = sio.loadmat(imagePath.replace('images_rgb', 'surface_normals').replace('rgb', 'surface_normals'))['imgNormals']
                elif options.dataset == 'Make3D':
                    image = ((img[0] + 0.5) * 255).astype(np.uint8)
                    gt_d = sio.loadmat(imagePath.replace('images', 'depths').replace('img', 'depth').replace('.jpg', '.mat'))['depthMap']
                    gt_n = global_gt['normal']
                else:
                    image = ((img[0] + 0.5) * 255).astype(np.uint8)
                    gt_d = global_gt['depth']
                    gt_n = global_gt['normal']
                    pass

                images.append(image)
                gtDepths.append(gt_d)
                gtNormals.append(gt_n)

                plane_data = sio.loadmat(imagePath.replace('images_rgb', 'planes').replace('rgb', 'plane_data'))['planeData']
                gt_s = (plane_data[0][0][0] - 1).astype(np.int32)
                planes = plane_data[0][0][1]
                numPlanes = planes.shape[0]
                gt_s[gt_s == numPlanes] = options.numOutputPlanes

                # print(gt_s.max())
                # print(gt_s.min())
                # print(numPlanes)

                # print(NUM_PLANES)
                # for planeIndex in xrange(numPlanes + 1):
                #     cv2.imwrite('test/mask_' + str(planeIndex) + '.png', drawMaskImage(gt_s == planeIndex))
                #     continue
                #cv2.imwrite('test/segmentation.png', drawSegmentationImage(gt_s, blackIndex=options.numOutputPlanes))
                #exit(1)

                gtSegmentations.append(gt_s)

                gt_semantics = sio.loadmat(imagePath.replace('images_rgb', 'labels_objects').replace('rgb', 'labels'))['imgObjectLabels']
                gt_semantics = id_map[gt_semantics]
                gtSemantics.append(gt_semantics)

                gtNumPlanes.append(numPlanes)
                continue

            gt_dict['image'] = np.array(images)
            gt_dict['depth'] = np.array(gtDepths)
            gt_dict['normal'] = np.array(gtNormals)
            gt_dict['segmentation'] = np.array(gtSegmentations)
            gt_dict['semantics'] = np.array(gtSemantics)
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

def evaluateAll(options):


    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass

    checkpoint_prefix = options.rootFolder + '/checkpoint/'

    method = options.methods[0]


    if 'ds0' not in method[0]:
        options.deepSupervisionLayers = ['res4b22_relu', ]
    else:
        options.deepSupervisionLayers = []
        pass
    options.predictConfidence = 0
    options.predictLocal = 0
    options.predictPixelwise = 1
    options.predictBoundary = 0
    options.anchorPlanes = 0
    if 'ps' in method[0] and 'hybrid_' not in method[0]:
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
    if options.hybrid != '1':
        options.checkpoint_dir = options.checkpoint_dir.replace('hybrid1', 'hybrid' + str(options.hybrid))
        pass

    print(options.checkpoint_dir)

    tf.reset_default_graph()

    options.batchSize = 1
    min_after_dequeue = 1000

    if options.dataset == 'SUNCG':
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_SUNCG_val.tfrecords'], num_epochs=10000)
    elif options.dataset == 'NYU_RGBD_raw':
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_nyu_rgbd_val.tfrecords'], num_epochs=1)
        options.deepSupervision = 0
        options.predictLocal = 0
    elif options.dataset == 'matterport':
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_matterport_val.tfrecords'], num_epochs=1)
    elif options.dataset == 'ScanNet':
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_scannet_val.tfrecords'], num_epochs=1)
    elif options.dataset == 'NYU_RGBD':
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_nyu_rgbd_labeled_val.tfrecords'], num_epochs=1)
    elif options.dataset == 'Make3D':
        filename_queue = tf.train.string_input_producer(['../planes_make3d_val.tfrecords'], num_epochs=1)
        pass

    if options.dataset != 'Make3D':
        reader = RecordReaderAll()
    else:
        reader = RecordReaderMake3D()
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

    print(options)

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        #var_to_restore = [v for v in var_to_restore if 'res4b22_relu_non_plane' not in v.name]
        loader = tf.train.Saver(var_to_restore)
        loader.restore(sess, "%s/checkpoint.ckpt"%(options.checkpoint_dir))
        #loader.restore(sess, options.fineTuningCheckpoint)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)


        planenet_results = []
        pixelwise_results = []

        if options.useCache == 1:
            planenet_results = np.load(options.test_dir + '/planenet_results.npy').tolist()
            pixelwise_results = np.load(options.test_dir + '/pixelwise_results.npy').tolist()
            pass

        try:
            predDepths = []
            predNormals = []
            predPlanes = []
            predSegmentations = []
            predNonPlaneDepths = []
            predNonPlaneMasks = []
            predNonPlaneNormals = []
            predBoundaries = []
            predDepthNormals = []
            for index in xrange(options.startIndex + options.numImages):
                if options.useCache == 1 and index < len(planenet_results):
                    continue
                if index % 10 == 0:
                    print(('image', index))
                    pass
                t0=time.time()

                img, global_gt, global_pred = sess.run([img_inp, global_gt_dict, global_pred_dict])

                if index < options.startIndex:
                    continue


                info = global_gt['info'][0]

                width_high_res = int(info[16])
                height_high_res = int(info[17])

                pred_p = global_pred['plane'][0]
                pred_s = global_pred['segmentation'][0]

                pred_np_m = global_pred['non_plane_mask'].squeeze()
                pred_np_d = global_pred['non_plane_depth'].squeeze()
                pred_np_n = global_pred['non_plane_normal'][0]

                pred_np_m = cv2.resize(pred_np_m, (width_high_res, height_high_res), interpolation=cv2.INTER_LINEAR)
                pred_np_m = np.expand_dims(pred_np_m, -1)

                pred_np_d = cv2.resize(pred_np_d, (width_high_res, height_high_res), interpolation=cv2.INTER_LINEAR)
                pred_np_n = cv2.resize(pred_np_n, (width_high_res, height_high_res), interpolation=cv2.INTER_LINEAR)
                pred_np_d = np.expand_dims(pred_np_d, -1)

                pred_s_high_res = []
                for planeIndex in xrange(pred_s.shape[-1]):
                    pred_s_high_res.append(cv2.resize(pred_s[:, :, planeIndex], (width_high_res, height_high_res), interpolation=cv2.INTER_LINEAR))
                    continue
                pred_s = np.stack(pred_s_high_res, axis=2)
                #pred_b = global_pred['boundary'][0]

                #predNonPlaneDepths.append(pred_np_d)
                #predNonPlaneNormals.append(pred_np_n)
                #predBoundaries.append(pred_b)

                all_segmentations = np.concatenate([pred_s, pred_np_m], axis=2)
                plane_depths = calcPlaneDepths(pred_p, width_high_res, height_high_res, info)
                all_depths = np.concatenate([plane_depths, pred_np_d], axis=2)

                segmentation = np.argmax(all_segmentations, 2)
                pred_d = all_depths.reshape(-1, options.numOutputPlanes + 1)[np.arange(width_high_res * height_high_res), segmentation.reshape(-1)].reshape((height_high_res, width_high_res))

                #plane_normals = calcPlaneNormals(pred_p, width_high_res, height_high_res)
                #all_normals = np.concatenate([plane_normals, np.expand_dims(pred_np_n, 2)], axis=2)
                #pred_n = all_normals.reshape(-1, options.numOutputPlanes + 1, 3)[np.arange(width_high_res * height_high_res), segmentation.reshape(-1)].reshape((height_high_res, width_high_res, 3))


                imagePath = global_gt['image_path'][0]

                #image = sio.loadmat(imagePath)['imgRgb']


                if options.dataset == 'NYU_RGBD':
                    gt_d = sio.loadmat(imagePath.replace('rgb', 'depth'))['imgDepth']
                elif options.dataset == 'Make3D':
                    gt_d = sio.loadmat(imagePath.replace('images', 'depths').replace('img', 'depth').replace('.jpg', '.mat'))['depthMap']
                    planenet_result = []
                    pred_d = cv2.resize(pred_d, (gt_d.shape[1], gt_d.shape[0]))
                    planenet_result.append(evaluateDepths(pred_d, gt_d, np.ones(gt_d.shape)))
                    planenet_results.append(planenet_result)
                    continue
                else:
                    gt_d = global_gt['depth']
                    pass

                if index < options.visualizeImages:
                    img = sio.loadmat(imagePath)['imgRgb']
                    #cv2.imwrite(options.test_dir + '/' + str(index) + '_image.png', ((img[0] + 0.5) * 255).astype(np.uint8))
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_image.png', img.astype(np.uint8))
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_segmentation.png', drawSegmentationImage(segmentation, blackIndex=options.numOutputPlanes))
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_pred.png', drawDepthImage(pred_d))
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_gt.png', drawDepthImage(gt_d))
                    pass

                #gt_d = sio.loadmat(imagePath.replace('rgb', 'depth'))['imgDepth']


                #gt_n = sio.loadmat(imagePath.replace('images_rgb', 'surface_normals').replace('rgb', 'surface_normals'))['imgNormals']

                # images.append(image)
                # gtDepths.append(gt_d)
                # gtNormals.append(gt_n)

                plane_data = sio.loadmat(imagePath.replace('images_rgb', 'planes').replace('rgb', 'plane_data'))['planeData']
                gt_s = (plane_data[0][0][0] - 1).astype(np.int32)
                planes = plane_data[0][0][1]
                numPlanes = planes.shape[0]
                gt_s[gt_s == numPlanes] = options.numOutputPlanes

                planeMask = gt_s < options.numOutputPlanes
                edgeMap = calcEdgeMap(gt_s, edgeWidth=5)
                if edgeMap.sum() == 0:
                    edgeMap[0] = True
                    edgeMap[-1] = True
                    edgeMap[:, 0] = True
                    edgeMap[:, -1] = True
                    pass

                planenet_result = []
                print('image')
                planenet_result.append(evaluateDepths(pred_d, gt_d, np.ones(gt_d.shape)))
                print('plane')
                planenet_result.append(evaluateDepths(pred_d, gt_d, np.ones(gt_d.shape), planeMask))
                print('edge')
                planenet_result.append(evaluateDepths(pred_d, gt_d, np.ones(gt_d.shape), planeMasks=edgeMap))
                planenet_results.append(planenet_result)

                pixelwise_result = []
                pred_np_d = np.squeeze(pred_np_d)
                print('pixelwise')
                pixelwise_result.append(evaluateDepths(pred_np_d, gt_d, np.ones(gt_d.shape)))
                pixelwise_result.append(evaluateDepths(pred_np_d, gt_d, np.ones(gt_d.shape), planeMasks=planeMask))
                pixelwise_result.append(evaluateDepths(pred_np_d, gt_d, np.ones(gt_d.shape), planeMasks=edgeMap))
                pixelwise_results.append(pixelwise_result)
                #exit(1)

                #cv2.imwrite('test/mask.png', drawMaskImage(edgeMap))

                #predDepthNormals.append(calcNormal(pred_d, info))
                continue

            #pred_dict['plane'] = np.array(predPlanes)
            #pred_dict['segmentation'] = np.array(predSegmentations)
            #pred_dict['depth'] = np.array(predDepths)
            #pred_dict['normal'] = np.array(predNormals)
            #pred_dict['np_depth'] = np.array(predNonPlaneDepths)
            #pred_dict['np_mask'] = np.array(predNonPlaneMasks)
            #pred_dict['np_normal'] = np.array(predNonPlaneNormals)
            #pred_dict['depth_normal'] = np.array(predDepthNormals)
            #pred_dict['boundary'] = np.array(predBoundaries)
            pass
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
            # if len(planenet_results) > 0:
            #     np.save(options.test_dir + '/planenet_results.npy', np.array(planenet_results))
            #     np.save(options.test_dir + '/pixelwise_results.npy', np.array(pixelwise_results))
            #     pass
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            pass

        planenet_results = np.array(planenet_results)
        pixelwise_results = np.array(pixelwise_results)
        np.save(options.test_dir + '/planenet_results.npy', planenet_results)
        np.save(options.test_dir + '/pixelwise_results.npy', pixelwise_results)
        print(planenet_results.mean(0))
        print(pixelwise_results.mean(0))

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()
        pass
    return

if __name__=='__main__':
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Planenet')
    parser.add_argument('--task', dest='task',
                        help='task type',
                        default='all', type=str)
    parser.add_argument('--numOutputPlanes', dest='numOutputPlanes',
                        help='the number of output planes',
                        default=20, type=int)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset name',
                        default='NYU_RGBD', type=str)
    parser.add_argument('--hybrid', dest='hybrid',
                        help='hybrid',
                        default='1', type=str)
    parser.add_argument('--visualizeImages', dest='visualizeImages',
                        help='visualize image',
                        default=10, type=int)
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
                        default='2', type=str)
    parser.add_argument('--rootFolder', dest='rootFolder',
                        help='root folder',
                        default='/mnt/vision/PlaneNet/', type=str)
    parser.add_argument('--highRes', dest='highRes',
                        help='evaluate on high resolution',
                        default=1, type=str)

    args = parser.parse_args()
    #args.hybrid = 'hybrid' + args.hybrid
    args.test_dir = 'evaluate/' + args.task + '/' + args.dataset + '/hybrid' + args.hybrid + '_' + args.methods
    if args.highRes == 1:
        args.test_dir += '_high_res'
        args.width = 561
        args.height = 427
    else:
        args.width = 256
        args.height = 192
        pass
    args.test_dir += '/'

    #args.visualizeImages = max(args.visualizeImages, args.numImages)
    args.visualizeImages = args.numImages
    if args.imageIndex >= 0:
        args.visualizeImages = 1
        args.numImages = 1
        pass

    args.titles = [ALL_TITLES[int(method)] for method in args.methods]
    args.methods = [ALL_METHODS[int(method)] for method in args.methods]
    print(args.titles)

    if args.task == 'plane':
        evaluatePlanePrediction(args)
    elif args.task == 'depth':
        evaluateDepthPrediction(args)
    elif args.task == 'all':
        evaluateAll(args)
        pass
