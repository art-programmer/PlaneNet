import tensorflow as tf
import numpy as np
#np.set_printoptions(precision=2, linewidth=200)
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
from SegmentationRefinement import *
from crfasrnn_layer import CrfRnnLayer

#ALL_METHODS = [('', ''), ('pb_pp', 'pixelwise_1'), ('pb_pp', 'pixelwise_2'), ('pb_pp', 'pixelwise_3'), ('pb_pp', 'semantics'), ('pb_pp', 'gt')]

ALL_TITLES = ['PlaneNet', 'Oracle NYU toolbox', 'NYU toolbox', 'Oracle Manhattan', 'Manhattan', 'Piecewise', 'Oracle Piecewise']

#bl0_dl0_ll1_pb_pp_sm0
#bl0_dl0_ll1_ds0_pb_pp
#ll1_pb_pp
#bl0_dl0_ll1_ds0_pb_pp
#ll1_pb_pp
#bl0_dl0_crfrnn10_sm0
#bl0_dl0_crfrnn-10_sm0
#bl2_ll1_bw0.5_pb_pp_sm0
#crf1_pb_pp
#bl0_dl0_bw0.5_pb_pp_ps_sm0
#planenet_hybrid3_bl0_ll1_bw0.5_pb_pp_ps_sm0
#planenet_hybrid3_ll1_pb_pp
#planenet_hybrid3_bl0_dl0_ll1_ds0_pb_pp

#bl0_dl0_bw0.5_pb_pp_ps_sm0
#bl0_ll1_bw0.5_pb_pp_ps_sm0
#ll1_bw0.5_pb_pp_sm0
#pb_pp_sm0
#bl0_dl0_ll1_bw0.5_pb_pp_sm0
#pb_pp

ALL_METHODS = [['planenet_hybrid3_bl0_dl0_ll1_pb_pp_sm0', '', 0, 0], ['planenet_hybrid3_bl0_dl0_crfrnn-10_sm0', '', 1, 0], ['sample_np10_hybrid3_bl0_dl0_ll1_hl2_ds0_crfrnn5_sm0', '', 1, 0], ['sample_np10_hybrid3_bl0_dl0_hl2_crfrnn5_sm0', '', 1, 0], ['sample_np10_hybrid3_bl0_dl0_hl2_ds0_crfrnn5_sm0', '', 1, 0], ['planenet_hybrid3_bl0_dl0_ds0_crfrnn5_sm0', '', 1, 0], ['sample_np10_hybrid3_bl0_dl0_ds0_crfrnn5_sm0', '', 1, 0]]


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
        r_inp.td('input ' + str(index))
        r_inp.td().img(src=path + '/' + str(index) + '_image.png')
        r_inp.td().img(src=path + '/' + str(index) + '_depth_gt.png')
        r_inp.td().img(src=path + '/' + str(index) + '_segmentation_gt.png')
        r_inp.td().img(src=path + '/' + str(index) + '_semantics_gt.png')                
        r_inp.td().img(src=path + '/' + str(index) + '_depth_gt_plane.png')
        r_inp.td().img(src=path + '/' + str(index) + '_depth_gt_diff.png')        
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

        r = t.tr()
        r.td('depth')
        for method_index, method in enumerate(titles):
            r.td().img(src=path + '/' + str(index) + '_depth_pred_' + str(method_index) + '.png')
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


            
    #predictions[2] = predictions[3]


            
    
    for image_index in xrange(options.visualizeImages):
        cv2.imwrite(options.test_dir + '/' + str(image_index) + '_image.png', gt_dict['image'][image_index])
        cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_gt.png', drawDepthImage(gt_dict['depth'][image_index]))
        cv2.imwrite(options.test_dir + '/' + str(image_index) + '_normal_gt.png', drawNormalImage(gt_dict['normal'][image_index]))        
        cv2.imwrite(options.test_dir + '/' + str(image_index) + '_segmentation_gt.png', drawSegmentationImage(np.concatenate([gt_dict['segmentation'][image_index], 1 - np.expand_dims(gt_dict['plane_mask'][image_index], -1)], axis=2), blackIndex=options.numOutputPlanes))
        cv2.imwrite(options.test_dir + '/' + str(image_index) + '_semantics_gt.png', drawSegmentationImage(gt_dict['semantics'][image_index], blackIndex=0))

        
        plane_depths = calcPlaneDepths(gt_dict['plane'][image_index], WIDTH, HEIGHT, gt_dict['info'][image_index])
        all_depths = np.concatenate([plane_depths, np.expand_dims(gt_dict['depth'][image_index], -1)], axis=2)
        depth = np.sum(all_depths * np.concatenate([gt_dict['segmentation'][image_index], 1 - np.expand_dims(gt_dict['plane_mask'][image_index], -1)], axis=2), axis=2)
        cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_gt_plane.png', drawDepthImage(depth))
        cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_gt_diff.png', drawMaskImage((depth - gt_dict['depth'][image_index]) * 5 + 0.5))        
        
        for method_index, pred_dict in enumerate(predictions):
            cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_dict['depth'][image_index]))

            if 'pixelwise' in options.methods[method_index][1]:
                continue
            segmentation = pred_dict['segmentation'][image_index]
            segmentation = np.concatenate([segmentation, pred_dict['np_mask'][image_index]], axis=2)
            numPlanes = options.numOutputPlanes
            if 'np10' in options.methods[method_index][0]:
                numPlanes = 10
            elif 'np15' in options.methods[method_index][0]:
                numPlanes = 15
                pass
            cv2.imwrite(options.test_dir + '/' + str(image_index) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage(segmentation, blackIndex=numPlanes))
            continue
        continue


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

                cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_d))
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
            new_pred_dict['num_planes'] = np.array(predNumPlanes)
            if method_index < len(predictions):
                predictions[method_index] = new_pred_dict
            else:
                predictions.append(new_pred_dict)
                pass
            #titles.append('pixelwise+semantics+RANSAC')
            pass

        if method[1] == 'crfrnn':
            numPlanes = options.numOutputPlanes
            if 'np10' in method[0]:
                numPlanes = 10
            elif 'np15' in method[0]:
                numPlanes = 15
                pass
            
            predSegmentations = []
            predDepths = []

            image_inp = tf.placeholder(tf.float32, shape=[1, HEIGHT, WIDTH, 3], name='image')
            segmentation_inp = tf.placeholder(tf.float32, shape=[1, HEIGHT, WIDTH, numPlanes + 1], name='segmentation')
            
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
                    print('crf rnn ' + str(image_index))
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
                    pred_s = one_hot(np.argmax(pred_s, axis=-1), numPlanes + 1)

                    
                    planeDepths = calcPlaneDepths(pred_dict['plane'][image_index], WIDTH, HEIGHT, gt_dict['info'][image_index])
                    allDepths = np.concatenate([planeDepths, pred_dict['np_depth'][image_index]], axis=2)
                    pred_d = np.sum(allDepths * pred_s, axis=-1)
                    
                    predSegmentations.append(pred_s)
                    predDepths.append(pred_d)
            
                    cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_d))            
                    cv2.imwrite(options.test_dir + '/' + str(image_index) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage(pred_s, blackIndex=numPlanes))

                    continue
                pass
            new_pred_dict = {}
            for key, value in pred_dict.iteritems():
                new_pred_dict[key] = value
                continue
            segmentations = np.array(predSegmentations)
            new_pred_dict['segmentation'] = segmentations[:, :, :, :numPlanes]
            new_pred_dict['non_plane_mask'] = segmentations[:, :, :, numPlanes:numPlanes + 1]
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
    
    #exit(1)
    
    #print(results)

    # depth = gt_dict['depth'][4]
    # cv2.imwrite(options.test_dir + '/test_depth_gt.png', drawDepthImage(depth))
    # pred_p, pred_s, pred_d = fitPlanes(depth, getSUNCGCamera(), numPlanes=20, planeAreaThreshold=3*4, numIterations=100, distanceThreshold=0.05, local=0.2)
    # cv2.imwrite(options.test_dir + '/test_depth.png', drawDepthImage(pred_d))
    # cv2.imwrite(options.test_dir + '/test_segmentation.png', drawSegmentationImage(pred_s))
    # exit(1)
    
    
    plotResults(gt_dict, predictions, options)
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
        #if method_index != 6:
        #continue
        
        numPlanes = pred_dict['segmentation'].shape[-1]
        segmentations = one_hot(np.argmax(np.concatenate([pred_dict['segmentation'], pred_dict['np_mask']], axis=3), axis=-1), numPlanes + 1)
        segmentations = segmentations[:, :, :, :numPlanes]
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
                if 'np10' in options.methods[method_index][0]:
                    predNumPlanes = 10
                elif 'np15' in options.methods[method_index][0]:
                    predNumPlanes = 15
                    pass
                pass

            # for planeIndex in xrange(gtDepths.shape[-1]):
            #     cv2.imwrite('test/depth_gt_' + str(planeIndex) + '.png', drawDepthImage(gtDepths[:, :, planeIndex]))
            #     cv2.imwrite('test/mask_gt_' + str(planeIndex) + '.png', drawMaskImage(gt_dict['segmentation'][image_index][:, :, planeIndex]))
            #     continue
            # for planeIndex in xrange(predDepths.shape[-1]):
            #     cv2.imwrite('test/depth_pred_' + str(planeIndex) + '.png', drawDepthImage(predDepths[:, :, planeIndex]))
            #     cv2.imwrite('test/mask_pred_' + str(planeIndex) + '.png', drawMaskImage(segmentations[image_index][:, :, planeIndex]))
            #     continue
            
            # cv2.imwrite('test/depth_diff_0.png', drawMaskImage(np.abs(gt_dict['depth'][image_index] - predDepths[:, :, 6]) * gt_dict['segmentation'][image_index][:, :, 0] * segmentations[image_index][:, :, 6] / 0.5))
            # cv2.imwrite('test/depth_diff_1.png', drawMaskImage(np.abs(gt_dict['depth'][image_index] - predDepths[:, :, 4]) * gt_dict['segmentation'][image_index][:, :, 0] * segmentations[image_index][:, :, 4] / 0.5))
            # cv2.imwrite('test/mask_0.png', drawMaskImage(gt_dict['segmentation'][image_index][:, :, 0] * segmentations[image_index][:, :, 6] > 0.5))
            # cv2.imwrite('test/mask_1.png', drawMaskImage(gt_dict['segmentation'][image_index][:, :, 0] * segmentations[image_index][:, :, 4] > 0.5))
            # print(np.abs(gt_dict['depth'][image_index] - predDepths[:, :, 6])[gt_dict['segmentation'][image_index][:, :, 0] * segmentations[image_index][:, :, 6] > 0.5].mean())
            # print(np.abs(gt_dict['depth'][image_index] - predDepths[:, :, 4])[gt_dict['segmentation'][image_index][:, :, 3] * segmentations[image_index][:, :, 4] > 0.5].mean())
            # print(np.abs(gt_dict['depth'][image_index] - predDepths[:, :, 5])[gt_dict['segmentation'][image_index][:, :, 1] * segmentations[image_index][:, :, 5] > 0.5].mean())            
            # exit(1)
            #pixelStatistics, planeStatistics = evaluatePlanePrediction(predDepths, segmentations[image_index], predNumPlanes, gtDepths, gt_dict['segmentation'][image_index], gt_dict['num_planes'][image_index], gtPlanes=gt_dict['plane'][image_index], predPlanes=pred_dict['plane'][image_index])

            #if method_index != 6:
            #continue
            pixelStatistics, planeStatistics = evaluatePlanePrediction(predDepths, segmentations[image_index], predNumPlanes, gtDepths, gt_dict['segmentation'][image_index], gt_dict['num_planes'][image_index])
            #print(predNumPlanes, gt_dict['num_planes'][image_index])
            #print(image_index, planeStatistics[4][5])
            #print(pred_dict['plane'][image_index])
            #print(segmentations[image_index].sum(axis=(0, 1)))
            #exit(1)
            #print(method_index)
            #print(planeStatistics[2][5])

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

    #exit(1)
    
    np.save(options.test_dir + '/pixel_curves.npy', np.array(pixel_metric_curves))
    np.save(options.test_dir + '/plane_curves.npy', np.array(plane_metric_curves))    

    
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
    #print(curve_labels)
    for metric_index, curves in enumerate(pixel_metric_curves):
        filename = options.test_dir + '/curve_pixel_' + curve_titles[metric_index].replace(' ', '_') + '.png'
        #print(metric_index)
        #print(curves)
        plotCurvesSimple(xs[metric_index], curves, filename = filename, xlabel=xlabels[metric_index], ylabel='pixel coverage', title=curve_titles[metric_index], labels=curve_labels)
        continue
    for metric_index, curves in enumerate(plane_metric_curves):
        filename = options.test_dir + '/curve_plane_' + curve_titles[metric_index].replace(' ', '_') + '.png'
        curves = [curve[:, 0] / curve[:, 1] for curve in curves]
        plotCurvesSimple(xs[metric_index], curves, filename = filename, xlabel=xlabels[metric_index], ylabel='plane accuracy', title=curve_titles[metric_index], labels=curve_labels)
        continue

    
def gridSearch(options):
    #writeHTML(options)
    #exit(1)

    if os.path.exists(options.result_filename):
        results = np.load(options.result_filename)
        results = results[()]
    else:
        assert(False)
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


    
    for image_index in xrange(options.visualizeImages):
        cv2.imwrite(options.test_dir + '/' + str(image_index) + '_image.png', gt_dict['image'][image_index])
        cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_gt.png', drawDepthImage(gt_dict['depth'][image_index]))
        cv2.imwrite(options.test_dir + '/' + str(image_index) + '_normal_gt.png', drawNormalImage(gt_dict['normal'][image_index]))        
        cv2.imwrite(options.test_dir + '/' + str(image_index) + '_segmentation_gt.png', drawSegmentationImage(np.concatenate([gt_dict['segmentation'][image_index], 1 - np.expand_dims(gt_dict['plane_mask'][image_index], -1)], axis=2), blackIndex=options.numOutputPlanes))
        cv2.imwrite(options.test_dir + '/' + str(image_index) + '_semantics_gt.png', drawSegmentationImage(gt_dict['semantics'][image_index], blackIndex=0))
        
        plane_depths = calcPlaneDepths(gt_dict['plane'][image_index], WIDTH, HEIGHT, gt_dict['info'][image_index])
        all_depths = np.concatenate([plane_depths, np.expand_dims(gt_dict['depth'][image_index], -1)], axis=2)
        depth = np.sum(all_depths * np.concatenate([gt_dict['segmentation'][image_index], 1 - np.expand_dims(gt_dict['plane_mask'][image_index], -1)], axis=2), axis=2)
        cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_gt_plane.png', drawDepthImage(depth))
        cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_gt_diff.png', drawMaskImage((depth - gt_dict['depth'][image_index]) * 5 + 0.5))        
        
        for method_index, pred_dict in enumerate(predictions):
            cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_dict['depth'][image_index]))

            if 'pixelwise' in options.methods[method_index][1]:
                continue
            segmentation = pred_dict['segmentation'][image_index]
            segmentation = np.concatenate([segmentation, pred_dict['np_mask'][image_index]], axis=2)

            numPlanes = options.numOutputPlanes
            if 'np10' in options.methods[method_index][0]:
                numPlanes = 10
            elif 'np15' in options.methods[method_index][0]:
                numPlanes = 15                
                pass
            cv2.imwrite(options.test_dir + '/' + str(image_index) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage(segmentation, blackIndex=numPlanes))
            continue
        continue


    #post processing
    for method_index, method in enumerate(options.methods):
        if len(method) == 3:
            pred_dict = predictions[method[2]]
        else:
            pred_dict = predictions[method_index]
            pass

        if 'pixelwise_2' in method[1] or 'pixelwise_3' in method[1]:
            bestScore = 0
            configurationIndex = 0
            for distanceCostThreshold in [0.3]:
                for smoothnessWeight in [0.01, 0.03, 0.05]:
                    for distanceThreshold in [0.2]:
                        parameters = {'distanceCostThreshold': distanceCostThreshold, 'smoothnessWeight': smoothnessWeight, 'distanceThreshold': distanceThreshold, 'semantics': True}
                        score = 0
                        for image_index in xrange(options.numImages):
                            #cv2.imwrite('test/normal.png', drawNormalImage(gt_dict['normal'][image_index]))
                            if '_2' in method[1]:
                                pred_p, pred_s = fitPlanesNYU(gt_dict['image'][image_index], gt_dict['depth'][image_index].squeeze(), gt_dict['normal'][image_index], gt_dict['semantics'][image_index], gt_dict['info'][image_index], numOutputPlanes=20, parameters=parameters)
                            else:
                                pred_d = pred_dict['np_depth'][image_index].squeeze()
                                pred_n = pred_dict['np_normal'][image_index].squeeze()
                                pred_p, pred_s = fitPlanesNYU(gt_dict['image'][image_index], pred_d, pred_n, pred_dict['semantics'][image_index], gt_dict['info'][image_index], numOutputPlanes=20, parameters=parameters)
                                pass

                            predNumPlanes = pred_p.shape[0]
                            gtDepths = calcPlaneDepths(gt_dict['plane'][image_index], WIDTH, HEIGHT, gt_dict['info'][image_index])
                            planeDepths = calcPlaneDepths(pred_p, WIDTH, HEIGHT, gt_dict['info'][image_index])
                            pixelStatistics, planeStatistics = evaluatePlanePrediction(planeDepths, pred_s, predNumPlanes, gtDepths, gt_dict['segmentation'][image_index], gt_dict['num_planes'][image_index])
                            #print(pixelStatistics)
                            #exit(1)
                            #planeStatistics = np.array(planeStatistics)[1]
                            #accuracy = (planeStatistics[3:8, 0].astype(np.float32) / np.maximum(planeStatistics[3:8, 1], 1e-4)).mean()

                            pixelStatistics = np.array(pixelStatistics)[1]
                            accuracy = pixelStatistics[3:8].mean()
                            score += accuracy

                            #cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_d))            
                            cv2.imwrite('test/segmentation_pred_' + str(image_index) + '.png', drawSegmentationImage(pred_s, blackIndex=options.numOutputPlanes))
                            #exit(1)
                            continue
                        score /= options.numImages
                        print(score, parameters)
                        configurationIndex += 1                    
                        #exit(1)
                        if score > bestScore:
                            bestScore = score
                            bestParameters = parameters
                            pass
                        continue
                    continue
                continue
            print(bestScore)
            print(bestParameters)
            exit(1)
                    
        if 'pixelwise_4' in method[1] or 'pixelwise_5' in method[1]:
            bestScore = 0
            configurationIndex = 0
            for distanceCostThreshold in [0.05]:
                for smoothnessWeight in [30]:
                    for offsetGap in [-0.1]:
                        parameters = {'distanceCostThreshold': distanceCostThreshold, 'smoothnessWeight': smoothnessWeight, 'offsetGap': abs(offsetGap), 'meanshift': offsetGap}

                        score = 0
                        for image_index in xrange(options.numImages):
                            if '_4' in method[1]:
                                pred_p, pred_s = fitPlanesManhattan(gt_dict['image'][image_index], gt_dict['depth'][image_index].squeeze(), gt_dict['normal'][image_index], gt_dict['info'][image_index], numOutputPlanes=20, parameters=parameters)
                            else:
                                pred_d = pred_dict['np_depth'][image_index].squeeze()
                                pred_n = pred_dict['np_normal'][image_index].squeeze()
                                pred_p, pred_s = fitPlanesManhattan(gt_dict['image'][image_index], pred_d, pred_n, gt_dict['info'][image_index], numOutputPlanes=20, parameters=parameters)
                                pass

                            predNumPlanes = pred_p.shape[0]
                            gtDepths = calcPlaneDepths(gt_dict['plane'][image_index], WIDTH, HEIGHT, gt_dict['info'][image_index])
                            planeDepths = calcPlaneDepths(pred_p, WIDTH, HEIGHT, gt_dict['info'][image_index])
                            pixelStatistics, planeStatistics = evaluatePlanePrediction(planeDepths, pred_s, predNumPlanes, gtDepths, gt_dict['segmentation'][image_index], gt_dict['num_planes'][image_index])
                            #print(pixelStatistics)
                            #exit(1)
                            #planeStatistics = np.array(planeStatistics)[1]
                            #accuracy = (planeStatistics[3:8, 0].astype(np.float32) / np.maximum(planeStatistics[3:8, 1], 1e-4)).mean()

                            pixelStatistics = np.array(pixelStatistics)[1]
                            accuracy = pixelStatistics[3:8].mean()
                            score += accuracy

                            #cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_d))            
                            #cv2.imwrite(options.test_dir + '/' + str(image_index) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage(pred_s, blackIndex=options.numOutputPlanes))
                            #exit(1)
                            continue
                        score /= options.numImages
                        print(score, parameters)
                        configurationIndex += 1                    
                        #exit(1)
                        if score > bestScore:
                            bestScore = score
                            bestParameters = parameters
                            pass
                        continue
                    continue
                continue
            print(bestScore)
            print(bestParameters)
            exit(1)

        if 'pixelwise_6' in method[1] or 'pixelwise_7' in method[1]:
            bestScore = 0
            configurationIndex = 0            
            for distanceCostThreshold in [0.1]:
                for smoothnessWeight in [300]:
                    for normalWeight in [1]:
                        for offset in [0.2]:
                            parameters = {'distanceCostThreshold': distanceCostThreshold, 'smoothnessWeight': smoothnessWeight, 'numProposals': 5, 'normalWeight': normalWeight, 'offsetGap': abs(offset), 'meanshift': offset}

                            score = 0
                            for image_index in xrange(options.numImages):
                                if '_6' in method[1]:
                                    pred_p, pred_s = fitPlanesPiecewise(gt_dict['image'][image_index], gt_dict['depth'][image_index].squeeze(), gt_dict['normal'][image_index], gt_dict['info'][image_index], numOutputPlanes=20, parameters=parameters)
                                else:
                                    pred_d = pred_dict['np_depth'][image_index].squeeze()                
                                    pred_n = pred_dict['np_normal'][image_index].squeeze()                
                                    pred_p, pred_s = fitPlanesPiecewise(gt_dict['image'][image_index], pred_d, pred_n, gt_dict['info'][image_index], numOutputPlanes=20, parameters=parameters)
                                    pass

                                predNumPlanes = pred_p.shape[0]
                                gtDepths = calcPlaneDepths(gt_dict['plane'][image_index], WIDTH, HEIGHT, gt_dict['info'][image_index])
                                planeDepths = calcPlaneDepths(pred_p, WIDTH, HEIGHT, gt_dict['info'][image_index])
                                pixelStatistics, planeStatistics = evaluatePlanePrediction(planeDepths, pred_s, predNumPlanes, gtDepths, gt_dict['segmentation'][image_index], gt_dict['num_planes'][image_index])
                                #print(pixelStatistics)
                                #exit(1)
                                #planeStatistics = np.array(planeStatistics)[1]
                                #accuracy = (planeStatistics[3:8, 0].astype(np.float32) / np.maximum(planeStatistics[3:8, 1], 1e-4)).mean()

                                pixelStatistics = np.array(pixelStatistics)[1]
                                accuracy = pixelStatistics[3:8].mean()
                                score += accuracy

                                #cv2.imwrite('test/depth_pred_' + str(configurationIndex) + '.png', drawDepthImage(pred_d))
                                cv2.imwrite('test/segmentation_pred_' + str(image_index) + '.png', drawSegmentationImage(pred_s, blackIndex=options.numOutputPlanes))
                                #exit(1)
                                continue
                            score /= options.numImages
                            print(score, parameters)
                            configurationIndex += 1

                            #exit(1)
                            if score > bestScore:
                                bestScore = score
                                bestParameters = parameters
                                pass
                            continue
                        continue
                    continue
                continue
            print(bestScore)
            print(bestParameters)
            exit(1)            

        if method[1] == 'crfrnn':

            parameterConfigurations = [(160., 3., 3.), (80., 3., 3.), (40., 3., 3.), (120., 3., 3.)]

            bestScore = 0
            for parameters in parameterConfigurations:
                tf.reset_default_graph()
                image_inp = tf.placeholder(tf.float32, shape=[1, HEIGHT, WIDTH, 3], name='image')
                segmentation_inp = tf.placeholder(tf.float32, shape=[1, HEIGHT, WIDTH, options.numOutputPlanes + 1], name='segmentation')
            
                refined_segmentation = CrfRnnLayer(image_dims=(HEIGHT, WIDTH), num_classes=21, theta_alpha=parameters[0], theta_beta=parameters[1], theta_gamma=parameters[2], num_iterations=10, name='crfrnn')([segmentation_inp, image_inp])
            
                config=tf.ConfigProto()
                config.gpu_options.allow_growth=True
                config.allow_soft_placement=True

                init_op = tf.group(tf.global_variables_initializer(),
                                   tf.local_variables_initializer())
                with tf.Session(config=config) as sess:
                    sess.run(init_op)

                    score = 0.
                    for image_index in xrange(options.numImages):
                        #if image_index != 1:
                        #continue
                        print('crf as rnn', image_index)
                        allSegmentations = np.concatenate([pred_dict['segmentation'][image_index], pred_dict['np_mask'][image_index]], axis=2)
                        img = gt_dict['image'][image_index:image_index + 1].astype(np.float32) - 128

                        pred_s = sess.run(refined_segmentation, feed_dict={segmentation_inp: np.expand_dims(allSegmentations, 0), image_inp: img})
                        pred_s = pred_s[0]

                        #pred_s = allSegmentations
                        
                        pred_s = one_hot(np.argmax(pred_s, axis=-1), options.numOutputPlanes + 1)


                        planeDepths = calcPlaneDepths(pred_dict['plane'][image_index], WIDTH, HEIGHT, gt_dict['info'][image_index])
                        allDepths = np.concatenate([planeDepths, pred_dict['np_depth'][image_index]], axis=2)
                        pred_d = np.sum(allDepths * pred_s, axis=-1)


                        # for planeIndex in xrange(options.numOutputPlanes):
                        #     cv2.imwrite('test/mask_' + str(planeIndex) + '.png', drawMaskImage(pred_s[:, :, planeIndex]))
                        #     cv2.imwrite('test/gt_mask_' + str(planeIndex) + '.png', drawMaskImage(gt_dict['segmentation'][image_index][:, :, planeIndex]))
                        #     continue
                        # cv2.imwrite('test/depth_pred.png', drawDepthImage(pred_d))
                        # cv2.imwrite('test/depth_gt.png', drawDepthImage(gt_dict['depth'][image_index]))                        
                        # cv2.imwrite('test/depth_diff.png', drawMaskImage(np.abs(pred_d - gt_dict['depth'][image_index]) * 5))
                        # exit(1)
                        predNumPlanes = options.numOutputPlanes
                        gtDepths = calcPlaneDepths(gt_dict['plane'][image_index], WIDTH, HEIGHT, gt_dict['info'][image_index])
                        pixelStatistics, planeStatistics = evaluatePlanePrediction(planeDepths, pred_s[:, :, :options.numOutputPlanes], predNumPlanes, gtDepths, gt_dict['segmentation'][image_index], gt_dict['num_planes'][image_index])
                        #print(pixelStatistics)
                        #exit(1)
                        #planeStatistics = np.array(planeStatistics)[1]
                        #accuracy = (planeStatistics[3:8, 0].astype(np.float32) / np.maximum(planeStatistics[3:8, 1], 1e-4)).mean()
                        pixelStatistics = np.array(pixelStatistics)[1]
                        # print(pixelStatistics)
                        # pixelStatistics[3:8].mean()
                        # exit(1)
                        accuracy = pixelStatistics[3:8].mean()
                        score += accuracy
                        
                        #cv2.imwrite(options.test_dir + '/' + str(image_index) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_d))            
                        #cv2.imwrite(options.test_dir + '/' + str(image_index) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage(pred_s, blackIndex=options.numOutputPlanes))
                        #exit(1)
                        continue
                    score /= options.numImages
                    print(score)
                    #exit(1)
                    if score > bestScore:
                        bestScore = score
                        bestParameters = parameters
                        pass
                    pass
                continue
            print(bestScore)
            print(bestParameters)
            pass
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
        if options.useCache == 2 and (len(method) < 4 or method[3] < 2):
            continue
        if method[0] == '':
            continue

        method_options = copy.deepcopy(options)
        if 'ds0' not in method[0]:
            method_options.deepSupervisionLayers = ['res4b22_relu', ]
        else:
            method_options.deepSupervisionLayers = []
            pass
        method_options.predictConfidence = 0
        method_options.predictLocal = 0
        method_options.predictPixelwise = 1
        method_options.predictBoundary = int('pb' in method[0])
        method_options.anchorPlanes = 0
        if 'ps' in method[0]:
            method_options.predictSemantics = 1
        else:
            method_options.predictSemantics = 0
            pass
        if 'crfrnn' in method[0]:
            method_options.crfrnn = 5
        else:
            method_options.crfrnn = 0
            pass
            
        if 'ap1' in method[0]:
            method_options.anchorPlanes = 1            
            pass

        method_options.numOutputPlanes = 20
        if 'np10' in method[0]:
            method_options.numOutputPlanes = 10
        elif 'np15' in method[0]:
            method_options.numOutputPlanes = 15
            pass
        
        method_options.checkpoint_dir = checkpoint_prefix + method[0]
        print(method_options.checkpoint_dir)
        
        method_options.suffix = method[1]

        method_names = [previous_method[0] for previous_method in methods[:method_index]]

        if method[0] in method_names:
            pred_dict = predictions[method_names.index(method[0])]
        elif method[0] == 'gt':
            pred_dict = gt_dict
        else:
            while True:
                try:
                    pred_dict = getPrediction(method_options)
                    break
                except:
                    continue
                continue
            pass

        # for image_index in xrange(method_options.visualizeImages):
        #     cv2.imwrite(method_options.test_dir + '/' + str(image_index) + '_depth_pred_' + str(method_index) + '.png', drawDepthImage(pred_dict['depth'][image_index]))
        #     cv2.imwrite(method_options.test_dir + '/' + str(image_index) + '_segmentation_pred_' + str(method_index) + '.png', drawSegmentationImage())
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
    #np.save(method_options.test_dir + '/curves.npy', curves)
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
    if 'sample' not in options.checkpoint_dir:
        global_pred_dict, local_pred_dict, deep_pred_dicts = build_graph(img_inp, img_inp, training_flag, options)
    else:
        global_pred_dict, local_pred_dict, deep_pred_dicts = build_graph_sample(img_inp, img_inp, training_flag, options)
        pass
        
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


                pred_b = global_pred['boundary'][0]
                predNonPlaneMasks.append(pred_np_m)                    
                predNonPlaneDepths.append(pred_np_d)
                predNonPlaneNormals.append(pred_np_n)
                predBoundaries.append(pred_b)
                    
                all_segmentations = np.concatenate([pred_np_m, pred_s], axis=2)
                plane_depths = calcPlaneDepths(pred_p, WIDTH, HEIGHT, global_gt['info'][0])
                all_depths = np.concatenate([pred_np_d, plane_depths], axis=2)

                segmentation = np.argmax(all_segmentations, 2)
                pred_d = all_depths.reshape(-1, options.numOutputPlanes + 1)[np.arange(WIDTH * HEIGHT), segmentation.reshape(-1)].reshape(HEIGHT, WIDTH)

                if 'semantics' in global_pred:
                    #cv2.imwrite('test/semantics.png', drawSegmentationImage(np.argmax(global_pred['semantics'][0], axis=-1)))
                    #exit(1)
                    predSemantics.append(np.argmax(global_pred['semantics'][0], axis=-1))
                else:
                    predSemantics.append(np.zeros((HEIGHT, WIDTH)))
                    pass
                                         
                predDepths.append(pred_d)
                predPlanes.append(pred_p)
                predSegmentations.append(pred_s)
                    
                continue
            pred_dict['plane'] = np.array(predPlanes)
            pred_dict['segmentation'] = np.array(predSegmentations)
            pred_dict['semantics'] = np.array(predSemantics)            
            pred_dict['depth'] = np.array(predDepths)
            pred_dict['np_depth'] = np.array(predNonPlaneDepths)
            pred_dict['np_normal'] = np.array(predNonPlaneNormals)
            pred_dict['np_mask'] = np.array(predNonPlaneMasks)
            pred_dict['boundary'] = np.array(predBoundaries)
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

                
                # print(global_gt['path'])
                # if index == 11:
                #     cv2.imwrite('test/mask.png', drawMaskImage(global_gt['non_plane_mask'].squeeze()))
                #     exit(1)
                image = ((img[0] + 0.5) * 255).astype(np.uint8)
                images.append(image)

                #cv2.imwrite(options.test_dir + '/' + str(index) + '_boundary.png', drawMaskImage(np.concatenate([global_gt['boundary'][0], np.zeros((HEIGHT, WIDTH, 1))], axis=2)))
                
                gt_d = global_gt['depth'].squeeze()
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
                        default='compare', type=str)
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
                        default='1111111', type=str)
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

    #args.titles = [ALL_TITLES[int(method)] for method in args.methods]
    #args.methods = [ALL_METHODS[int(method)] for method in args.methods]
    titles = []
    for methodIndex, flag in enumerate(args.methods):
        if int(flag) == 0:
            continue
        titles.append(ALL_TITLES[methodIndex])
        pass
    args.titles = titles

    methods = []
    for methodIndex, flag in enumerate(args.methods):
        if int(flag) == 0:
            continue
        method = ALL_METHODS[methodIndex]
        method[3] = int(flag)
        methods.append(method)
        pass
    args.methods = methods
    
    args.result_filename = args.test_dir + '/results_' + str(0) + '.npy'
    print(args.titles)
    
    if args.task == 'compare':
        evaluatePlanes(args)
    elif args.task == 'depth':
        evaluateDepthPrediction(args)
    elif args.task == 'search':
        gridSearch(args)
        pass
