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
    
    pred_dict = getResults(options)[0]
    print(pred_dict.keys())

    saving = True
    if pred_dict['image'].shape[0] != options.numImages:
        saving = False
        pass
        
    
    for pred_dict in predictions:
        for key, value in pred_dict.iteritems():
            if value.shape[0] > options.numImages:
                pred_dict[key] = value[:options.numImages]
                pass
            continue
        continue

    #methods = ['planenet', 'pixelwise+RANSAC', 'GT+RANSAC']


            
    #predictions[2] = predictions[3]


    
    for image_index in xrange(options.visualizeImages):
        if args.imageIndex >= 0 and image_index + options.startIndex != args.imageIndex:
            continue
        cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_image.png', pred_dict['image'][image_index])
        info = pred_dict['info'][image_index]
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
            segmentationImageBlended = (segmentationImage * 0.7 + pred_dict['image'][image_index] * 0.3).astype(np.uint8)
            cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_segmentation_pred_blended_' + str(method_index) + '.png', segmentationImageBlended)

            segmentationImageBlended = np.minimum(segmentationImage * 0.3 + pred_dict['image'][image_index] * 0.7, 255).astype(np.uint8)            
            if options.imageIndex >= 0:
                if options.suffix == 'texture':
                    for planeIndex in xrange(options.numOutputPlanes):
                        cv2.imwrite('test/mask_' + str(planeIndex) + '.png', drawMaskImage(segmentation == planeIndex))
                        continue

                    
                    resultImage = copyLogo(options.test_dir, image_index + options.startIndex, pred_dict['image'][image_index], pred_dict['depth'][image_index], pred_dict['plane'][image_index], segmentation, pred_dict['info'][image_index])
                    #resultImage = copyWallTexture(options.test_dir, image_index + options.startIndex, gt_dict['image'][image_index], pred_dict['depth'][image_index], pred_dict['plane'][image_index], segmentation, gt_dict['info'][image_index], [7, 9])
                    cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_result.png', resultImage)
                    writePLYFile(options.test_dir, image_index + options.startIndex, pred_dict['image'][image_index], pred_dict['depth'][image_index], segmentation, pred_dict['plane'][image_index], pred_dict['info'][image_index])
                elif options.suffix == 'dump':
                    newPlanes = []
                    newSegmentation = np.zeros(segmentation.shape)
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
                    
                    np.save('test/planes.npy', np.stack(newPlanes, axis=0))
                    #print(global_gt['non_plane_mask'].shape)
                    np.save('test/segmentation.npy', newSegmentation)
                    cv2.imwrite('test/image.png', pred_dict['image'][image_index])
                    depth = pred_dict['depth'][image_index]
                    np.save('test/depth.npy', depth)
                    info = pred_dict['info'][image_index]
                    normal = calcNormal(depth, info)
                    np.save('test/normal.npy', normal)
                    np.save('test/info.npy', info)
                    exit(1)
                                    
                else:
                    np_mask = (segmentation == options.numOutputPlanes).astype(np.float32)
                    np_depth = pred_dict['np_depth'][image_index].squeeze()
                    np_depth = cv2.resize(np_depth, (np_mask.shape[1], np_mask.shape[0]))
                    cv2.imwrite(options.test_dir + '/' + str(image_index + options.startIndex) + '_np_depth_pred_' + str(method_index) + '.png', drawDepthImage(np_depth * np_mask))
                    writePLYFile(options.test_dir, image_index + options.startIndex, segmentationImageBlended, pred_dict['depth'][image_index], segmentation, pred_dict['plane'][image_index], pred_dict['info'][image_index])
                    pass
                exit(1)
                pass
            continue
        continue

    writeHTML(options)
    exit(1)    
    return


def getResults(options):
    checkpoint_prefix = options.rootFolder + '/checkpoint/'

    methods = options.methods
    predictions = []

    if os.path.exists(options.result_filename) and options.useCache == 1:
        predictions = np.load(options.result_filename)
        return predictions
    
    

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
        
        options.suffix = method[1]

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
    results = predictions

    if options.useCache != -1:
        np.save(options.result_filename, results)
        pass
    pass
    
    return results

def getPrediction(options):
    tf.reset_default_graph()
    
    options.batchSize = 1

    img_inp = tf.placeholder(tf.float32, shape=[1, HEIGHT, WIDTH, 3], name='image')
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
                

    image_list = glob.glob('../my_images/*.jpg') + glob.glob('../my_images/*.png')
    
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
            images = []
            infos = []
            for index in xrange(min(options.startIndex + options.numImages, len(image_list))):
                if index % 10 == 0:
                    print(('image', index))
                    pass
                t0=time.time()

                img_ori = cv2.imread(image_list[index])
                images.append(img_ori)
                img = cv2.resize(img, (HEIGHT, WIDTH))
                img = np.expand_dims(img, 0)
                global_pred = sess.run([global_pred_dict], feed_dict={img_inp: img})

                if index < options.startIndex:
                    continue                


                pred_p = global_pred['plane'][0]
                pred_s = global_pred['segmentation'][0]
                
                pred_np_m = global_pred['non_plane_mask'][0]
                pred_np_d = global_pred['non_plane_depth'][0]
                pred_np_n = global_pred['non_plane_normal'][0]
                
                #if global_gt['info'][0][19] > 1 and global_gt['info'][0][19] < 4 and False:
                #pred_np_n = calcNormal(pred_np_d.squeeze(), global_gt['info'][0])
                #pass


                #pred_b = global_pred['boundary'][0]
                predNonPlaneMasks.append(pred_np_m)                    
                predNonPlaneDepths.append(pred_np_d)
                predNonPlaneNormals.append(pred_np_n)
                #predBoundaries.append(pred_b)
                    
                all_segmentations = np.concatenate([pred_s, pred_np_m], axis=2)

                info = np.zeros(20)
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
                
                plane_depths = calcPlaneDepths(pred_p, width_high_res, height_high_res, info)

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
            pred_dict['image'] = np.array(images)
            pred_dict['info'] = np.array(infos)
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


if __name__=='__main__':
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Planenet')
    parser.add_argument('--task', dest='task',
                        help='task type',
                        default='custom', type=str)
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

    if args.imageIndex >= 0 and args.suffix != '':
        args.test_dir += '/' + args.suffix + '/'
        pass
    
    print(args.titles)
    
    if args.task == 'custom':
        evaluatePlanes(args)
    elif args.task == 'depth':
        evaluateDepthPrediction(args)
    elif args.task == 'search':
        gridSearch(args)
        pass
