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
#from SegmentationRefinement import *

HEIGHT=192
WIDTH=256
NUM_PLANES = 20

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def writeRecordFile(split, dataset):
    
    batchSize = 8
    numOutputPlanes = 20
    if split == 'train':
        reader = RecordReaderAll()
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_' + dataset + '_train_new.tfrecords'], num_epochs=1)
        img_inp, global_gt_dict, _ = reader.getBatch(filename_queue, numOutputPlanes=numOutputPlanes, batchSize=batchSize, random=False, getLocal=True)
        writer = tf.python_io.TFRecordWriter('/mnt/vision/PlaneNet/planes_' + dataset + '_train_temp.tfrecords')
        numImages = 50000
    else:
        reader = RecordReaderAll()
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_' + dataset + '_val.tfrecords'], num_epochs=1)
        img_inp, global_gt_dict, _ = reader.getBatch(filename_queue, numOutputPlanes=numOutputPlanes, batchSize=batchSize, random=False, getLocal=True)
        writer = tf.python_io.TFRecordWriter('/mnt/vision/PlaneNet/planes_' + dataset + '_val_temp.tfrecords')
        numImages = 1000
        pass
    

    parser = argparse.ArgumentParser(description='Planenet')
    options = parser.parse_args()
    options.deepSupervisionLayers = ['res4b22_relu', ]
    options.predictConfidence = 0
    options.predictLocal = 0
    options.predictPixelwise = 1
    options.predictBoundary = 1
    options.predictSemantics = 0    
    options.anchorPlanes = 0
    options.numOutputPlanes = 20
    options.batchSize = 8
    options.useNonPlaneDepth = 1
    

    training_flag = tf.constant(False, tf.bool)
    
    options.gpu_id = 0
    global_pred_dict, _, _ = build_graph(img_inp, img_inp, training_flag, options)    

    var_to_restore = [v for v in tf.global_variables()]

    config=tf.ConfigProto()
    config.allow_soft_placement=True    
    #config.gpu_options.allow_growth=True
    config.gpu_options.per_process_gpu_memory_fraction=0.9    
    
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    #numPlanesArray = []
    
    with tf.Session(config=config) as sess:
        sess.run(init_op)        
        loader = tf.train.Saver(var_to_restore)
        loader.restore(sess, '/mnt/vision/PlaneNet/checkpoint/planenet_hybrid3_ll1_pb_pp/checkpoint.ckpt')
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            for _ in xrange(numImages / batchSize):
                img, global_gt, pred_dict = sess.run([img_inp, global_gt_dict, global_pred_dict])

                print(_)                
                for batchIndex in xrange(batchSize):
                    numPlanes = global_gt['num_planes'][batchIndex]
                    if numPlanes == 0:
                        print(_)
                        print('no plane')
                        continue
                    
                    image = ((img[batchIndex] + 0.5) * 255).astype(np.uint8)

                    gt_s = np.concatenate([global_gt['segmentation'][batchIndex], global_gt['non_plane_mask'][batchIndex]], axis=-1)
                    segmentation = np.argmax(gt_s, axis=-1).astype(np.uint8).squeeze()
                    boundary = global_gt['boundary'][batchIndex].astype(np.uint8)
                    semantics = global_gt['semantics'][batchIndex].astype(np.uint8)


                    planes = global_gt['plane'][batchIndex]
                    if np.isnan(planes).any():
                        print(global_gt['image_path'][batchIndex])
                        planes, segmentation, numPlanes = removeSmallSegments(planes, np.zeros((HEIGHT, WIDTH, 3)), global_gt['depth'][batchIndex].squeeze(), np.zeros((HEIGHT, WIDTH, 3)), np.argmax(global_gt['segmentation'][batchIndex], axis=-1), global_gt['semantics'][batchIndex], global_gt['info'][batchIndex], global_gt['num_planes'][batchIndex])
                        if np.isnan(planes).any():
                            np.save('temp/plane.npy', global_gt['plane'][batchIndex])                        
                            np.save('temp/depth.npy', global_gt['depth'][batchIndex])
                            np.save('temp/segmentation.npy', global_gt['segmentation'][batchIndex])
                            np.save('temp/info.npy', global_gt['info'][batchIndex])
                            np.save('temp/num_planes.npy', global_gt['num_planes'][batchIndex])
                            print('why')
                            pass
                        exit(1)                        
                        pass

                    
                    #if _ * batchSize + batchIndex < 29:
                    #continue
                    
                    
                    pred_s = np.concatenate([pred_dict['segmentation'][batchIndex], pred_dict['non_plane_mask'][batchIndex]], axis=-1)
                    #pred_s[:, :, numOutputPlanes] -= 0.1
                    pred_s = one_hot(np.argmax(pred_s, axis=-1), numOutputPlanes + 1)
                    #planes, segmentation, numPlanes = filterPlanes(planes, gt_s, global_gt['depth'][batchIndex].squeeze(), global_gt['info'][batchIndex], pred_s)
                    planes, segmentation, numPlanes = filterPlanes(planes, gt_s, global_gt['depth'][batchIndex].squeeze(), global_gt['info'][batchIndex])


                    #cv2.imwrite('test/segmentation_' + str(batchIndex) + '_ori.png', drawSegmentationImage(gt_s, blackIndex=20))
                    #cv2.imwrite('test/segmentation_' + str(batchIndex) + '_pred.png', drawSegmentationImage(pred_s, blackIndex=20))
                    #cv2.imwrite('test/segmentation_' + str(batchIndex) + '_new.png', drawSegmentationImage(segmentation, blackIndex=20))

                    # plane_depths = calcPlaneDepths(planes, WIDTH, HEIGHT, global_gt['info'][batchIndex])
                    # all_depths = np.concatenate([plane_depths, global_gt['depth'][batchIndex]], axis=2)
                    # depth = np.sum(all_depths * one_hot(segmentation.astype(np.int32), numOutputPlanes + 1), axis=2)
                    # cv2.imwrite('test/segmentation_' + str(batchIndex) + '_depth.png', drawDepthImage(depth))


                    # if batchIndex == 6:
                    #     print(planes)
                    #     exit(1)
                    #continue

                    if numPlanes == 0:
                        continue

                    #print(global_gt['num_planes'][batchIndex], numPlanes)
                    #numPlanesArray.append(numPlanes)
                    
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'image_path': _bytes_feature(global_gt['image_path'][batchIndex]),
                        'image_raw': _bytes_feature(image.tostring()),
                        'depth': _float_feature(global_gt['depth'][batchIndex].reshape(-1)),
                        'normal': _float_feature(global_gt['normal'][batchIndex].reshape(-1)),
                        'semantics_raw': _bytes_feature(semantics.tostring()),
                        'plane': _float_feature(planes.reshape(-1)),
                        'num_planes': _int64_feature([numPlanes]),
                        'segmentation_raw': _bytes_feature(segmentation.tostring()),
                        'boundary_raw': _bytes_feature(boundary.tostring()),
                        #'plane_relation': _float_feature(planeRelations.reshape(-1)),
                        'info': _float_feature(global_gt['info'][batchIndex])}))
                    
                    writer.write(example.SerializeToString())
                    continue
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
    #np.save('results/num_planes.npy', np.array(numPlanesArray))
    return

    
if __name__=='__main__':
    #writeRecordFile('val', 'matterport')
    #writeRecordFile('val', 'scannet')
    # writeRecordFile('train', 'matterport')    
    #writeRecordFile('train', 'scannet')
    #writeRecordFile('train', 'nyu_rgbd')
    writeRecordFile('train', 'scannet')

