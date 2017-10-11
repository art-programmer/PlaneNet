import tensorflow as tf
import numpy as np
import cv2
import random
import PIL.Image
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules import *
from utils import *
from RecordReader3D import *

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
        reader = RecordReader3D()
        filename_queue = tf.train.string_input_producer(['../planes_' + dataset + '_train.tfrecords'], num_epochs=1)
        img_inp, global_gt_dict, _ = reader.getBatch(filename_queue, numOutputPlanes=numOutputPlanes, batchSize=batchSize, random=False, getLocal=True)
        writer = tf.python_io.TFRecordWriter('/mnt/vision/PlaneNet/planes_' + dataset + '_train.tfrecords')
        numImages = 50000
    else:
        reader = RecordReader3D()
        filename_queue = tf.train.string_input_producer(['../planes_' + dataset + '_val.tfrecords'], num_epochs=1)
        img_inp, global_gt_dict, _ = reader.getBatch(filename_queue, numOutputPlanes=numOutputPlanes, batchSize=batchSize, random=False, getLocal=True)
        writer = tf.python_io.TFRecordWriter('/mnt/vision/PlaneNet/planes_' + dataset + '_val.tfrecords')
        numImages = 1000
        pass
    
        
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    #segmentation_gt, plane_mask = fitPlaneMasksModule(global_gt_dict['plane'], global_gt_dict['depth'], global_gt_dict['normal'], width=WIDTH, height=HEIGHT, normalDotThreshold=np.cos(np.deg2rad(5)), distanceThreshold=0.05, closing=True, one_hot=True)
    #global_gt_dict['segmentation'] = tf.argmax(tf.concat([segmentation_gt, 1 - plane_mask], axis=3), axis=3)

    segmentation_gt = tf.cast(tf.equal(global_gt_dict['segmentation'], tf.reshape(tf.range(NUM_PLANES), (1, 1, 1, -1))), tf.float32)
    plane_mask = tf.cast(tf.less(global_gt_dict['segmentation'], NUM_PLANES), tf.float32)
    global_gt_dict['boundary'] = findBoundaryModuleSmooth(global_gt_dict['depth'], segmentation_gt, plane_mask, global_gt_dict['smooth_boundary'], max_depth_diff = 0.1, max_normal_diff = np.sqrt(2 * (1 - np.cos(np.deg2rad(20)))))
    
    
    # info = np.zeros(20)
    # info[0] = 5.1885790117450188e+02
    # info[2] = 3.2558244941119034e+02 - 40
    # info[5] = 5.1946961112127485e+02
    # info[6] = 2.5373616633400465e+02 - 44
    # info[10] = 1
    # info[15] = 1
    # info[16] = 561
    # info[17] = 427
    # info[18] = 1000
    # info[19] = 1
    
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for _ in xrange(numImages / batchSize):
                print(_)
                img, global_gt = sess.run([img_inp, global_gt_dict])
                for batchIndex in xrange(batchSize):
                    image = ((img[batchIndex] + 0.5) * 255).astype(np.uint8)
                    segmentation = global_gt['segmentation'][batchIndex].astype(np.uint8).squeeze()
                    boundary = global_gt['boundary'][batchIndex].astype(np.uint8)

                 
                    #boundary = np.concatenate([boundary, np.zeros((HEIGHT, WIDTH, 1))], axis=2)

                    info = global_gt['info'][batchIndex]
                    datasetIndex = 2
                    if dataset == 'scannet':
                        datasetIndex = 3
                        pass
                    info = np.concatenate([info[3:], info[:3], np.ones(1) * datasetIndex])

                    # cv2.imwrite('test/segmentation_' + str(batchIndex) + '.png', drawSegmentationImage(segmentation, planeMask = segmentation < 20, black=True))
                    # cv2.imwrite('test/boundary_' + str(batchIndex) + '.png', drawMaskImage(boundary))
                    # cv2.imwrite('test/image_' + str(batchIndex) + '.png', image)               
                    
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'image_path': _bytes_feature(global_gt['image_path'][batchIndex]),
                        'image_raw': _bytes_feature(image.tostring()),
                        'depth': _float_feature(global_gt['depth'][batchIndex].reshape(-1)),
                        'normal': _float_feature(np.zeros((HEIGHT * WIDTH * 3))),
                        'plane': _float_feature(global_gt['plane'][batchIndex].reshape(-1)),
                        'num_planes': _int64_feature([global_gt['num_planes'][batchIndex]]),
                        'segmentation_raw': _bytes_feature(segmentation.tostring()),        
                        'boundary_raw': _bytes_feature(boundary.tostring()),
                        #'plane_relation': _float_feature(planeRelations.reshape(-1)),
                        'info': _float_feature(info),
                    }))
                    
                    writer.write(example.SerializeToString())
                    continue
                #exit(1)
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
    return

    
if __name__=='__main__':
    #writeRecordFile('val', 'matterport')
    writeRecordFile('val', 'scannet')
    # writeRecordFile('train', 'matterport')    
    # writeRecordFile('train', 'scannet')    

