
import tensorflow as tf
import numpy as np
import cv2
import random
import PIL.Image
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from modules import *
from RecordReaderAll import *
from utils import *

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
        filename_queue = tf.train.string_input_producer(['../../Data/PlaneNet/planes_' + dataset + '_train.tfrecords'], num_epochs=1)
        img_inp, global_gt_dict, _ = reader.getBatch(filename_queue, numOutputPlanes=numOutputPlanes, batchSize=batchSize, random=False, getLocal=True)
        numImages = 50000
    else:
        reader = RecordReaderAll()
        filename_queue = tf.train.string_input_producer(['../../Data/PlaneNet/planes_' + dataset + '_val.tfrecords'], num_epochs=1)
        img_inp, global_gt_dict, _ = reader.getBatch(filename_queue, numOutputPlanes=numOutputPlanes, batchSize=batchSize, random=False, getLocal=True)
        numImages = 1000
        pass
    
        
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for _ in xrange(numImages / batchSize):
                img, global_gt = sess.run([img_inp, global_gt_dict])
                if _ % 500 == 0:
                    print(_)
                    pass
                for batchIndex in xrange(batchSize):
                    imagePath = global_gt['image_path'][batchIndex]
                    if '/mnt/vision/' in imagePath:
                        imagePath = imagePath.replace('/mnt/vision/', '../../Data/')
                    elif '/home/chenliu/Projects/Data/' in imagePath:
                        imagePath = imagePath.replace('/home/chenliu/Projects/Data/', '../../Data/')
                        pass
                    
                    tokens = imagePath.split('/')
                    if not os.path.exists('/'.join(tokens[:-2])):
                        os.system('mkdir ' + '/'.join(tokens[:-2]))
                        pass
                    annotationPath = '/'.join(tokens[:-2]) + '/annotation_new/'

                    info = global_gt['info'][batchIndex]
                    info[1] = info[5]
                    info[3] = info[6]
                    info[4] = info[16]
                    info[5] = info[17]
                    info[6] = info[18]
                    info[7] = info[8] = info[9] = 0
                    info = info[:10]
                    if not os.path.exists(annotationPath + '/info.npy'):
                        np.save(annotationPath + '/info.npy', info)
                        pass
                    continue
                    
                    if os.path.exists(annotationPath + tokens[-1].replace('color.jpg', 'planes.npy')):
                        continue

                    image = ((img[batchIndex] + 0.5) * 255).astype(np.uint8)                    
                    numPlanes = global_gt['num_planes'][batchIndex]
                    if numPlanes == 0:
                        continue
                    segmentation = np.argmax(np.concatenate([global_gt['segmentation'][batchIndex], global_gt['non_plane_mask'][batchIndex]], axis=-1), axis=-1).astype(np.uint8).squeeze()
                    #boundary = global_gt['boundary'][batchIndex].astype(np.uint8)
                    #semantics = global_gt['semantics'][batchIndex].astype(np.uint8)
                    planes = global_gt['plane'][batchIndex]
                    
                    if not os.path.exists(annotationPath):
                        os.system('mkdir ' + annotationPath)
                        pass
                    cv2.imwrite(annotationPath + tokens[-1].replace('color.jpg', 'segmentation.png'), segmentation)
                    np.save(annotationPath + tokens[-1].replace('color.jpg', 'planes.npy'), planes[:numPlanes])
                    cv2.imwrite(annotationPath + tokens[-1], image)
                    depth = np.round(global_gt['depth'][batchIndex] * 1000).astype(np.uint16)
                    cv2.imwrite(annotationPath + tokens[-1].replace('color.jpg', 'depth.png'), depth)
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
    return

    
if __name__=='__main__':
    #writeRecordFile('val', 'matterport')
    #writeRecordFile('val', 'scannet')
    # writeRecordFile('train', 'matterport')    
    #writeRecordFile('train', 'scannet')
    #writeRecordFile('train', 'nyu_rgbd')
    writeRecordFile('val', 'scannet')    
