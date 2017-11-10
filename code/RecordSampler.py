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
from RecordReaderAll import *
from SegmentationRefinement import *

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
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_' + dataset + '_train.tfrecords'], num_epochs=1)
        img_inp, global_gt_dict, _ = reader.getBatch(filename_queue, numOutputPlanes=numOutputPlanes, batchSize=batchSize, random=False, getLocal=True)
        numImages = 5000
        writer = tf.python_io.TFRecordWriter('/mnt/vision/PlaneNet/planes_' + dataset + '_train_sample_' + str(numImages) + '.tfrecords')

    else:
        reader = RecordReaderAll()
        filename_queue = tf.train.string_input_producer(['/mnt/vision/PlaneNet/planes_' + dataset + '_val.tfrecords'], num_epochs=1)
        img_inp, global_gt_dict, _ = reader.getBatch(filename_queue, numOutputPlanes=numOutputPlanes, batchSize=batchSize, random=False, getLocal=True)
        writer = tf.python_io.TFRecordWriter('/mnt/vision/PlaneNet/planes_' + dataset + '_val_sample.tfrecords')
        numImages = 100
        pass
    
        
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    
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
                    
                    segmentation = np.argmax(np.concatenate([global_gt['segmentation'][batchIndex], global_gt['non_plane_mask'][batchIndex]], axis=-1), axis=-1).astype(np.uint8).squeeze()
                    #boundary = global_gt['boundary'][batchIndex].astype(np.uint8)
                    semantics = global_gt['semantics'][batchIndex].astype(np.uint8)
                    boundary = global_gt['boundary'][batchIndex].astype(np.uint8)

                    planes = global_gt['plane'][batchIndex]
                    numPlanes = global_gt['num_planes'][batchIndex]

                    # cv2.imwrite('test/image.png', image)
                    # cv2.imwrite('test/segmentation.png', drawSegmentationImage(segmentation))
                    # exit(1)
                    
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
                        'info': _float_feature(global_gt['info'][batchIndex]),
                    }))
                    
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
    return

    
if __name__=='__main__':
    #writeRecordFile('val', 'matterport')
    #writeRecordFile('val', 'scannet')
    # writeRecordFile('train', 'matterport')    
    writeRecordFile('train', 'scannet')    

