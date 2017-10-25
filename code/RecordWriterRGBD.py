import tensorflow as tf
import numpy as np
import cv2
import random
import PIL.Image
import glob
import scipy.io as sio
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

def writeExample(writer, imagePath):
    #img = np.array(Image.open(imagePath['image']))
    #img = cv2.imread(imagePath['image'])


    #img = sio.loadmat(imagePath['image'])['imgRgb']
    #img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)    
    #depth = sio.loadmat(imagePath['depth'])['imgDepth']
    #depth = cv2.resize(depth, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    #normal = sio.loadmat(imagePath['normal'])['imgNormals']
    #normal = cv2.resize(normal, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)


    img = cv2.imread(imagePath['image'])
    img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)    
    
    height = img.shape[0]
    width = img.shape[1]
    img_raw = img.tostring()
    
    depth = cv2.imread(imagePath['depth'], -1).astype(np.float32) / 255 * 10
    invalidMask = (depth < 1e-4).astype(np.float32)
    invalidMask = (cv2.resize(invalidMask, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR) > 1e-4).astype(np.bool)
    depth = cv2.resize(depth, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    depth[invalidMask] = 0
    
    normal = cv2.imread(imagePath['normal']).astype(np.float32) / (255 / 2) - 1
    normal = cv2.resize(normal, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)    
    
    plane_data = sio.loadmat(imagePath['plane'])['planeData']
    segmentation = (plane_data[0][0][0] - 1).astype(np.int32)
    
    segmentation = cv2.resize(segmentation, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
    #print(segmentation.shape)
    planes = plane_data[0][0][1]
    planes = planes[:, :3] * planes[:, 3:4]
    numPlanes = planes.shape[0]
    if numPlanes > NUM_PLANES:
        return
    
    if numPlanes < NUM_PLANES:
        segmentation[segmentation == numPlanes] = NUM_PLANES
        planes = np.concatenate([planes, np.zeros((NUM_PLANES - numPlanes, 3))], axis=0)
        pass


    info = np.zeros(20)
    info[0] = 5.1885790117450188e+02
    info[2] = 3.2558244941119034e+02 - 40
    info[5] = 5.1946961112127485e+02
    info[6] = 2.5373616633400465e+02 - 44
    info[10] = 1
    info[15] = 1
    info[16] = 561
    info[17] = 427
    info[18] = 1000
    info[19] = 1
    
    #print(segmentation.shape)
    #exit(1)
    example = tf.train.Example(features=tf.train.Features(feature={
        'image_path': _bytes_feature(imagePath['image']),
        'image_raw': _bytes_feature(img_raw),
        'depth': _float_feature(depth.reshape(-1)),
        'normal': _float_feature(normal.reshape(-1)),
        'plane': _float_feature(planes.reshape(-1)),
        'num_planes': _int64_feature([numPlanes]),
        'segmentation_raw': _bytes_feature(segmentation.tostring()),
        'info': _float_feature(info),
    }))
    writer.write(example.SerializeToString())
    return


def writeRecordFile(tfrecords_filename, imagePaths):
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    for index, imagePath in enumerate(imagePaths):
        print(index)
        if index % 100 == 0:
            pass
        writeExample(writer, imagePath)
        #cv2.imwrite('test/image_' + str(index) + '.png', img)
        #cv2.imwrite('test/segmentation_' + str(index) + '.png', drawSegmentationImage(segmentation, planeMask=segmentation < segmentation.max(), black=True))
        #if index == 10:
        #break
        continue
    writer.close()
    return


if __name__=='__main__':
    imagePaths = glob.glob('/mnt/vision2/NYU_RGBD/train/color_*.png')
    imagePaths = [{'image': imagePath, 'depth': imagePath.replace('color', 'depth'), 'normal': imagePath.replace('color', 'normal'), 'plane': imagePath.replace('color', 'plane').replace('png', 'mat')} for imagePath in imagePaths]
    
    # splits = sio.loadmat('/mnt/vision/NYU_RGBD/splits.mat')
    # trainInds = splits['trainNdxs'].reshape(-1).tolist()
    # imagePaths = []
    # for index in trainInds:
    #     imagePath = '/mnt/vision/NYU_RGBD/images_rgb/rgb_%06d.mat' % (index)
    #     imagePaths.append({'image': imagePath, 'depth': imagePath.replace('rgb', 'depth'), 'normal': imagePath.replace('images_rgb', 'surface_normals').replace('rgb', 'surface_normals'), 'plane': imagePath.replace('images_rgb', 'planes').replace('rgb', 'plane_data')})
    #     continue
        
    print(len(imagePaths))
    # #exit(1)
    random.shuffle(imagePaths)
    writeRecordFile('../planes_nyu_rgbd_train.tfrecords', imagePaths)
    #exit(1)
    
    # testInds = splits['testNdxs'].reshape(-1).tolist()
    # imagePaths = []
    # for index in testInds:
    #     imagePath = '/mnt/vision/NYU_RGBD/images_rgb/rgb_%06d.mat' % (index)
    #     imagePaths.append({'image': imagePath, 'depth': imagePath.replace('rgb', 'depth'), 'normal': imagePath.replace('images_rgb', 'surface_normals').replace('rgb', 'surface_normals'), 'plane': imagePath.replace('images_rgb', 'planes').replace('rgb', 'plane_data')})
    #     continue
        
    # print(len(imagePaths))
    # #exit(1)
    # random.shuffle(imagePaths)
    # writeRecordFile('../planes_nyu_rgbd_val.tfrecords', imagePaths)

    
    
    #reader.readRecordFile()


    # # The op for initializing the variables.
    # init_op = tf.group(tf.global_variables_initializer(),
    #                    tf.local_variables_initializer())

    # with tf.Session()  as sess:

    #     sess.run(init_op)

    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)

    #     for i in xrange(3):
    #         image, plane, plane_mask = sess.run([image_inp, plane_inp, plane_mask_inp])
    #         print(image.shape)
    #         print(plane)
    #         print(plane_mask.shape)
    #         for index in xrange(image.shape[0]):
    #             print(plane[index])
    #             cv2.imwrite('test/record_' + str(index) + '_image.png', ((image[index] + 0.5) * 255).astype(np.uint8))
    #             cv2.imwrite('test/record_' + str(index) + '_mask.png', (plane_mask[index, :, :, 0] * 255).astype(np.uint8))
    #             continue
    #         exit(1)
    #         continue
    #     pass
    # exit(1)

#func = partial(writeExample, writer, 1)
#pool.map(func, self.imagePaths[self.numTrainingImages:])
#pool.close()
#pool.join()
#writer.close()
