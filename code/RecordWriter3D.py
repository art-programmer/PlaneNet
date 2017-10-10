import tensorflow as tf
import numpy as np
import cv2
import random
import PIL.Image
import glob
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
    img = cv2.imread(imagePath['image'])
    img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)

    height = img.shape[0]
    width = img.shape[1]
    img_raw = img.tostring()


    with open(imagePath['info']) as info_file:
        #with open('/mnt/vision/ScanNet/scene') as pose_file:
        info = np.zeros(3 + 4 * 4)
        line_index = 0
        for line in info_file:
            line = line.split(' ')
            if line[0] == 'm_depthWidth':
                info[0] = int(line[2])
            elif line[0] == 'm_depthHeight':
                info[1] = int(line[2])
            elif line[0] == 'm_depthShift':
                info[2] = int(line[2])
            elif line[0] == 'm_calibrationDepthIntrinsic':
                for i in xrange(16):
                    info[3 + i] = float(line[2 + i])
                    continue
                pass
            line_index += 1
            continue
        pass
    
    depth = np.array(PIL.Image.open(imagePath['depth'])).astype(np.float32) / info[2]
    depth = cv2.resize(depth, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)

    globalSegmentation = cv2.imread(imagePath['segmentation'])
    mask = globalSegmentation.sum(axis=2)
    non_plane_mask = mask == 255 * 3
    boundary_mask = (mask == 0).astype(np.uint8)

    if non_plane_mask.sum() + boundary_mask.sum() > mask.shape[0] * mask.shape[1] * 0.5:
        return
    
    globalSegmentation = globalSegmentation[:, :, 2] * (256 * 256) + globalSegmentation[:, :, 1] * 256 + globalSegmentation[:, :, 0]
    globalSegmentation = (np.round(globalSegmentation.astype(np.float32) / 100) - 1).astype(np.int32)

    segments, unique_counts = np.unique(globalSegmentation, return_counts=True)
    segments = segments.tolist()
    unique_counts.tolist()

    #print(segments)

    segmentList = zip(segments, unique_counts)
    segmentList = [segment for segment in segmentList if segment[0] not in [-1, 167771]]

    if len(segmentList) == 0 or len(segmentList) > NUM_PLANES:
        if len(segmentList) > NUM_PLANES and False:
            print('num planes ' + str(len(segmentList)))
            cv2.imwrite('test/image.png', img)            
            cv2.imwrite('test/segmentation.png', drawSegmentationImage(globalSegmentation))
            cv2.imwrite('test/boundary.png', drawMaskImage(boundary_mask))
            cv2.imwrite('test/non_plane.png', drawMaskImage(non_plane_mask))
            cv2.imwrite('test/depth.png', drawDepthImage(depth))
            print(imagePath['segmentation'])
            exit(1)
            pass
        return
        
    #segmentList = sorted(segmentList, key=lambda x:-x[1])
    #segmentList = segmentList[:min(len(segmentList), NUM_PLANES)]
    segments, unique_counts = zip(*segmentList)
    segments = list(segments)
    unique_counts = list(unique_counts)

    
    globalPlanes = np.load(imagePath['plane'])
    numGlobalPlanes = globalPlanes.shape[0]
    globalPlaneRelations = np.load(imagePath['plane_relation'])
    segmentation = np.zeros(globalSegmentation.shape)
    planes = []
    for segmentIndex, globalSegmentIndex in enumerate(segments):
        segmentation[globalSegmentation == globalSegmentIndex] = segmentIndex + 1
        planes.append(globalPlanes[globalSegmentIndex])
        continue
    planes = np.array(planes)
    numPlanes = planes.shape[0]
    planeMapping = np.zeros((NUM_PLANES, numGlobalPlanes))
    planeMapping[np.arange(numPlanes), segments] = 1
    planeRelations = np.matmul(planeMapping, np.matmul(globalPlaneRelations, np.transpose(planeMapping)))

    segmentation = segmentation.astype(np.uint8)
    segmentation[non_plane_mask] = NUM_PLANES + 1
    
    kernel = np.ones((3, 3), np.uint8)
    kernel[0][0] = kernel[2][0] = kernel[0][2] = kernel[2][2] = 0

    ori_boundary_mask = boundary_mask
    for _ in xrange(2):
        segmentation = segmentation + cv2.dilate(segmentation, kernel) * boundary_mask
        boundary_mask = boundary_mask * (segmentation == 0)
        continue
    smooth_boundary = cv2.resize(np.maximum(ori_boundary_mask - boundary_mask, 0), (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
    
    segmentation -= 1
    segmentation = cv2.resize(segmentation, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)

    
    with open(imagePath['pose']) as pose_file:
        #with open('/mnt/vision/ScanNet/scene') as pose_file:
        pose = []
        for line in pose_file:
            line = line.split(' ')
            for value in line:
                pose.append(float(value))
                continue
            continue
        pose = np.array(pose).reshape([4, 4])
        pass

    pose = np.linalg.inv(pose)
    temp = pose[1].copy()
    pose[1] = pose[2]
    pose[2] = -temp

    planes = transformPlanes(planes, pose)

    if numPlanes < NUM_PLANES:
        planes = np.concatenate([planes, np.zeros((NUM_PLANES - numPlanes, 3))], axis=0)
        pass

    
    example = tf.train.Example(features=tf.train.Features(feature={
        'image_path': _bytes_feature(imagePath['image']),
        'image_raw': _bytes_feature(img_raw),
        'depth': _float_feature(depth.reshape(-1)),
        'plane': _float_feature(planes.reshape(-1)),
        'num_planes': _int64_feature([numPlanes]),
        'segmentation_raw': _bytes_feature(segmentation.tostring()),        
        'smooth_boundary_raw': _bytes_feature(smooth_boundary.tostring()),        
        'plane_relation': _float_feature(planeRelations.reshape(-1)),
        'info': _float_feature(info.reshape(-1)),        
    }))
    writer.write(example.SerializeToString())
    return


def writeRecordFile(tfrecords_filename, imagePaths):
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    for index, imagePath in enumerate(imagePaths):
        if index % 100 == 0:
            print(index)            
            pass
        writeExample(writer, imagePath)
        continue
    writer.close()
    return


if __name__=='__main__':
    #imagePaths = glob.glob('/home/chenliu/Projects/Data/ScanNet/*/annotation/segmentation/frame-*.segmentation.png')
    #imagePaths = glob.glob('/mnt/vision/ScanNet/*/annotation/segmentation/frame-*.segmentation.png')

    datasets = {'scannet': '/mnt/vision/ScanNet/data/', 'matterport': '/mnt/vision/matterport/data/v1/scans/'}

    for dataset, ROOT_FOLDER in datasets.iteritems():
        if dataset == 'matterport':
            continue
        all_scene_ids = os.listdir(ROOT_FOLDER)

        all_scene_ids = os.listdir(ROOT_FOLDER)

        scene_ids = all_scene_ids[int(len(all_scene_ids) * 0.9):]
        segmentationPaths = []
        for scene_id in scene_ids:
            segmentationPaths += glob.glob(ROOT_FOLDER + scene_id + '/annotation/segmentation*/frame-*.segmentation.png')
            continue

        imagePaths = []
        for segmentationPath in segmentationPaths:
            framePath = segmentationPath.replace('annotation/segmentation', 'frames')
            imagePath = {'image': framePath.replace('segmentation.png', 'color.jpg'), 'depth': framePath.replace('segmentation.png', 'depth.pgm'), 'segmentation': segmentationPath, 'plane': '/'.join(segmentationPath.split('/')[:-2]) + '/planes.npy', 'plane_relation': '/'.join(segmentationPath.split('/')[:-2]) + '/plane_relations.npy', 'pose': framePath.replace('segmentation.png', 'pose.txt'), 'info': '/'.join(framePath.split('/')[:-1]) + '/_info.txt'}
            imagePaths.append(imagePath)
            continue
        random.shuffle(imagePaths)
    
        writeRecordFile('/mnt/vision/planes_matterport_val.tfrecords', imagePaths)

    
        scene_ids = all_scene_ids[:int(len(all_scene_ids) * 0.9)]
        segmentationPaths = []
        for scene_id in scene_ids:
            segmentationPaths += glob.glob(ROOT_FOLDER + scene_id + '/annotation/segmentation_*/frame-*.segmentation.png')
            continue

        imagePaths = []
        for segmentationPath in segmentationPaths:
            framePath = segmentationPath.replace('annotation/segmentation', 'frames')
            imagePath = {'image': framePath.replace('segmentation.png', 'color.jpg'), 'depth': framePath.replace('segmentation.png', 'depth.pgm'), 'segmentation': segmentationPath, 'plane': '/'.join(segmentationPath.split('/')[:-2]) + '/planes.npy', 'plane_relation': '/'.join(segmentationPath.split('/')[:-2]) + '/plane_relations.npy', 'pose': framePath.replace('segmentation.png', 'pose.txt'), 'info': '/'.join(framePath.split('/')[:-1]) + '/_info.txt'}
            imagePaths.append(imagePath)
            continue
        #print(len(imagePaths))
        #exit(1)
        random.shuffle(imagePaths)
    
        writeRecordFile('/mnt/vision/planes_matterport_train.tfrecords', imagePaths)
        continue
    


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
