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
from RecordReaderSUNCG import *
import csv
import json
import glob

np.set_printoptions(precision=2, linewidth=200)


HEIGHT=192
WIDTH=256
NUM_OBJECTS = 10

MOVING_AVERAGE_DECAY = 0.99
dataFolder = '../Data/SUNCG/'

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def writeRecordFile(split, scene_paths):

    nyu_id_map = {}
    with open('../Data/ScanNet/tasks/scannet-labels.combined.tsv') as label_file:
        label_reader = csv.reader(label_file, delimiter='\t')
        for line_index, line in enumerate(label_reader):
            if line_index > 0:
                if line[4] != '' and line[7] != '':
                    nyu_id_map[line[7]] = int(line[4])
                    pass
                pass
            continue
        label_file.close()
        pass

    model_id_map = {}
    with open(dataFolder + '/SUNCGtoolbox/metadata/ModelCategoryMapping.csv') as label_file:
        label_reader = csv.reader(label_file, delimiter=',')
        for line in label_reader:
            if line[1] != '' and line[5] in nyu_id_map:
                model_id_map[line[1]] = nyu_id_map[line[5]]
                pass
            continue
        pass

    if split == 'train':
        numImagesThreshold = 50000
    else:
        numImagesThreshold = 1000
        pass


    writer = tf.python_io.TFRecordWriter('SUNCG_anchor_' + split + '.tfrecords')


    # reader = RecordReaderSUNCG()
    # batchSize = 8

    # if split == 'train':
    #     filename_queue = tf.train.string_input_producer([rootFolder + '/planes_SUNCG_train.tfrecords'], num_epochs=1)
    #     writer = tf.python_io.TFRecordWriter('SUNCG_train.tfrecords')
    # else:
    #     filename_queue = tf.train.string_input_producer([rootFolder + '/planes_SUNCG_val.tfrecords'], num_epochs=1)
    #     writer = tf.python_io.TFRecordWriter('SUNCG_val.tfrecords')
    #     pass
    # img_inp, global_gt_dict, local_gt_dict = reader.getBatch(filename_queue, batchSize=batchSize, random=False)

    # config=tf.ConfigProto()
    # config.gpu_options.allow_growth=True
    # config.allow_soft_placement=True

    # init_op = tf.group(tf.global_variables_initializer(),
    #                    tf.local_variables_initializer())

    # testdir = 'test/'

    # with tf.Session(config=config) as sess:
    #     sess.run(init_op)

    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #     try:
    #         for _ in xrange(100000):
    #             print(_)
    #             img, gt_dict = sess.run([img_inp, global_gt_dict])
    #             for batchIndex in xrange(batchSize):

    # # print(global_p[batchIndex])
    # # print(original_p[batchIndex])
    # # exit(1)

    # image_path = gt_dict['image_path'][batchIndex]


    if True:
        if True:
            if True:
                info = np.array([517.97, 0., 320., 0., 0., 517.97, 240., 0., 0., 0., 1., 0., 0., 0., 0., 1., 640., 480., 1000., 0.])
                directionsArray = np.array([[1, 1, 1],
                                            [1, 1, -1],
                                            [1, -1, 1],
                                            [1, -1, -1],
                                            [-1, 1, 1],
                                            [-1, 1, -1],
                                            [-1, -1, 1],
                                            [-1, -1, -1]])

                image_paths = []
                for scene_path in scene_paths:
                    image_paths += glob.glob(dataFolder + '/' + scene_path + '/*_mlt.png')
                    continue

                # if split == 'test':
                #     #image_paths.append('../Data/SUNCG//50f5ec027f344b3fb2e700f32be5b2f7/000018_mlt.png')
                #     #image_paths.append('../Data/SUNCG//1e716f245eb94b3f514e2394c52a9786/000017_mlt.png')
                #     image_paths = ['../Data/SUNCG//ce10a93439b96392c77b08ac288f6315/000001_mlt.png']
                #     pass

                random.shuffle(image_paths)
                print(len(image_paths))
                numImages = 0
                for _, image_path in enumerate(image_paths):
                    print(image_path)
                    try:
                        tokens = image_path.split('/')

                        image = cv2.imread(dataFolder + '/' + tokens[4] + '/' + tokens[5])
                        image = cv2.resize(image, (WIDTH, HEIGHT))

                        #image = ((img[batchIndex] + 0.5) * 255).astype(np.uint8)
                        img_raw = image.tostring()
                        depth = cv2.imread(dataFolder + '/' + tokens[4] + '/' + tokens[5].replace('mlt', 'depth'), -1).astype(np.float32) / 1000
                        depth = cv2.resize(depth, (WIDTH, HEIGHT))
                        #depth = gt_dict['depth'][batchIndex]
                        #info = gt_dict['info'][batchIndex]

                        scene_path = tokens[4]
                        cameras = []
                        with open(dataFolder + '/' + tokens[4] + '/room_camera.txt') as camera_file:
                            for line in camera_file:
                                cameras.append([float(value) for value in line.strip().split(' ') if value != ''])
                                continue
                            pass
                        roomIDs = []
                        with open(dataFolder + '/' + tokens[4] + '/room_camera_name.txt') as camera_file:
                            for line in camera_file:
                                roomIDs.append('_'.join(line.split('#')[1].split('_')[:2]))
                                continue
                            pass
                        imageIndex = int(tokens[5].split('_')[0])
                        if imageIndex >= len(cameras) or imageIndex >= len(roomIDs):
                            continue
                        camera = np.array(cameras[imageIndex])
                        roomID = roomIDs[imageIndex]
                        center = camera[:3]
                        towards = camera[3:6]
                        towards /= np.linalg.norm(towards)
                        up = camera[6:9]
                        up /= np.linalg.norm(up)
                        right = np.cross(towards, up)
                        rotation = np.stack([towards, up, right], axis=1)
                        xfov = camera[9]
                        yfov = camera[10]

                        objects = []
                        roomBBox = np.zeros(6)
                        with open(dataFolder + '/house/' + tokens[4] + '/house.json', 'r') as f:
                            house = json.load(f)
                            for level in house['levels']:
                                for node in level['nodes']:
                                    if node['type'] == 'Object' and 'bbox' in node and 'modelId' in node and node['modelId'] in model_id_map and 'transform' in node:
                                        #mins = np.array(node['bbox']['min'])
                                        #maxs = np.array(node['bbox']['max'])
                                        #center = (mins + maxs) / 2
                                        #sizes = maxs - mins
                                        objectTransform = np.array(node['transform']).reshape((4, 4))
                                        objectRotation = objectTransform[:3, :3]
                                        objectTowards = objectRotation[:, 0]
                                        objectUp = objectRotation[:, 1]
                                        objectMins = np.array(node['bbox']['min'])
                                        objectMaxs = np.array(node['bbox']['max'])
                                        objectCenter = (objectMins + objectMaxs) / 2
                                        if np.all(objectMaxs - objectMins < 1e-4):
                                            continue
                                        #objectSizes = objectMaxs - objectMins
                                        objectCorners = np.expand_dims(objectCenter, 0) + np.expand_dims(objectMaxs - objectMins, 0) / 2 * directionsArray
                                        objectCorners = np.matmul(objectCorners, objectRotation)
                                        objectSizes = objectCorners.max(0) - objectCorners.min(0)

                                        objects.append(objectCenter.tolist() + objectSizes.tolist() + objectTowards.tolist() + objectUp.tolist() + [float(model_id_map[node['modelId']])])
                                        pass
                                    if node['type'] == 'Room' and node['id'] == roomID:
                                        roomBBox = node['bbox']['min'] + node['bbox']['max']
                                        pass
                                    continue
                                continue
                            pass
                        if len(objects) == 0:
                            continue
                        objects = np.array(objects, dtype=np.float64)

                        objects = objects[np.logical_and(np.all(objects[:, :3] >= roomBBox[:3], axis=1), np.all(objects[:, :3] <= roomBBox[3:6], axis=1))]


                        points = objects[:, :3]
                        #points = points - objects[:, 3:4] * objects[:, 6:9] / 2 * ((np.sum(points * objects[:, 6:9], axis=-1, keepdims=True) < 0).astype(np.float32) * 2 - 1)

                        transformedPoints = np.matmul(points - np.expand_dims(center, 0), rotation)
                        angleY = np.arctan(transformedPoints[:, 1] / transformedPoints[:, 0])
                        angleZ = np.arctan(transformedPoints[:, 2] / transformedPoints[:, 0])
                        validMask = np.logical_and(np.logical_and(transformedPoints[:, 0] > 0, np.abs(angleY) < yfov), np.abs(angleZ) < xfov)
                        objectCenters = transformedPoints[validMask]
                        objects = objects[validMask]
                        objects[:, :3] = objectCenters

                        directions = objects[:, 6:12].reshape((-1, 3))
                        transformedDirections = np.matmul(directions, rotation)
                        transformedDirections /= np.maximum(np.linalg.norm(transformedDirections, axis=1, keepdims=True), 1e-4)
                        objects[:, 6:12] = transformedDirections.reshape((-1, 6))

                        if _ % 100 == 0 or split == 'test':
                            print(np.linalg.norm(np.cross(objectTowards, objectUp)))
                            print(objects)
                            cv2.imwrite('test/objects.png', drawObjectImage(image, objects, info))
                            if split == 'test':
                                exit(1)
                                pass
                            pass

                        # objects = objects[np.logical_and(np.all(objects[:, :3] >= roomBBox[:3], axis=1), np.all(objects[:, 3:6] <= roomBBox[3:6], axis=1))]
                        # points = objects[:, :6].reshape((-1, 3))

                        # transformedPoints = np.matmul((points - np.expand_dims(center, 0)), rotation)
                        # angleY = np.arctan(transformedPoints[:, 1] / transformedPoints[:, 0])
                        # angleZ = np.arctan(transformedPoints[:, 2] / transformedPoints[:, 0])
                        # validMask = np.logical_and(np.logical_and(transformedPoints[:, 0] > 0, np.abs(angleY) < yfov), np.abs(angleZ) < xfov)
                        # if _ % 100 == 0:
                        #     validPoints = transformedPoints[validMask]
                        #     u = (validPoints[:, 2] / validPoints[:, 0] * info[0] + info[2]) / info[16] * WIDTH
                        #     v = (-validPoints[:, 1] / validPoints[:, 0] * info[5] + info[6]) / info[17] * HEIGHT
                        #     points = np.round(np.stack([u, v], axis=1)).astype(np.int32)
                        #     #print(points)
                        #     for point in points:
                        #         cv2.circle(image, (point[0], point[1]), radius=3, color=(255, 255, 255), thickness=3)
                        #         continue
                        #     cv2.imwrite('test/objects.png', image)
                        #     cv2.imwrite('test/depth.png', drawDepthImage(depth))
                        #     pass

                        # validMask = np.any(validMask.reshape((-1, 2)), axis=1)
                        # objectPoints = transformedPoints.reshape((-1, 6))[validMask]
                        # objectCenters = (objectPoints[:, :3] + objectPoints[:, 3:]) / 2
                        # objectSizes = np.abs(objectPoints[:, 3:] - objectPoints[:, :3])
                        # objectTypes = objects[:, 6][validMask]
                        # objects = np.concatenate([objectCenters, objectSizes, np.expand_dims(objectTypes, -1)], axis=1)
                        if objects.shape[0] > NUM_OBJECTS:
                            continue
                            objects = objects[np.random.choice(np.arange(objects.shape[0]), NUM_OBJECTS, replace=False)]
                            pass
                        if objects.shape[0] < NUM_OBJECTS:
                            objects = np.concatenate([objects, np.zeros((NUM_OBJECTS - objects.shape[0], objects.shape[-1]))], axis=0)
                            pass

                        example = tf.train.Example(features=tf.train.Features(feature={
                            'image_raw': _bytes_feature(img_raw),
                            'image_path': _bytes_feature(image_path),
                            #'normal': _float_feature(normal.reshape(-1)),
                            'depth': _float_feature(depth.reshape(-1)),
                            'info': _float_feature(info.reshape(-1)),
                            'objects': _float_feature(objects.reshape(-1)),
                        }))
                        writer.write(example.SerializeToString())
                        numImages += 1
                    except:
                        pass
                    if numImages >= numImagesThreshold:
                        break
                    continue
                pass
            pass
        pass
    return


def checkData(scene_paths):

    nyu_id_map = {}
    with open('../Data/ScanNet/tasks/scannet-labels.combined.tsv') as label_file:
        label_reader = csv.reader(label_file, delimiter='\t')
        for line_index, line in enumerate(label_reader):
            if line_index > 0:
                if line[4] != '' and line[7] != '':
                    nyu_id_map[line[7]] = int(line[4])
                    pass
                pass
            continue
        label_file.close()
        pass

    model_id_map = {}
    with open(dataFolder + '/SUNCGtoolbox/metadata/ModelCategoryMapping.csv') as label_file:
        label_reader = csv.reader(label_file, delimiter=',')
        for line in label_reader:
            if line[1] != '' and line[5] in nyu_id_map:
                model_id_map[line[1]] = nyu_id_map[line[5]]
                pass
            continue
        pass

    info = np.array([517.97, 0., 320., 0., 0., 517.97, 240., 0., 0., 0., 1., 0., 0., 0., 0., 1., 640., 480., 1000., 0.])

    directionsArray = np.array([[1, 1, 1],
                                [1, 1, -1],
                                [1, -1, 1],
                                [1, -1, -1],
                                [-1, 1, 1],
                                [-1, 1, -1],
                                [-1, -1, 1],
                                [-1, -1, -1]])

    for sceneIndex, scene_path in enumerate(scene_paths):
        image_paths = glob.glob(dataFolder + '/' + scene_path + '/*_mlt.png')

        for _, image_path in enumerate(image_paths):
            print(image_path)
            tokens = image_path.split('/')

            image = cv2.imread(dataFolder + '/' + tokens[4] + '/' + tokens[5])
            image = cv2.resize(image, (WIDTH, HEIGHT))

            #image = ((img[batchIndex] + 0.5) * 255).astype(np.uint8)
            img_raw = image.tostring()
            depth = cv2.imread(dataFolder + '/' + tokens[4] + '/' + tokens[5].replace('mlt', 'depth'), -1).astype(np.float32) / 1000
            depth = cv2.resize(depth, (WIDTH, HEIGHT))

            scene_path = tokens[4]
            cameras = []
            with open(dataFolder + '/' + tokens[4] + '/room_camera.txt') as camera_file:
                for line in camera_file:
                    cameras.append([float(value) for value in line.strip().split(' ') if value != ''])
                    continue
                pass
            roomIDs = []
            with open(dataFolder + '/' + tokens[4] + '/room_camera_name.txt') as camera_file:
                for line in camera_file:
                    roomIDs.append('_'.join(line.split('#')[1].split('_')[:2]))
                    continue
                pass
            imageIndex = int(tokens[5].split('_')[0])
            if imageIndex >= len(cameras) or imageIndex >= len(roomIDs):
                continue
            if imageIndex != 22:
                continue

            camera = np.array(cameras[imageIndex])
            roomID = roomIDs[imageIndex]
            center = camera[:3]
            towards = camera[3:6]
            towards /= np.linalg.norm(towards)
            up = camera[6:9]
            up /= np.linalg.norm(up)
            right = np.cross(towards, up)
            rotation = np.stack([towards, up, right], axis=1)
            xfov = camera[9]
            yfov = camera[10]

            objects = []
            roomBBox = np.zeros(6)
            with open(dataFolder + '/house/' + tokens[4] + '/house.json', 'r') as f:
                house = json.load(f)
                for level in house['levels']:
                    for node in level['nodes']:
                        if node['type'] == 'Object' and 'bbox' in node and 'modelId' in node and node['modelId'] in model_id_map and 'transform' in node:
                            objectTransform = np.array(node['transform']).reshape((4, 4))
                            objectRotation = objectTransform[:3, :3]
                            objectTowards = objectRotation[:, 0]
                            objectUp = objectRotation[:, 1]
                            objectMins = np.array(node['bbox']['min'])
                            objectMaxs = np.array(node['bbox']['max'])
                            if np.all(objectMaxs - objectMins < 1e-4):
                                continue
                            objectCenter = (objectMins + objectMaxs) / 2

                            #objectSizes = objectMaxs - objectMins
                            objectCorners = np.expand_dims(objectCenter, 0) + np.expand_dims(objectMaxs - objectMins, 0) / 2 * directionsArray
                            objectCorners = np.matmul(objectCorners, objectRotation)
                            objectSizes = objectCorners.max(0) - objectCorners.min(0)

                            objects.append(objectCenter.tolist() + objectSizes.tolist() + objectTowards.tolist() + objectUp.tolist() + [float(model_id_map[node['modelId']])])
                            pass
                        if node['type'] == 'Room' and node['id'] == roomID:
                            roomBBox = node['bbox']['min'] + node['bbox']['max']
                            pass
                        continue
                    continue
                pass

            objects = np.array(objects, dtype=np.float64)


            objects = objects[np.logical_and(np.all(objects[:, :3] >= roomBBox[:3], axis=1), np.all(objects[:, :3] <= roomBBox[3:6], axis=1))]

            directions = objects[:, 6:12].reshape((-1, 3))
            transformedDirections = np.matmul(directions, rotation)
            transformedDirections /= np.maximum(np.linalg.norm(transformedDirections, axis=1, keepdims=True), 1e-4)
            objects[:, 6:12] = transformedDirections.reshape((-1, 6))

            points = objects[:, :3]
            points = points - objects[:, 3:4] * objects[:, 6:9] / 2 * ((np.sum(points * objects[:, 6:9], axis=-1, keepdims=True) < 0).astype(np.float32) * 2 - 1)

            transformedPoints = np.matmul(points - np.expand_dims(center, 0), rotation)
            angleY = np.arctan(transformedPoints[:, 1] / transformedPoints[:, 0])
            angleZ = np.arctan(transformedPoints[:, 2] / transformedPoints[:, 0])
            validMask = np.logical_and(np.logical_and(transformedPoints[:, 0] > 0, np.abs(angleY) < yfov), np.abs(angleZ) < xfov)
            objectCenters = transformedPoints[validMask]
            objects = objects[validMask]


            objects[:, :3] = objectCenters


            #corners = np.expand_dims(objects[:, :3], 1) + np.expand_dims(objects[:, 3:6], 1) / 2 * np.expand_dims(directionsArray, 0)
            #print(corners.shape, transformedDirections.shape)
            #transformedCorners = np.tensordot(corners, rotation, axes=([2], [1]))
            #transformedCorners = np.matmul(corners, rotation)
            #objects[:, 3:6] = transformedCorners.max(1) - transformedCorners.min(1)
            #print(np.linalg.norm(np.cross(objectTowards, objectUp)))

            # print(objects[:, 12])
            # drawObject3D('test/' + str(sceneIndex) + '_' + str(imageIndex) + '.ply', objects, axis_aligned=False)
            # cv2.imwrite('test/' + str(sceneIndex) + '_' + str(imageIndex) + '.png', drawObjectImage(image, objects, info))
            # cv2.imwrite('test/' + str(sceneIndex) + '_' + str(imageIndex) + '_image.png', image)
            # exit(1)
            continue
        continue
    return


if __name__=='__main__':
    scene_paths = os.listdir(dataFolder)
    #checkData(['00298efe1bfeead6b172f25f0386b23a'])
    #exit(1)
    random.shuffle(scene_paths)
    writeRecordFile('val', scene_paths[int(len(scene_paths) * 0.9):])
    writeRecordFile('train', scene_paths[:int(len(scene_paths) * 0.9)])
    #writeRecordFile('test', [])
    pass
