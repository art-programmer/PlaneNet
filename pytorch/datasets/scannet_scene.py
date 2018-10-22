import numpy as np
import glob
import cv2
import os

from utils import *

## This class handle one scene of the scannet dataset and provide interface for dataloaders
class ScanNetScene():
    def __init__(self, options, scenePath, scene_id):
        self.options = options

        self.loadCached = False
        self.scannetVersion = 2
        
        if not self.loadCached:
            self.metadata = np.zeros(10)

            if self.scannetVersion == 1:
                with open(scenePath + '/frames/_info.txt') as f:
                    for line in f:
                        line = line.strip()
                        tokens = [token for token in line.split(' ') if token.strip() != '']
                        if tokens[0] == "m_calibrationColorIntrinsic":
                            intrinsics = np.array([float(e) for e in tokens[2:]])
                            intrinsics = intrinsics.reshape((4, 4))
                            self.metadata[0] = intrinsics[0][0]
                            self.metadata[1] = intrinsics[1][1]
                            self.metadata[2] = intrinsics[0][2]
                            self.metadata[3] = intrinsics[1][2]                    
                        elif tokens[0] == "m_colorWidth":
                            self.colorWidth = int(tokens[2])
                        elif tokens[0] == "m_colorHeight":
                            self.colorHeight = int(tokens[2])
                        elif tokens[0] == "m_depthWidth":
                            self.depthWidth = int(tokens[2])
                        elif tokens[0] == "m_depthHeight":
                            self.depthHeight = int(tokens[2])
                        elif tokens[0] == "m_depthShift":
                            self.depthShift = int(tokens[2])
                        elif tokens[0] == "m_frames.size":
                            self.numImages = int(tokens[2])
                            pass
                        continue
                    pass
                self.imagePaths = glob.glob(scenePath + '/frames/frame-*color.jpg')
            else:
                with open(scenePath + '/' + scene_id + '.txt') as f:
                    for line in f:
                        line = line.strip()
                        tokens = [token for token in line.split(' ') if token.strip() != '']
                        if tokens[0] == "fx_color":
                            self.metadata[0] = float(tokens[2])
                        if tokens[0] == "fy_color":
                            self.metadata[1] = float(tokens[2])
                        if tokens[0] == "mx_color":
                            self.metadata[2] = float(tokens[2])                            
                        if tokens[0] == "my_color":
                            self.metadata[3] = float(tokens[2])                            
                        elif tokens[0] == "colorWidth":
                            self.colorWidth = int(tokens[2])
                        elif tokens[0] == "colorHeight":
                            self.colorHeight = int(tokens[2])
                        elif tokens[0] == "depthWidth":
                            self.depthWidth = int(tokens[2])
                        elif tokens[0] == "depthHeight":
                            self.depthHeight = int(tokens[2])
                        elif tokens[0] == "numDepthFrames":
                            self.numImages = int(tokens[2])
                            pass
                        continue
                    pass
                self.depthShift = 1000.0
                self.imagePaths = glob.glob(scenePath + '/frames/color/*.jpg')                
                pass
                    
            self.metadata[4] = self.colorWidth
            self.metadata[5] = self.colorHeight
            self.planes = np.load(scenePath + '/annotation/planes.npy')

            #self.imagePaths = [imagePath for imagePath in self.imagePaths if os.path.exists(imagePath.replace('frames/', 'annotation/segmentation/').replace('color.jpg', 'segmentation.png')) and os.path.exists(imagePath.replace('color.jpg', 'depth.pgm')) and os.path.exists(imagePath.replace('color.jpg', 'pose.txt'))]
            
        else:
            self.metadata = np.load(scenePath + '/annotation_new/info.npy')
            self.imagePaths = glob.glob(scenePath + '/annotation_new/frame-*.segmentation.png')
            pass
        
        # self.imagePaths = []
        # for imageIndex in xrange(self.numImages):
        #     self.imagePaths.append('%s/frames/frame-%06d.color.jpg'%(scenePath, imageIndex))
        #     continue
        return

    def getItemCached(self, imageIndex):
        segmentationPath = self.imagePaths[imageIndex]
        imagePath = segmentationPath.replace('annotation_new/', 'frames/').replace('segmentation.png', 'color.jpg')
        image = cv2.imread(imagePath)
        depth = cv2.imread(imagePath.replace('color.jpg', 'depth.pgm'), -1).astype(np.float32) / self.metadata[6]
        extrinsics_inv = []
        with open(imagePath.replace('color.jpg', 'pose.txt'), 'r') as f:
            for line in f:
                extrinsics_inv += [float(value) for value in line.strip().split(' ') if value.strip() != '']
                continue
            pass
        extrinsics_inv = np.array(extrinsics_inv).reshape((4, 4))
        extrinsics = np.linalg.inv(extrinsics_inv)
        temp = extrinsics[1].copy()
        extrinsics[1] = extrinsics[2]
        extrinsics[2] = -temp

        segmentation = cv2.imread(segmentationPath, -1).astype(np.int32)
        #segmentation = segmentation[:, :, 2] * 256 * 256 + segmentation[:, :, 1] * 256 + segmentation[:, :, 0]

        planes = np.load(segmentationPath.replace('segmentation.png', 'planes.npy'))

        info = [image, planes, segmentation, depth, self.metadata]

        if False:
            print(planes)
            print(depth.min(), depth.max())
            cv2.imwrite('test/image.png', image)
            cv2.imwrite('test/depth_ori.png', drawDepthImage(depth))
            cv2.imwrite('test/segmentation.png', drawSegmentationImage(segmentation))
            # for index in range(segmentation.max() + 1):
            #     print(index, newPlanes[index])
            #     cv2.imwrite('test/mask_' + str(index) + '.png', (segmentation == index).astype(np.uint8) * 255)
            #     continue

            #planeDepths = calcPlaneDepths(planes, segmentation, 192, self.metadata)
            exit(1)

        return info

    def transformPlanes(self, transformation, planes):
        planeOffsets = np.linalg.norm(planes, axis=-1, keepdims=True)
        
        centers = planes
        centers = np.concatenate([centers, np.ones((planes.shape[0], 1))], axis=-1)
        newCenters = np.transpose(np.matmul(transformation, np.transpose(centers)))
        newCenters = newCenters[:, :3] / newCenters[:, 3:4]

        refPoints = planes - planes / np.maximum(planeOffsets, 1e-4)
        refPoints = np.concatenate([refPoints, np.ones((planes.shape[0], 1))], axis=-1)
        newRefPoints = np.transpose(np.matmul(transformation, np.transpose(refPoints)))
        newRefPoints = newRefPoints[:, :3] / newRefPoints[:, 3:4]

        planeNormals = newRefPoints - newCenters
        planeNormals /= np.linalg.norm(planeNormals, axis=-1, keepdims=True)
        planeOffsets = np.sum(newCenters * planeNormals, axis=-1, keepdims=True)
        newPlanes = planeNormals * planeOffsets
        return newPlanes
        
    def __getitem__(self, imageIndex):
        if self.loadCached:
            return self.getItemCached(imageIndex)
        
        imagePath = self.imagePaths[imageIndex]

        if self.scannetVersion == 1:
            segmentationPath = imagePath.replace('frames/', 'annotation/segmentation/').replace('color.jpg', 'segmentation.png')
            depthPath = imagePath.replace('color.jpg', 'depth.pgm')
            posePath = imagePath.replace('color.jpg', 'pose.txt')
        else:
            segmentationPath = imagePath.replace('frames/color/', 'annotation/segmentation/').replace('.jpg', '.png')
            depthPath = imagePath.replace('color', 'depth').replace('.jpg', '.png')
            posePath = imagePath.replace('color', 'pose').replace('.jpg', '.txt')
            pass
        
        image = cv2.imread(imagePath)
        depth = cv2.imread(depthPath, -1).astype(np.float32) / self.depthShift

        extrinsics_inv = []
        with open(posePath, 'r') as f:
            for line in f:
                extrinsics_inv += [float(value) for value in line.strip().split(' ') if value.strip() != '']
                continue
            pass
        extrinsics_inv = np.array(extrinsics_inv).reshape((4, 4))
        extrinsics = np.linalg.inv(extrinsics_inv)
        
        segmentation = cv2.imread(segmentationPath, -1).astype(np.int32)
        segmentation = segmentation[:, :, 2] * 256 * 256 + segmentation[:, :, 1] * 256 + segmentation[:, :, 0]
        
        segmentation = segmentation / 100 - 1
        segments, counts = np.unique(segmentation, return_counts=True)
        segmentList = zip(segments.tolist(), counts.tolist())
        segmentList = [segment for segment in segmentList if segment[0] not in [-1, 167771]]
        segmentList = sorted(segmentList, key=lambda x:-x[1])
        
        newPlanes = []
        newSegmentation = np.full(segmentation.shape, fill_value=-1, dtype=np.int32)
        for newIndex, (oriIndex, count) in enumerate(segmentList):
            if count < (segmentation.shape[0] * segmentation.shape[1]) * 0.02:
                continue
            newPlanes.append(self.planes[oriIndex])
            newSegmentation[segmentation == oriIndex] = newIndex
            continue

        newPlanes = np.array(newPlanes)

        temp = extrinsics[1].copy()
        extrinsics[1] = extrinsics[2]
        extrinsics[2] = -temp

        if len(newPlanes) > 0:
            newPlanes = self.transformPlanes(extrinsics, newPlanes)
            pass

        info = [image, newPlanes, newSegmentation, depth, self.metadata]

        if False:
            print(newPlanes)
            print(depth.min(), depth.max())
            cv2.imwrite('test/image.png', image)
            cv2.imwrite('test/depth_ori.png', drawDepthImage(depth))
            cv2.imwrite('test/segmentation.png', drawSegmentationImage(newSegmentation))
            for index in range(newSegmentation.max() + 1):
                print(index, newPlanes[index])
                cv2.imwrite('test/mask_' + str(index) + '.png', (newSegmentation == index).astype(np.uint8) * 255)
                continue
            #planeDepths = calcPlaneDepths(planes, segmentation, 192, self.metadata)
            exit(1)
        
        return info
