import numpy as np
import PIL.Image
import copy
import sys
import os
import cv2
import scipy.ndimage as ndimage
#import pydensecrf.densecrf as dcrf
#from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    #create_pairwise_gaussian, unary_from_softmax
from skimage import segmentation
#from skimage.future import graph

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layers import PlaneDepthLayer
#from layers import PlaneNormalLayer
from SegmentationRefinement import *

class ColorPalette:
    def __init__(self, numColors):
        np.random.seed(1)
        self.colorMap = np.random.randint(255, size = (numColors, 3))
        #self.colorMap[0] = 0
        return

    def getColorMap(self):
        return self.colorMap
    
    def getColor(self, index):
        if index >= colorMap.shape[0]:
            return np.random.randint(255, size = (3))
        else:
            return self.colorMap[index]
            pass

def writePointCloud(filename, pointCloud, color = [255, 255, 255]):
    with open(filename, 'w') as f:
        header = """ply
format ascii 1.0
element vertex """
        header += str(pointCloud.shape[0])
        header += """
property float x
property float y
property float z
property uchar red                                     { start of vertex color }
property uchar green
property uchar blue
end_header
"""
        f.write(header)
        for point in pointCloud:
            for value in point:
                f.write(str(value) + ' ')
                continue
            for value in color:
                f.write(str(value) + ' ')
                continue
            f.write('\n')
            continue
        f.close()
        pass
    return


def writeClusteringPointCloud(filename, pointCloud, clusters):
    with open(filename, 'w') as f:
        header = """ply
format ascii 1.0
element vertex """
        header += str(pointCloud.shape[0])
        header += """
property float x
property float y
property float z
property uchar red                                     { start of vertex color }
property uchar green
property uchar blue
end_header
"""
        colorMap = np.random.randint(255, size = clusters.shape)
        assignment = np.argmin(np.linalg.norm(pointCloud.reshape(-1, 1, 3).repeat(clusters.shape[0], 1)[:] - clusters, 2, 2), 1)
        f.write(header)
        for pointIndex, point in enumerate(pointCloud):
            for value in point:
                f.write(str(value) + ' ')
                continue
            color = colorMap[assignment[pointIndex]]
            for value in color:
                f.write(str(value) + ' ')
                continue
            f.write('\n')
            continue
        f.close()
        pass
    return


def writeNearestNeighbors(filename, pointCloudSource, pointCloudTarget):
    with open(filename, 'w') as f:
        header = """ply
format ascii 1.0
element vertex """
        header += str((pointCloudSource.shape[0] + pointCloudTarget.shape[0] + pointCloudSource.shape[0]) * 4)
        header += """
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face """
        header += str(pointCloudSource.shape[0] + pointCloudTarget.shape[0] + pointCloudSource.shape[0])
        header += """
property list uchar int vertex_index
end_header
"""
        f.write(header)
        
        sourceColor = [0, 255, 0]
        targetColor = [0, 0, 255]
        colorMap = np.random.randint(255, size = pointCloudSource.shape)
        
        # for pointIndex, point in enumerate(pointCloudSource):
        #     for value in point:
        #         f.write(str(value) + ' ')
        #         continue
        #     color = sourceColor
        #     for value in color:
        #         f.write(str(value) + ' ')
        #         continue
        #     f.write('\n')
        #     continue

        # for pointIndex, point in enumerate(pointCloudTarget):
        #     for value in point:
        #         f.write(str(value) + ' ')
        #         continue
        #     color = targetColor
        #     for value in color:
        #         f.write(str(value) + ' ')
        #         continue
        #     f.write('\n')
        #     continue        

        planeSize = 0.1
        for planeType, planes in enumerate([pointCloudSource, pointCloudTarget]):
            for planeIndex, plane in enumerate(planes):
                planeD = np.linalg.norm(plane)
                planeNormal = -plane / planeD

                maxNormalDim = np.argmax(np.abs(plane))
                allDims = [0, 1, 2]
                allDims.remove(maxNormalDim)
                dim_1, dim_2 = allDims
                for delta_1, delta_2 in [(-planeSize, -planeSize), (planeSize, -planeSize), (planeSize, planeSize), (-planeSize, planeSize)]:
                    point = copy.deepcopy(plane)
                    point[dim_1] += delta_1
                    point[dim_2] += delta_2
                    point[maxNormalDim] = (-planeD - planeNormal[dim_1] * point[dim_1] - planeNormal[dim_2] * point[dim_2]) / planeNormal[maxNormalDim]

                    for value in point:
                        f.write(str(value) + ' ')
                        continue
                    if planeType == 0:
                        color = sourceColor
                    else:
                        color = targetColor
                        pass
                    
                    for value in color:
                        f.write(str(value) + ' ')
                        continue
                    f.write('\n')
                    continue
                continue
            continue

        assignment = np.argmin(np.linalg.norm(pointCloudSource.reshape(-1, 1, 3).repeat(pointCloudTarget.shape[0], 1)[:] - pointCloudTarget, 2, 2), 1)

        planeSize = 0.01
        lineColor = [255, 0, 0]
        for planeIndex, planeSource in enumerate(pointCloudSource):
            planeD = np.linalg.norm(planeSource)
            planeNormal = -planeSource / planeD            

            maxNormalDim = np.argmax(np.abs(planeSource))
            allDims = [0, 1, 2]
            allDims.remove(maxNormalDim)
            dim_1, dim_2 = allDims
            minNormalDim = np.argmin(np.abs(planeSource))

            for delta in [-planeSize, planeSize]:
                point = copy.deepcopy(planeSource)
                point[minNormalDim] += delta
                point[maxNormalDim] = (-planeD - planeNormal[dim_1] * point[dim_1] - planeNormal[dim_2] * point[dim_2]) / planeNormal[maxNormalDim]
                for value in point:
                    f.write(str(value) + ' ')
                    continue
                color = lineColor
                for value in color:
                    f.write(str(value) + ' ')
                    continue
                f.write('\n')
                continue

            planeTarget = pointCloudTarget[assignment[planeIndex]]
            planeDTarget = np.linalg.norm(plane)
            planeNormalTarget = -plane / planeD
            planeD = np.linalg.norm(planeTarget)
            planeNormal = -planeTarget / planeD            

            for delta in [planeSize, -planeSize]:
                point = copy.deepcopy(planeTarget)
                point[minNormalDim] += delta
                point[maxNormalDim] = (-planeD - planeNormal[dim_1] * point[dim_1] - planeNormal[dim_2] * point[dim_2]) / planeNormal[maxNormalDim]
                for value in point:
                    f.write(str(value) + ' ')
                    continue
                color = lineColor
                for value in color:
                    f.write(str(value) + ' ')
                    continue
                f.write('\n')
                continue
            continue

        for index in xrange(pointCloudSource.shape[0] + pointCloudTarget.shape[0] + pointCloudSource.shape[0]):
            planeIndex = index * 4
            f.write('4 ' + str(planeIndex + 0) + ' ' + str(planeIndex + 1) + ' ' + str(planeIndex + 2) + ' ' + str(planeIndex + 3) + '\n')
            continue

        # for pointIndexSource, point in enumerate(pointCloudSource):
    #     pointIndexTarget = assignment[pointIndexSource]
#     f.write(str(pointIndexSource) + ' ' + str(pointIndexTarget + pointCloudSource.shape[0]) + ' ')
        #     color = colorMap[pointIndexSource]
    #     for value in color:
#         f.write(str(value) + ' ')
        #         continue
    #     f.write('\n')
#     continue


        f.close()
        pass
    return


def evaluatePlanes(planes, filename = None, depths = None, normals = None, invalidMask = None, outputFolder = None, outputIndex = 0, colorMap = None):
    if filename != None:
        if 'mlt' not in filename:
            filename = filename.replace('color', 'mlt')
            pass
        normalFilename = filename.replace('mlt', 'norm_camera')
        normals = np.array(PIL.Image.open(normalFilename)).astype(np.float) / 255 * 2 - 1
        norm = np.linalg.norm(normals, 2, 2)
        for c in xrange(3):
            normals[:, :, c] /= norm
            continue
        
        depthFilename = filename.replace('mlt', 'depth')
        depths = np.array(PIL.Image.open(depthFilename)).astype(np.float) / 1000
        # if len(depths.shape) == 3:
        #     depths = depths.mean(2)
        #     pass
        maskFilename = filename.replace('mlt', 'valid')    
        invalidMask = np.array(PIL.Image.open(maskFilename))
        invalidMask = invalidMask < 128
        invalidMask += depths > 10
        pass

    height = normals.shape[0]
    width = normals.shape[1]
    focalLength = 517.97
    urange = np.arange(width).reshape(1, -1).repeat(height, 0) - width * 0.5
    vrange = np.arange(height).reshape(-1, 1).repeat(width, 1) - height * 0.5
    ranges = np.array([urange / focalLength, np.ones(urange.shape), -vrange / focalLength]).transpose([1, 2, 0])
    
    X = depths / focalLength * urange
    Y = depths
    Z = -depths / focalLength * vrange
    d = -(normals[:, :, 0] * X + normals[:, :, 1] * Y + normals[:, :, 2] * Z)    


    normalDotThreshold = np.cos(np.deg2rad(30))
    distanceThreshold = 50000
    
    reconstructedNormals = np.zeros(normals.shape)
    reconstructedDepths = np.zeros(depths.shape)
    segmentationImage = np.zeros((height, width, 3))
    distanceMap = np.ones((height, width)) * distanceThreshold
    occupancyMask = np.zeros((height, width)).astype(np.bool)
    segmentationTest = np.zeros((height, width))
    y = 297
    x = 540
    for planeIndex, plane in enumerate(planes):
        planeD = np.linalg.norm(plane)
        planeNormal = -plane / planeD

        normalXYZ = np.dot(ranges, planeNormal)
        normalXYZ = np.reciprocal(normalXYZ)
        planeY = -normalXYZ * planeD

        distance = np.abs(planeNormal[0] * X + planeNormal[1] * Y + planeNormal[2] * Z + planeD) / np.abs(np.dot(normals, planeNormal))
        #distance = np.abs(planeY - depths)
        
        mask = (distance < distanceMap) * (planeY > 0) * (np.abs(np.dot(normals, planeNormal)) > normalDotThreshold) * (np.abs(planeY - depths) < 0.5)
        occupancyMask += mask
        
        reconstructedNormals[mask] = planeNormal
        
        
        #if planeNormal[2] > 0.9:
        #print(planeD)
        #print(planeNormal)
        # minDepth = depths.min()
        # maxDepth = depths.max()
        # print(depths[300][300])
        # print(planeY[300][300])
        # print(depths[350][350])
        # print(planeY[350][350])
        # PIL.Image.fromarray((np.maximum(np.minimum((planeY - minDepth) / (maxDepth - minDepth), 1), 0) * 255).astype(np.uint8)).save(outputFolder + '/plane.png')
        # exit(1)
        #pass
        reconstructedDepths[mask] = planeY[mask]
        if colorMap != None and planeIndex in colorMap:
            segmentationImage[mask] = colorMap[planeIndex]
        else:
            segmentationImage[mask] = np.random.randint(255, size=(3,))
            pass
        distanceMap[mask] = distance[mask]
        segmentationTest[mask] = planeIndex + 1
        #print((planeIndex, planeY[y][x], distance[y][x], np.abs(np.dot(normals, planeNormal))[y][x]))
        continue

    # print(distanceMap.mean())
# print(distanceMap.max())
    # print(np.abs(reconstructedDepths - depths)[occupancyMask].max())
# print(pow(reconstructedDepths - depths, 2)[True - invalidMask].mean())
    # exit(1)

    # planeIndex = segmentationTest[y][x]
# print(normals[y][x])
    # plane = planes[int(planeIndex)]
# planeD = np.linalg.norm(plane)
    # planeNormal = -plane / planeD
# print((planeNormal, planeD))
    # print(depths[y][x])
# print(reconstructedDepths[y][x])
    # print(segmentationTest[y][x])

    if outputFolder != None:
        depths[invalidMask] = 0
        normals[invalidMask] = 0
        reconstructedDepths[invalidMask] = 0
        reconstructedNormals[invalidMask] = 0
        minDepth = depths.min()
        maxDepth = depths.max()
        #print(minDepth)
        #print(maxDepth)
        PIL.Image.fromarray(((depths - minDepth) / (maxDepth - minDepth) * 255).astype(np.uint8)).save(outputFolder + '/' + str(outputIndex) + '_depth.png')
        PIL.Image.fromarray((np.maximum(np.minimum((reconstructedDepths - minDepth) / (maxDepth - minDepth), 1), 0) * 255).astype(np.uint8)).save(outputFolder + '/' + str(outputIndex) + '_depth_reconstructed.png')
        #PIL.Image.fromarray((np.maximum(np.minimum((reconstructedDepths - depths) / (distanceThreshold), 1), 0) * 255).astype(np.uint8)).save(outputFolder + '/depth_' + str(outputIndex) + '_diff.png')
        PIL.Image.fromarray(((normals + 1) / 2 * 255).astype(np.uint8)).save(outputFolder + '/' + str(outputIndex) + '_normal_.png')
        PIL.Image.fromarray(((reconstructedNormals + 1) / 2 * 255).astype(np.uint8)).save(outputFolder + '/' + str(outputIndex) + '_normal_reconstructed.png')
        PIL.Image.fromarray(segmentationImage.astype(np.uint8)).save(outputFolder + '/' + str(outputIndex) + '_plane_segmentation.png')
        #depthImage = ((depths - minDepth) / (maxDepth - minDepth) * 255).astype(np.uint8)
        #PIL.Image.fromarray((invalidMask * 255).astype(np.uint8)).save(outputFolder + '/mask.png')
        #exit(1)
    else:
        occupancy = (occupancyMask > 0.5).astype(np.float32).sum() / (1 - invalidMask).sum()
        invalidMask += np.invert(occupancyMask)
        #PIL.Image.fromarray(invalidMask.astype(np.uint8) * 255).save(outputFolder + '/mask.png')
        reconstructedDepths = np.maximum(np.minimum(reconstructedDepths, 10), 0)
        depthError = pow(reconstructedDepths - depths, 2)[np.invert(invalidMask)].mean()
        #depthError = distanceMap.mean()
        normalError = np.arccos(np.maximum(np.minimum(np.sum(reconstructedNormals * normals, 2), 1), -1))[np.invert(invalidMask)].mean()
        #normalError = pow(np.linalg.norm(reconstructedNormals - normals, 2, 2), 2)[True - invalidMask].mean()
        #print((depthError, normalError, occupancy))
        # print(depths.max())
        # print(depths.min())
        # print(reconstructedDepths.max())
        # print(reconstructedDepths.min())
        # print(occupancy)
        # exit(1)
        
        #reconstructedDepths[np.invert(occupancyMask)] = depths[np.invert(occupancyMask)]
        return depthError, normalError, occupancy, segmentationTest, reconstructedDepths, occupancyMask
    return


def evaluatePlanesSeparately(planes, filename, outputFolder = None, outputIndex = 0):
    if 'mlt' not in filename:
        filename = filename.replace('color', 'mlt')
        pass
    colorImage = np.array(PIL.Image.open(filename))
    normalFilename = filename.replace('mlt', 'norm_camera')
    normals = np.array(PIL.Image.open(normalFilename)).astype(np.float) / 255 * 2 - 1
    height = normals.shape[0]
    width = normals.shape[1]
    norm = np.linalg.norm(normals, 2, 2)
    for c in xrange(3):
        normals[:, :, c] /= norm
        continue

    
    depthFilename = filename.replace('mlt', 'depth')
    depths = np.array(PIL.Image.open(depthFilename)).astype(np.float) / 1000
    # if len(depths.shape) == 3:
    #     depths = depths.mean(2)
    #     pass
    
    focalLength = 517.97
    urange = np.arange(width).reshape(1, -1).repeat(height, 0) - width * 0.5
    vrange = np.arange(height).reshape(-1, 1).repeat(width, 1) - height * 0.5
    ranges = np.array([urange / focalLength, np.ones(urange.shape), -vrange / focalLength]).transpose([1, 2, 0])
    X = depths / focalLength * urange
    Y = depths
    Z = -depths / focalLength * vrange
    d = -(normals[:, :, 0] * X + normals[:, :, 1] * Y + normals[:, :, 2] * Z)

    
    maskFilename = filename.replace('mlt', 'valid')    
    invalidMask = np.array(PIL.Image.open(maskFilename))
    # if len(invalidMask.shape) == 3:
    #     invalidMask = invalidMask.mean(2)
    #     pass
    invalidMask = invalidMask < 128
    invalidMask += depths > 10


    normalDotThreshold = np.cos(np.deg2rad(15))
    distanceThreshold = 0.15
    colorPalette = ColorPalette(len(planes))
    for planeIndex, plane in enumerate(planes):
        planeD = np.linalg.norm(plane)
        planeNormal = -plane / planeD

        distance = np.abs(planeNormal[0] * X + planeNormal[1] * Y + planeNormal[2] * Z + planeD)

        normalXYZ = np.dot(ranges, planeNormal)
        normalXYZ = np.reciprocal(normalXYZ)
        planeY = -normalXYZ * planeD
        
        mask = (planeY > 0) * (np.abs(np.dot(normals, planeNormal)) > normalDotThreshold) * (distance < distanceThreshold)

        maxDepth = 10
        minDepth = 0
        #PIL.Image.fromarray((np.minimum(np.maximum((planeY - minDepth) / (maxDepth - minDepth), 0), 1) * 255).astype(np.uint8)).save(outputFolder + '/plane_depth_' + str(planeIndex) + '.png')
        #PIL.Image.fromarray(((planeNormal.reshape(1, 1, 3).repeat(height, 0).repeat(width, 1) + 1) / 2 * 255).astype(np.uint8)).save(outputFolder + '/plane_normal_' + str(planeIndex) + '.png')
        planeImage = colorImage * 0.3
        planeImage[mask] += colorPalette.getColor(planeIndex) * 0.7
        PIL.Image.fromarray(planeImage.astype(np.uint8)).save(outputFolder + '/plane_mask_' + str(planeIndex) + '_' + str(outputIndex) + '.png')
        #PIL.Image.fromarray(mask.astype(np.uint8) * 255).save(outputFolder + '/mask_' + str(planeIndex) + '.png')
        continue
    return

def residual2Planes(residualPlanes, predefinedPlanes):
    numClusters = predefinedPlanes.shape[0]
    planes = []
    for residualPlane in residualPlanes:
        gridIndex = int(residualPlane[0]) / numClusters
        planeIndex = int(residualPlane[0]) % numClusters
        planes.append(predefinedPlanes[planeIndex] + residualPlane[1:])
        continue
    return planes

def residual2PlanesGlobal(residualPlanes, predefinedPlanes):
    numClusters = predefinedPlanes.shape[0]
    planes = []
    for residualPlane in residualPlanes:
        planeIndex = int(residualPlane[0])
        planes.append(predefinedPlanes[planeIndex] + residualPlane[1:])
        continue
    return planes


def getPlaneInfo(planes):
    imageWidth = 640
    imageHeight = 480
    focalLength = 517.97
    urange = np.arange(imageWidth).reshape(1, -1).repeat(imageHeight, 0) - imageWidth * 0.5
    vrange = np.arange(imageHeight).reshape(-1, 1).repeat(imageWidth, 1) - imageHeight * 0.5
    ranges = np.array([urange / focalLength, np.ones(urange.shape), -vrange / focalLength]).transpose([1, 2, 0])
    
    planeDepths = PlaneDepthLayer(planes, ranges)
    planeNormals = PlaneNormalLayer(planes, ranges)
    return planeDepths, planeNormals

def getProbability(image, segmentation):
    width = image.shape[1]
    height = image.shape[0]
    numPlanes = segmentation.shape[0]
    probabilities = np.exp(segmentation)
    probabilities = probabilities / probabilities.sum(0)
    # The input should be the negative of the logarithm of probability values
    # Look up the definition of the softmax_to_unary for more information
    unary = unary_from_softmax(probabilities)

    # The inputs should be C-continious -- we are using Cython wrapper
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF(height * width, numPlanes)
    
    d.setUnaryEnergy(unary)

    # This potential penalizes small pieces of segmentation that are
    # spatially isolated -- enforces more spatially consistent segmentations
    # feats = create_pairwise_gaussian(sdims=(10, 10), shape=(height, width))
    # d.addPairwiseEnergy(feats, compat=300,
    #                                         kernel=dcrf.DIAG_KERNEL,
    #                                         normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features --
    # because the segmentation that we get from CNN are too coarse
    # and we can use local color features to refine them
        
    feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
                                      img=image, chdim=2)
    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
        
    Q = d.inference(50)

    inds = np.argmax(Q, axis=0).reshape((height, width))
    probabilities = np.zeros((height * width, numPlanes))
    probabilities[np.arange(height * width), inds.reshape(-1)] = 1
    probabilities = probabilities.reshape([height, width, -1])
    #print(res.shape)
    return probabilities

def getProbabilityMax(segmentation):
    width = segmentation.shape[2]
    height = segmentation.shape[1]
    numPlanes = segmentation.shape[0]
    inds = np.argmax(segmentation.reshape([-1, height * width]), axis=0)
    probabilities = np.zeros((height * width, numPlanes))
    probabilities[np.arange(height * width), inds] = 1
    probabilities = probabilities.reshape([height, width, -1])
    return probabilities

def evaluateNormal(predNormal, gtSegmentations, numPlanes, gtPlanes):
    var = 0
    numImages = predNormal.shape[0]
    totalNumPlanes = 0
    normal = np.linalg.norm(predNormal, axis=-1)
    for imageIndex in xrange(numImages):
        for planeIndex in xrange(numPlanes[imageIndex]):
            mask = gtSegmentations[imageIndex, :, :, planeIndex]
            if mask.sum() == 0:
                continue
            normals = predNormal[imageIndex][mask.astype(np.bool)]
            averageNormal = np.mean(normals, axis=0, keepdims=True)
            averageNormal /= np.linalg.norm(averageNormal)
            degrees = np.rad2deg(np.arccos(np.sum(normals * averageNormal, axis=-1)))
            var += degrees.mean()
            totalNumPlanes += 1
            continue
        continue
    var /= totalNumPlanes
    print(var)
    return var
        
def evaluateDepths(predDepths, gtDepths, validMasks, planeMasks=True, printInfo=True):
    masks = np.logical_and(np.logical_and(validMasks, planeMasks), gtDepths > 1e-4)
    
    numPixels = float(masks.sum())
    
    rmse = np.sqrt((pow(predDepths - gtDepths, 2) * masks).sum() / numPixels)
    #rmse_log = np.sqrt((pow(np.log(1 + np.abs(predDepths - gtDepths)), 2) * masks).sum() / numPixels)
    log10 = (np.abs(np.log10(np.maximum(predDepths, 1e-4)) - np.log10(np.maximum(gtDepths, 1e-4))) * masks).sum() / numPixels
    rel = (np.abs(predDepths - gtDepths) / np.maximum(gtDepths, 1e-4) * masks).sum() / numPixels
    deltas = np.maximum(predDepths / np.maximum(gtDepths, 1e-4), gtDepths / np.maximum(predDepths, 1e-4)) + (1 - masks.astype(np.float32)) * 10000
    accuracy_1 = (deltas < 1.25).sum() / numPixels
    accuracy_2 = (deltas < pow(1.25, 2)).sum() / numPixels
    accuracy_3 = (deltas < pow(1.25, 3)).sum() / numPixels
    recall = float(masks.sum()) / validMasks.sum()
    #print((rms, recall))
    if printInfo:
        print(('evaluate', rmse, log10, rel, accuracy_1, accuracy_2, accuracy_3, recall))
        pass
    return rmse, accuracy_1
    #return rel, log10, rms, accuracy_1, accuracy_2, accuracy_3, recall

def drawDepthImage(depth):
    #return cv2.applyColorMap(np.clip(depth / 10 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    return 255 - np.clip(depth / 5 * 255, 0, 255).astype(np.uint8)

def drawDepthImageOverlay(image, depth):
    #return cv2.applyColorMap(np.clip(depth / 10 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    depth = np.clip(depth / min(np.max(depth), 10) * 255, 0, 255).astype(np.uint8)
    imageOverlay = np.stack([image[:, :, 1], depth, image[:, :, 2]], axis=2)
    return imageOverlay 
    
def drawNormalImage(normal):
    return ((normal + 1) / 2 * 255).astype(np.uint8)
    
def drawSegmentationImage(segmentations, randomColor=None, numColors=22, blackIndex=-1):
    if segmentations.ndim == 2:
        numColors = max(numColors, segmentations.max() + 2)
    else:
        numColors = max(numColors, segmentations.shape[2] + 2)
        pass
    randomColor = ColorPalette(numColors).getColorMap()
    if blackIndex >= 0:
        randomColor[blackIndex] = 0
        pass
    width = segmentations.shape[1]
    height = segmentations.shape[0]
    if segmentations.ndim == 3:
        #segmentation = (np.argmax(segmentations, 2) + 1) * (np.max(segmentations, 2) > 0.5)
        segmentation = np.argmax(segmentations, 2)
    else:
        segmentation = segmentations
        pass
    segmentation = segmentation.astype(np.int)
    return randomColor[segmentation.reshape(-1)].reshape((height, width, 3))

def drawMaskImage(mask):
    return (np.clip(mask * 255, 0, 255)).astype(np.uint8)

def drawDiffImage(values_1, values_2, threshold):
    #return cv2.applyColorMap(np.clip(np.abs(values_1 - values_2) / threshold * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    return np.clip(np.abs(values_1 - values_2) / threshold * 255, 0, 255).astype(np.uint8)


def getSuperpixels(depth, normal, width, height, numPlanes=50, numGlobalPlanes = 10):
    depth = np.expand_dims(depth, -1)

    urange = (np.arange(width, dtype=np.float32) / (width + 1) - 0.5) / focalLength * 641
    urange = np.tile(np.reshape(urange, [1, -1]), [height, 1])
    vrange = (np.arange(height, dtype=np.float32) / (height + 1) - 0.5) / focalLength * 481
    vrange = np.tile(np.reshape(vrange, [-1, 1]), [1, width])
    
    ranges = np.stack([urange, np.ones([height, width]), -vrange], axis=2)
    #ranges = np.expand_dims(ranges, 0)

    planeImage = np.sum(normal * ranges, axis=2, keepdims=True) * depth * normal
    planeImage = planeImage / 10 * 1000

    superpixels = segmentation.slic(planeImage, compactness=30, n_segments=400)
    g = graph.rag_mean_color(planeImage, superpixels, mode='similarity')
    planeSegmentation = graph.cut_normalized(superpixels, g)
    return planeSegmentation, superpixels


def calcPlaneDepths(planes, width, height, info):
    urange = np.arange(width, dtype=np.float32).reshape(1, -1).repeat(height, 0) / (width + 1) * (info[16] + 1) - info[2]
    vrange = np.arange(height, dtype=np.float32).reshape(-1, 1).repeat(width, 1) / (height + 1) * (info[17] + 1) - info[6]
    ranges = np.array([urange / info[0], np.ones(urange.shape), -vrange / info[5]]).transpose([1, 2, 0])
    planeDepths = PlaneDepthLayer(planes, ranges)
    return planeDepths

def calcPlaneNormals(planes, width, height):
    planeNormals = -planes / np.maximum(np.linalg.norm(planes, axis=-1, keepdims=True), 1e-4)
    return np.tile(planeNormals.reshape([1, 1, -1, 3]), [height, width, 1, 1])


def writePLYFile(folder, index, image, depth, segmentation, boundaries):
    imageFilename = str(index) + '_image.png'
    cv2.imwrite(folder + '/' + imageFilename, image)

    focalLength = 517.97
    width = image.shape[1]
    height = image.shape[0]
        
    faces = []
    minDepthDiff = 0.15
    maxDepthDiff = 0.3
    occlusionBoundary = boundaries[:, :, 1]
        
    for y in xrange(height - 1):
        for x in xrange(width - 1):
            segmentIndex = segmentation[y][x]
            if segmentIndex == -1:
                continue
            if segmentIndex == 0:
                continue
            depths = [depth[y][x], depth[y + 1][x], depth[y + 1][x + 1]]
            if segmentIndex > 0 or (max([occlusionBoundary[y][x], occlusionBoundary[y + 1][x], occlusionBoundary[y + 1][x + 1]]) < 0.5 and (max(depths) - min(depths)) < maxDepthDiff):
                if segmentation[y + 1][x] == segmentIndex and segmentation[y + 1][x + 1] == segmentIndex:
                    if min(depths) > 0 and max(depths) < 10:
                        faces.append((x, y, x, y + 1, x + 1, y + 1))
                        pass
                    pass
                elif max(depths) - min(depths) < minDepthDiff:
                    faces.append((x, y, x, y + 1, x + 1, y + 1))
                    pass
                pass

                        
            depths = [depth[y][x], depth[y][x + 1], depth[y + 1][x + 1]]                        
            if segmentIndex > 0 or (max([occlusionBoundary[y][x], occlusionBoundary[y][x + 1], occlusionBoundary[y + 1][x + 1]]) < 0.5 and (max(depths) - min(depths)) < maxDepthDiff):
                if segmentation[y][x + 1] == segmentIndex and segmentation[y + 1][x + 1] == segmentIndex:
                    if min(depths) > 0 and max(depths) < 10:
                        faces.append((x, y, x + 1, y + 1, x + 1, y))
                        pass
                    pass
                elif max(depths) - min(depths) < minDepthDiff:
                    faces.append((x, y, x + 1, y + 1, x + 1, y))
                    pass
                pass
            continue
        continue

    #print(len(faces))
    with open(folder + '/' + str(index) + '_model.ply', 'w') as f:
        header = """ply
format ascii 1.0
comment VCGLIB generated
comment TextureFile """
        header += imageFilename
        header += """
element vertex """
        header += str(width * height)
        header += """
property float x
property float y
property float z
element face """
        header += str(len(faces))
        header += """
property list uchar int vertex_indices
property list uchar float texcoord
end_header
"""
        f.write(header)
        for y in xrange(height):
            for x in xrange(width):
                segmentIndex = segmentation[y][x]
                if segmentIndex == -1:
                    f.write("0.0 0.0 0.0\n")
                    continue
                Y = depth[y][x]
                X = Y / focalLength * (x - width / 2) / width * 640
                Z = -Y / focalLength * (y - height / 2) / height * 480
                f.write(str(X) + ' ' +    str(Z) + ' ' + str(-Y) + '\n')
                continue
            continue


        for face in faces:
            f.write('3 ')
            for c in xrange(3):
                f.write(str(face[c * 2 + 1] * width + face[c * 2]) + ' ')
                continue
            f.write('6 ')                     
            for c in xrange(3):
                f.write(str(float(face[c * 2]) / width) + ' ' + str(1 - float(face[c * 2 + 1]) / height) + ' ')
                continue
            f.write('\n')
            continue
        f.close()
        pass
	return  

def writeHTML(folder, numImages):
    from html import HTML

    h = HTML('html')
    h.p('Results')
    h.br()
    #suffixes = ['', '_crf_1']
    #folders = ['test_all_resnet_v2' + suffix + '/' for suffix in suffixes]
    for index in xrange(numImages):
        t = h.table(border='1')
        r_inp = t.tr()
        r_inp.td('input')
        r_inp.td().img(src=str(index) + '_image.png')
        r_inp.td().img(src='one.png')
        r_inp.td().img(src='one.png')
        #r_inp.td().img(src='one.png')
        r_inp.td().img(src=str(index) + '_model.png')

        r_gt = t.tr()
        r_gt.td('gt')
        r_gt.td().img(src=str(index) + '_segmentation_gt.png')
        r_gt.td().img(src=str(index) + '_depth.png')
        r_gt.td().img(src='one.png')
        r_gt.td().img(src=str(index) + '_normal.png')
                
        #r_gt.td().img(src=folders[0] + str(index) + '_depth_gt.png')
        #r_gt.td().img(src=folders[0] + '_depth_gt_diff.png')
        #r_gt.td().img(src=folders[0] + str(index) + '_normal_gt.png')

        r_pred = t.tr()
        r_pred.td('pred')
        r_pred.td().img(src=str(index) + '_segmentation_pred.png')
        r_pred.td().img(src=str(index) + '_depth_pred.png')
        r_pred.td().img(src=str(index) + '_depth_pred_diff.png')
        r_pred.td().img(src=str(index) + '_normal_pred.png')

        h.br()
        continue

    html_file = open(folder + '/index.html', 'w')
    html_file.write(str(h))
    html_file.close()
    return


def writeHTMLRGBD(filename, numImages=10):
    from html import HTML

    #0.227 0.194 0.163 0.157 0.100
    #0.196 0.150 0.143 0.488 0.082
    h = HTML('html')
    h.p('Results')
    h.br()
    path = ''
    folders = ['pred', 'local_02', 'local_05', 'local_07', 'local_10', 'local_12']
    second_folders = ['pred_local_02', 'pred_local_05', 'pred_local_07', 'pred_local_10', 'pred_local_12']
    for index in xrange(numImages):
        firstFolder = path + folders[0]
        t = h.table(border='1')
        r_inp = t.tr()
        r_inp.td('input')
        r_inp.td().img(src=firstFolder + '/' + str(index) + '_image.png')
        r_inp.td().img(src=firstFolder + '/' + str(index) + '_depth.png')

        r = t.tr()
        r.td('PlaneNet prediction')
        r.td().img(src=firstFolder + '/' + str(index) + '_segmentation_pred.png')
        r.td().img(src=firstFolder + '/' + str(index) + '_depth_pred.png')

        r = t.tr()
        r.td('segmentation')
        for folder in folders[1:6]:
            r.td().img(src=folder + '/' + str(index) + '_segmentation_pred.png')
            continue

        r = t.tr()
        r.td('depth')
        for folder in folders[1:6]:
            folder = path + folder
            r.td().img(src=folder + '/' + str(index) + '_depth_pred.png')
            continue

        r = t.tr()
        r.td('pixelwise prediction')
        r.td().img(src=path + second_folders[0] + '/' + str(index) + '_depth.png')

        r = t.tr()
        r.td('segmentation')
        for folder in second_folders[0:5]:
            r.td().img(src=folder + '/' + str(index) + '_segmentation_pred.png')
            continue

        r = t.tr()
        r.td('depth')
        for folder in second_folders[0:5]:
            folder = path + folder
            r.td().img(src=folder + '/' + str(index) + '_depth_pred.png')
            continue
        
        h.br()
        continue

    html_file = open(filename, 'w')
    html_file.write(str(h))
    html_file.close()
    return

def writeHTMLPlane(filename, numImages=10):
    from html import HTML

    #0.227 0.194 0.163 0.157 0.100
    #0.196 0.150 0.143 0.488 0.082
    h = HTML('html')
    h.p('Results')
    h.br()
    path = ''
    folders = ['PlaneNet_hybrid', 'PlaneNet', 'GT_RANSAC', 'pixelwise_RANSAC']
    for index in xrange(numImages):
        firstFolder = path + folders[0]
        t = h.table(border='1')
        r_inp = t.tr()
        r_inp.td('input')
        r_inp.td().img(src=firstFolder + '/' + str(index) + '_image.png')
        r_inp.td().img(src=firstFolder + '/' + str(index) + '_depth.png')

        # r = t.tr()
        # r.td('PlaneNet prediction')
        # r.td().img(src=firstFolder + '/' + str(index) + '_segmentation_pred.png')
        # r.td().img(src=firstFolder + '/' + str(index) + '_depth_pred.png')

        r = t.tr()
        r.td('segmentation')
        for folder in folders[0:6]:
            r.td().img(src=folder + '/' + str(index) + '_segmentation_pred.png')
            continue

        r = t.tr()
        r.td('depth')
        for folder in folders[0:6]:
            folder = path + folder
            r.td().img(src=folder + '/' + str(index) + '_depth_pred.png')
            continue
        
        h.br()
        continue

    html_file = open(filename, 'w')
    html_file.write(str(h))
    html_file.close()
    return

def getNYURGBDCamera():
    camera = {}
    camera['fx'] = 5.1885790117450188e+02
    camera['fy'] = 5.1946961112127485e+02
    camera['cx'] = 3.2558244941119034e+02 - 40
    camera['cy'] = 2.5373616633400465e+02 - 44
    camera['width'] = 561
    camera['height'] = 427
    camera['depth_shift'] = 1000    
    return camera

def getSUNCGCamera():
    camera = {}
    camera['fx'] = 517.97
    camera['fy'] = 517.97
    camera['cx'] = 320
    camera['cy'] = 240
    camera['width'] = 640
    camera['height'] = 480
    camera['depth_shift'] = 1000    
    return camera

def get3DCamera():
    camera = {}
    camera['fx'] = 1075
    camera['fy'] = 1075
    camera['cx'] = 637
    camera['cy'] = 508
    camera['width'] = 1280
    camera['height'] = 1024
    camera['depth_shift'] = 4000    
    return camera

def getCameraFromInfo(info):
    camera = {}
    camera['fx'] = info[0]
    camera['fy'] = info[5]
    camera['cx'] = info[2]
    camera['cy'] = info[6]
    camera['width'] = info[16]
    camera['height'] = info[17]
    camera['depth_shift'] = info[18]    
    return camera

def fitPlane(points):
    if points.shape[0] == points.shape[1]:
        return np.linalg.solve(points, np.ones(points.shape[0]))
    else:
        return np.linalg.lstsq(points, np.ones(points.shape[0]))[0]

def fitPlanes(depth, info, numPlanes=50, planeAreaThreshold=3*4, numIterations=100, distanceThreshold=0.05, local=-1):
    camera = getCameraFromInfo(info)
    width = depth.shape[1]
    height = depth.shape[0]

    #camera = getNYURGBDCamera()
    #camera = getSUNCGCamera()

    urange = (np.arange(width, dtype=np.float32) / width * camera['width'] - camera['cx']) / camera['fx']
    urange = urange.reshape(1, -1).repeat(height, 0)
    vrange = (np.arange(height, dtype=np.float32) / height * camera['height'] - camera['cy']) / camera['fy']
    vrange = vrange.reshape(-1, 1).repeat(width, 1)
    X = depth * urange
    Y = depth
    Z = -depth * vrange

    # print(depth[:10, :10])
    # print(X[:10, :10])
    # print(Z[:10, :10])
    # print(urange[:10, :10])
    # print(vrange[:10, :10])
    # exit(1)
    XYZ = np.stack([X, Y, Z], axis=2).reshape(-1, 3)
    XYZ = XYZ[depth.reshape(-1) != 0]
    planes = []
    planePointsArray = []
    for planeIndex in xrange(numPlanes):
        maxNumInliers = planeAreaThreshold
        for iteration in xrange(numIterations):
            if local > 0:
                sampledPoint = XYZ[np.random.randint(XYZ.shape[0], size=(1))]
                sampledPoints = XYZ[np.linalg.norm(XYZ - sampledPoint, 2, 1) < local]
                if sampledPoints.shape[0] < 3:
                    continue
                elif sampledPoints.shape[0] > 3:
                    sampledPoints = sampledPoints[np.random.choice(np.arange(sampledPoints.shape[0]), size=(3))]
                    pass
            else:
                sampledPoints = XYZ[np.random.choice(np.arange(XYZ.shape[0]), size=(3))]
                pass
            
            try:
                plane = fitPlane(sampledPoints)
                pass
            except:
                continue
            diff = np.abs(np.matmul(XYZ, plane) - np.ones(XYZ.shape[0])) / np.linalg.norm(plane)
            numInliers = np.sum(diff < distanceThreshold)
            if numInliers > maxNumInliers:
                maxNumInliers = numInliers
                bestPlane = plane
                pass
            continue
        if maxNumInliers == planeAreaThreshold:
            break
        planes.append(bestPlane)

        diff = np.abs(np.matmul(XYZ, bestPlane) - np.ones(XYZ.shape[0])) / np.linalg.norm(bestPlane)
        inlierIndices = diff < distanceThreshold
        inliersPoints = XYZ[inlierIndices]
        planePointsArray.append(inliersPoints)
        XYZ = XYZ[np.logical_not(inlierIndices)]
        if XYZ.shape[0] < planeAreaThreshold:
            break
        continue

    planes = np.array(planes)
    if len(planes.shape) == 0:
        planes = np.zeros((numPlanes, 3))
        pass
    if len(planes.shape) == 1:
        if planes.shape[0] == 3:
            planes = np.expand_dims(planes, 0)
        else:
            planes = np.zeros((numPlanes, 3))
            pass
        pass
    
    planeSegmentation = np.ones(depth.shape) * numPlanes
    for planeIndex, planePoints in enumerate(planePointsArray):
        planeDepth = planePoints[:, 1]
        u = np.round((planePoints[:, 0] / planeDepth * camera['fx'] + camera['cx']) / camera['width'] * width).astype(np.int32)
        v = np.round((-planePoints[:, 2] / planeDepth * camera['fy'] + camera['cy']) / camera['height'] * height).astype(np.int32)
        planeSegmentation[v, u] = planeIndex
        continue


    planesD = 1 / np.linalg.norm(planes, 2, axis=1, keepdims=True)
    planes *= pow(planesD, 2)

    if planes.shape[0] < numPlanes:
        #print(planes.shape)
        planes = np.concatenate([planes, np.zeros((numPlanes - planes.shape[0], 3))], axis=0)
        pass

    planeDepths = calcPlaneDepths(planes, width, height, info)
    
    allDepths = np.concatenate([planeDepths, np.zeros((height, width, 1))], axis=2)
    depthPred = allDepths.reshape(-1, numPlanes + 1)[np.arange(width * height), planeSegmentation.astype(np.int32).reshape(-1)].reshape(height, width)

    return planes, planeSegmentation, depthPred


def fitPlanesSegmentation(depth, segmentation, info, numPlanes=50, numPlanesPerSegment=3, planeAreaThreshold=3*4, numIterations=100, distanceThreshold=0.05, local=-1):
    
    camera = getCameraFromInfo(info)
    width = depth.shape[1]
    height = depth.shape[0]

    urange = (np.arange(width, dtype=np.float32) / width * camera['width'] - camera['cx']) / camera['fx']
    urange = urange.reshape(1, -1).repeat(height, 0)
    vrange = (np.arange(height, dtype=np.float32) / height * camera['height'] - camera['cy']) / camera['fy']
    vrange = vrange.reshape(-1, 1).repeat(width, 1)
    X = depth * urange
    Y = depth
    Z = -depth * vrange

    allXYZ = np.stack([X, Y, Z], axis=2).reshape(-1, 3)

    planes = []
    planePointIndices = []

    for segmentIndex in xrange(segmentation.max()):
        segmentMask = np.logical_and(segmentation == segmentIndex, depth != 0)
        XYZ = allXYZ[segmentMask.reshape(-1)]
        if XYZ.shape[0] < planeAreaThreshold:
            continue
        for planeIndex in xrange(numPlanesPerSegment):
            maxNumInliers = planeAreaThreshold
            for iteration in xrange(numIterations):
                if local > 0:
                    sampledPoint = XYZ[np.random.randint(XYZ.shape[0], size=(1))]
                    sampledPoints = XYZ[np.linalg.norm(XYZ - sampledPoint, 2, 1) < local]

                    if sampledPoints.shape[0] < 3:
                        continue
                    elif sampledPoints.shape[0] > 3:
                        sampledPoints = sampledPoints[np.random.choice(np.arange(sampledPoints.shape[0]), size=(3))]
                        pass
                else:
                    sampledPoints = XYZ[np.random.choice(np.arange(XYZ.shape[0]), size=(3))]
                    pass
                
                try:
                    plane = fitPlane(sampledPoints)
                    pass
                except:
                    continue
                diff = np.abs(np.matmul(XYZ, plane) - np.ones(XYZ.shape[0])) / np.linalg.norm(plane)
                numInliers = np.sum(diff < distanceThreshold)
                if numInliers > maxNumInliers:
                    maxNumInliers = numInliers
                    bestPlane = plane
                    pass
                continue
            if maxNumInliers == planeAreaThreshold:
                break
            planes.append(bestPlane)

            diff = np.abs(np.matmul(XYZ, bestPlane) - np.ones(XYZ.shape[0])) / np.linalg.norm(bestPlane)
            inlierIndices = diff < distanceThreshold
            inliersPoints = XYZ[inlierIndices]
            planePointIndices.append(inliersPoints)
            XYZ = XYZ[np.logical_not(inlierIndices)]
            if XYZ.shape[0] < planeAreaThreshold:
                break
            continue
        continue

    if len(planes) > numPlanes:
        planeList = zip(planes, planePointIndices)
        planeList = sorted(planeList, key=lambda x:-len(x[1]))
        planeList = planeList[:numPlanes]
        planes, planePointIndices = zip(*planeList)
        pass

    
    
    planeSegmentation = np.ones(depth.shape) * numPlanes
    for planeIndex, planePoints in enumerate(planePointIndices):
        planeDepth = planePoints[:, 1]
        u = np.round((planePoints[:, 0] / planeDepth * camera['fx'] + camera['cx']) / camera['width'] * width).astype(np.int32)
        v = np.round((-planePoints[:, 2] / planeDepth * camera['fy'] + camera['cy']) / camera['height'] * height).astype(np.int32)
        planeSegmentation[v, u] = planeIndex
        continue

    planes = np.array(planes)    
    planesD = 1 / np.linalg.norm(planes, 2, 1, keepdims=True)
    planes *= pow(planesD, 2)
    if planes.shape[0] < numPlanes:
        planes = np.concatenate([planes, np.zeros((numPlanes - planes.shape[0], 3))], axis=0)
        pass

    planeDepths = calcPlaneDepths(planes, width, height, info)

    
    allDepths = np.concatenate([planeDepths, np.zeros((height, width, 1))], axis=2)
    depthPred = allDepths.reshape([height * width, numPlanes + 1])[np.arange(width * height), planeSegmentation.astype(np.int32).reshape(-1)].reshape(height, width)

    return planes, planeSegmentation, depthPred


def fitPlanesNYU(image, depth, normal, semantics, info, numPlanes=20, planeAreaThreshold=500, distanceThreshold=0.05, local=-1):
    camera = getCameraFromInfo(info)
    width = depth.shape[1]
    height = depth.shape[0]

    #camera = getNYURGBDCamera()
    #camera = getSUNCGCamera()

    urange = (np.arange(width, dtype=np.float32) / width * camera['width'] - camera['cx']) / camera['fx']
    urange = urange.reshape(1, -1).repeat(height, 0)
    vrange = (np.arange(height, dtype=np.float32) / height * camera['height'] - camera['cy']) / camera['fy']
    vrange = vrange.reshape(-1, 1).repeat(width, 1)
    X = depth * urange
    Y = depth
    Z = -depth * vrange

    # print(depth[:10, :10])
    # print(X[:10, :10])
    # print(Z[:10, :10])
    # print(urange[:10, :10])
    # print(vrange[:10, :10])
    # exit(1)
    XYZ = np.stack([X, Y, Z], axis=2).reshape(-1, 3)
    #XYZ = XYZ[depth.reshape(-1) != 0]
    planes = []
    planeMasks = []
    invalidDepthMask = depth < 1e-4
    for y in xrange(5, 10, height):
        for x in xrange(5, 10, width):
            if invalidDepthMask[y][x]:
                continue
            sampledPoint = XYZ[y * width + x]
            sampledPoints = XYZ[np.linalg.norm(XYZ - sampledPoint, 2, 1) < local]
            if sampledPoints.shape[0] < 3:
                continue
            sampledPoints = sampledPoints[np.random.choice(np.arange(sampledPoints.shape[0]), size=(3))]
            try:
                plane = fitPlane(sampledPoints)
                pass
            except:
                continue
            
            diff = np.abs(np.matmul(XYZ, plane) - np.ones(XYZ.shape[0])) / np.linalg.norm(plane)
            inlierIndices = diff < distanceThreshold
            if np.sum(inlierIndices) < planeAreaThreshold:
                continue
            
            planes.append(plane)
            planeMasks.append(inlierIndices.reshape((height, width)))
            continue
        continue
    
    planes = np.array(planes)

    planeList = zip(planes, planeMasks)
    planeList = sorted(planeList, key=lambda x:-x[1].sum())
    planes, planeMasks = zip(*planeList)

    availableMask = np.ones((height, width), np.bool)
    validPlanes = []
    validPlaneMasks = []
    
    for planeIndex, plane in enumerate(planes):
        planeMask = planeMasks[planeIndex]
        validMask = np.logical_and(planeMask, availableMask)
        if validMask.sum() < planeMask.sum() * 0.5:
            continue
        validPlanes.append(plane)
        validPlaneMasks.append(planeMask)
        availableMask -= planeMask
        continue
    planes = np.array(validPlanes)
    planesD = 1 / np.maximum(np.linalg.norm(planes, 2, 1, keepdims=True), 1e-4)
    planes *= pow(planesD, 2)
    
    planeMasks = np.stack(validPlaneMasks, axis=2)
    print('number of planes: ' + str(planes.shape[0]))
    
    planeSegmentation = getSegmentationsGraphCut(planes, image, depth, normal, info)

    planeSegmentation[planeSegmentation == planes.shape[0]] = numPlanes

    if planes.shape[0] < numPlanes:
        planes = np.concatenate([planes, np.zeros((numPlanes - planes.shape[0], 3))], axis=0)
        pass

    planeDepths = calcPlaneDepths(planes, width, height, info)
    
    allDepths = np.concatenate([planeDepths, np.expand_dims(depth, -1)], axis=2)
    depthPred = allDepths.reshape([height * width, numPlanes + 1])[np.arange(width * height), planeSegmentation.astype(np.int32).reshape(-1)].reshape(height, width)


    planeNormals = calcPlaneNormals(planes, width, height)
    allNormals = np.concatenate([np.expand_dims(normal, 2), planeNormals], axis=2)
    normalPred = allNormals.reshape(-1, numPlanes + 1, 3)[np.arange(width * height), planeSegmentation.reshape(-1)].reshape((height, width, 3))
    
    return planes, planeSegmentation, depthPred, normalPred


def fitPlanesPoints(points, segmentation, groupSegments, numPlanes=50, numPlanesPerSegment=3, numPlanesPerGroup=8, segmentRatio=0.1, planeAreaThreshold=100, numIterations=100, distanceThreshold=0.05, local=-1):
    allXYZ = points.reshape(-1, 3)

    
    # planeDiffThreshold = distanceThreshold    
    # planes = np.load('test/planes.pny.npy')
    # planePointIndices = np.load('test/plane_indices.pny.npy')
    # planeIndex_1 = 1
    # planeIndex_2 = 2
    # for planeIndex_1 in [1, ]:
    #     for planeIndex_2 in [2, 5, 7, 10]:
    #         plane_1 = planes[planeIndex_1]
    #         plane_2 = planes[planeIndex_2]
    #         points_1 = allXYZ[planePointIndices[planeIndex_1]]
    #         points_2 = allXYZ[planePointIndices[planeIndex_2]]

    #         diff_1 = np.abs(np.matmul(points_2, plane_1) - np.ones(points_2.shape[0])) / np.linalg.norm(plane_1)
    #         diff_2 = np.abs(np.matmul(points_1, plane_2) - np.ones(points_1.shape[0])) / np.linalg.norm(plane_2)
    #         print(np.sum(diff_1 < planeDiffThreshold), diff_1.shape[0])
    #         print(np.sum(diff_2 < planeDiffThreshold), diff_2.shape[0])
    #         print((diff_1.mean(), diff_2.mean()))
    #         # if np.sum(diff_1 < planeDiffThreshold) > diff_1.shape[0] * inlierThreshold or np.sum(diff_2 < planeDiffThreshold) > diff_2.shape[0] * inlierThreshold:
    #         #     planesDiff[planeIndex][otherPlaneIndex] = 1
    #         #     planesDiff[otherPlaneIndex][planeIndex] = 1
    #         #     pass

            
    #         # if min(diff_1.mean(), diff_2.mean()) < planeDiffThreshold:
    #         #     planesDiff[planeIndex][otherPlaneIndex] = 1
    #         #     planesDiff[otherPlaneIndex][planeIndex] = 1
    #         #     pass
            
    #         continue
    #     continue
    
    # print(planes / np.linalg.norm(planes, axis=1, keepdims=True))
    # exit(1)
    
    planes = []
    planePointIndices = []
    groupNumPlanes = []
    for groupIndex, group in enumerate(groupSegments):
        groupPlanes = []
        groupPlanePointIndices = []
        for segmentIndex in group:
            segmentMask = segmentation == segmentIndex
            # planes.append(np.ones(3))
            # planePointIndices.append(segmentMask.nonzero()[0])
            # continue

            XYZ = allXYZ[segmentMask.reshape(-1)]
            numPoints = XYZ.shape[0]

            if numPoints <= planeAreaThreshold:
                if numPoints > 0:
                    groupPlanes.append(np.random.random(3))
                    groupPlanePointIndices.append(segmentMask.nonzero()[0])
                    pass
                continue

            for planeIndex in xrange(numPlanesPerSegment):
                maxNumInliers = 0
                for iteration in xrange(numIterations):
                    if local > 0:
                        sampledPoint = XYZ[np.random.randint(XYZ.shape[0], size=(1))]
                        sampledPoints = XYZ[np.linalg.norm(XYZ - sampledPoint, axis=1) < local]

                        if sampledPoints.shape[0] < 3:
                            continue
                        elif sampledPoints.shape[0] > 3:
                            sampledPoints = sampledPoints[np.random.choice(np.arange(sampledPoints.shape[0]), size=(3))]
                            pass
                        pass
                    else:
                        sampledPoints = XYZ[np.random.choice(np.arange(XYZ.shape[0]), size=(3))]
                        pass
                
                    try:
                        plane = fitPlane(sampledPoints)
                        pass
                    except:
                        continue
                    diff = np.abs(np.matmul(XYZ, plane) - np.ones(XYZ.shape[0])) / np.linalg.norm(plane)
                    numInliers = np.sum(diff < distanceThreshold)
                    if numInliers > maxNumInliers:
                        maxNumInliers = numInliers
                        bestPlane = plane
                        pass
                    continue

                groupPlanes.append(bestPlane)
            
                diff = np.abs(np.matmul(XYZ, bestPlane) - np.ones(XYZ.shape[0])) / np.linalg.norm(bestPlane)
                inlierIndices = diff < distanceThreshold
                if inlierIndices.sum() > numPoints * (1 - segmentRatio):
                    inlierIndices = np.ones(diff.shape, dtype=np.bool)
                    pass
                pointIndices = segmentMask.nonzero()[0][inlierIndices]
                groupPlanePointIndices.append(pointIndices)
                segmentMask[pointIndices] = 0
                
                XYZ = XYZ[np.logical_not(inlierIndices)]
                if XYZ.shape[0] <= planeAreaThreshold:
                    if XYZ.shape[0] > 0:
                        groupPlanes.append(np.random.random(3))
                        groupPlanePointIndices.append(segmentMask.nonzero()[0])
                        pass
                    break
                continue
            continue
        if len(groupPlanes) == 0:
            continue
        
        # planeList = zip(groupPlanes, groupPlanePointIndices)
        # planeList = sorted(planeList, key=lambda x:-len(x[1]))
        # groupPlanes, groupPlanePointIndices = zip(*planeList)

        # groupMask = np.zeros(segmentation.shape, np.bool)
        # for segmentIndex in group:
        #     groupMask = np.logical_or(groupMask, segmentation == segmentIndex)
        #     continue

        # groupPlanePointIndices = []
        # for plane in groupPlanes:
        #     groupPoints = allXYZ[groupMask]
        #     diff = np.abs(np.matmul(groupPoints, plane) - np.ones(groupPoints.shape[0])) / np.linalg.norm(plane)
        #     inlierIndices = diff < distanceThreshold
        #     pointIndices = groupMask.nonzero()[0][inlierIndices]
        #     groupPlanePointIndices.append(pointIndices)
        #     groupMask[pointIndices] = 0
        #     continue

        # if len(groupPlanes) > numPlanesPerGroup:
        #     planeList = zip(groupPlanes, groupPlanePointIndices)
        #     planeList = sorted(planeList, key=lambda x:-len(x[1]))
        #     planeList = planeList[:numPlanesPerGroup]
        #     groupPlanes, groupPlanePointIndices = zip(*planeList)
        #     pass

        
        numPointsOri = 0
        for indices in groupPlanePointIndices:
            numPointsOri += len(indices)
            continue
        
        groupPlanes, groupPlanePointIndices = mergePlanes(points, groupPlanes, groupPlanePointIndices, planeDiffThreshold=distanceThreshold, planeAreaThreshold=planeAreaThreshold)
        if len(groupPlanes) > 1:
            groupPlanes, groupPlanePointIndices = mergePlanes(points, groupPlanes, groupPlanePointIndices, planeDiffThreshold=distanceThreshold, planeAreaThreshold=planeAreaThreshold)
            pass
        #groupPlanes, groupPlanePointIndices = mergePlanes(points, groupPlanes, groupPlanePointIndices, planeDiffThreshold=distanceThreshold)
        
        # if groupIndex == 14:
        #     groupPlanes, groupPlanePointIndices = mergePlanes(points, groupPlanes, groupPlanePointIndices, planeDiffThreshold=distanceThreshold)
        #     planesTest = np.array(groupPlanes)
        #     np.save('test/planes.npy', planesTest)
        #     np.save('test/plane_indices.npy', groupPlanePointIndices)
        #     print(planesTest / np.linalg.norm(planesTest, axis=1, keepdims=True))
        #     #exit(1)
        #     pass

        numPoints = 0
        for indices in groupPlanePointIndices:
            numPoints += len(indices)
            continue
        planes += groupPlanes
        planePointIndices += groupPlanePointIndices
        groupNumPlanes.append(len(groupPlanes))
        #planeSegmentation[groupPlaneSegmentation >= 0] = groupPlaneSegmentation[groupPlaneSegmentation >= 0] + numPlanes
        continue

    #planes = np.concatenate(planes, axis=0)

    # if len(planes) > numPlanes:
    #     planeList = zip(planes, planePointsArray)
    #     planeList = sorted(planeList, key=lambda x:-len(x[1]))
    #     planeList = planeList[:numPlanes]
    #     planes, planePointsArray = zip(*planeList)
    #     pass


    # if len(planes) > numPlanes:
    #     planeList = zip(planes, planePointIndices)
    #     planeList = sorted(planeList, key=lambda x:-len(x[1]))
    #     planeList = planeList[:numPlanes]
    #     planes, planePointIndices = zip(*planeList)
    #     pass


    if len(planes) == 0:
        return np.array([]), np.ones(segmentation.shape).astype(np.int32) * (-1), []
    
    planes = np.array(planes)
    print('number of planes: ' + str(planes.shape[0]))

    # print(planes)
    # for v in planePointsArray:
    #     print(len(v))
    # if planes.shape[0] < numPlanes:
    #     planes = np.concatenate([planes, np.zeros((numPlanes - planes.shape[0], 3))], axis=0)
    #     pass
    
    
    planeSegmentation = np.ones(segmentation.shape) * (-1)
    for planeIndex, planePoints in enumerate(planePointIndices):
        planeSegmentation[planePoints] = planeIndex
        continue

    planesD = 1 / np.linalg.norm(planes, 2, 1, keepdims=True)
    planes *= pow(planesD, 2)

    return planes, planeSegmentation, groupNumPlanes

def mergePlanes3D(points, planes, planePointIndices, planeDiffThreshold = 0.05, planeAngleThreshold = 30, inlierThreshold = 0.9, planeAreaThreshold = 100):

    # planesD = np.linalg.norm(planes, axis=1, keepdims=True)
    # planes = planes / pow(planesD, 2)
    # planesDiff = (np.linalg.norm(np.expand_dims(planes, 1) - np.expand_dims(planes, 0), axis=2) < planeDiffThreshold).astype(np.int32)

    planeList = zip(planes, planePointIndices)
    planeList = sorted(planeList, key=lambda x:-len(x[1]))
    planes, planePointIndices = zip(*planeList)

    groupedPlanes = []
    groupedPlanePointIndices = []
    
    numPlanes = len(planes)
    usedPlaneConfidence = np.ones(numPlanes, np.float32) * (-1)
    usedPlaneMap = np.ones(numPlanes, np.int32) * (-1)
    
    for planeIndex, plane in enumerate(planes):
        if usedPlaneConfidence[planeIndex] > 0:
            continue
        usedPlaneConfidence[planeIndex] = 1
        usedPlaneMap[planeIndex] = planeIndex
        XYZ = points[planePointIndices[planeIndex]]
        for otherPlaneIndex in xrange(planeIndex + 1, numPlanes):
            #if planeIndex not in [0, 1, 2, ]:
            #break
            #if usedPlanes[otherPlaneIndex]:
            #continue
            #otherPlane = planes[otherPlaneIndex]
            #if np.abs(np.dot(plane, otherPlane)) / (np.linalg.norm(plane) * np.linalg.norm(otherPlane)) < np.cos(np.deg2rad(planeAngleThreshold)):
            #continue

            otherXYZ = points[planePointIndices[otherPlaneIndex]]

            diff_1 = np.abs(np.matmul(otherXYZ, plane) - np.ones(otherXYZ.shape[0])) / np.linalg.norm(plane)
            #diff_2 = np.abs(np.matmul(XYZ, otherPlane) - np.ones(XYZ.shape[0])) / np.linalg.norm(otherPlane)
            ratio = float(np.sum(diff_1 < planeDiffThreshold)) / diff_1.shape[0]
            if ratio > max(inlierThreshold, usedPlaneConfidence[otherPlaneIndex]):
                usedPlaneConfidence[otherPlaneIndex] = ratio
                usedPlaneMap[otherPlaneIndex] = planeIndex
                pass
            continue
        continue

    for planeIndex in xrange(numPlanes):
        mergedPlanes = (usedPlaneMap == planeIndex).nonzero()[0].tolist()
        if len(mergedPlanes) == 0:
            continue
        pointIndices = []
        for mergedPlaneIndex in mergedPlanes:
            pointIndices += planePointIndices[mergedPlaneIndex].tolist()
            continue
        if len(pointIndices) <= planeAreaThreshold:
            continue
        XYZ = points[pointIndices]
        ranges = XYZ.max(0) - XYZ.min(0)
        ranges.sort()
        #print((planeIndex, ranges.tolist() + XYZ.mean(0).tolist()))
        if ranges[1] < 0.2:
            continue
        #continue
        #print(XYZ.shape[0])
        #print(ranges)
        #if ranges.max() * 5 >= XYZ.shape[0]:
        #continue
        plane = fitPlane(XYZ)
        groupedPlanes.append(plane)
        groupedPlanePointIndices.append(np.array(pointIndices))
        continue
    
    return groupedPlanes, groupedPlanePointIndices

def mergePlanesBackup(points, planes, planePointIndices, planeDiffThreshold = 0.05, planeAngleThreshold = 30, inlierThreshold = 0.8):

    # planesD = np.linalg.norm(planes, axis=1, keepdims=True)
    # planes = planes / pow(planesD, 2)
    # planesDiff = (np.linalg.norm(np.expand_dims(planes, 1) - np.expand_dims(planes, 0), axis=2) < planeDiffThreshold).astype(np.int32)
    numPlanes = len(planes)
    planesDiff = np.diag(np.ones(numPlanes))
    for planeIndex, plane in enumerate(planes):
        for otherPlaneIndex in xrange(planeIndex + 1, numPlanes):
            otherPlane = planes[otherPlaneIndex]
            if np.abs(np.dot(plane, otherPlane)) / (np.linalg.norm(plane) * np.linalg.norm(otherPlane)) < np.cos(np.deg2rad(planeAngleThreshold)):
                continue
            XYZ = points[planePointIndices[planeIndex]]
            otherXYZ = points[planePointIndices[otherPlaneIndex]]
            diff_1 = np.abs(np.matmul(otherXYZ, plane) - np.ones(otherXYZ.shape[0])) / np.linalg.norm(plane)
            diff_2 = np.abs(np.matmul(XYZ, otherPlane) - np.ones(XYZ.shape[0])) / np.linalg.norm(otherPlane)

            if np.sum(diff_1 < planeDiffThreshold) > diff_1.shape[0] * inlierThreshold or np.sum(diff_2 < planeDiffThreshold) > diff_2.shape[0] * inlierThreshold:
                planesDiff[planeIndex][otherPlaneIndex] = 1
                planesDiff[otherPlaneIndex][planeIndex] = 1
                pass

            
            # if min(diff_1.mean(), diff_2.mean()) < planeDiffThreshold:
            #     planesDiff[planeIndex][otherPlaneIndex] = 1
            #     planesDiff[otherPlaneIndex][planeIndex] = 1
            #     pass
            
            
            # if diff_1.mean() < planeDiffThreshold:
            #     planesDiff[planeIndex][otherPlaneIndex] = 1
            #     pass
            # if diff_2.mean() < planeDiffThreshold:
            #     planesDiff[otherPlaneIndex][planeIndex] = 1
            #     pass
            continue
        continue

    # while True:
    #     #nextPlanesDiff = (np.matmul(planesDiff, planesDiff) > 0.5).astype(np.int32)
    #     nextPlanesDiff = ((planesDiff + np.matmul(np.transpose(planesDiff), planesDiff)) > 0.5).astype(np.int32)
    #     if np.all(nextPlanesDiff == planesDiff):
    #         break
    #     planesDiff = nextPlanesDiff
    #     continue

    #print(planesDiff)
    usedPlanes = np.zeros(planesDiff.shape[0])
    uniquePlanesDiff = []
    for planeIndex in xrange(planesDiff.shape[0]):
        planesMask = np.maximum(planesDiff[planeIndex] - usedPlanes, 0)
        if planesMask[planeIndex] > 0:
            uniquePlanesDiff.append(planesMask)
            usedPlanes += planesMask
            pass
        continue
    planesDiff = np.array(uniquePlanesDiff)
    #print(planesDiff)

    
    # diffMatrix = np.diag(np.ones(planesDiff.shape[0])).astype(np.float64)
    # diffMatrix -= np.tril(np.ones(planesDiff.shape), -1)
    # planesDiff = np.maximum(np.matmul(diffMatrix, planesDiff), 0)
    # planesDiff *= np.expand_dims(np.diag(planesDiff), -1)
    # planesDiff = np.unique(planesDiff, axis=0)
    
    groupedPlanes = []
    groupedPlanePointIndices = []
    for groupIndex in xrange(planesDiff.shape[0]):
        if planesDiff[groupIndex].sum() == 0:
            continue
        segmentIndices = planesDiff[groupIndex].nonzero()[0]
        pointIndices = []
        for segmentIndex in segmentIndices.tolist():
            pointIndices += planePointIndices[segmentIndex].tolist()
            continue
        XYZ = points[pointIndices]
        plane = fitPlane(XYZ)
        groupedPlanes.append(plane)
        groupedPlanePointIndices.append(np.array(pointIndices))
        continue
    
    return groupedPlanes, groupedPlanePointIndices


def evaluatePlaneSegmentation(predPlanes, predSegmentations, gtPlanes, gtSegmentations, gtNumPlanes, prefix = '', numOutputPlanes = 20):
    if len(gtSegmentations.shape) == 3:
        gtSegmentations = (np.expand_dims(gtSegmentations, -1) == np.arange(numOutputPlanes)).astype(np.float32)
        pass
    if len(predSegmentations.shape) == 3:
        predSegmentations = (np.expand_dims(predSegmentations, -1) == np.arange(numOutputPlanes)).astype(np.float32)
        pass
    
    width = predSegmentations.shape[2]
    height = predSegmentations.shape[1]

    planeDiffs = np.linalg.norm(np.expand_dims(gtPlanes, 2) - np.expand_dims(predPlanes, 1), axis=3)
    #print(gtPlanes[0])
    #print(predPlanes[0])
    #print(planeDiffs[0])

    planeAreas = np.sum(np.sum(gtSegmentations, axis=1), axis=1)
    intersection = np.sum((np.expand_dims(gtSegmentations, -1) * np.expand_dims(predSegmentations, 3) > 0.5).astype(np.float32), axis=(1, 2))
    union = np.sum((np.expand_dims(gtSegmentations, -1) + np.expand_dims(predSegmentations, 3) > 0.5).astype(np.float32), axis=(1, 2))
    planeIOUs = intersection / np.maximum(union, 1e-4)

    planeMask = np.expand_dims(np.arange(predPlanes.shape[1]), 0) < np.expand_dims(gtNumPlanes, 1)
    for index, numPlanes in enumerate(gtNumPlanes.tolist()):
        planeDiffs[index, numPlanes:] = 1000000
        planeIOUs[index, numPlanes:] = -1
        pass
    
    totalNumPlanes = gtNumPlanes.sum()

    numPixels = planeAreas.sum(1)
    
    # planeDistanceThreshold = 0.5
    # diffMask = (planeDiffs < planeDistanceThreshold).astype(np.float32)
    # maxIOU = np.max(planeIOUs * diffMask, axis=2)
    # IOU = 0.5
    # print(maxIOU[0])
    # print(planeMask[0])
    # print(((maxIOU >= IOU) * planeMask).sum(1).astype(np.float32))
    # print(gtNumPlanes)
    # print(float(((maxIOU >= IOU) * planeMask).sum()) / totalNumPlanes)
    
    # exit(1)
    
    pixel_curves = []
    plane_curves = []
    for planeDistanceThreshold in [0.1, 0.3, 0.5]:
        diffMask = (planeDiffs < planeDistanceThreshold).astype(np.float32)
        maxIOU = np.max(planeIOUs * diffMask, axis=2)
        stride = 0.1
        planeRecalls = []
        pixelRecalls = []
        for step in xrange(int(1 / stride + 1)):
            IOU = step * stride
            pixelRecalls.append((np.minimum((intersection * (planeIOUs >= IOU).astype(np.float32) * diffMask).sum(2), planeAreas).sum(1) / numPixels).mean())
            planeRecalls.append(float(((maxIOU >= IOU) * planeMask).sum()) / totalNumPlanes)            
            continue
        
        pixel_curves.append(pixelRecalls)
        plane_curves.append(planeRecalls)
        pass
    
    for IOUThreshold in [0.3, 0.5, 0.7]:
        IOUMask = (planeIOUs > IOUThreshold).astype(np.float32)
        minDiff = np.min(planeDiffs * IOUMask + 1000000 * (1 - IOUMask), axis=2)
        stride = 0.05
        planeRecalls = []
        pixelRecalls = []
        for step in xrange(int(0.5 / stride + 1)):
            diff = step * stride
            pixelRecalls.append((np.minimum((intersection * (planeDiffs <= diff).astype(np.float32) * IOUMask).sum(2), planeAreas).sum(1) / numPixels).mean())
            planeRecalls.append(float(((minDiff <= diff) * planeMask).sum()) / totalNumPlanes)
            continue
        pixel_curves.append(pixelRecalls)
        plane_curves.append(planeRecalls)
        pass

    
    if prefix == '':
        return pixel_curves, plane_curves
    else:
        np.save(prefix + 'curves.npy', pixel_curves + plane_curves)
        return

def plotCurves(x, ys, filename = 'test/test.png', xlabel='', ylabel='', title='', labels=[]):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    for index, y in enumerate(ys):
        plt.plot(x, y, figure=fig, label=labels[index])
        continue
    plt.legend(loc='upper right')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)    
    plt.ylim((-0.1, 1.1))
    plt.savefig(filename)
    return
    
def transformPlanes(planes, transformation):
    
    centers = planes
    planesD = np.maximum(np.linalg.norm(planes, axis=1, keepdims=True), 1e-4)
    refPoints = centers - planes / planesD
    
    centers = np.concatenate([centers, np.ones((planes.shape[0], 1))], axis=1)
    refPoints = np.concatenate([refPoints, np.ones((planes.shape[0], 1))], axis=1)
    
    newCenters = np.transpose(np.matmul(transformation, np.transpose(centers)))
    newRefPoints = np.transpose(np.matmul(transformation, np.transpose(refPoints)))

    newCenters = newCenters[:, :3] / newCenters[:, 3:4]
    newRefPoints = newRefPoints[:, :3] / newRefPoints[:, 3:4]

    planeNormals = newRefPoints - newCenters
    planesD = -np.sum(newCenters * planeNormals, axis=1, keepdims=True)
    newPlanes = -planeNormals * planesD
    return newPlanes


def softmax(values):
    exp = np.exp(values - values.max())
    return exp / exp.sum(-1, keepdims=True)

def one_hot(values, depth):
    maxInds = values.reshape(-1)
    results = np.zeros([maxInds.shape[0], depth])
    results[np.arange(maxInds.shape[0]), maxInds] = 1
    results = results.reshape(list(values.shape) + [depth])
    return results

def sigmoid(values):
    return 1 / (1 + np.exp(-values))

def sortSegmentations(segmentations, planes, planesTarget):
    diff = np.linalg.norm(np.expand_dims(planes, 1) - np.expand_dims(planesTarget, 0), axis=2)
    planeMap = one_hot(np.argmin(diff, axis=-1), depth=diff.shape[-1])
    #print(planeMap)
    segmentationsTarget = np.matmul(segmentations, planeMap)
    return segmentationsTarget, np.matmul(planes.transpose(), planeMap).transpose()

def refitPlanes(planes, segmentation, depth, info, numOutputPlanes=20, planeAreaThreshold=6*8):
    camera = getCameraFromInfo(info)
    width = depth.shape[1]
    height = depth.shape[0]

    #camera = getNYURGBDCamera()
    #camera = getSUNCGCamera()

    urange = (np.arange(width, dtype=np.float32) / width * camera['width'] - camera['cx']) / camera['fx']
    urange = urange.reshape(1, -1).repeat(height, 0)
    vrange = (np.arange(height, dtype=np.float32) / height * camera['height'] - camera['cy']) / camera['fy']
    vrange = vrange.reshape(-1, 1).repeat(width, 1)
    X = depth * urange
    Y = depth
    Z = -depth * vrange

    XYZ = np.stack([X, Y, Z], axis=2)
    
    validDepthMask = depth > 1e-4
    
    newPlaneInfo = []
    for planeIndex in xrange(numOutputPlanes):
        mask = segmentation == planeIndex
        points = XYZ[np.logical_and(cv2.erode(mask.astype(np.float32), np.ones((3, 3))) > 0.5, validDepthMask)]
        if points.shape[0] >= 3:
            try:
                plane = fitPlane(points)
                plane /= pow(np.linalg.norm(plane), 2)
                newPlaneInfo.append((plane, mask, points.shape[0]))
                #newPlaneInfo.append((planes[planeIndex], mask, points.shape[0]))                
            except:
                pass
            pass
        continue
    
    numPlanes = len(newPlaneInfo)
    if numPlanes == 0:
        return np.zeros((numOutputPlanes, 3)), newSegmentation, numPlanes
    
    newPlaneInfo = sorted(newPlaneInfo, key=lambda x: -x[2])

    newPlanes = []
    newSegmentation = np.ones(segmentation.shape, dtype=np.uint8) * numOutputPlanes
    for planeIndex, planeInfo in enumerate(newPlaneInfo):
        newPlanes.append(planeInfo[0])
        newSegmentation[planeInfo[1]] = planeIndex
        continue
    
    newPlanes = np.array(newPlanes)
    if numPlanes < numOutputPlanes:
        newPlanes = np.concatenate([newPlanes, np.zeros((numOutputPlanes - numPlanes, 3))], axis=0)
        pass
    
    return newPlanes, newSegmentation, numPlanes

# def filterPlanesPred(planes, segmentations, depth, info, segmentationsTarget, numOutputPlanes=20, nonPlaneRatioThreshold=0.7, coveredPlaneRatioThreshold=0.5, planeDistanceThreshold=0.05, planeAngleThreshold=np.cos(np.deg2rad(20))):

#     camera = getCameraFromInfo(info)
#     width = depth.shape[1]
#     height = depth.shape[0]

#     #camera = getNYURGBDCamera()
#     #camera = getSUNCGCamera()

#     urange = (np.arange(width, dtype=np.float32) / width * camera['width'] - camera['cx']) / camera['fx']
#     urange = urange.reshape(1, -1).repeat(height, 0)
#     vrange = (np.arange(height, dtype=np.float32) / height * camera['height'] - camera['cy']) / camera['fy']
#     vrange = vrange.reshape(-1, 1).repeat(width, 1)
#     X = depth * urange
#     Y = depth
#     Z = -depth * vrange

#     XYZ = np.stack([X, Y, Z], axis=2)
    
#     validDepthMask = depth > 1e-4
    
#     #numPlanes = planes.shape[0]
#     validPlaneInfo = []
#     emptyMaskTarget = segmentationsTarget[:, :, numOutputPlanes]
#     emptyMask = segmentations[:, :, numOutputPlanes]    
#     for planeIndex in xrange(numOutputPlanes):
#         mask = segmentations[:, :, planeIndex]
#         if (emptyMaskTarget * mask).sum() < mask.sum() * nonPlaneRatioThreshold:
#             points = XYZ[np.logical_and(cv2.erode(mask, np.ones((3, 3))) > 0.5, validDepthMask)]
#             if points.shape[0] >= 3:
#                 try:
#                     plane = fitPlane(points)
#                     plane /= pow(np.linalg.norm(plane), 2)
#                     validPlaneInfo.append((plane, mask, points))                    
#                 except:
#                     emptyMask += mask
#                 pass
#             else:
#                 emptyMask += mask
#         else:
#             emptyMask += mask
#             pass
#         continue
    
#     #validPlaneInfo = sorted(validPlaneInfo, key=lambda x:-x[2])

#     for planeIndex, planeInfo in enumerate(validPlaneInfo):
#         cv2.imwrite('test/mask_' + str(planeIndex) + '.png', drawMaskImage(planeInfo[1]))
#         print(planeIndex, planeInfo[0])
#         continue
    
#     emptyMask = (emptyMask > 0.5).astype(np.float32)
#     for planeIndexTarget in xrange(numOutputPlanes):
#         maskTarget = segmentationsTarget[:, :, planeIndexTarget]
#         excludedMask = ((maskTarget + emptyMask) > 0.5).astype(np.float32)
#         coveredPlanes = []
#         for planeIndex, planeInfo in enumerate(validPlaneInfo):
#             mask = planeInfo[1]
#             area = mask.sum()
#             if (maskTarget * mask).sum() / area > coveredPlaneRatioThreshold:
#                 coveredPlanes.append((planeIndex, planeInfo[0], planeInfo[2]))
#                 pass
#             continue
#         if len(coveredPlanes) <= 1:
#             continue
        
#         coveredPlanes = sorted(coveredPlanes, key=lambda x:-x[2].shape[0])

        
#         majorPlane = coveredPlanes[0][1]
#         majorPlaneD = np.linalg.norm(majorPlane)
#         majorPlaneNormal = majorPlane / majorPlaneD
#         mergedPlanes = [coveredPlanes[0][0], ]
#         for planeInfo in coveredPlanes[1:]:
#             #if np.linalg.norm(planeInfo[1] - majorPlane) < planeDistanceThreshold:
#             distance = np.abs(np.sum(planeInfo[2] * majorPlaneNormal, axis=-1) - majorPlaneD)
#             print(distance.mean())
#             print(distance.max())
#             planeNormal = planeInfo[1] / np.linalg.norm(planeInfo[1])            
#             print(np.sum(planeNormal * majorPlaneNormal), planeAngleThreshold)
#             exit(1)
#             if distance.mean() < planeDistanceThreshold and np.sum(planeNormal * majorPlaneNormal) > planeAngleThreshold:
#                 mergedPlanes.append(planeInfo[0])
#                 pass
#             continue
#         if mergedPlanes <= 1:
#             continue
#         newValidPlaneInfo = []
#         mergedPlaneMask = np.zeros(emptyMask.shape)
#         for planeIndex, planeInfo in enumerate(validPlaneInfo):
#             if planeIndex not in mergedPlanes:
#                 if (excludedMask * planeInfo[1]).sum() < planeInfo[1].sum() * nonPlaneRatioThreshold:
#                     newValidPlaneInfo.append(planeInfo)
#                     pass
#             else:
#                 mergedPlaneMask += planeInfo[1]
#                 pass
#             continue
#         cv2.erode(mergedPlaneMask, np.ones((3, 3)))        
#         mergedPlaneMask = mergedPlaneMask > 0.5
#         points = XYZ[np.logical_and(mergedPlaneMask, validDepthMask)]
#         if points.shape[0] >= 3:
#             try:
#                 mergedPlane = fitPlane(points)
#                 mergedPlane = mergedPlane / pow(np.linalg.norm(mergedPlane), 2)
#                 newValidPlaneInfo.append((mergedPlane, mergedPlaneMask.astype(np.float32), points))
#             except:
#                 pass
#             pass
#         validPlaneInfo = newValidPlaneInfo
#         continue

#     validPlaneInfo = sorted(validPlaneInfo, key=lambda x: -x[1].sum())
    
#     newPlanes = []
#     newSegmentation = np.ones(emptyMask.shape) * numOutputPlanes
#     for planeIndex, planeInfo in enumerate(validPlaneInfo):
#         newPlanes.append(planeInfo[0])
#         newSegmentation[planeInfo[1].astype(np.bool)] = planeIndex
#         continue
#     numPlanes = len(newPlanes)
#     if numPlanes == 0:
#         return np.zeros((numOutputPlanes, 3)), newSegmentation, numPlanes
    
#     newPlanes = np.array(newPlanes)
#     if numPlanes < numOutputPlanes:
#         newPlanes = np.concatenate([newPlanes, np.zeros((numOutputPlanes - numPlanes, 3))], axis=0)
#         pass
    
#     return newPlanes, newSegmentation.astype(np.uint8), numPlanes

def filterPlanes(planes, segmentations, depth, info, numOutputPlanes=20, coveredPlaneRatioThreshold=0.5, planeDistanceThreshold=0.05, normalDotThreshold=np.cos(np.deg2rad(20)), planeFittingThreshold = 0.03):

    camera = getCameraFromInfo(info)
    width = depth.shape[1]
    height = depth.shape[0]

    #camera = getNYURGBDCamera()
    #camera = getSUNCGCamera()

    urange = (np.arange(width, dtype=np.float32) / width * camera['width'] - camera['cx']) / camera['fx']
    urange = urange.reshape(1, -1).repeat(height, 0)
    vrange = (np.arange(height, dtype=np.float32) / height * camera['height'] - camera['cy']) / camera['fy']
    vrange = vrange.reshape(-1, 1).repeat(width, 1)
    X = depth * urange
    Y = depth
    Z = -depth * vrange

    XYZ = np.stack([X, Y, Z], axis=2)
    
    validDepthMask = depth > 1e-4

    
    
    #numPlanes = planes.shape[0]
    validPlaneInfo = []
    for planeIndex in xrange(numOutputPlanes):
        mask = segmentations[:, :, planeIndex]
        points = XYZ[np.logical_and(cv2.erode(mask, np.ones((3, 3)), 2) > 0.5, validDepthMask)]
        if points.shape[0] >= 3:
            try:
                plane = fitPlane(points)
                plane /= pow(np.linalg.norm(plane), 2)
                #plane = planes[planeIndex]
                
                #planeD = np.linalg.norm(plane)
                #diff = np.abs(np.sum(points * (plane / planeD), axis=-1) - planeD).mean()
                #validMask = np.abs(np.sum(points * (plane / planeD), axis=-1) - planeD) < planeDistanceThreshold

                #diff = (np.abs((np.sum(points * plane, axis=-1) - 1) / np.linalg.norm(plane)) > 0.05).astype(np.float32).mean()

                #print(planeIndex, diff, mask.sum())
                #cv2.imwrite('test/mask_' + str(planeIndex) + '.png', drawMaskImage(mask))                
                #if diff < planeFittingThreshold:
                validPlaneInfo.append((plane, mask, points))
                #    pass
            except:
                pass
            pass
        continue

    #validPlaneInfo = sorted(validPlaneInfo, key=lambda x:-x[2])

    validPlaneInfo = sorted(validPlaneInfo, key=lambda x: -x[2].shape[0])

    if False:
        for planeIndex, planeInfo in enumerate(validPlaneInfo):
            cv2.imwrite('test/mask_' + str(planeIndex) + '.png', drawMaskImage(planeInfo[1]))
            print(planeIndex, planeInfo[0], planeInfo[3], planeInfo[1].sum())
            continue
        pass

    newPlaneInfo = []
    usedPlaneMask = np.zeros(len(validPlaneInfo), dtype=np.bool)
    for majorPlaneIndex, majorPlaneInfo in enumerate(validPlaneInfo):
        if usedPlaneMask[majorPlaneIndex]:
            continue
        usedPlaneMask[majorPlaneIndex] = True
        
        majorPlane = majorPlaneInfo[0]
        majorPlaneD = np.linalg.norm(majorPlane)
        majorPlaneNormal = majorPlane / majorPlaneD

        mergedPlaneMask = majorPlaneInfo[1].copy()
        planeMerged = False
        for planeIndex, planeInfo in enumerate(validPlaneInfo):
            if planeIndex <= majorPlaneIndex or usedPlaneMask[planeIndex]:
                continue
            
            fittingDiff = np.abs(np.sum(planeInfo[2] * majorPlaneNormal, axis=-1) - majorPlaneD)
            planeNormal = planeInfo[0] / np.linalg.norm(planeInfo[0])
            normalDot = np.sum(planeNormal * majorPlaneNormal)
            
            #print(majorPlaneIndex, planeIndex)
            #print(majorPlane, planeInfo[0])
            #print(fittingDiff.mean(), (fittingDiff < 0.05).astype(np.float32).mean(), normalDot, normalDotThreshold)

            if fittingDiff.mean() < planeDistanceThreshold and normalDot > normalDotThreshold:
                #print('merge', majorPlaneIndex, planeIndex)
                mergedPlaneMask += planeInfo[1]
                usedPlaneMask[planeIndex] = True                                
                planeMerged = True
                pass
            continue

        if planeMerged:
            mergedPlaneMask = (mergedPlaneMask > 0.5).astype(np.float32)
            pass
        
        newPlaneInfo.append((majorPlaneInfo[0], mergedPlaneMask))
        continue

    newPlaneInfo = sorted(newPlaneInfo, key=lambda x: -x[1].sum())
    
    newPlanes = []
    newSegmentation = np.ones((height, width), dtype=np.uint8) * numOutputPlanes
    for planeIndex, planeInfo in enumerate(newPlaneInfo):
        area = planeInfo[1].sum()
        xs = planeInfo[1].max(0).nonzero()[0]
        ys = planeInfo[1].max(1).nonzero()[0]
        length = np.sqrt(pow(xs.max() - xs.min() + 1, 2) + pow(ys.max() - ys.min() + 1, 2))
        if area < (width * height / 100.) or area / length < 10:
            continue
        newSegmentation[planeInfo[1].astype(np.bool)] = len(newPlanes)        
        newPlanes.append(planeInfo[0])
        continue
    numPlanes = len(newPlanes)
    if numPlanes == 0:
        return np.zeros((numOutputPlanes, 3)), newSegmentation, numPlanes
    
    newPlanes = np.array(newPlanes)
    if numPlanes < numOutputPlanes:
        newPlanes = np.concatenate([newPlanes, np.zeros((numOutputPlanes - numPlanes, 3))], axis=0)
        pass
    
    return newPlanes, newSegmentation, numPlanes


def getSegmentationsGraphCut(planes, image, depth, normal, info):

    height = depth.shape[0]
    width = depth.shape[1]

    # planeMap = []
    # planeMasks = []
    # for planeIndex in xrange(numPlanes):
    #     planeMask = segmentation == planeIndex
    #     if planeMask.sum() < 6 * 8:
    #         continue
    #     planeMap.append(planeIndex)
    #     semantic = np.bincount(semantics[planeMask]).argmax()
    #     for _ in xrange(2):
    #         planeMask = cv2.dilate(planeMask.astype(np.float32), np.ones((3, 3), dtype=np.float32))
    #         continue        
    #     planeMask = np.logical_and(np.logical_or(semantics == semantic, semantics == 0), planeMask).astype(np.float32)
    #     for _ in xrange(1):
    #         planeMask = cv2.dilate(planeMask, np.ones((3, 3), dtype=np.float32))
    #         continue
    #     planeMasks.append(planeMask)
    #     continue
    # planeMap = one_hot(np.array(planeMap), depth=planes.shape[0])
    #planes = np.matmul(planeMap, planes)
    #planeMasks = np.stack(planeMasks, 2).reshape((-1, numPlanes))

    numPlanes = planes.shape[0]
    
    #if numPlanes < numOutputPlanes:
    #planeMasks = np.concatenate([planeMasks, np.zeros((height, width, numOutputPlanes - numPlanes))], axis=2)
    #pass
    
    #print(info)
    camera = getCameraFromInfo(info)
    urange = (np.arange(width, dtype=np.float32) / (width) * (camera['width']) - camera['cx']) / camera['fx']
    urange = urange.reshape(1, -1).repeat(height, 0)
    vrange = (np.arange(height, dtype=np.float32) / (height) * (camera['height']) - camera['cy']) / camera['fy']
    vrange = vrange.reshape(-1, 1).repeat(width, 1)
    
    X = depth * urange
    Y = depth
    Z = -depth * vrange

    points = np.stack([X, Y, Z], axis=2)
    planes = planes[:numPlanes]
    planesD = np.linalg.norm(planes, axis=1, keepdims=True)
    planeNormals = planes / np.maximum(planesD, 1e-4)

    distanceCostThreshold = 0.05

    #distanceCost = 1 - np.exp(-np.abs(np.tensordot(points, planeNormals, axes=([2, 1])) - np.reshape(planesD, [1, 1, -1])) / distanceCostThreshold)
    #distanceCost = np.concatenate([distanceCost, np.ones((height, width, 1)) * (1 - np.exp(-1))], axis=2)
    distanceCost = np.abs(np.tensordot(points, planeNormals, axes=([2, 1])) - np.reshape(planesD, [1, 1, -1])) / distanceCostThreshold
    distanceCost = np.concatenate([distanceCost, np.ones((height, width, 1))], axis=2)
    #cv2.imwrite('test/mask.png', drawMaskImage(np.minimum(distanceCost[:, :, 2] /  5, 1)))
    #distanceCost[:, :, numPlanes:numOutputPlanes] = 10000

    normalCost = 0
    if info[19] <= 1:
        normalCostThreshold = 1 - np.cos(20)        
        normalCost = (1 - np.tensordot(normal, planeNormals, axes=([2, 1]))) / normalCostThreshold
        #normalCost[:, :, numPlanes:] = 10000
        normalCost = np.concatenate([normalCost, np.ones((height, width, 1))], axis=2)
        pass

    unaryCost = distanceCost + normalCost
    unaryCost *= np.expand_dims((depth > 1e-4).astype(np.float32), -1)
    unaries = -unaryCost.reshape((-1, numPlanes + 1))

    #unaries[:, :numPlanes] -= (1 - planeMasks) * 10000
    
    #print(planes)
    #print(distanceCost[150][200])
    #print(unaryCost[150][200])
    # print(np.argmax(-unaryCost[60][150]))

    #cv2.imwrite('test/depth.png', drawDepthImage(depth))
    cv2.imwrite('test/segmentation.png', drawSegmentationImage(unaries.reshape((height, width, -1)), blackIndex=numPlanes))
    #cv2.imwrite('test/mask.png', drawSegmentationImage(planeMasks.reshape((height, width, -1))))
    #exit(1)


    nodes = np.arange(height * width).reshape((height, width))

    image = image.astype(np.float32)
    colors = image.reshape((-1, 3))
    #deltas = [(0, 1), (1, 0), (-1, 1), (1, 1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    # intensityDifference = np.zeros((height * width))
    # for delta in deltas:
    #     deltaX = delta[0]
    #     deltaY = delta[1]
    #     partial_nodes = nodes[max(-deltaY, 0):min(height - deltaY, height), max(-deltaX, 0):min(width - deltaX, width)].reshape(-1)
    #     intensityDifference[partial_nodes] += np.sum(pow(colors[partial_nodes] - colors[partial_nodes + (deltaY * width + deltaX)], 2), axis=1)
    #     continue

    # intensityDifference = intensityDifference.reshape((height, width))
    # intensityDifference[1:height - 1, 1:width - 1] /= 8
    # intensityDifference[1:height - 1, 0] /= 5
    # intensityDifference[1:height - 1, width - 1] /= 5
    # intensityDifference[0, 1:width - 1] /= 5
    # intensityDifference[height - 1, 1:width - 1] /= 5    
    # intensityDifference[0][0] /= 3
    # intensityDifference[0][width - 1] /= 3
    # intensityDifference[height - 1][0] /= 3
    # intensityDifference[height - 1][width - 1] /= 3
    # intensityDifference = intensityDifference.reshape(-1)
    

    deltas = [(0, 1), (1, 0), (-1, 1), (1, 1)]    
    
    intensityDifferenceSum = 0.0
    intensityDifferenceCount = 0
    for delta in deltas:
        deltaX = delta[0]
        deltaY = delta[1]
        partial_nodes = nodes[max(-deltaY, 0):min(height - deltaY, height), max(-deltaX, 0):min(width - deltaX, width)].reshape(-1)
        intensityDifferenceSum += np.sum(pow(colors[partial_nodes] - colors[partial_nodes + (deltaY * width + deltaX)], 2))
        intensityDifferenceCount += partial_nodes.shape[0]
        continue
    intensityDifference = intensityDifferenceSum / intensityDifferenceCount

    
    edges = []
    edges_features = []
    pairwise_matrix = 1 - np.diag(np.ones(numPlanes + 1))

    for delta in deltas:
        deltaX = delta[0]
        deltaY = delta[1]
        partial_nodes = nodes[max(-deltaY, 0):min(height - deltaY, height), max(-deltaX, 0):min(width - deltaX, width)].reshape(-1)
        edges.append(np.stack([partial_nodes, partial_nodes + (deltaY * width + deltaX)], axis=1))

        colorDiff = np.sum(pow(colors[partial_nodes] - colors[partial_nodes + (deltaY * width + deltaX)], 2), axis=1)
        
        pairwise_cost = np.expand_dims(pairwise_matrix, 0) * np.reshape(1 + 45 * np.exp(-colorDiff / intensityDifference), [-1, 1, 1])
        #pairwise_cost = np.expand_dims(pairwise_matrix, 0) * np.ones(np.reshape(1 + 45 * np.exp(-colorDiff / np.maximum(intensityDifference[partial_nodes], 1e-4)), [-1, 1, 1]).shape)
        edges_features.append(-pairwise_cost)
        continue

    edges = np.concatenate(edges, axis=0)
    edges_features = np.concatenate(edges_features, axis=0)

    refined_segmentation = inference_ogm(unaries * 50, edges_features, edges, return_energy=False, alg='alphaexp')
    #print(pairwise_matrix)
    #refined_segmentation = inference_ogm(unaries * 5, -pairwise_matrix, edges, return_energy=False, alg='alphaexp')
    refined_segmentation = refined_segmentation.reshape([height, width])
    
    return refined_segmentation
