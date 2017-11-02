import numpy as np
from pystruct.inference import get_installed, inference_ogm, inference_dispatch
from utils import *

def findProposals(segmentations, numProposals = 5):
    height = segmentations.shape[0]
    width = segmentations.shape[1]
    segmentationsNeighbors = []
    windowSize = 5
    segmentationsPadded = np.pad(segmentations, ((windowSize, windowSize), (windowSize, windowSize), (0, 0)), mode='constant')
    for shiftX in xrange(windowSize * 2 + 1):
        for shiftY in xrange(windowSize * 2 + 1):
            segmentationsNeighbors.append(segmentationsPadded[shiftY:shiftY + height, shiftX:shiftX + width, :])
            continue
        continue
    segmentationsNeighbors = np.max(np.stack(segmentationsNeighbors, axis=3), axis=3)
    proposals = np.argpartition(-segmentationsNeighbors, numProposals)[:, :, :numProposals]
    return proposals


def refineSegmentation(image, allSegmentations, allDepths, boundaries, numOutputPlanes=20, numIterations=20, numProposals=5):
    height = allSegmentations.shape[0]
    width = allSegmentations.shape[1]

    #allSegmentations = np.concatenate([planeSegmentations, nonPlaneSegmentation], axis=2)
    #allDepths = np.concatenate([planeDepths, nonPlaneDepth], axis=2)
    proposals = findProposals(allSegmentations, numProposals=numProposals)
    #proposals = np.sort(proposals, axis=-1)
    proposalSegmentations = readProposalInfo(allSegmentations, proposals)
    proposalDepths = readProposalInfo(allDepths, proposals)

    # print(allDepths[80][75])
    # print(proposals[80][75])    
    # print(proposalDepths[80][75])
    # exit(1)
    proposals = proposals.reshape((-1, numProposals))
    proposalDepths = proposalDepths.reshape((-1, numProposals))
    smoothBoundaries = boundaries[:, :, 0].reshape(-1)
    occlusionBoundaries = boundaries[:, :, 1].reshape(-1)
    
    maxDepthDiff = 0.1
    unaries = proposalSegmentations.reshape((-1, numProposals))
    nodes = np.arange(height * width).reshape((height, width))
    #deltas = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    deltas = [(0, 1), (1, 0), (-1, 1), (1, 1)]
    edges = []
    edges_features = []
    colors = image.reshape((-1, 3)).astype(np.float32)

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
    
    for delta in deltas:
        deltaX = delta[0]
        deltaY = delta[1]
        partial_nodes = nodes[max(-deltaY, 0):min(height - deltaY, height), max(-deltaX, 0):min(width - deltaX, width)].reshape(-1)
        neighbor_nodes = partial_nodes + (deltaY * width + deltaX)
        edges.append(np.stack([partial_nodes, neighbor_nodes], axis=1))

        #depth_1_1 = proposalDepths[max(-deltaY, 0):min(height - deltaY, height), max(-deltaX, 0):min(width - deltaX, width)].reshape((-1, numProposals))
        #depth_2_2 = proposalDepths[max(deltaY, 0):min(height + deltaY, height), max(deltaX, 0):min(width + deltaX, width)].reshape((-1, numProposals))

        #label_1 = proposals[max(-deltaY, 0):min(height - deltaY, height), max(-deltaX, 0):min(width - deltaX, width)].reshape((-1, numProposals))
        #label_2 = proposals[max(deltaY, 0):min(height + deltaY, height), max(deltaX, 0):min(width + deltaX, width)].reshape((-1, numProposals))
        
        #smooth_boundary = np.maximum(boundaries[max(-deltaY, 0):min(height - deltaY, height), max(-deltaX, 0):min(width - deltaX, width), 0].reshape(-1), boundaries[max(deltaY, 0):min(height + deltaY, height), max(deltaX, 0):min(width + deltaX, width), 0].reshape((-1)))
        #occlusion_boundary = np.maximum(boundaries[max(-deltaY, 0):min(height - deltaY, height), max(-deltaX, 0):min(width - deltaX, width), 1].reshape(-1), boundaries[max(deltaY, 0):min(height + deltaY, height), max(deltaX, 0):min(width + deltaX, width), 1].reshape((-1)))

        depth_1_1 = proposalDepths[partial_nodes]
        depth_2_2 = proposalDepths[neighbor_nodes]        
        depth_diff = np.abs(np.expand_dims(depth_1_1, -1) - np.expand_dims(depth_2_2, 1))
        depth_diff = np.clip(np.abs(depth_diff) / maxDepthDiff, 0, 1)
        
        label_1 = proposals[partial_nodes]
        label_2 = proposals[neighbor_nodes]
        label_diff = (np.expand_dims(label_1, -1) != np.expand_dims(label_2, 1)).astype(np.float32)
        #label_diff = np.clip(np.abs(label_diff), 0, 1)

        color_1 = colors[partial_nodes]
        color_2 = colors[neighbor_nodes]
        color_diff = pow(color_1 - color_2, 2).sum(1).reshape((-1, 1, 1))
        smooth_boundary = np.maximum(smoothBoundaries[partial_nodes], smoothBoundaries[neighbor_nodes])
        occlusion_boundary = np.maximum(occlusionBoundaries[partial_nodes], occlusionBoundaries[neighbor_nodes])

        #pairwise_cost = label_diff * depth_diff * smooth_boundary.reshape((-1, 1, 1)) + label_diff * np.clip(1 - smooth_boundary - occlusion_boundary, 0.1, 1).reshape((-1, 1, 1))

        pairwise_cost = label_diff * (1 + 50 * depth_diff + 20 * np.exp(-color_diff / intensityDifference))

        x = 75
        y = 88
        print((y - deltaY, x - deltaX))
        print(proposals[(y - deltaY) * width + (x - deltaX)])
        print(proposals[y * width + x])
        #index = neighbor_nodes == y * width + x
        index = partial_nodes == y * width + x
        print(pairwise_cost[index])
        print(depth_diff[index])
        print(depth_1_1[index])
        print(depth_2_2[index])                
        print(color_diff[index])
        
        #print(intensityDifference)
        #exit(1)
        edges_features.append(pairwise_cost)
        #print((edges[-1][partial_nodes == y * width + x][0][1] % width, edges[-1][partial_nodes == y * width + x][0][1] / width))
        #print(edges_features[-1][partial_nodes == y * width + x])
        #mask = edges[-1][:, 1] == y * width + x
        #print((edges[-1][mask][0][0] % width, edges[-1][mask][0][0] / width))
        #print(edges_features[-1][mask])
        continue

    
    edges = np.concatenate(edges, axis=0)
    edges_features = np.concatenate(edges_features, axis=0)
    refined_segmentation = inference_ogm(unaries * 10, edges_features, edges, return_energy=False, alg='trw')
    #refined_segmentation = inference_dispatch(unaries, edges_features * 0, edges, ('unary'))
    #refined_segmentation = np.argmin(unaries, axis=1)

    refined_segmentation = refined_segmentation.reshape([height, width, 1])
    refined_segmentation = readProposalInfo(proposals, refined_segmentation)
    refined_segmentation = refined_segmentation.reshape([height, width])

    # print((x, y))
    # print(proposals[y][x])
    # print((x + 1, y))
    # print(proposals[y][x + 1])
    # print((x - 1, y + 1))
    # print(proposals[y + 1][x - 1])
    # print((x, y + 1))
    # print(proposals[y + 1][x])
    # print((x + 1, y + 1))
    # print(proposals[y + 1][x + 1])
    # print(unaries[y * width + x])
    # print(refined_segmentation[y][x])
    return refined_segmentation
    #return proposals.reshape([height, width])


def getSegmentationsTRWS(planes, image, depth, normal, segmentation, semantics, info, numPlanes):
    numOutputPlanes = planes.shape[0]
    height = depth.shape[0]
    width = depth.shape[1]
    numProposals = 3


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
    distanceCost = np.abs(np.tensordot(points, planeNormals, axes=([2, 1])) - np.reshape(planesD, [1, 1, -1])) / distanceCostThreshold

    normalCost = 0
    if info[19] <= 1:
        normalCostThreshold = 1 - np.cos(20)        
        normalCost = (1 - np.tensordot(normal, planeNormals, axes=([2, 1]))) / normalCostThreshold
        pass
    
    planeMasks = []
    for planeIndex in xrange(numPlanes):
        #print(np.bincount(semantics[segmentation == planeIndex]))
        planeMaskOri = segmentation == planeIndex
        semantic = np.bincount(semantics[planeMaskOri]).argmax()
        #print(semantic)
        planeMask = cv2.dilate((np.logical_and(np.logical_or(semantics == semantic, planeMaskOri), distanceCost[:, :, planeIndex])).astype(np.uint8), np.ones((3, 3), dtype=np.uint8)).astype(np.float32)
        planeMasks.append(planeMask)
        #cv2.imwrite('test/mask_' + str(planeIndex) + '.png', drawMaskImage(segmentation == planeIndex))
        #cv2.imwrite('test/mask_2.png', drawMaskImage(semantics == semantic))
        continue
    
    planeMasks = np.stack(planeMasks, 2)

    unaryCost = distanceCost + (1 - planeMasks) * 10000
    unaryCost = np.concatenate([unaryCost, np.ones((height, width, 1))], axis=2)

    proposals = np.argpartition(unaryCost, numProposals)[:, :, :numProposals]
    unaries = -readProposalInfo(unaryCost, proposals).reshape((-1, numProposals))

    # refined_segmentation = np.argmax(unaries, axis=1)
    # refined_segmentation = refined_segmentation.reshape([height, width, 1])
    # refined_segmentation = readProposalInfo(proposals, refined_segmentation)
    # refined_segmentation = refined_segmentation.reshape([height, width])
    # refined_segmentation[refined_segmentation == numPlanes] = numOutputPlanes
    

    proposals = proposals.reshape((-1, numProposals))
    #cv2.imwrite('test/segmentation.png', drawSegmentationImage(unaries.reshape((height, width, -1)), blackIndex=numOutputPlanes))

    nodes = np.arange(height * width).reshape((height, width))

    image = image.astype(np.float32)
    colors = image.reshape((-1, 3))
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

    for delta in deltas:
        deltaX = delta[0]
        deltaY = delta[1]
        partial_nodes = nodes[max(-deltaY, 0):min(height - deltaY, height), max(-deltaX, 0):min(width - deltaX, width)].reshape(-1)
        edges.append(np.stack([partial_nodes, partial_nodes + (deltaY * width + deltaX)], axis=1))

        labelDiff = (np.expand_dims(proposals[partial_nodes], -1) != np.expand_dims(proposals[partial_nodes + (deltaY * width + deltaX)], 1)).astype(np.float32)
        colorDiff = np.sum(pow(colors[partial_nodes] - colors[partial_nodes + (deltaY * width + deltaX)], 2), axis=1)
        pairwise_cost = labelDiff * np.reshape(1 + 45 * np.exp(-colorDiff / intensityDifference), [-1, 1, 1])
        #pairwise_cost = np.expand_dims(pairwise_matrix, 0) * np.ones(np.reshape(1 + 45 * np.exp(-colorDiff / np.maximum(intensityDifference[partial_nodes], 1e-4)), [-1, 1, 1]).shape)
        edges_features.append(-pairwise_cost)
        continue

    edges = np.concatenate(edges, axis=0)
    edges_features = np.concatenate(edges_features, axis=0)

    refined_segmentation = inference_ogm(unaries * 10, edges_features, edges, return_energy=False, alg='trw')
    #print(pairwise_matrix)
    #refined_segmentation = inference_ogm(unaries * 5, -pairwise_matrix, edges, return_energy=False, alg='alphaexp')
    refined_segmentation = refined_segmentation.reshape([height, width, 1])    
    refined_segmentation = readProposalInfo(proposals, refined_segmentation)
    refined_segmentation = refined_segmentation.reshape([height, width])
    refined_segmentation[refined_segmentation == numPlanes] = numOutputPlanes
    return refined_segmentation


def removeSmallSegments(planes, image, depth, normal, segmentation, semantics, info, numPlanes, planeAreaThreshold = 100, useAllEmpty=False):
    from skimage import measure
    
    numOutputPlanes = planes.shape[0]
    height = depth.shape[0]
    width = depth.shape[1]
    
    validDepthMask = (depth > 1e-4).astype(np.float32)

    camera = getCameraFromInfo(info)
    urange = (np.arange(width, dtype=np.float32) / (width) * (camera['width']) - camera['cx']) / camera['fx']
    urange = urange.reshape(1, -1).repeat(height, 0)
    vrange = (np.arange(height, dtype=np.float32) / (height) * (camera['height']) - camera['cy']) / camera['fy']
    vrange = vrange.reshape(-1, 1).repeat(width, 1)
    
    X = depth * urange
    Y = depth
    Z = -depth * vrange

    points = np.stack([X, Y, Z], axis=2)
    
    
    planeMap = []
    planeMasks = []
    planeScopes = []    
    emptyMask = segmentation == numOutputPlanes

    newSegmentation = np.ones((height, width), dtype=np.uint8) * numOutputPlanes
    for planeIndex in xrange(numPlanes):
        planeMask = segmentation == planeIndex
        planeMaskDilated = cv2.dilate(planeMask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)).astype(np.bool)
        planeMaskDilated = planeMaskDilated
        
        components = measure.label(planeMaskDilated, background=0)
        isValid = False
        for label in xrange(components.max() + 1):
            mask = components == label
            maskEroded = cv2.erode(mask.astype(np.float32), np.ones((3, 3), dtype=np.float32), iterations=2)
            #print((planeIndex, maskEroded.sum()))
            if maskEroded.sum() < planeAreaThreshold:
                mask = np.logical_and(mask, planeMask)
                emptyMask[mask] = True
                planeMask = np.logical_and(planeMask, np.logical_not(mask))
                #     cv2.imwrite('test/component_' + str(label) + '.png', drawMaskImage(mask))  
                #     cv2.imwrite('test/component_' + str(label) + '_removed.png', drawMaskImage(planeMask))
                #     pass
            else:
                isValid = True
                pass
            continue
        planeScope = planeMask.astype(np.float32)
        planeMask = np.logical_and(planeMask, validDepthMask)
        if not isValid or planeMask.sum() < planeAreaThreshold:
            continue
        
        newSegmentation[planeMask] = len(planeMap)
        planeMap.append(planeIndex)
        planeMasks.append(planeMask)
        
        #semantic = np.bincount(semantics[planeMask]).argmax()
        #planeScope = np.logical_and(np.logical_or(semantics == semantic, semantics == 0), planeMaskDilated).astype(np.float32)
        planeScope = cv2.dilate(planeScope, np.ones((3, 3), dtype=np.float32), iterations=3)
        planeScopes.append(planeScope)

        # if planeIndex == 9:
        #     cv2.imwrite('test/mask_' + str(len(planeMap) - 1) + '_dilated.png', drawMaskImage(planeMaskDilated))
        #     cv2.imwrite('test/mask_' + str(len(planeMap) - 1) + '.png', drawMaskImage(planeMask))
        #     exit(1)
                
        continue
    if len(planeMap) == 0:
        return np.zeros((numOutputPlanes, 3)), np.ones((height, width)) * numOutputPlanes
    
    #planeMap = one_hot(np.array(planeMap), depth=planes.shape[0])
    #planes = np.matmul(planeMap, planes)
    planes = []
    for planeIndex, planeMask in enumerate(planeMasks):
        plane = fitPlane(points[planeMask])
        plane = plane / pow(np.linalg.norm(plane), 2)
        planes.append(plane)
        # if planeIndex == 1:
        #     print(points[planeMask])
        #     print(plane)
        #     cv2.imwrite('test/mask_' + str(planeIndex) + '.png', drawMaskImage(planeMask))
        #     exit(1)
        continue
    
    planes = np.array(planes)
    #print(planes.shape)
    numPlanes = planes.shape[0]
    planeMasks = np.stack(planeMasks, 2)
    planeScopes = np.stack(planeScopes, 2)

    if useAllEmpty:
        #planeScopes = np.expand_dims(emptyMask, -1)
        planeScopes = np.ones((height, width, 1))
        pass

    planes = planes[:numPlanes]
    planesD = np.linalg.norm(planes, axis=1, keepdims=True)
    planeNormals = planes / np.maximum(planesD, 1e-4)

    distanceThreshold = 0.1
    distance = np.abs(np.tensordot(points, planeNormals, axes=([2, 1])) - np.reshape(planesD, [1, 1, -1]))
    #invalidFittingMask = (distance.reshape((-1, numPlanes))[np.arange(width * height), newSegmentation.reshape(-1)] > distanceThreshold).reshape((height, width))
    invalidFittingMask = np.sum(distance * planeMasks, axis=2) > distanceThreshold
    #invalidFittingMask = np.argmin(distance, axis=-1) != newSegmentation
    emptyMask += np.logical_and(invalidFittingMask, validDepthMask)
    distance = distance + (1 - planeScopes) * 10000
    distance = np.concatenate([distance, np.expand_dims(emptyMask.astype(np.float32), -1) * distanceThreshold], axis=2)
    #cv2.imwrite('test/mask.png', drawMaskImage(emptyMask))
    filledSegmentation = np.argmin(distance[emptyMask], axis=-1)
    filledSegmentation[filledSegmentation == numPlanes] = numOutputPlanes
    newSegmentation[emptyMask] = filledSegmentation
    
    planes = np.concatenate([planes, np.zeros((numOutputPlanes - planes.shape[0], 3))], axis=0)

    return planes, newSegmentation, numPlanes
