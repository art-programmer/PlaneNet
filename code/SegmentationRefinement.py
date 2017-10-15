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

def readProposalInfo(info, proposals):
    numProposals = proposals.shape[-1]
    outputShape = list(info.shape)
    outputShape[-1] = numProposals
    info = info.reshape([-1, info.shape[-1]])
    proposals = proposals.reshape([-1, proposals.shape[-1]])
    proposalInfo = []
    for proposal in xrange(numProposals):
        proposalInfo.append(info[np.arange(info.shape[0]), proposals[:, proposal]])
        continue
    proposalInfo = np.stack(proposalInfo, axis=1).reshape(outputShape)
    return proposalInfo

def refineSegmentation(allSegmentations, allDepths, boundaries, numOutputPlanes = 20, numIterations=20, numProposals=5):
    height = allSegmentations.shape[0]
    width = allSegmentations.shape[1]

    #allSegmentations = np.concatenate([planeSegmentations, nonPlaneSegmentation], axis=2)
    #allDepths = np.concatenate([planeDepths, nonPlaneDepth], axis=2)
    proposals = findProposals(allSegmentations, numProposals=numProposals)
    proposalSegmentations = readProposalInfo(allSegmentations, proposals)
    proposalDepths = readProposalInfo(allDepths, proposals)

    maxDepthDiff = 0.3
    unaries = proposalSegmentations.reshape((-1, numProposals))
    nodes = np.arange(height * width).reshape((height, width))
    #deltas = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    x = 150
    y = 125

    deltas = [(0, 1), (1, 0), (-1, 1), (1, 1)]
    edges = []
    edges_features = []
    for delta in deltas:
        deltaX = delta[0]
        deltaY = delta[1]
        partial_nodes = nodes[max(-deltaY, 0):min(height - deltaY, height), max(-deltaX, 0):min(width - deltaX, width)].reshape(-1)
        edges.append(np.stack([partial_nodes, partial_nodes + (deltaY * width + deltaX)], axis=1))
        depth_1_1 = proposalDepths[max(-deltaY, 0):min(height - deltaY, height), max(-deltaX, 0):min(width - deltaX, width)].reshape((-1, numProposals))
        depth_2_2 = proposalDepths[max(deltaY, 0):min(height + deltaY, height), max(deltaX, 0):min(width + deltaX, width)].reshape((-1, numProposals))
        depth_diff = np.expand_dims(depth_1_1, -1) - np.expand_dims(depth_2_2, 1)
        depth_diff = np.clip(np.abs(depth_diff) / maxDepthDiff, 0, 1)

        label_1 = proposals[max(-deltaY, 0):min(height - deltaY, height), max(-deltaX, 0):min(width - deltaX, width)].reshape((-1, numProposals))
        label_2 = proposals[max(deltaY, 0):min(height + deltaY, height), max(deltaX, 0):min(width + deltaX, width)].reshape((-1, numProposals))
        label_diff = np.expand_dims(label_1, -1) - np.expand_dims(label_2, 1)
        label_diff = np.clip(np.abs(label_diff), 0, 1)

        smooth_boundary = np.maximum(boundaries[max(-deltaY, 0):min(height - deltaY, height), max(-deltaX, 0):min(width - deltaX, width), 0].reshape(-1), boundaries[max(deltaY, 0):min(height + deltaY, height), max(deltaX, 0):min(width + deltaX, width), 0].reshape((-1)))
        occlusion_boundary = np.maximum(boundaries[max(-deltaY, 0):min(height - deltaY, height), max(-deltaX, 0):min(width - deltaX, width), 1].reshape(-1), boundaries[max(deltaY, 0):min(height + deltaY, height), max(deltaX, 0):min(width + deltaX, width), 1].reshape((-1)))
        pairwise_cost = label_diff * depth_diff * smooth_boundary.reshape((-1, 1, 1)) + label_diff * np.clip(1 - smooth_boundary - occlusion_boundary, 0.1, 1).reshape((-1, 1, 1))
        edges_features.append(-pairwise_cost)
        #print((edges[-1][partial_nodes == y * width + x][0][1] % width, edges[-1][partial_nodes == y * width + x][0][1] / width))
        #print(edges_features[-1][partial_nodes == y * width + x])
        mask = edges[-1][:, 1] == y * width + x
        #print((edges[-1][mask][0][0] % width, edges[-1][mask][0][0] / width))
        #print(edges_features[-1][mask])
        continue

    edges = np.concatenate(edges, axis=0)
    edges_features = np.concatenate(edges_features, axis=0)
    refined_segmentation = inference_ogm(unaries, edges_features * 5, edges, return_energy=False, alg='trw')
    #refined_segmentation = inference_dispatch(unaries, edges_features * 0, edges, ('unary'))
    #refined_segmentation = np.argmin(unaries, axis=1)

    refined_segmentation = refined_segmentation.reshape([height, width, 1])
    refined_segmentation = readProposalInfo(proposals, refined_segmentation)
    refined_segmentation = refined_segmentation.reshape([height, width])

    print((x, y))
    print(proposals[y][x])
    print((x + 1, y))
    print(proposals[y][x + 1])
    print((x - 1, y + 1))
    print(proposals[y + 1][x - 1])
    print((x, y + 1))
    print(proposals[y + 1][x])
    print((x + 1, y + 1))
    print(proposals[y + 1][x + 1])
    print(unaries[y * width + x])
    print(refined_segmentation[y][x])
    return refined_segmentation
    #return proposals.reshape([height, width])


def getSegmentationsGraphCut(planes, image, depth, normal, info, numPlanes):
    numOutputPlanes = planes.shape[0]
    height = depth.shape[0]
    width = depth.shape[1]

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
    planesD = np.linalg.norm(planes, axis=1, keepdims=True)
    planeNormals = planes / np.maximum(planesD, 1e-4)

    distanceCostThreshold = 0.05

    distanceCost = 1 - np.exp(-np.abs(np.tensordot(points, planeNormals, axes=([2, 1])) - np.reshape(planesD, [1, 1, -1])) / distanceCostThreshold)
    distanceCost[:, :, numPlanes:] = 10000
    distanceCost = np.concatenate([distanceCost, np.ones((height, width, 1)) * (1 - np.exp(-1))], axis=2)
    normalCost = 0
    if info[19] <= 1:
        normalCostThreshold = 1 - np.cos(20)        
        normalCost = (1 - np.tensordot(normal, planeNormals, axes=([2, 1]))) / normalCostThreshold
        normalCost[:, :, numPlanes:] = 10000
        normalCost = np.concatenate([normalCost, np.ones((height, width, 1))], axis=2)
        pass

    unaryCost = distanceCost + normalCost
    unaryCost *= np.expand_dims((depth > 1e-4).astype(np.float32), -1)
    unaries = -unaryCost.reshape((-1, numOutputPlanes + 1))

    # print(distanceCost[60][150])
    # print(unaryCost[60][150])
    # print(np.argmax(-unaryCost[60][150]))

    #cv2.imwrite('test/segmentation.png', drawSegmentationImage(-unaryCost, blackIndex=numOutputPlanes))
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
    pairwise_matrix = 1 - np.diag(np.ones(numOutputPlanes + 1))

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
