import numpy as np
from pystruct.inference import get_installed, inference_ogm, inference_dispatch
from utils import *

NUM_LAYERS = 3
NUM_PLANES = 15

def getConcaveHullProposal(solution, segmentations, planeDepths):
    height = solution.shape[0]
    width = solution.shape[1]
    
    LEFT_WALLS = [0, 5, 6, 11, 18]
    RIGHT_WALLS = [4, 10, 7, 19]
    FLOORS = [14]
    CEILINGS = []    
    LAYOUT_PLANES = [CEILINGS, FLOORS, LEFT_WALLS + RIGHT_WALLS]

    planeAreaThresholds = [WIDTH * HEIGHT / 400, WIDTH * HEIGHT / 400, WIDTH * HEIGHT / 400]

    layoutPlanes = []
    for layoutIndex, planeInds in enumerate(LAYOUT_PLANES):
        planeCandidates = []
        for planeIndex in planeInds:
            area = (segmentations[:, :, planeIndex]).sum()
            #area = (segmentation == planeIndex).sum()
            if area > planeAreaThresholds[layoutIndex]:
                planeCandidates.append(planeIndex)
                pass
            continue
        if len(planeCandidates) == 0:
            continue
        numSelectedPlanes = min(int(layoutIndex == 2) * 2 + 1, len(planeCandidates))
        planeInds = np.random.choice(planeCandidates, numSelectedPlanes, replace=np.random.random() > 0.5)
        selectedPlanes += planeInds.tolist()
        continue
    layoutPlanes = np.array(selectedPlanes)
    layoutPlaneDepths = planeDepths[:, :, layoutPlanes]
    layoutSegmentation = np.argmin(layoutPlaneDepths, axis=-1)
    #remainingPlaneInds = np.arange(numPlanes).tolist()
    proposal = solution.copy()
    backgroundMask = np.zeros(proposal.shape, dtype=np.bool)
    for layoutIndex, planeIndex in enumerate(layoutPlanes):
        layoutSegmentation[layoutSegmentation == layoutIndex] = planeIndex
        for layer in xrange(1, NUM_LAYERS):
            proposal[:, :, layer][proposal[:, :, layer] == planeIndex] = NUM_PLANES
            continue
        backgroundMask = np.logical_or(backgroundMask, proposal[0] == planeIndex)
        #remainingPlaneInds.remove(planeIndex)
        continue
    backgroundMask = np.logical_not(backgroundMask).astype(np.int32)
    if backgroundMask.sum() > 0:
        for layer in xrange(NUM_LAYERS - 1, 0, -1):
            proposal[:, :, layer][backgroundMask] = proposal[:, :, layer - 1][backgroundMask]
            continue
        pass
    proposal[0] = layoutSegmentation
    return proposal

    
def getProposals(solution, planes, segmentation, segmentations, planeDepths, iteration):
    height = solution.shape[1]
    width = solution.shape[2]
    
    numProposals = 3
    if iteration == 0:
        proposal = np.full((height, width, NUM_LAYERS), NUM_PLANES)
        proposal[:, :, 0] = segmentation
        return [proposal, ]
    elif iteration == 1:
        return [solution, getConcaveHullProposal()]
    else:
        return
    
def decompose(image, depth, normal, info, planes, segmentation):
    numPlanes = planes.shape[0]
    NUM_PLANES = max(NUM_PLANES, numPlanes)
    segmentation[segmentation == numPlanes] = NUM_PLANES
    
    height = depth.shape[0]
    width = depth.shape[1]
    segmentations = (np.expand_dims(segmentation, -1) == np.arange(numPlanes).reshape([1, 1, -1])).astype(np.float32)
    
    planeDepths = calcPlaneDepths(planes, width, height, info)    
    allDepths = np.concatenate([planeDepths, np.zeros(height, width, 1)], axis=2)
    
    camera = getCameraFromInfo(info)
    urange = (np.arange(width, dtype=np.float32) / (width) * (camera['width']) - camera['cx']) / camera['fx']
    urange = urange.reshape(1, -1).repeat(height, 0)
    vrange = (np.arange(height, dtype=np.float32) / (height) * (camera['height']) - camera['cy']) / camera['fy']
    vrange = vrange.reshape(-1, 1).repeat(width, 1)
    
    X = depth * urange
    Y = depth
    Z = -depth * vrange


    normals = normal.reshape((-1, 3))
    normals = normals / np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-4)
    validMask = np.logical_and(np.linalg.norm(normals, axis=-1) > 1e-4, depth.reshape(-1) > 1e-4)
    
    points = np.stack([X, Y, Z], axis=2).reshape(-1, 3)
    
    planesD = np.linalg.norm(planes, axis=1, keepdims=True)
    planeNormals = planes / np.maximum(planesD, 1e-4)

    distanceCostThreshold = 0.1
    distanceCost = np.abs(np.tensordot(points, planeNormals, axes=([1, 1])) - np.reshape(planesD, [1, -1])) / distanceCostThreshold

    #valid_normals = normals[validMask]
    normalCostThreshold = 1 - np.cos(np.deg2rad(30))        
    normalCost = (1 - np.abs(np.tensordot(normals, planeNormals, axes=([1, 1])))) / normalCostThreshold

    normalWeight = 1    
    
    unaryCost = distanceCost + normalCost * normalWeight
    unaryCost *= np.expand_dims(validMask.astype(np.float32), -1)    
    #unaryCost = unaryCost.reshape((height * width, -1))


    image = image.astype(np.float32)
    colors = image.reshape((-1, 3))
    deltas = [(0, 1), (1, 0)]    
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
    maxDepthDiff = 0.3
    depthGap = 0.05
    
    solution = []
    for iteration in xrange(2):
        proposals = getProposals(solution, planes, segmentation, segmentations, planeDepths, iteration)
        numProposals = len(proposals)
        if numProposals == 1:
            solution = proposals[0]
            continue
        
        visibleSegmentations = []
        for proposal in proposals:
            visibleSegmentation = proposal[0]
            for layer in xrange(1, NUM_LAYERS):
                mask = proposal[:, :, layer] < NUM_PLANES
                visibleSegmentation[mask] = proposal[:, :, layer][mask]
                continue
            visibleSegmentations.append(visibleSegmentation)
            continue
        visibleSegmentations = np.stack(visibleSegmentations, axis=2).reshape((-1, numProposals))
        unaries = -readProposalInfo(unaryCost, visibleSegmentations)

        proposalDepths = []
        for proposal in proposals:
            proposalDepths.append(readProposalInfo(proposalDepths, proposal))
            continue
        proposalDepths = np.stack(proposalDepths, axis=-1)
        proposalDepths.reshape((width * height, NUM_LAYERS, numProposals))

        conflictDepthMask = np.zeros((width * height, numProposals), dtype=np.bool)
        for layer in xrange(1, NUM_LAYERS):
            conflictDepthMask = np.logical_or(conflictDepthMask, proposalDepths[:, layer, :] > proposalDepths[:, layer - 1, :] + depthGap)
            continue
        unaries += conflictDepthMask.astype(np.float32) * 100
        
        proposals = np.stack(proposals, axis=-1).reshape((width * height, NUM_LAYERS, numProposals))
        #cv2.imwrite('test/segmentation.png', drawSegmentationImage(unaries.reshape((height, width, -1)), blackIndex=numOutputPlanes))

        nodes = np.arange(height * width).reshape((height, width))

    
        edges = []
        edges_features = []

        for delta in deltas:
            deltaX = delta[0]
            deltaY = delta[1]
            partial_nodes = nodes[max(-deltaY, 0):min(height - deltaY, height), max(-deltaX, 0):min(width - deltaX, width)].reshape(-1)
            edges.append(np.stack([partial_nodes, partial_nodes + (deltaY * width + deltaX)], axis=1))

            
            #pairwise_cost = np.zeros((partial_nodes.shape[0], numProposals, numProposals))
            
            labelDiff = (np.expand_dims(proposals[partial_nodes], -1) != np.expand_dims(proposals[partial_nodes + (deltaY * width + deltaX)], -2)).astype(np.float32)

            depth_1 = np.expand_dims(proposalDepths[partial_nodes], -1)
            depth_2 = np.expand_dims(proposalDepths[partial_nodes + (deltaY * width + deltaX)], -2)
            emptyMask = np.logical_or(np.logical_and(depth_1 > 1e-4, depth_2 < 1e-4), np.logical_and(depth_1 < 1e-4, depth_2 > 1e-4))
            
            depthDiff = np.abs(depth_1 - depth_2) / maxDepthDiff * (1 - emptyMask) + emptyMask * 0.01

            visibleLabelDiff = (np.expand_dims(visibleSegmentations[partial_nodes], -1) != np.expand_dims(visibleSegmentations[partial_nodes + (deltaY * width + deltaX)], -2)).astype(np.float32)
            colorDiff = np.sum(pow(colors[partial_nodes] - colors[partial_nodes + (deltaY * width + deltaX)], 2), axis=1)
            #depth_diff = np.clip(np.abs(depth_diff) / maxDepthDiff, 0, 1)
            #depth_2_2 = proposalDepths[max(deltaY, 0):min(height + deltaY, height), max(deltaX, 0):min(width + deltaX, width)].reshape((-1, numProposals))
            
            pairwise_cost = (labelDiff * depthDiff).sum(1) + visibleLabelDiff * np.reshape(0.02 + np.exp(-colorDiff / intensityDifference), [-1, 1, 1])
             np.reshape(1 + 45 * np.exp(-colorDiff / intensityDifference), [-1, 1, 1])
            #pairwise_cost = np.expand_dims(pairwise_matrix, 0) * np.ones(np.reshape(1 + 45 * np.exp(-colorDiff / np.maximum(intensityDifference[partial_nodes], 1e-4)), [-1, 1, 1]).shape)
            edges_features.append(-pairwise_cost)
            continue
        
        edges = np.concatenate(edges, axis=0)
        edges_features = np.concatenate(edges_features, axis=0)

        solution = inference_ogm(unaries * 10, edges_features, edges, return_energy=False, alg='trw')
        solution = np.tile(solution.reshape([height * width, 1, 1]), [1, NUM_LAYERS, 1])
        solution = readProposalInfo(proposals, solution).reshape((height, width, NUM_LAYERS))
        continue
    
    return solution


