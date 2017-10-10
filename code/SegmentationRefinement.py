import numpy as np
from pystruct.inference import get_installed, inference_ogm, inference_dispatch

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
