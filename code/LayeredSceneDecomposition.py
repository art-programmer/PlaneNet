import numpy as np
from pystruct.inference import get_installed, inference_ogm, inference_dispatch
from utils import *

def drawSolution(layeredSegmentations, numPlanes, index):
    for layer in xrange(layeredSegmentations.shape[-1]):
        segmentation = layeredSegmentations[:, :, layer]
        if index >= 0:
            cv2.imwrite('test/' + str(index) + '_segmentation_' + str(layer) + '.png', drawSegmentationImage(segmentation, blackIndex=numPlanes))
        else:
            cv2.imwrite('test/segmentation_' + str(layer) + '.png', drawSegmentationImage(segmentation, blackIndex=numPlanes))
            pass
        continue
    return

def getConcaveHullProposal(solution, depth, segmentations, planeDepths, NUM_LAYERS=3, NUM_PLANES=15, height=192, width=256):
    layoutPlanes = []
    if False:
        LEFT_WALLS = [0, 5, 6, 11, 18]
        RIGHT_WALLS = [4, 10, 7, 19]
        FLOORS = [14]
        CEILINGS = []    
        LAYOUT_PLANES = [CEILINGS, FLOORS, LEFT_WALLS + RIGHT_WALLS]

        planeAreaThresholds = [WIDTH * HEIGHT / 400, WIDTH * HEIGHT / 400, WIDTH * HEIGHT / 400]

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
            layoutPlanes += planeInds.tolist()
            continue    
    else:
        conflictAreaThreshold = width * height / 100
        depthGap = 0.05
        bestPlaneIndex = -1
        minNumConflict = width * height
        for planeIndex in xrange(planeDepths.shape[-1]):
            planeDepth = planeDepths[:, :, planeIndex]
            numConflicts = np.logical_and(np.logical_and(planeDepth < depth - depthGap, depth > 1e-4), planeDepth > 1e-4).sum()
            if numConflicts < conflictAreaThreshold:
                layoutPlanes.append(planeIndex)
                pass
            if numConflicts <= minNumConflict:
                minNumConflict = numConflicts
                bestPlaneIndex = planeIndex
                pass
            continue
        if len(layoutPlanes) == 0:
            layoutPlanes.append(bestPlaneIndex)
            pass
        layoutPlanes = np.random.choice(layoutPlanes, min(5 - int(np.floor(pow(np.random.random(), 2) * 5)), len(layoutPlanes)), replace=False)
        pass


    layoutPlanes = np.array(layoutPlanes)
    layoutPlaneDepths = planeDepths[:, :, layoutPlanes]
    layoutSegmentation = np.argmin(layoutPlaneDepths, axis=-1)
    
    #remainingPlaneInds = np.arange(numPlanes).tolist()
    proposal = np.split(solution.copy(), NUM_LAYERS, axis=2)
    proposal = [np.squeeze(v) for v in proposal]

    backgroundMask = np.zeros((height, width), dtype=np.bool)
    newLayoutSegmentation = layoutSegmentation.copy()
    for layoutIndex, planeIndex in enumerate(layoutPlanes):
        print(layoutIndex, planeIndex)
        newLayoutSegmentation[layoutSegmentation == layoutIndex] = planeIndex
        for layer in xrange(1, NUM_LAYERS):
            proposal[layer][proposal[layer] == planeIndex] = NUM_PLANES
            continue
        backgroundMask = np.logical_or(backgroundMask, proposal[0] == planeIndex)
        #remainingPlaneInds.remove(planeIndex)
        continue

    
    backgroundMask = np.logical_not(backgroundMask)
    if backgroundMask.sum() > 0:
        for layer in xrange(NUM_LAYERS - 1, 0, -1):
            proposal[layer][backgroundMask] = proposal[layer - 1][backgroundMask]
            continue
        pass

    # cv2.imwrite('test/segmentation_0.png', drawSegmentationImage(layoutSegmentation, blackIndex=numPlanes))    
    # cv2.imwrite('test/segmentation_2.png', drawSegmentationImage(solution[:, :, 0], blackIndex=numPlanes))    
    # cv2.imwrite('test/segmentation_1.png', drawSegmentationImage(proposal[1], blackIndex=numPlanes))    
    # exit(1)

    proposal[0] = newLayoutSegmentation
    proposal = np.stack(proposal, axis=2)
    return proposal

def getExpansionProposals(solution, depth, segmentations, planeDepths, NUM_LAYERS=3, NUM_PLANES=15, height=192, width=256):
 
    layoutPlanes = np.unique(solution[:, :, 0])
    if NUM_PLANES in layoutPlanes:
        assert(False)
        #layoutPlanes = np.delete(layoutPlanes, np.argwhere(layoutPlanes == NUM_PLANES))   
        pass

    #print((solution[:, :, 0] == 2).sum())
    #drawSolution(solution, NUM_PLANES, -1)
    #exit(1)
    
    # layoutDepth = np.zeros((height, width))
    # for layoutPlane in layoutPlanes:
    #     mask = solution[:, :, 0] == layoutPlane
    #     layoutDepth[mask] = planeDepths[:, :, layoutPlane][mask]
    #     continue

    # depthGap = 0.1
    # smoothLayout = True
    # if len(layoutPlanes) > 1:
    #     smoothLayout = ((np.abs(layoutDepth[1:] - layoutDepth[:-1]) > depthGap).sum() + (np.abs(layoutDepth[:, 1:] - layoutDepth[:, :-1]) > depthGap).sum()) > 0
    #     pass

    # print(smoothLayout)
    
    # if smoothLayout == True:
    #     if len(layoutPlanes) == NUM_PLANES:
    #         return []
    #     while True:
    #         selectedPlaneIndex = np.random.randint(NUM_PLANES)
    #         if selectedPlaneIndex in layoutPlanes:
    #             continue
    #         break
    # else:
    if True:
        #selectedPlaneIndex = np.random.randint(NUM_PLANES)
        #selectedPlaneIndex = 3
        layoutProposal = getConcaveHullProposal(solution, depth, segmentations, planeDepths, NUM_LAYERS=3, NUM_PLANES=15, height=192, width=256)
        validLayoutPlanes = np.unique(layoutProposal[:, :, 0])


        if len(validLayoutPlanes) == len(layoutPlanes):
            while True:
                selectedPlaneIndex = np.random.randint(NUM_PLANES)
                if selectedPlaneIndex in layoutPlanes:
                    continue
                break
            smoothLayout = True
        else:
            invalidLayoutPlanes = []
            for planeIndex in layoutPlanes:
                if planeIndex not in validLayoutPlanes:
                    invalidLayoutPlanes.append(planeIndex)
                    pass
                continue
            selectedPlaneIndex = np.random.choice(invalidLayoutPlanes)

            print('expand layout plane', selectedPlaneIndex)
            proposals = [layoutProposal, ]
            for layer in xrange(1, NUM_LAYERS):
                proposal = np.split(layoutProposal.copy(), NUM_LAYERS, axis=2)    
                proposal = [np.squeeze(v) for v in proposal]

                for otherLayer in xrange(1, NUM_LAYERS):
                    if otherLayer == layer:
                        continue
                    mask = proposal[otherLayer] == selectedPlaneIndex
                    if mask.sum() == 0:
                        continue
                    proposal[otherLayer][mask] = NUM_PLANES
                    continue
                proposal[layer].fill(selectedPlaneIndex)
                proposals.append(np.stack(proposal, axis=-1))
                continue
            return proposals
        pass

    print('expand plane', selectedPlaneIndex)
    
    
    proposals = []
    for layer in xrange(NUM_LAYERS):
        proposal = np.split(solution.copy(), NUM_LAYERS, axis=2)    
        proposal = [np.squeeze(v) for v in proposal]

        for otherLayer in xrange(NUM_LAYERS):
            if otherLayer == layer:
                continue
            mask = proposal[otherLayer] == selectedPlaneIndex
            if mask.sum() == 0:
                continue
            if otherLayer == 0:
                layoutPlanes = np.delete(layoutPlanes, np.argwhere(layoutPlanes == selectedPlaneIndex))
                layoutPlaneDepths = planeDepths[:, :, layoutPlanes]
                layoutSegmentation = np.argmin(layoutPlaneDepths, axis=-1)
                newLayoutSegmentation = layoutSegmentation.copy()
                for layoutIndex, planeIndex in enumerate(layoutPlanes):
                    newLayoutSegmentation[layoutSegmentation == layoutIndex] = planeIndex
                    continue
                proposal[0] = newLayoutSegmentation
            else:
                proposal[otherLayer][mask] = NUM_PLANES
                pass
            continue
        proposal[layer].fill(selectedPlaneIndex)
        proposals.append(np.stack(proposal, axis=-1))
        continue
    
    return proposals

def getLayerSwapProposals(solution, NUM_LAYERS=3, NUM_PLANES=15, height=192, width=256):
    proposals = []
    for layer in xrange(1, NUM_LAYERS):
        proposal = np.split(solution.copy(), NUM_LAYERS, axis=2)    
        proposal = [np.squeeze(v) for v in proposal]

        layerPlanes = np.unique(proposal[layer])
        if len(layerPlanes) == 0:
            continue
        selectedPlaneIndex = np.random.choice(layerPlanes)
        while True:
            otherLayer = np.random.randint(NUM_LAYERS)
            if otherLayer == layer:
                continue
            break
        
        if layer != 1:
            continue
        else:
            selectedPlaneIndex = 2
            otherLayer = 2
            pass

            
        mask = proposal[layer] == selectedPlaneIndex
        proposal[layer][mask] = NUM_PLANES
        proposal[otherLayer][mask] = selectedPlaneIndex
        proposals.append(np.stack(proposal, axis=-1))
        continue
    
    return proposals
    

    
def getProposals(solution, planes, segmentation, segmentations, planeDepths, iteration, NUM_LAYERS=3, NUM_PLANES=15, height=192, width=256):

    numProposals = 3
    if iteration == 0:
        proposal = np.full((height, width, NUM_LAYERS), NUM_PLANES)
        proposal[:, :, 0] = segmentation
        return [proposal, ]
    elif iteration == 1:
        return [solution, getConcaveHullProposal(solution, depth, segmentations, planeDepths, NUM_LAYERS, NUM_PLANES, height, width)]
    elif iteration == 2:
        return [solution] + getExpansionProposals(solution, depth, segmentations, planeDepths, NUM_LAYERS, NUM_PLANES, height, width)
    elif iteration == 3:
        return [solution] + getLayerSwapProposals(solution, NUM_LAYERS, NUM_PLANES, height, width)
    elif iteration == 4:
        return [solution] + getExpansionProposals(solution, depth, segmentations, planeDepths, NUM_LAYERS, NUM_PLANES, height, width)
    elif iteration == 5:
        return [solution] + getLayerSwapProposals(solution, NUM_LAYERS, NUM_PLANES, height, width)
    return
    
def decompose(image, depth, normal, info, planes, segmentation):
    NUM_PLANES = planes.shape[0]
    #segmentation[segmentation == numPlanes] = 
    NUM_LAYERS = 3
    
    height = depth.shape[0]
    width = depth.shape[1]
    
    segmentations = (np.expand_dims(segmentation, -1) == np.arange(NUM_PLANES).reshape([1, 1, -1])).astype(np.float32)
    
    planeDepths = calcPlaneDepths(planes, width, height, info)

    # for planeIndex in xrange(NUM_PLANES):
    #     cv2.imwrite('test/depth_' + str(planeIndex) + '.png', drawDepthImage(planeDepths[:, :, planeIndex]))
    #     continue
    
    allDepths = np.concatenate([planeDepths, np.zeros((height, width, 1))], axis=2)
    
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
    unaryCost = np.concatenate([unaryCost, np.full((unaryCost.shape[0], 1), 100)], axis=1)
    #unaryCost = unaryCost.reshape((height * width, -1))


    nodes = np.arange(height * width).reshape((height, width))    
    
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
    for iteration in xrange(6):
        if os.path.exists('test/solution_' + str(iteration) + '.npy') and iteration <= 4:
            solution = np.load('test/solution_' + str(iteration) + '.npy')
            continue

        proposals = getProposals(solution, planes, segmentation, segmentations, planeDepths, iteration, NUM_LAYERS=NUM_LAYERS, NUM_PLANES=NUM_PLANES, height=height, width=width)

        numProposals = len(proposals)

        if numProposals == 1:
            solution = proposals[0]
            continue
        
        proposals = [proposals[0]]
        numProposals = len(proposals)
                
        for proposalIndex, proposal in enumerate(proposals):
            drawSolution(proposal, NUM_PLANES, proposalIndex)
            continue

        #if iteration == 2:
        #exit(1)
            
        visibleSegmentations = []
        for proposal in proposals:
            visibleSegmentation = proposal[:, :, 0].copy()
            for layer in xrange(1, NUM_LAYERS):
                mask = proposal[:, :, layer] < NUM_PLANES
                visibleSegmentation[mask] = proposal[:, :, layer][mask]
                continue
            visibleSegmentations.append(visibleSegmentation)
            continue
        visibleSegmentations = np.stack(visibleSegmentations, axis=-1).reshape((-1, numProposals))

        #cv2.imwrite('test/segmentation_0.png', drawSegmentationImage(visibleSegmentations[:, 0].reshape((height, width)), blackIndex=NUM_PLANES))
        #cv2.imwrite('test/segmentation_1.png', drawSegmentationImage(visibleSegmentations[:, 1].reshape((height, width)), blackIndex=NUM_PLANES))        
        
        unaries = readProposalInfo(unaryCost, visibleSegmentations)

        proposalDepths = []
        for proposal in proposals:
            proposalDepths.append(readProposalInfo(allDepths, proposal))
            continue
        proposalDepths = np.stack(proposalDepths, axis=-1)
        proposalDepths = proposalDepths.reshape((width * height, NUM_LAYERS, numProposals))

        conflictDepthMask = np.zeros((width * height, numProposals), dtype=np.bool)
        for layer in xrange(1, NUM_LAYERS):
            conflictDepthMask = np.logical_or(conflictDepthMask, np.logical_and(proposalDepths[:, layer, :] > proposalDepths[:, layer - 1, :] + depthGap, proposalDepths[:, layer - 1, :] > 1e-4))
            continue
        unaries += conflictDepthMask.astype(np.float32) * 100
        
        #cv2.imwrite('test/mask_0.png', drawMaskImage((unaries[:, 1] - unaries[:, 0]).reshape((height, width)) / 2 + 0.5))
        #exit(1)
        #print((unaries[:, 1] - unaries[:, 0]).min())        

        
        #print((unaries[:, 1] - unaries[:, 0]).max())
        #print((unaries[:, 1] - unaries[:, 0]).min())        
        
        proposals = np.stack(proposals, axis=-1).reshape((width * height, NUM_LAYERS, numProposals))

        #empty background cost
        unaries += (proposals[:, 0, :] == NUM_PLANES).astype(np.float32) * 100

        #print((unaries[:, 1] - unaries[:, 0]).max())
        #print((unaries[:, 1] - unaries[:, 0]).min())        
        #exit(1)
        #cv2.imwrite('test/segmentation.png', drawSegmentationImage(unaries.reshape((height, width, -1)), blackIndex=numOutputPlanes))
        #cv2.imwrite('test/mask_0.png', drawMaskImage((unaries[:, 1] - unaries[:, 0]).reshape((height, width)) / 2 + 0.5))
        #exit(1)
        
        edges = []
        edges_features = []

        for deltaIndex, delta in enumerate(deltas):
            deltaX = delta[0]
            deltaY = delta[1]
            partial_nodes = nodes[max(-deltaY, 0):min(height - deltaY, height), max(-deltaX, 0):min(width - deltaX, width)].reshape(-1)
            edges.append(np.stack([partial_nodes, partial_nodes + (deltaY * width + deltaX)], axis=1))

            
            #pairwise_cost = np.zeros((partial_nodes.shape[0], numProposals, numProposals))

            labels_1 = np.expand_dims(proposals[partial_nodes], -1)
            labels_2 = np.expand_dims(proposals[partial_nodes + (deltaY * width + deltaX)], -2)
            labelDiff = (labels_1 != labels_2).astype(np.float32)

            depth_1 = np.expand_dims(proposalDepths[partial_nodes], -1)
            depth_2 = np.expand_dims(proposalDepths[partial_nodes + (deltaY * width + deltaX)], -2)

            visibleLabels_1 = np.expand_dims(visibleSegmentations[partial_nodes], -1)
            visibleLabels_2 = np.expand_dims(visibleSegmentations[partial_nodes + (deltaY * width + deltaX)], -2)
            
            
            emptyMask_1 = np.logical_and(depth_1 > 1e-4, depth_2 < 1e-4)
            emptyMask_2 = np.logical_and(depth_1 < 1e-4, depth_2 > 1e-4)
            emptyMask = np.logical_or(emptyMask_1, emptyMask_2)

            cutMask = (labelDiff - (np.expand_dims(labels_1, -3) != np.expand_dims(labels_2, -4)).astype(np.float32).min(-3)) * (labels_1 < NUM_PLANES).astype(np.float32)

            
            visibleEmptyMask = np.logical_or(np.logical_and(emptyMask_1, (labels_1 == np.expand_dims(visibleLabels_1, -3))), np.logical_and(emptyMask_2, (labels_2 == np.expand_dims(visibleLabels_2, -3))))
            invisibleEmptyMask = np.logical_and(emptyMask, np.logical_not(visibleEmptyMask))
            
            depthDiff = np.abs(depth_1 - depth_2) / maxDepthDiff * (1 - emptyMask) + invisibleEmptyMask * (0.05 / maxDepthDiff)


            visibleLabelDiff = (visibleLabels_1 != visibleLabels_2)
            visibleLabelDiff = np.logical_or(visibleEmptyMask.max(1), visibleLabelDiff).astype(np.float32)
            
            colorDiff = np.sum(pow(colors[partial_nodes] - colors[partial_nodes + (deltaY * width + deltaX)], 2), axis=-1)
            #depth_diff = np.clip(np.abs(depth_diff) / maxDepthDiff, 0, 1)
            #depth_2_2 = proposalDepths[max(deltaY, 0):min(height + deltaY, height), max(deltaX, 0):min(width + deltaX, width)].reshape((-1, numProposals))
            
            pairwise_cost = (labelDiff * depthDiff).sum(1) + visibleLabelDiff * np.reshape(0.02 + np.exp(-colorDiff / intensityDifference), [-1, 1, 1])
            np.reshape(1 + 45 * np.exp(-colorDiff / intensityDifference), [-1, 1, 1])
            #pairwise_cost = np.expand_dims(pairwise_matrix, 0) * np.ones(np.reshape(1 + 45 * np.exp(-colorDiff / np.maximum(intensityDifference[partial_nodes], 1e-4)), [-1, 1, 1]).shape)
            edges_features.append(-pairwise_cost)

            #print(pairwise_cost.shape)
            #print(pairwise_cost.max())

            debug = False
            if debug:
                cv2.imwrite('test/cost_diff.png', drawMaskImage((unaries[:, 1] - unaries[:, 0]).reshape((height, width)) / 2 + 0.5))

                #exit(1)
                if deltaIndex == 0:
                    cv2.imwrite('test/cost_color.png', drawMaskImage((visibleLabelDiff * np.reshape(0.02 + np.exp(-colorDiff / intensityDifference), (-1, 1, 1)))[:, 0, 1].reshape((height - 1, width))))

                    #cv2.imwrite('test/segmentation.png', drawSegmentationImage(proposals[partial_nodes, 0, 1].reshape((height - 1, width))))
                    #diff = (labelDiff * depthDiff).reshape((height - 1, width, NUM_LAYERS, numProposals, numProposals))
                    diff = (invisibleEmptyMask).reshape((height - 1, width, NUM_LAYERS, numProposals, numProposals))
                    #diff = (labelDiff).reshape((height - 1, width, NUM_LAYERS, numProposals, numProposals))

                    for proposalIndex_1 in xrange(numProposals):
                        for proposalIndex_2 in xrange(numProposals):
                            for layer in xrange(NUM_LAYERS):
                                cv2.imwrite('test/cost_' + str(proposalIndex_1) + str(proposalIndex_2) + str(layer) + str(deltaIndex) + '.png', drawMaskImage(diff[:, :, layer, proposalIndex_1, proposalIndex_2]))
                                continue
                            continue
                        continue
                if deltaIndex == 1 and True:
                    #cv2.imwrite('test/segmentation.png', drawSegmentationImage(proposals[partial_nodes, 0, 1].reshape((height - 1, width))))
                    
                    #diff = (labelDiff * depthDiff).reshape((height, width - 1, NUM_LAYERS, numProposals, numProposals))
                    diff = (invisibleEmptyMask).reshape((height, width - 1, NUM_LAYERS, numProposals, numProposals))
                    #diff = (labelDiff).reshape((height - 1, width, NUM_LAYERS, numProposals, numProposals))

                    for proposalIndex_1 in xrange(numProposals):
                        for proposalIndex_2 in xrange(numProposals):
                            for layer in xrange(NUM_LAYERS):
                                cv2.imwrite('test/cost_' + str(proposalIndex_1) + str(proposalIndex_2) + str(layer) + str(deltaIndex) + '.png', drawMaskImage(diff[:, :, layer, proposalIndex_1, proposalIndex_2]))
                                continue
                            continue
                        continue
                    #exit(1)
                pass
            continue
        
        edges = np.concatenate(edges, axis=0)
        edges_features = np.concatenate(edges_features, axis=0)


        labelCost = True
        if labelCost and numProposals > 1:
            labelCosts = np.full(NUM_PLANES * NUM_LAYERS, numProposals, 10000)
            labelCosts[:, 0] = 0
            labelCosts[:, 1] = 1000
            unaries = np.concatenate([unaries, labelCosts], axis=0)

        solution, energy = inference_ogm(-unaries * 0.1, edges_features, edges, return_energy=True, alg='trw')
        print(energy)

        cv2.imwrite('test/solution_' + str(iteration) + '.png', drawSegmentationImage(solution.reshape((height, width))))
        
        solution = np.tile(solution.reshape([height * width, 1, 1]), [1, NUM_LAYERS, 1])
        solution = readProposalInfo(proposals, solution).reshape((height, width, NUM_LAYERS))

        np.save('test/solution_' + str(iteration) + '.npy', solution)
        continue
    
    return solution


planes = np.load('test/planes.npy')
segmentation = np.load('test/segmentation.npy')
image = cv2.imread('test/image.png')
depth = np.load('test/depth.npy')
normal = np.load('test/normal.npy')
info = np.load('test/info.npy')
numPlanes = planes.shape[0]
# cv2.imwrite('test/segmentation.png', drawSegmentationImage(segmentation, blackIndex=numPlanes))
# cv2.imwrite('test/depth.png', drawDepthImage(depth))
# for planeIndex in xrange(numPlanes):
#     cv2.imwrite('test/mask_' + str(planeIndex) + '.png', drawMaskImage(segmentation == planeIndex))
#     continue
# exit(1)
width = 256
height = 192
segmentation = cv2.resize(segmentation, (width, height), interpolation=cv2.INTER_NEAREST)
depth = cv2.resize(depth, (width, height))
normal = cv2.resize(normal, (width, height))
image = cv2.resize(image, (width, height))
layeredSegmentations = decompose(image, depth, normal, info, planes, segmentation)
drawSolution(layeredSegmentations, numPlanes, -1)
