import xml.etree.ElementTree as ET
import numpy as np
#np.set_printoptions(precision=3, threshold=np.nan)
import cv2
import sys
import os
#sys.path.append('/home/chenliu/Projects/PlaneNet/code')
from plyfile import PlyData, PlyElement
import json
import zipfile
import glob
from download_scannet import *

#ROOT_FOLDER = '/mnt/vision/ScanNet/data/'
ROOT_FOLDER = '/home/chenliu/Projects/server/Data/ScanNet/data/'


class ColorPalette:
    def __init__(self, numColors):
        np.random.seed(2)
        #self.colorMap = np.random.randint(255, size = (numColors, 3))
        #self.colorMap[0] = 0


        self.colorMap = np.array([[255, 0, 0],
                                  [0, 255, 0],
                                  [0, 0, 255],
                                  [80, 128, 255],
                                  [255, 230, 180],
                                  [255, 0, 255],
                                  [0, 255, 255],
                                  [100, 0, 0],
                                  [0, 100, 0],
                                  [255, 255, 0],
                                  [50, 150, 0],
                                  [200, 255, 255],
                                  [255, 200, 255],
                                  [128, 128, 80],
                                  [0, 50, 128],
                                  [0, 100, 100],
                                  [0, 255, 128],
                                  [0, 128, 255],
                                  [255, 0, 128],
                                  [128, 0, 255],
                                  [255, 128, 0],
                                  [128, 255, 0],
        ])

        if numColors > self.colorMap.shape[0]:
            self.colorMap = np.concatenate([self.colorMap, np.random.randint(255, size = (numColors - self.colorMap.shape[0], 3))], axis=0)
            pass

        return

    def getColorMap(self):
        return self.colorMap

    def getColor(self, index):
        if index >= colorMap.shape[0]:
            return np.random.randint(255, size = (3))
        else:
            return self.colorMap[index]
            pass

def writePointCloudFace(filename, points, faces):
    with open(filename, 'w') as f:
        header = """ply
format ascii 1.0
element vertex """
        header += str(len(points))
        header += """
property float x
property float y
property float z
property uchar red                                     { start of vertex color }
property uchar green
property uchar blue
element face """
        header += str(len(faces))
        header += """
property list uchar int vertex_index
end_header
"""
        f.write(header)
        for point in points:
            for value in point[:3]:
                f.write(str(value) + ' ')
                continue
            for value in point[3:]:
                f.write(str(int(value)) + ' ')
                continue
            f.write('\n')
            continue
        for face in faces:
            f.write('3 ' + str(face[0]) + ' ' + str(face[1]) + ' ' + str(face[2]) + '\n')
            continue        
        f.close()
        pass
    return

def loadClassMap():
    classMap = {}
    classLabelMap = {}
    with open(ROOT_FOLDER + '/tasks/scannet-labels.combined.tsv') as info_file:
        line_index = 0
        for line in info_file:
            if line_index > 0:
                line = line.split('\t')
                
                key = line[1].strip()                
                classMap[key] = line[7].strip()
                classMap[key + 's'] = line[7].strip()

                if line[4].strip() != '':
                    label = int(line[4].strip())
                else:
                    label = -1
                    pass
                classLabelMap[key] = label
                classLabelMap[key + 's'] = label                    
                pass
            line_index += 1
            continue
        pass
    return classMap, classLabelMap


def mergePlanesNew(points, planes, planePointIndices, planeSegments, segmentNeighbors, numPlanes, planeDiffThreshold = 0.05, planeAngleThreshold = 30, inlierThreshold = 0.9, planeAreaThreshold = 10, orthogonalThreshold = np.cos(np.deg2rad(60)), parallelThreshold = np.cos(np.deg2rad(30)), debug=False):


    fittingErrorThreshold = planeDiffThreshold
    
    planeFittingErrors = []
    for plane, pointIndices in zip(planes, planePointIndices):
        XYZ = points[pointIndices]
        planeNorm = np.linalg.norm(plane)
        if planeNorm == 0:
            planeFittingErrors.append(fittingErrorThreshold)
            continue
        diff = np.abs(np.matmul(XYZ, plane) - np.ones(XYZ.shape[0])) / planeNorm
        planeFittingErrors.append(diff.mean())
        continue
    
    planeList = zip(planes, planePointIndices, planeSegments, planeFittingErrors)
    planeList = sorted(planeList, key=lambda x:x[3])

    ## Merge two planes if they are neighbors and the merged plane has small fitting error
    while len(planeList) > 0:
        hasChange = False
        planeIndex = 0

        if debug:
            for index, planeInfo in enumerate(sorted(planeList, key=lambda x:-len(x[1]))):
                print(index, planeInfo[0] / np.linalg.norm(planeInfo[0]), planeInfo[2], planeInfo[3])
                continue
            pass
        
        while planeIndex < len(planeList):
            plane, pointIndices, segments, fittingError = planeList[planeIndex]
            if fittingError > fittingErrorThreshold:
                break
            neighborSegments = []
            for segment in segments:
                if segment in segmentNeighbors:
                    neighborSegments += segmentNeighbors[segment]
                    pass
                continue
            neighborSegments += list(segments)
            neighborSegments = set(neighborSegments)
            bestNeighborPlane = (fittingErrorThreshold, -1, None)
            for neighborPlaneIndex, neighborPlane in enumerate(planeList):
                if neighborPlaneIndex <= planeIndex:
                    continue
                if not bool(neighborSegments & neighborPlane[2]):
                    continue
                dotProduct = np.abs(np.dot(neighborPlane[0], plane) / np.maximum(np.linalg.norm(neighborPlane[0]) * np.linalg.norm(plane), 1e-4))
                newPointIndices = np.concatenate([neighborPlane[1], pointIndices], axis=0)
                XYZ = points[newPointIndices]
                if dotProduct > parallelThreshold and len(neighborPlane[1]) > len(pointIndices) * 0.5:
                    newPlane = fitPlane(XYZ)                    
                else:
                    newPlane = plane
                    pass
                #newPlane = plane
                diff = np.abs(np.matmul(XYZ, newPlane) - np.ones(XYZ.shape[0])) / np.linalg.norm(newPlane)
                newFittingError = diff.mean()
                if debug:
                    print(len(planeList), planeIndex, neighborPlaneIndex, newFittingError, plane / np.linalg.norm(plane), neighborPlane[0] / np.linalg.norm(neighborPlane[0]), dotProduct, orthogonalThreshold)
                    pass
                if dotProduct < orthogonalThreshold:
                    continue                
                if newFittingError < bestNeighborPlane[0]:
                    newPlaneInfo = [newPlane, newPointIndices, segments.union(neighborPlane[2]), newFittingError]
                    bestNeighborPlane = (newFittingError, neighborPlaneIndex, newPlaneInfo)
                    pass
                continue
            if bestNeighborPlane[1] != -1:
                newPlaneList = planeList[:planeIndex] + planeList[planeIndex + 1:bestNeighborPlane[1]] + planeList[bestNeighborPlane[1] + 1:]
                newFittingError, newPlaneIndex, newPlane = bestNeighborPlane
                for newPlaneIndex in range(len(newPlaneList)):
                    if (newPlaneIndex == 0 and newPlaneList[newPlaneIndex][3] > newFittingError) \
                       or newPlaneIndex == len(newPlaneList) - 1 \
                       or (newPlaneList[newPlaneIndex][3] < newFittingError and newPlaneList[newPlaneIndex + 1][3] > newFittingError):
                        newPlaneList.insert(newPlaneIndex, newPlane)
                        break                    
                    continue
                if len(newPlaneList) == 0:
                    newPlaneList = [newPlane]
                    pass
                planeList = newPlaneList
                hasChange = True
            else:
                planeIndex += 1
                pass
            continue
        if not hasChange:
            break
        continue

    planeList = sorted(planeList, key=lambda x:-len(x[1]))

    
    minNumPlanes, maxNumPlanes = numPlanes
    if minNumPlanes == 1 and len(planeList) == 0:
        if debug:
            print('at least one plane')
            pass
    elif len(planeList) > maxNumPlanes:
        if debug:
            print('too many planes', len(planeList), maxNumPlanes)
            pass
        planeList = planeList[:maxNumPlanes]
        pass
    
    groupedPlanes, groupedPlanePointIndices, groupedPlaneSegments, groupedPlaneFittingErrors = zip(*planeList)
    groupNeighbors = []
    for planeIndex, planeSegments in enumerate(groupedPlaneSegments):
        neighborSegments = []
        for segment in planeSegments:
            if segment in segmentNeighbors:            
                neighborSegments += segmentNeighbors[segment]
                pass
            continue
        neighborSegments += list(planeSegments)        
        neighborSegments = set(neighborSegments)
        neighborPlaneIndices = []
        for neighborPlaneIndex, neighborPlaneSegments in enumerate(groupedPlaneSegments):
            if neighborPlaneIndex == planeIndex:
                continue
            if bool(neighborSegments & neighborPlaneSegments):
                plane = groupedPlanes[planeIndex]
                neighborPlane = groupedPlanes[neighborPlaneIndex]
                if np.linalg.norm(plane) * np.linalg.norm(neighborPlane) < 1e-4:
                    continue
                dotProduct = np.abs(np.dot(plane, neighborPlane) / np.maximum(np.linalg.norm(plane) * np.linalg.norm(neighborPlane), 1e-4))
                if dotProduct < orthogonalThreshold:
                    neighborPlaneIndices.append(neighborPlaneIndex)
                    pass
                pass
            continue
        groupNeighbors.append(neighborPlaneIndices)
        continue

    if debug and len(groupedPlanes) > 1:
        print('merging result', [len(pointIndices) for pointIndices in groupedPlanePointIndices], groupedPlaneFittingErrors, groupNeighbors)
        pass
    
    planeList = zip(groupedPlanes, groupedPlanePointIndices, groupNeighbors)
    return planeList


def readMesh(scene_id):

    filename = ROOT_FOLDER + scene_id + '/' + scene_id + '.aggregation.json'
    data = json.load(open(filename, 'r'))
    aggregation = np.array(data['segGroups'])

    high_res = False

    if high_res:
        filename = ROOT_FOLDER + scene_id + '/' + scene_id + '_vh_clean.labels.ply'
    else:
        filename = ROOT_FOLDER + scene_id + '/' + scene_id + '_vh_clean_2.labels.ply'
        pass

    plydata = PlyData.read(filename)
    vertices = plydata['vertex']
    points = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=1)
    faces = np.array(plydata['face']['vertex_indices'])
    
    semanticSegmentation = vertices['label']


    if high_res:
        filename = ROOT_FOLDER + scene_id + '/' + scene_id + '_vh_clean.segs.json'
    else:
        filename = ROOT_FOLDER + scene_id + '/' + scene_id + '_vh_clean_2.0.010000.segs.json'
        pass

    data = json.load(open(filename, 'r'))
    segmentation = np.array(data['segIndices'])

    groupSegments = []
    groupLabels = []
    for segmentIndex in xrange(len(aggregation)):
        groupSegments.append(aggregation[segmentIndex]['segments'])
        groupLabels.append(aggregation[segmentIndex]['label'])
        continue

    segmentation = segmentation.astype(np.int32)

    uniqueSegments = np.unique(segmentation).tolist()
    numSegments = 0
    for segments in groupSegments:
        for segmentIndex in segments:
            if segmentIndex in uniqueSegments:
                uniqueSegments.remove(segmentIndex)
                pass
            continue
        numSegments += len(segments)
        continue

    for segment in uniqueSegments:
        groupSegments.append([segment, ])
        groupLabels.append('unannotated')
        continue

    numGroups = len(groupSegments)
    numPoints = segmentation.shape[0]    
    numPlanes = 1000

    ## Segment connections for plane merging later
    segmentEdges = []
    for faceIndex in xrange(faces.shape[0]):
        face = faces[faceIndex]
        segment_1 = segmentation[face[0]]
        segment_2 = segmentation[face[1]]
        segment_3 = segmentation[face[2]]
        if segment_1 != segment_2 or segment_1 != segment_3:
            if segment_1 != segment_2 and segment_1 != -1 and segment_2 != -1:
                segmentEdges.append((min(segment_1, segment_2), max(segment_1, segment_2)))
                pass
            if segment_1 != segment_3 and segment_1 != -1 and segment_3 != -1:
                segmentEdges.append((min(segment_1, segment_3), max(segment_1, segment_3)))
                pass
            if segment_2 != segment_3 and segment_2 != -1 and segment_3 != -1:
                segmentEdges.append((min(segment_2, segment_3), max(segment_2, segment_3)))                
                pass
            pass
        continue
    segmentEdges = list(set(segmentEdges))


    numPlanes = 200
    numPlanesPerSegment = 2
    segmentRatio = 0.1
    planeAreaThreshold = 10
    numIterations = 100
    numIterationsPair = 1000
    planeDiffThreshold = 0.05
    fittingErrorThreshold = planeDiffThreshold

    ## Specify the minimum and maximum number of planes for each object
    labelNumPlanes = {'wall': [1, 3], 
                      'floor': [1, 1],
                      'cabinet': [1, 5],
                      'bed': [1, 5],
                      'chair': [1, 2],
                      'sofa': [1, 10],
                      'table': [1, 5],
                      'door': [1, 2],
                      'window': [1, 2],
                      'bookshelf': [1, 5],
                      'picture': [1, 1],
                      'counter': [1, 10],
                      'blinds': [0, 0],
                      'desk': [1, 10],
                      'shelf': [1, 5],
                      'shelves': [1, 5],                      
                      'curtain': [0, 0],
                      'dresser': [1, 5],
                      'pillow': [0, 0],
                      'mirror': [0, 0],
                      'entrance': [1, 1],
                      'floor mat': [1, 1],                      
                      'clothes': [0, 0],
                      'ceiling': [1, 5],
                      'book': [0, 1],
                      'books': [0, 1],                      
                      'refridgerator': [1, 5],
                      'television': [1, 1], 
                      'paper': [0, 1],
                      'towel': [0, 1],
                      'shower curtain': [0, 1],
                      'box': [1, 5],
                      'whiteboard': [1, 5],
                      'person': [0, 0],
                      'night stand': [1, 5],
                      'toilet': [0, 5],
                      'sink': [0, 5],
                      'lamp': [0, 1],
                      'bathtub': [0, 5],
                      'bag': [0, 1],
                      'otherprop': [0, 5],
                      'otherstructure': [0, 5],
                      'otherfurniture': [0, 5],                      
                      'unannotated': [0, 5],
                      '': [0, 0],
    }
    nonPlanarGroupLabels = ['bicycle', 'bottle', 'water bottle']
    nonPlanarGroupLabels = {label: True for label in nonPlanarGroupLabels}
    
    verticalLabels = ['wall', 'door', 'cabinet']
    classMap, classLabelMap = loadClassMap()
    allXYZ = points.reshape(-1, 3)

    segmentNeighbors = {}
    for segmentEdge in segmentEdges:
        if segmentEdge[0] not in segmentNeighbors:
            segmentNeighbors[segmentEdge[0]] = []
            pass
        segmentNeighbors[segmentEdge[0]].append(segmentEdge[1])
        
        if segmentEdge[1] not in segmentNeighbors:
            segmentNeighbors[segmentEdge[1]] = []
            pass
        segmentNeighbors[segmentEdge[1]].append(segmentEdge[0])
        continue

    planeGroups = []
    print('num groups', len(groupSegments))

    debug = False    
    debugIndex = -1

    ## A group corresponds to an instance in the ScanNet annotation
    for groupIndex, group in enumerate(groupSegments):
        if debugIndex != -1 and groupIndex != debugIndex:
            continue
        if groupLabels[groupIndex] in nonPlanarGroupLabels:
            groupLabel = groupLabels[groupIndex]
            minNumPlanes, maxNumPlanes = 0, 0
        elif groupLabels[groupIndex] == 'unannotated':
            groupLabel = 'unannotated'
            minNumPlanes, maxNumPlanes = labelNumPlanes[groupLabel]
        elif groupLabels[groupIndex] in classMap:
            groupLabel = classMap[groupLabels[groupIndex]]
            minNumPlanes, maxNumPlanes = labelNumPlanes[groupLabel]            
        else:
            minNumPlanes, maxNumPlanes = 0, 0
            groupLabel = ''
            pass

        if maxNumPlanes == 0:
            pointMasks = []
            for segmentIndex in group:
                pointMasks.append(segmentation == segmentIndex)
                continue
            pointIndices = np.any(np.stack(pointMasks, 0), 0).nonzero()[0]
            groupPlanes = [[np.zeros(3), pointIndices, []]]
            planeGroups.append(groupPlanes)
            continue
        groupPlanes = []
        groupPlanePointIndices = []
        groupPlaneSegments = []


        ## A group contains multiple segments and we run RANSAC for each segment
        for segmentIndex in group:
            segmentMask = segmentation == segmentIndex
            segmentIndices = segmentMask.nonzero()[0]

            XYZ = allXYZ[segmentMask.reshape(-1)]
            numPoints = XYZ.shape[0]

            segmentPlanes = []
            segmentPlanePointIndices = []

            for c in range(2):
                if c == 0:
                    ## First try to fit one plane to see if the entire segment is one plane
                    plane = fitPlane(XYZ)
                    diff = np.abs(np.matmul(XYZ, plane) - np.ones(XYZ.shape[0])) / np.linalg.norm(plane)
                    if diff.mean() < fittingErrorThreshold:
                        segmentPlanes.append(plane)
                        segmentPlanePointIndices.append(segmentIndices)
                        break
                else:
                    ## Run ransac                    
                    for planeIndex in range(numPlanesPerSegment):
                        if len(XYZ) < planeAreaThreshold:
                            continue
                        bestPlaneInfo = [None, 0, None]
                        for iteration in range(min(XYZ.shape[0], numIterations)):
                            sampledPoints = XYZ[np.random.choice(np.arange(XYZ.shape[0]), size=(3), replace=False)]
                            try:
                                plane = fitPlane(sampledPoints)
                                pass
                            except:
                                continue
                            diff = np.abs(np.matmul(XYZ, plane) - np.ones(XYZ.shape[0])) / np.linalg.norm(plane)
                            inlierMask = diff < planeDiffThreshold
                            numInliers = inlierMask.sum()
                            if numInliers > bestPlaneInfo[1]:
                                bestPlaneInfo = [plane, numInliers, inlierMask]
                                pass
                            continue

                        if bestPlaneInfo[1] < planeAreaThreshold:
                            break

                        
                        pointIndices = segmentIndices[bestPlaneInfo[2]]
                        #bestPlane = bestPlaneInfo[0]
                        bestPlane = fitPlane(XYZ[bestPlaneInfo[2]])
                        
                        segmentPlanes.append(bestPlane)                
                        segmentPlanePointIndices.append(pointIndices)

                        outlierMask = np.logical_not(bestPlaneInfo[2])
                        segmentIndices = segmentIndices[outlierMask]
                        XYZ = XYZ[outlierMask]
                        continue
                    pass
                continue
            
            if sum([len(indices) for indices in segmentPlanePointIndices]) < numPoints * 0.5:
                print('not enough fitted points')
                if len(segmentIndices) >= planeAreaThreshold:
                    groupPlanes.append(np.zeros(3))
                    groupPlanePointIndices.append(segmentIndices)
                    groupPlaneSegments.append(set([segmentIndex]))
                    pass
            else:
                groupPlanes += segmentPlanes
                groupPlanePointIndices += segmentPlanePointIndices
                for _ in range(len(segmentPlanes)):
                    groupPlaneSegments.append(set([segmentIndex]))
                    continue
                pass
            continue
            
        if len(groupPlanes) > 0:
            ## Merge planes of each instance
            groupPlanes = mergePlanesNew(points, groupPlanes, groupPlanePointIndices, groupPlaneSegments, segmentNeighbors, numPlanes=(minNumPlanes, maxNumPlanes), planeDiffThreshold=planeDiffThreshold, planeAreaThreshold=planeAreaThreshold, debug=debugIndex != -1)
            pass

        if debug:
            print('group', groupIndex, groupLabels[groupIndex], groupLabel, len(groupPlanes))
            pass
        
        planeGroups.append(groupPlanes)
        continue
    
    
    if debug:
        #colorMap = np.random.randint(255, size=(segmentation.max() + 2, 3))
        colorMap = ColorPalette(segmentation.max() + 2).getColorMap()
        colorMap[-1] = 0
        colorMap[-2] = 255
        annotationFolder = 'test/'
        #colorMap = np.tile(np.expand_dims(np.arange(256), -1), [1, 3])
    else:
        #colorMap = ColorPalette(segmentation.max() + 2).getColorMap()
        numPlanes = sum([len(group) for group in planeGroups])
        #print('num planes', numPlanes)
        #exit(1)
        segmentationColor = (np.arange(numPlanes) + 1) * 100
        colorMap = np.stack([segmentationColor / (256 * 256), segmentationColor / 256 % 256, segmentationColor % 256], axis=1)
        colorMap[-1] = 255
        annotationFolder = ROOT_FOLDER + scene_id + '/annotation/'
        pass


    if debug:
        colors = colorMap[segmentation]
        writePointCloudFace(annotationFolder + '/segments.ply', np.concatenate([points, colors], axis=-1), faces)

        groupedSegmentation = np.full(segmentation.shape, fill_value=-1)
        for segmentIndex in xrange(len(aggregation)):
            indices = aggregation[segmentIndex]['segments']
            for index in indices:
                groupedSegmentation[segmentation == index] = segmentIndex
                continue
            continue
        groupedSegmentation = groupedSegmentation.astype(np.int32)
        colors = colorMap[groupedSegmentation]
        writePointCloudFace(annotationFolder + '/groups.ply', np.concatenate([points, colors], axis=-1), faces)
        pass

    
    planes = []
    planePointIndices = []
    for index, group in enumerate(planeGroups):
        groupPlanes, groupPlanePointIndices, groupNeighbors = zip(*group)

        planes += groupPlanes
        planePointIndices += groupPlanePointIndices
        continue
    

    planeSegmentation = np.full(segmentation.shape, fill_value=-1, dtype=np.int32)
    for planeIndex, planePoints in enumerate(planePointIndices):
        if np.linalg.norm(planes[planeIndex]) < 1e-4:
            planeSegmentation[planePoints] = -2
        else:
            planeSegmentation[planePoints] = planeIndex
            pass
        continue


    if debug:
        groupSegmentation = np.full(segmentation.shape, fill_value=-1, dtype=np.int32)        
        structureSegmentation = np.full(segmentation.shape, fill_value=-1, dtype=np.int32)
        typeSegmentation = np.full(segmentation.shape, fill_value=-1, dtype=np.int32)
        for planeIndex, planePoints in enumerate(planePointIndices):
            if len(planeInfo[planeIndex]) > 1:
                structureSegmentation[planePoints] = planeInfo[planeIndex][1][0]
                typeSegmentation[planePoints] = np.maximum(typeSegmentation[planePoints], planeInfo[planeIndex][1][1] - 2)
                pass
            groupSegmentation[planePoints] = planeInfo[planeIndex][0][0]
            continue

        colors = colorMap[groupSegmentation]    
        writePointCloudFace(annotationFolder + '/group.ply', np.concatenate([points, colors], axis=-1), faces)

        colors = colorMap[structureSegmentation]    
        writePointCloudFace(annotationFolder + '/structure.ply', np.concatenate([points, colors], axis=-1), faces)

        colors = colorMap[typeSegmentation]    
        writePointCloudFace(annotationFolder + '/type.ply', np.concatenate([points, colors], axis=-1), faces)
        pass


    planes = np.array(planes)
    print('number of planes: ', planes.shape[0])    
    planesD = 1.0 / np.maximum(np.linalg.norm(planes, axis=-1, keepdims=True), 1e-4)
    planes *= pow(planesD, 2)

    ## Remove boundary faces for rendering purpose
    removeIndices = []
    for faceIndex in xrange(faces.shape[0]):
        face = faces[faceIndex]
        segment_1 = planeSegmentation[face[0]]
        segment_2 = planeSegmentation[face[1]]
        segment_3 = planeSegmentation[face[2]]
        if segment_1 != segment_2 or segment_1 != segment_3:
            removeIndices.append(faceIndex)
            pass
        continue
    faces = np.delete(faces, removeIndices)
    colors = colorMap[planeSegmentation]    
    writePointCloudFace(annotationFolder + '/planes.ply', np.concatenate([points, colors], axis=-1), faces)

    if debug:
        exit(1)
        pass
    
    np.save(annotationFolder + '/planes.npy', planes)
        
    return


if __name__=='__main__':

    scene_ids = os.listdir(ROOT_FOLDER)
    scene_ids = scene_ids

    for scene_id in scene_ids:
        #scene_id = 'scene0002_00'
        if not os.path.exists(ROOT_FOLDER + '/' + scene_id + '/annotation'):
            os.system('mkdir -p ' + ROOT_FOLDER + '/' + scene_id + '/annotation')
            pass
        if not os.path.exists(ROOT_FOLDER + '/' + scene_id + '/annotation/segmentation'):
            os.system('mkdir -p ' + ROOT_FOLDER + '/' + scene_id + '/annotation/segmentation')
            pass
        print(scene_id)
        ## Download if not exists
        if not os.path.exists(ROOT_FOLDER + '/' + scene_id + '/' + scene_id + '.aggregation.json'):
            print('download')
            download_release([scene_id], ROOT_FOLDER, FILETYPES, use_v1_sens=True)
            pass
        print('plane fitting', scene_id)
        if not os.path.exists(ROOT_FOLDER + '/' + scene_id + '/annotation/planes.ply'):
            readMesh(scene_id)
            pass

        ## Use a C++ program built upon OpenGL to render the 3D plane fitting results to each view
        if len(glob.glob(ROOT_FOLDER + '/' + scene_id + '/annotation/segmentation/*.png')) < len(glob.glob(ROOT_FOLDER + '/' + scene_id + '/frames/pose/*.txt')):
            cmd = '../../Renderer/Renderer --scene_id=' + scene_id + ' --root_folder=' + ROOT_FOLDER
            os.system(cmd)
            pass
        continue

