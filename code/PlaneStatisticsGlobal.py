import numpy as np
import PIL.Image
import random
import scipy.ndimage as ndimage
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import copy
import skimage.measure
import os

class PlaneStatistics:
  def __init__(self, width, height, stride):
    self.width = width
    self.height = height
    self.stride = stride
    self.planeParametersArray = []
    self.predefinedPlanes = []
    self.residualImages = np.zeros((4, self.height, self.width))
    self.positivePlaneThreshold = self.stride * self.stride * 0.3
    self.planeAreaThreshold = 40 * 30
    return
  
  def addPlaneInfo(self, normalFilename, maskFilename, depthFilename):
    normals = np.array(PIL.Image.open(normalFilename)).astype(np.float) / 255 * 2 - 1
    norm = np.linalg.norm(normals, 2, 2)
    for c in xrange(3):
      normals[:, :, c] /= norm
      continue
    
    invalidMask = (np.array(PIL.Image.open(maskFilename)) < 128)

    sampleRatio = 3
    azimuthAngleImage = (-np.round(np.rad2deg(np.arctan2(normals[:, :, 1], normals[:, :, 0])) / sampleRatio).astype(np.int) * sampleRatio + 360) % 360

    altitudeAngleImage = (np.round(np.rad2deg(np.arctan2(np.sign(-normals[:, :, 1]) * np.linalg.norm(normals[:, :, :2], 2, 2), normals[:, :, 2])) / sampleRatio).astype(np.int) * sampleRatio + 360) % 360

    orthogonalThreshold = 5
    orthogonalAzimuthMask_1 = ((azimuthAngleImage - 0) < orthogonalThreshold) + ((360 - azimuthAngleImage) < orthogonalThreshold)
    orthogonalAzimuthMask_2 = np.abs(azimuthAngleImage - 180) < orthogonalThreshold
    azimuthAngleImage[orthogonalAzimuthMask_1] = 0
    azimuthAngleImage[orthogonalAzimuthMask_2] = 180
    altitudeAngleImage[orthogonalAzimuthMask_1 + orthogonalAzimuthMask_2] = 0

    orthogonalAltitudeMask_1 = ((altitudeAngleImage - 0) < orthogonalThreshold) + ((360 - altitudeAngleImage) < orthogonalThreshold)
    orthogonalAltitudeMask_2 = np.abs(altitudeAngleImage - 180) < orthogonalThreshold
    altitudeAngleImage[orthogonalAltitudeMask_1] = 0
    altitudeAngleImage[orthogonalAltitudeMask_2] = 180
    azimuthAngleImage[orthogonalAltitudeMask_1 + orthogonalAltitudeMask_2] = 0

    azimuthAngleImage[invalidMask] = 360
    altitudeAngleImage[invalidMask] = 360
    
    
    sampleRatio = 5
    depths = np.array(PIL.Image.open(depthFilename)).astype(np.float) / 1000
    focalLength = 517.97
    urange = np.arange(self.width).reshape(1, -1).repeat(self.height, 0) - self.width * 0.5
    vrange = np.arange(self.height).reshape(-1, 1).repeat(self.width, 1) - self.height * 0.5
    X = depths / focalLength * urange
    Y = depths
    Z = -depths / focalLength * vrange
    d = -(normals[:, :, 0] * X + normals[:, :, 1] * Y + normals[:, :, 2] * Z)
    dImage = np.round(d / (10. / 360) / sampleRatio).astype(np.int) * sampleRatio
    dImage[dImage < 0] = 0
    dImage[dImage > 360] = 360
    dImage[invalidMask] = 360    

    valueMaps = [azimuthAngleImage, altitudeAngleImage, dImage]
    planes = []
    values_1, counts_1 = np.unique(valueMaps[0], return_counts=True)

    self.mask = np.zeros((self.height, self.width)) == 1
    
    for index_1, value_1 in enumerate(values_1):
      if counts_1[index_1] < self.planeAreaThreshold or value_1 == 360:
        continue
      mask_1 = valueMaps[0] == value_1

      values_2, counts_2 = np.unique(valueMaps[1][mask_1], return_counts=True)
      for index_2, value_2 in enumerate(values_2):
        if counts_2[index_2] < self.planeAreaThreshold or value_2 == 360:
          continue
        mask_2 = mask_1 * (valueMaps[1] == value_2)
        values_3, counts_3 = np.unique(valueMaps[2][mask_2], return_counts=True)
        for index_3, value_3 in enumerate(values_3):
          if counts_3[index_3] < self.planeAreaThreshold or value_3 == 360:
            continue
          mask_3 = mask_2 * (valueMaps[2] == value_3)
          mask_3 = ndimage.binary_erosion(mask_3).astype(mask_3.dtype)
          if mask_3.sum() < self.planeAreaThreshold:
            continue

          normal = np.array([normals[:, :, 0][mask_3].mean(), normals[:, :, 1][mask_3].mean(), normals[:, :, 2][mask_3].mean()])
          normal /= np.linalg.norm(normal, 2)
          dPlane = (-(normal[0] * X + normal[1] * Y + normal[2] * Z))[mask_3].mean()

          self.mask += mask_3
          
          azimuth = np.arctan2(-normal[1], normal[0])
          altitude = np.arctan2(np.sign(-normal[1]) * np.linalg.norm(normal[:2]), normal[2])
          #planes.append(((azimuth, altitude, dPlane), mask_3))
          planes.append(((-normal[0] * dPlane, -normal[1] * dPlane, -normal[2] * dPlane), mask_3))
          
          # azimuthAngleImage = np.arctan2(-normals[:, :, 1], normals[:, :, 0])
          # altitudeAngleImage = np.arctan2(np.sign(-normals[:, :, 1]) * np.linalg.norm(normals[:, :, :2], 2, 2), normals[:, :, 2])

          # dImage = -(normals[:, :, 0] * X + normals[:, :, 1] * Y + normals[:, :, 2] * Z)
          # dImage *= self.depthScaleFactor

          #PIL.Image.fromarray(mask_1.astype(np.uint8) * 255).save('test/mask_1.png')
          #PIL.Image.fromarray(mask_2.astype(np.uint8) * 255).save('test/mask_2.png')
          #PIL.Image.fromarray(mask_3.astype(np.uint8) * 255).save('test/mask_3.png')
          continue
        continue
      continue

    for plane in planes:
      planeParameters = plane[0]
      self.planeParametersArray.append(planeParameters)      
      continue



    if False:
      self.planeParametersArray = np.array(self.planeParametersArray)
      planeNormals = copy.deepcopy(self.planeParametersArray)
      planeD = np.linalg.norm(planeNormals, 2, 1)
      for c in xrange(3):
        planeNormals[:, c] /= planeD
        continue
      
      normalXYZ = np.array([urange / focalLength, np.ones(urange.shape), -vrange / focalLength])

      normalXYZ = np.dot(normalXYZ.transpose([1, 2, 0]), planeNormals.transpose())
      normalXYZ = np.reciprocal(normalXYZ)

      XYZ = np.array([X, Y, Z])
      planeXYZ = np.zeros(XYZ.shape)
      for i in xrange(self.planeParametersArray.shape[0]):

        mask = planes[i][1]
        planeY = normalXYZ[:, :, i] * planeD[i]
        planeX = planeY * urange / focalLength
        planeZ = -planeY * vrange / focalLength

        planeXYZ[0][mask] = planeX[mask]
        planeXYZ[1][mask] = planeY[mask]
        planeXYZ[2][mask] = planeZ[mask]
        continue

      for c in xrange(3):
        inputImage = XYZ[c]
        cMin = inputImage.min()
        cMax = inputImage.max()
        PIL.Image.fromarray(((inputImage - cMin) / (cMax - cMin) * 255).astype(np.uint8)).save('test/' + str(c) + '.png')
        reconstructed = planeXYZ[c]
        PIL.Image.fromarray(((reconstructed - cMin) / (cMax - cMin) * 255).astype(np.uint8)).save('test/' + str(c) + '_reconstructed.png')
        continue


      planeImage = np.zeros((self.height, self.width, 3))
      for plane in planes:
        mask = plane[1]
        for c in xrange(3):
          planeImage[:, :, c][mask] = random.randint(0, 255)
          #planeImage[:, :, c][mask] = max(min(round((plane[0][c] + 1) / 2 * 255), 255), 0)
          continue
        continue
      PIL.Image.fromarray(planeImage.astype(np.uint8)).save('test/plane.png')
      exit(1)
      pass
    
  def generatePlaneGroundTruthFitting(self, normalFilename, maskFilename, depthFilename, useGlobal = True):
    normals = np.array(PIL.Image.open(normalFilename)).astype(np.float) / 255 * 2 - 1

    height = self.height
    width = self.width
    
    norm = np.linalg.norm(normals, 2, 2)
    for c in xrange(3):
      normals[:, :, c] /= norm
      continue


    depths = np.array(PIL.Image.open(depthFilename)).astype(np.float) / 1000
    focalLength = 517.97
    urange = np.arange(width).reshape(1, -1).repeat(height, 0) - width * 0.5
    vrange = np.arange(height).reshape(-1, 1).repeat(width, 1) - height * 0.5
    X = depths / focalLength * urange
    Y = depths
    Z = -depths / focalLength * vrange

    invalidMask = (np.array(PIL.Image.open(maskFilename)) < 128)
    invalidMask += depths > 10
  
    #if outputFolder != None:
    #XYZ = np.array([X, Y, Z])
    diffNormals = np.ones((height, width)) * (-1)
    segmentationNormals = np.zeros((height, width))

    diffDepths = np.ones(depths.shape) * 1000000
    segmentationDepths = np.zeros((height, width))

    diffD = np.ones((height, width)) * 1000000
    segmentationD = np.zeros((height, width))
    
    ranges = np.array([urange / focalLength, np.ones(urange.shape), -vrange / focalLength]).transpose([1, 2, 0])

    predefinedPlanes = self.predefinedPlanes
    for planeIndex, plane in enumerate(predefinedPlanes):
      planeD = np.linalg.norm(plane)
      planeNormal = -plane / planeD

      mask = np.dot(normals, planeNormal) > diffNormals
      diffNormals[mask] = np.dot(normals, planeNormal)[mask]
      segmentationNormals[mask] = planeIndex
      
      normalXYZ = np.dot(ranges, planeNormal)
      normalXYZ = np.reciprocal(normalXYZ)
      planeY = -normalXYZ * planeD

      mask = pow(planeY - depths, 2) < diffDepths
      diffDepths[mask] = pow(planeY - depths, 2)[mask]
      segmentationDepths[mask] = planeIndex

      D = pow(X * planeNormal[0] + Y * planeNormal[1] + Z * planeNormal[2] + planeD, 2)
      mask = D < diffD
      diffD[mask] = D[mask]
      segmentationD[mask] = planeIndex
      continue

    segmentation = segmentationD

    residualPlanes = []
    segmentationImage = np.zeros((self.height, self.width, 3))
    for clusterIndex in xrange(self.numClusters):
      mask = segmentation == clusterIndex
      mask = ndimage.binary_erosion(mask).astype(mask.dtype)
      if mask.sum() < self.planeAreaThreshold:
        continue
      normal = np.array([normals[:, :, 0][mask].mean(), normals[:, :, 1][mask].mean(), normals[:, :, 2][mask].mean()])
      normal /= np.linalg.norm(normal, 2)
      dPlane = (-(normal[0] * X + normal[1] * Y + normal[2] * Z))[mask].mean()
      predefinedPlane = predefinedPlanes[clusterIndex]
      residualPlanes.append((clusterIndex, -normal[0] * dPlane - predefinedPlane[0], -normal[1] * dPlane - predefinedPlane[1], -normal[2] * dPlane - predefinedPlane[2]))
      segmentationImage[mask] = np.random.randint(255, size = (3, ))
      continue
    PIL.Image.fromarray(segmentationImage.astype(np.uint8)).save('test/segmentation.png')
    exit(1)
    planeFilename = normalFilename.replace('norm_camera.png', 'plane_global.npy')
    np.save(planeFilename, residualPlanes)

    return
  

  def generatePlaneGroundTruth(self, normalFilename, maskFilename, depthFilename, useGlobal = True):
    planeFilename = normalFilename.replace('norm_camera.png', 'plane_global.npy')
    if os.path.exists(planeFilename):
      self.planeFilenames.append(planeFilename)
      return
    
    normals = np.array(PIL.Image.open(normalFilename)).astype(np.float) / 255 * 2 - 1
    norm = np.linalg.norm(normals, 2, 2)
    for c in xrange(3):
      normals[:, :, c] /= norm
      continue
    
    invalidMask = (np.array(PIL.Image.open(maskFilename)) < 128)

    sampleRatio = 3
    azimuthAngleImage = (-np.round(np.rad2deg(np.arctan2(normals[:, :, 1], normals[:, :, 0])) / sampleRatio).astype(np.int) * sampleRatio + 360) % 360

    altitudeAngleImage = (np.round(np.rad2deg(np.arctan2(np.sign(-normals[:, :, 1]) * np.linalg.norm(normals[:, :, :2], 2, 2), normals[:, :, 2])) / sampleRatio).astype(np.int) * sampleRatio + 360) % 360

    orthogonalThreshold = 5
    orthogonalAzimuthMask_1 = ((azimuthAngleImage - 0) < orthogonalThreshold) + ((360 - azimuthAngleImage) < orthogonalThreshold)
    orthogonalAzimuthMask_2 = np.abs(azimuthAngleImage - 180) < orthogonalThreshold
    azimuthAngleImage[orthogonalAzimuthMask_1] = 0
    azimuthAngleImage[orthogonalAzimuthMask_2] = 180
    altitudeAngleImage[orthogonalAzimuthMask_1 + orthogonalAzimuthMask_2] = 0

    orthogonalAltitudeMask_1 = ((altitudeAngleImage - 0) < orthogonalThreshold) + ((360 - altitudeAngleImage) < orthogonalThreshold)
    orthogonalAltitudeMask_2 = np.abs(altitudeAngleImage - 180) < orthogonalThreshold
    altitudeAngleImage[orthogonalAltitudeMask_1] = 0
    altitudeAngleImage[orthogonalAltitudeMask_2] = 180
    azimuthAngleImage[orthogonalAltitudeMask_1 + orthogonalAltitudeMask_2] = 0

    azimuthAngleImage[invalidMask] = 360
    altitudeAngleImage[invalidMask] = 360
    
    sampleRatio = 5
    depths = np.array(PIL.Image.open(depthFilename)).astype(np.float) / 1000
    focalLength = 517.97
    urange = np.arange(self.width).reshape(1, -1).repeat(self.height, 0) - self.width * 0.5
    vrange = np.arange(self.height).reshape(-1, 1).repeat(self.width, 1) - self.height * 0.5
    X = depths / focalLength * urange
    Y = depths
    Z = -depths / focalLength * vrange
    d = -(normals[:, :, 0] * X + normals[:, :, 1] * Y + normals[:, :, 2] * Z)
    dImage = np.round(d / (10. / 360) / sampleRatio).astype(np.int) * sampleRatio
    dImage[dImage < 0] = 0
    dImage[dImage > 360] = 360
    dImage[invalidMask] = 360    

    valueMaps = [azimuthAngleImage, altitudeAngleImage, dImage]
    planes = []
    values_1, counts_1 = np.unique(valueMaps[0], return_counts=True)

    for index_1, value_1 in enumerate(values_1):
      if counts_1[index_1] < self.planeAreaThreshold or value_1 == 360:
        continue
      mask_1 = valueMaps[0] == value_1

      values_2, counts_2 = np.unique(valueMaps[1][mask_1], return_counts=True)
      for index_2, value_2 in enumerate(values_2):
        if counts_2[index_2] < self.planeAreaThreshold or value_2 == 360:
          continue
        mask_2 = mask_1 * (valueMaps[1] == value_2)
        values_3, counts_3 = np.unique(valueMaps[2][mask_2], return_counts=True)
        for index_3, value_3 in enumerate(values_3):
          if counts_3[index_3] < self.planeAreaThreshold or value_3 == 360:
            continue
          mask_3 = mask_2 * (valueMaps[2] == value_3)
          mask_3 = ndimage.binary_erosion(mask_3).astype(mask_3.dtype)
          if mask_3.sum() < self.planeAreaThreshold:
            continue

          normal = np.array([normals[:, :, 0][mask_3].mean(), normals[:, :, 1][mask_3].mean(), normals[:, :, 2][mask_3].mean()])
          normal /= np.linalg.norm(normal, 2)
          dPlane = (-(normal[0] * X + normal[1] * Y + normal[2] * Z))[mask_3].mean()
          planes.append(((-normal[0] * dPlane, -normal[1] * dPlane, -normal[2] * dPlane), mask_3))
          continue
        continue
      continue

    if False:
      planeImage = np.zeros((self.height, self.width, 3))
      for plane in planes:
        mask = plane[1]
        for c in xrange(3):
          planeImage[:, :, c][mask] = random.randint(0, 255)
          #planeImage[:, :, c][mask] = max(min(round((plane[0][c] + 1) / 2 * 255), 255), 0)
          continue
        continue
      PIL.Image.fromarray(planeImage.astype(np.uint8)).save('test/plane.png')
      exit(1)
      
    #planes = [planes[0]]
    residualPlanes = []
    if True:
      for plane in planes:
        mask = skimage.measure.block_reduce(plane[1], (32, 32), np.mean).reshape(-1)
        residualPlanes.append(np.append(plane[0], mask))
        continue
      pass
    elif useGlobal:
      residualPlaneMap = {}
      for plane in planes:
        planeParameters = np.array(plane[0])
        predefinedPlanes = self.predefinedPlanes
        diff = planeParameters.reshape(1, 3).repeat(predefinedPlanes.shape[0], 0) - predefinedPlanes
        diffSum = np.linalg.norm(diff, 2, 1)
        #diffSum = np.abs(diff[:, 1])
        planeIndex = np.argmin(diffSum)
        
        planeArea = plane[1].sum()
        if planeIndex not in residualPlaneMap or planeArea > residualPlaneMap[planeIndex][1]:
          residualPlaneMap[planeIndex] = (diff[planeIndex].tolist(), planeArea)
          pass
        continue
      for planeIndex, residualPlane in residualPlaneMap.items():
        residualPlanes.append([planeIndex, ] + residualPlane[0])
        continue
      pass
    else:
      for plane in planes:
        planeParameters = np.array(plane[0])
        mask = plane[1]
        for cell in xrange(self.width * self.width / (self.stride * self.stride)):
          gridX = int(cell) % (self.width / self.stride)
          gridY = int(cell) / (self.width / self.stride)
          intersection = mask[gridY * self.stride:(gridY + 1) * self.stride, gridX * self.stride:(gridX + 1) * self.stride].sum()
          if intersection > self.positivePlaneThreshold:
            predefinedPlanes = self.predefinedPlanes
            diff = planeParameters.reshape(1, 3).repeat(predefinedPlanes.shape[0], 0) - predefinedPlanes
            diffSum = np.linalg.norm(diff, 2, 1)
            #diffSum = np.abs(diff[:, 1])
            planeIndex = np.argmin(diffSum)
            index = cell * self.numClusters + planeIndex
            residualPlanes.append([index, ] + diff[planeIndex].tolist())
            pass
          continue
        continue
      pass

    residualPlanes = np.array(residualPlanes)
    #planeFilename = normalFilename.replace('norm_camera.png', 'plane_global.npy')
    np.save(planeFilename, residualPlanes)
    self.planeFilenames.append(planeFilename)


    if False:
      invalidMask += Y > 10
      X[invalidMask] = 0
      Y[invalidMask] = 3
      Z[invalidMask] = 0
      
      planeParametersArray = []
      for plane in planes:
        planeParametersArray.append(plane[0])
        continue
      planeParametersArray = np.array(planeParametersArray)
      planeNormals = copy.deepcopy(planeParametersArray)
      planeD = np.linalg.norm(planeNormals, 2, 1)
      for c in xrange(3):
        planeNormals[:, c] /= planeD
        continue
      
      normalXYZ = np.array([urange / focalLength, np.ones(urange.shape), -vrange / focalLength])

      normalXYZ = np.dot(normalXYZ.transpose([1, 2, 0]), planeNormals.transpose())
      normalXYZ = np.reciprocal(normalXYZ)

      XYZ = np.array([X, Y, Z])
      planeXYZ = np.zeros(XYZ.shape)
      for i in xrange(planeParametersArray.shape[0]):

        mask = planes[i][1]
        planeY = normalXYZ[:, :, i] * planeD[i]
        planeX = planeY * urange / focalLength
        planeZ = -planeY * vrange / focalLength

        planeXYZ[0][mask] = planeX[mask]
        planeXYZ[1][mask] = planeY[mask]
        planeXYZ[2][mask] = planeZ[mask]
        continue

      for c in xrange(3):
        inputImage = XYZ[c]
        cMin = inputImage.min()
        cMax = inputImage.max()
        PIL.Image.fromarray(((inputImage - cMin) / (cMax - cMin) * 255).astype(np.uint8)).save('test/' + str(c) + '.png')
        reconstructed = planeXYZ[c]
        PIL.Image.fromarray(((reconstructed - cMin) / (cMax - cMin) * 255).astype(np.uint8)).save('test/' + str(c) + '_reconstructed.png')
        continue


      planeImage = np.zeros((self.height, self.width, 3))
      for plane in planes:
        mask = plane[1]
        for c in xrange(3):
          planeImage[:, :, c][mask] = random.randint(0, 255)
          #planeImage[:, :, c][mask] = max(min(round((plane[0][c] + 1) / 2 * 255), 255), 0)
          continue
        continue
      PIL.Image.fromarray(planeImage.astype(np.uint8)).save('test/plane.png')
      exit(1)
      pass

    return

  def evaluatePredefinedPlanes(self, normalFilename, maskFilename, depthFilename):
    normals = np.array(PIL.Image.open(normalFilename)).astype(np.float) / 255 * 2 - 1
    norm = np.linalg.norm(normals, 2, 2)
    for c in xrange(3):
      normals[:, :, c] /= norm
      continue
    
    invalidMask = (np.array(PIL.Image.open(maskFilename)) < 128)

    azimuthAngleImage = np.arctan2(-normals[:, :, 1], normals[:, :, 0])
    altitudeAngleImage = np.arctan2(np.sign(-normals[:, :, 1]) * np.linalg.norm(normals[:, :, :2], 2, 2), normals[:, :, 2])


    depths = np.array(PIL.Image.open(depthFilename)).astype(np.float) / 1000
    focalLength = 517.97
    urange = np.arange(self.width).reshape(1, -1).repeat(self.height, 0) - self.width * 0.5
    vrange = np.arange(self.height).reshape(-1, 1).repeat(self.width, 1) - self.height * 0.5
    X = depths / focalLength * urange
    Y = depths
    Z = -depths / focalLength * vrange
    d = -(normals[:, :, 0] * X + normals[:, :, 1] * Y + normals[:, :, 2] * Z)
    #d *= self.depthScaleFactor


    inputParameters = np.array([X, Y, Z]).reshape(-1, self.height, self.width, 1).repeat(self.numClusters, 3)

    #invalidMask += (True - self.mask)

    diff = inputParameters - self.predefinedPlanesImage
    diff = pow(diff, 2)
    for c in xrange(diff.shape[0]):
      diff[c][invalidMask] = 0
      continue

    for c in xrange(diff.shape[0]):
      self.residualImages[c] += diff[c].min(2)
      continue
    self.residualImages[3] += np.linalg.norm(diff, 2, 0).min(2)



    if True:
      PIL.Image.fromarray((invalidMask * 255).astype(np.uint8)).save('test/mask.png')
      XYZ = np.array([X, Y, Z])
      for c in xrange(3):
        inputImage = XYZ[c]
        cMin = inputImage.min()
        cMax = inputImage.max()
        inputImage[invalidMask] = 0
        PIL.Image.fromarray(((inputImage - cMin) / (cMax - cMin) * 255).astype(np.uint8)).save('test/' + str(c) + '.png')
        reconstructed = np.zeros(inputImage.shape)
        diffImage = np.ones(inputImage.shape) * 10000
        segmentation = np.zeros((self.height, self.width, 3)).astype(np.uint8)
        for clusterIndex in xrange(self.numClusters):
          planeImage = self.predefinedPlanesImage[c, :, :, clusterIndex]
          mask = np.abs(planeImage - inputImage) < diffImage
          reconstructed[mask] = planeImage[mask]
          diffImage[mask] = np.abs(planeImage - inputImage)[mask]
          segmentation[mask] = (np.random.rand(3) * 255).astype(np.uint8)
          continue
        reconstructed[invalidMask] = 0
        PIL.Image.fromarray(((reconstructed - cMin) / (cMax - cMin) * 255).astype(np.uint8)).save('test/' + str(c) + '_reconstructed_2.png')
        PIL.Image.fromarray(segmentation).save('test/' + str(c) + '_segmentation_2.png')
        continue

      #assignment = np.argmin(diff.sum(0), 2).reshape(-1)

      for c in xrange(3):
        assignment = np.argmin(diff[c], 2)
        cMin = XYZ[c].min()
        cMax = XYZ[c].max()
        planesImage = self.predefinedPlanesImage[c]
        #reconstructed = .reshape(-1, self.numClusters)[np.arange(self.height * self.width), assignment].reshape(self.height, self.width)
        #cImage = self.predefinedPlanesImage[c][assignment]
        reconstructed = np.zeros(XYZ[c].shape)
        segmentation = np.zeros((self.height, self.width, 3)).astype(np.uint8)
        for clusterIndex in xrange(self.numClusters):
          mask = assignment == clusterIndex
          reconstructed[mask] = planesImage[:, :, clusterIndex][mask]
          segmentation[mask] = (np.random.rand(3) * 255).astype(np.uint8)
          continue
        inputImage[invalidMask] = 0
        reconstructed[invalidMask] = 0
        PIL.Image.fromarray(((XYZ[c] - cMin) / (cMax - cMin) * 255).astype(np.uint8)).save('test/' + str(c) + '.png')
        PIL.Image.fromarray((np.maximum(np.minimum((reconstructed - cMin) / (cMax - cMin), 1), 0) * 255).astype(np.uint8)).save('test/' + str(c) + '_reconstructed.png')
        PIL.Image.fromarray(segmentation).save('test/' + str(c) + '_segmentation.png')
        continue
      exit(1)
      pass
 
    return

  def finishAddingPlaneInfo(self):
    self.planeParametersArray = np.array(self.planeParametersArray)
    return

  def finishEvaluatingPredefinedPlanes(self, numImages):
    self.residualImages /= numImages
    return

  def savePlaneInfo(self, folder):
    np.save(folder + '/plane_parameters', self.planeParametersArray)
    return

  def loadPlaneInfo(self, folder):
    self.planeParametersArray = np.load(folder + '/plane_parameters.npy')

    if self.planeParametersArray.shape[1] == 4:
      for c in xrange(3):
        self.planeParametersArray[:, c] *= -self.planeParametersArray[:, 3]
        continue
      self.planeParametersArray = self.planeParametersArray[:, :3]
      pass

    return

  def saveResiduals(self, folder):
    np.save(folder + '/residual', self.residualImages)
    return

  def loadResiduals(self, folder):
    self.residualImages = np.load(folder + '/residual.npy')
    return

  def savePredefinedPlanes(self, folder, numClusters):
    np.save(folder + '/predefined_planes_' + str(numClusters), self.predefinedPlanes)
    return

  def loadPredefinedPlanes(self, folder, numClusters):
    self.predefinedPlanes = np.load(folder + '/predefined_planes_' + str(numClusters) + '.npy')
    self.numClusters = self.predefinedPlanes.shape[0]
    return

  def clusterPlanes(self, numClusters):
    self.numClusters = numClusters

    planeParameters = copy.deepcopy(self.planeParametersArray)    
    
    kmeans = KMeans(n_clusters = self.numClusters).fit(planeParameters)
    
    self.predefinedPlanes = np.array(kmeans.cluster_centers_)
    return

  def startEvaluatingPredefinedPlanes(self):
    normals = copy.deepcopy(self.predefinedPlanes)
    d = np.linalg.norm(normals, 2, 1)
    for c in xrange(3):
      normals[:, c] /= d
      continue
    #normals *= -1
    
    focalLength = 517.97
    urange = np.arange(self.width).reshape(1, -1).repeat(self.height, 0) - self.width * 0.5
    vrange = np.arange(self.height).reshape(-1, 1).repeat(self.width, 1) - self.height * 0.5
    XYZ = np.array([urange / focalLength, np.ones(urange.shape), -vrange / focalLength])

    normalXYZ = np.dot(XYZ.transpose([1, 2, 0]), normals.transpose())    
    Y = np.reciprocal(normalXYZ)
    X = np.zeros(Y.shape)
    Z = np.zeros(Y.shape)
    
    for i in xrange(self.numClusters):
      clusterY = Y[:, :, i] * d[i]
      Y[:, :, i] = clusterY
      X[:, :, i] = clusterY * urange / focalLength
      Z[:, :, i] = -clusterY * vrange / focalLength
      continue

    self.predefinedPlanesImage = np.array([X, Y, Z])
    
    # for i in xrange(self.numClusters):
    #   for c in xrange(3):
    #     if c == 1:
    #       PIL.Image.fromarray((np.minimum(self.predefinedPlanesImage[c, :, :, i] * 0.25, 1) * 255).astype(np.uint8)).save('test/' + str(i) + '_' + str(c) + '.png')
    #     else:
    #       PIL.Image.fromarray((np.maximum(np.minimum((self.predefinedPlanesImage[c, :, :, i] + 4) * 0.125, 1), 0) * 255).astype(np.uint8)).save('test/' + str(i) + '_' + str(c) + '.png')
    #       pass
    #     continue
    #   continue
    
          
    # dImage = np.linalg.norm(self.predefinedPlanesImage, 2, 0)
    # normalImage = np.zeros((self.height, self.width, 3, self.numClusters))
    # for c in xrange(3):
    #   normalImage[:, :, c, :] = self.predefinedPlanesImage[c] / d
    #   continue
    
    # for i in xrange(self.numClusters):
    #   print(normalImage[0, 0, :, i])
    #   print(dImage[0, 0, i])
    #   PIL.Image.fromarray(((normalImage[:, :, :, i] + 1) / 2 * 255).astype(np.uint8)).save('test/normal_' + str(i) + '.png')
    #   PIL.Image.fromarray((np.minimum(dImage[:, :, i] * 0.3, 1) * 255).astype(np.uint8)).save('test/d_' + str(i) + '.png')
    #   continue
    # exit(1)
    # #self.predefinedPlanesImage = self.predefinedPlanes.reshape(3, 1, 1, self.numClusters).repeat(self.height, 1).repeat(self.width, 2)

    return

  def startGeneratingPlaneGfroundTruth(self):
    self.planeFilenames = []

  def finishGeneratingPlaneGroundTruth(self, folder):
    with open(folder + '/image_list.txt', 'w') as f:
      for filename in self.planeFilenames:
        f.write(filename)
        f.write('\n')
        continue
      f.close()
      pass
    return

  def evaluatePlaneGroundTruth(self, normalFilename, maskFilename, depthFilename, planeFilename):
    
    normals = np.array(PIL.Image.open(normalFilename)).astype(np.float) / 255 * 2 - 1
    norm = np.linalg.norm(normals, 2, 2)
    for c in xrange(3):
      normals[:, :, c] /= norm
      continue
    
    invalidMask = (np.array(PIL.Image.open(maskFilename)) < 128)

    depths = np.array(PIL.Image.open(depthFilename)).astype(np.float) / 1000
    focalLength = 517.97
    urange = np.arange(self.width).reshape(1, -1).repeat(self.height, 0) - self.width * 0.5
    vrange = np.arange(self.height).reshape(-1, 1).repeat(self.width, 1) - self.height * 0.5
    X = depths / focalLength * urange
    Y = depths
    Z = -depths / focalLength * vrange
    #d = -(normals[:, :, 0] * X + normals[:, :, 1] * Y + normals[:, :, 2] * Z)

    residualPlanes = np.load(planeFilename)

    XYZ = np.array([X, Y, Z])
    planesXYZ = np.zeros(XYZ.shape)
    diffImage = np.ones(XYZ.shape) * 10000
    ranges = np.array([urange / focalLength, np.ones(urange.shape), -vrange / focalLength]).transpose([1, 2, 0])


    for residualPlane in residualPlanes:
      #residualPlane[1:] = 0
      gridIndex = int(residualPlane[0]) / self.numClusters
      planeIndex = int(residualPlane[0]) % self.numClusters
      plane = self.predefinedPlanes[planeIndex] + residualPlane[1:]
      #print(plane)
      planeD = np.linalg.norm(plane)
      planeNormal = plane / planeD

      
      normalXYZ = np.dot(ranges, planeNormal)
      normalXYZ = np.reciprocal(normalXYZ)

      planeY = normalXYZ * planeD
      planeX = planeY * urange / focalLength
      planeZ = -planeY * vrange / focalLength
      
      planeXYZ = [planeX, planeY, planeZ]
      for c in xrange(3):
        mask = np.abs(planeXYZ[c] - XYZ[c]) < diffImage[c]
        planesXYZ[c][mask] = planeXYZ[c][mask]
        diffImage[c][mask] = np.abs(planeXYZ[c] - XYZ[c])[mask]
        continue
      continue
    
    for c in xrange(3):
      inputImage = XYZ[c]
      cMin = inputImage.min()
      cMax = inputImage.max()
      PIL.Image.fromarray(((inputImage - cMin) / (cMax - cMin) * 255).astype(np.uint8)).save('test/' + str(c) + '.png')
      reconstructed = planesXYZ[c]
      PIL.Image.fromarray(((reconstructed - cMin) / (cMax - cMin) * 255).astype(np.uint8)).save('test/' + str(c) + '_reconstructed.png')
      continue

    return

  
def evaluatePlanes(planes, filename, outputFolder = None):
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
  focalLength = 517.97
  urange = np.arange(width).reshape(1, -1).repeat(height, 0) - width * 0.5
  vrange = np.arange(height).reshape(-1, 1).repeat(width, 1) - height * 0.5
  X = depths / focalLength * urange
  Y = depths
  Z = -depths / focalLength * vrange
  d = -(normals[:, :, 0] * X + normals[:, :, 1] * Y + normals[:, :, 2] * Z)

  
  maskFilename = filename.replace('mlt', 'valid')
  invalidMask = (np.array(PIL.Image.open(maskFilename)) < 128)
  invalidMask += depths > 10

  
  #if outputFolder != None:
  #XYZ = np.array([X, Y, Z])
  reconstructedNormals = np.zeros(normals.shape)
  diffNormals = np.ones((height, width)) * (-1)
  segmentationNormals = np.zeros((height, width, 3))
  reconstructedDepths = np.zeros(depths.shape)
  diffDepths = np.ones(depths.shape) * 1000000
  segmentationDepths = np.zeros((height, width, 3))
  
  ranges = np.array([urange / focalLength, np.ones(urange.shape), -vrange / focalLength]).transpose([1, 2, 0])

  for planeIndex, plane in enumerate(planes):
    planeD = np.linalg.norm(plane)
    planeNormal = -plane / planeD


    mask = np.dot(normals, planeNormal) > diffNormals
    reconstructedNormals[mask] = planeNormal
    diffNormals[mask] = np.dot(normals, planeNormal)[mask]
    segmentationNormals[mask] = np.random.randint(255, size=(3,))
    
    normalXYZ = np.dot(ranges, planeNormal)
    normalXYZ = np.reciprocal(normalXYZ)

    planeY = -normalXYZ * planeD

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
    if planeIndex in [1, 4]:
      print(planeY[113][251])
      continue
    mask = pow(planeY - Y, 2) < diffDepths
    reconstructedDepths[mask] = planeY[mask]
    diffDepths[mask] = pow(planeY - Y, 2)[mask]
    segmentationDepths[mask] = np.random.randint(255, size=(3,))
    continue

  
  if outputFolder != None:
    depths[invalidMask] = 0
    normals[invalidMask] = 0
    reconstructedDepths[invalidMask] = 0
    reconstructedNormals[invalidMask] = 0
    minDepth = depths.min()
    maxDepth = depths.max()
    print(minDepth)
    print(maxDepth)
    PIL.Image.fromarray(((depths - minDepth) / (maxDepth - minDepth) * 255).astype(np.uint8)).save(outputFolder + '/depth.png')
    PIL.Image.fromarray((np.maximum(np.minimum((reconstructedDepths - minDepth) / (maxDepth - minDepth), 1), 0) * 255).astype(np.uint8)).save(outputFolder + '/depth_reconstructed.png')
    PIL.Image.fromarray(segmentationDepths.astype(np.uint8)).save(outputFolder + '/depth_segmentation.png')
    PIL.Image.fromarray(((normals + 1) / 2 * 255).astype(np.uint8)).save(outputFolder + '/normal.png')
    PIL.Image.fromarray(((reconstructedNormals + 1) / 2 * 255).astype(np.uint8)).save(outputFolder + '/normal_reconstructed.png')
    PIL.Image.fromarray(segmentationNormals.astype(np.uint8)).save(outputFolder + '/normal_segmentation.png')
    depthImage = ((depths - minDepth) / (maxDepth - minDepth) * 255).astype(np.uint8)
    #PIL.Image.fromarray((invalidMask * 255).astype(np.uint8)).save(outputFolder + '/mask.png')
    exit(1)
    pass
  return diffDepths.mean(), diffNormals.mean()
  
