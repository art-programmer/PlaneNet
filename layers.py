import os
import numpy as np

#DEBUG = False

class RangesLayer(object):
  def __init__(self, width, height):

    focalLength = 517.97
    urange = np.arange(width).reshape(1, -1).repeat(height, 0) - width * 0.5
    vrange = np.arange(height).reshape(-1, 1).repeat(width, 1) - height * 0.5
    self.ranges = np.array([urange / focalLength / width * 640, np.ones(urange.shape), -vrange / focalLength / height * 480]).transpose([1, 2, 0])
    return
      
  def forward(self):
    return self.ranges


def PlaneDepthLayer(planes, ranges):
  batchSize = 1
  if len(planes.shape) == 3:
    batchSize = planes.shape[0]
    planes = planes.reshape(planes.shape[0] * planes.shape[1], planes.shape[2])
    pass
  
  planesD = np.linalg.norm(planes, 2, 1)
  planesD = np.maximum(planesD, 1e-4)
  planesNormal = -planes / planesD.reshape(-1, 1).repeat(3, 1)

  normalXYZ = np.dot(ranges, planesNormal.transpose())
  normalXYZ[normalXYZ == 0] = 1e-4
  normalXYZ = 1 / normalXYZ
  depths = -normalXYZ
  depths[:, :] *= planesD
  if batchSize > 1:
    depths = depths.reshape(depths.shape[0], depths.shape[1], batchSize, -1).transpose([2, 0, 1, 3])
    pass
  depths[(depths < 0) + (depths > 10)] = 10
  #depths[depths < 0] = 0
  #depths[depths > 10] = 10
  return depths


def PlaneNormalLayer(planes, ranges):
  batchSize = 1
  if len(planes.shape) == 3:
    batchSize = planes.shape[0]
    planes = planes.reshape(planes.shape[0] * planes.shape[1], planes.shape[2])
    pass
  planesD = np.linalg.norm(planes, 2, 1)
  planesD = np.maximum(planesD, 1e-4)
  planesNormal = -planes / planesD.reshape(-1, 1).repeat(3, 1)
  normals = planesNormal.reshape(1, 1, -1, 3).repeat(ranges.shape[0], 0).repeat(ranges.shape[1], 1)
  if batchSize > 1:
    normals = normals.reshape(normals.shape[0], normals.shape[1], batchSize, -1, 3).transpose([2, 0, 1, 3, 4])
    pass
  return normals
