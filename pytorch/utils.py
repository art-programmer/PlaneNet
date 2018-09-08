import numpy as np
import cv2

#MEAN_STD = [[0.29010095242892997, 0.32808144844279574, 0.28696394422942517], [0.1829540508368939, 0.18656561047509476, 0.18447508988480435]]
MEAN_STD = np.array([[0.5, 0.5, 0.5], [1, 1, 1]], dtype=np.float32)
MAX_DEPTH = 10

## Global color mapping
class ColorPalette:
    def __init__(self, numColors):
        np.random.seed(1)

        self.colorMap = np.array([[255, 0, 0],
                                  [0, 255, 0],
                                  [0, 0, 255],
                                  [80, 128, 255],
                                  [128, 0, 255],
                                  [255, 0, 255],
                                  [0, 255, 255],
                                  [255, 0, 128],
                                  [255, 255, 0],
                                  [0, 128, 255],
                                  [50, 150, 0],
                                  [200, 255, 255],
                                  [255, 200, 255],
                                  [128, 128, 80],
                                  [0, 50, 128],
                                  [0, 100, 100],
                                  [0, 255, 128],
                                  [100, 0, 0],
                                  [0, 100, 0],
                                  [255, 230, 180],
                                  [255, 128, 0],
                                  [128, 255, 0],
        ], dtype=np.uint8)
        self.colorMap = np.maximum(self.colorMap, 1)

        if numColors > self.colorMap.shape[0]:
            self.colorMap = np.concatenate([self.colorMap, np.random.randint(255, size = (numColors - self.colorMap.shape[0], 3), dtype=np.uint8)], axis=0)
            pass

        #self.colorMap = np.random.randint(255, size = (numColors, 3), dtype=np.uint8)
        #self.colorMap[0] = np.maximum(self.colorMap[0], 1)
        return

    def getColorMap(self):
        return self.colorMap

    def getColor(self, index):
        if index >= colorMap.shape[0]:
            return np.random.randint(255, size = (3), dtype=np.uint8)
        else:
            return self.colorMap[index]
            pass
        return
    
## Draw segmentation image. The input could be either HxW or HxWxC
def drawSegmentationImage(segmentations, numColors=42, blackIndex=-1, blackThreshold=-1):
    if segmentations.ndim == 2:
        numColors = max(numColors, segmentations.max() + 2)
    else:
        if blackThreshold > 0:
            segmentations = np.concatenate([segmentations, np.ones((segmentations.shape[0], segmentations.shape[1], 1)) * blackThreshold], axis=2)
            blackIndex = segmentations.shape[2] - 1
            pass

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

    segmentation = segmentation.astype(np.int32)
    return randomColor[segmentation.reshape(-1)].reshape((height, width, 3))

## Draw depth image
def drawDepthImage(depth):
    depthImage = np.clip(depth / 5 * 255, 0, 255).astype(np.uint8)
    depthImage = cv2.applyColorMap(255 - depthImage, colormap=cv2.COLORMAP_JET)
    return depthImage

## Math operations
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


## Fit a 3D plane from points
def fitPlane(points):
    if points.shape[0] == points.shape[1]:
        return np.linalg.solve(points, np.ones(points.shape[0]))
    else:
        return np.linalg.lstsq(points, np.ones(points.shape[0]))[0]
    return


## Metadata to intrinsics
def metadataToIntrinsics(metadata):
    intrinsics = np.zeros((3, 3))
    intrinsics[0][0] = metadata[0]
    intrinsics[1][1] = metadata[1]
    intrinsics[0][2] = metadata[2]
    intrinsics[1][2] = metadata[3]
    intrinsics[2][2] = 1
    return intrinsics

## The function to compute plane depths from plane parameters
def calcPlaneDepths(planes, width, height, metadata):
    urange = (np.arange(width, dtype=np.float32).reshape(1, -1).repeat(height, 0) / (width + 1) * (metadata[4] + 1) - metadata[2]) / metadata[0]
    vrange = (np.arange(height, dtype=np.float32).reshape(-1, 1).repeat(width, 1) / (height + 1) * (metadata[5] + 1) - metadata[3]) / metadata[1]
    ranges = np.stack([urange, np.ones(urange.shape), -vrange], axis=-1)
    
    planeOffsets = np.linalg.norm(planes, axis=-1, keepdims=True)
    planeNormals = planes / np.maximum(planeOffsets, 1e-4)

    normalXYZ = np.dot(ranges, planeNormal.transpose())
    normalXYZ[normalXYZ == 0] = 1e-4
    planeDepths = planeOffsets / normalXYZ
    planeDepths = np.clip(planeDepths, 0, MAX_DEPTH)
    return planeDepths
