import numpy as np
import cv2

def horizontalFlip(image, planes, segmentation, depth, metadata):
    image = image[:, ::-1]
    depth = depth[:, ::-1]
    segmentation = segmentation[:, ::-1]
    metadata[2] = image.shape[1] - metadata[2]
    if len(planes) > 0:
        planes[:, 0] *= -1
        pass
    return image, planes, segmentation, depth, metadata

def cropPatch(box, imageSizes, image, planes, segmentation, depth, metadata):
    mins, ranges = box
    image = cv2.resize(image[mins[1]:mins[1] + ranges[1], mins[0]:mins[0] + ranges[0]], (imageSizes[0], imageSizes[1]))
    depth = cv2.resize(depth[mins[1]:mins[1] + ranges[1], mins[0]:mins[0] + ranges[0]], (imageSizes[0], imageSizes[1]))
    segmentation = cv2.resize(segmentation[mins[1]:mins[1] + ranges[1], mins[0]:mins[0] + ranges[0]], (imageSizes[0], imageSizes[1]), interpolation=cv2.INTER_NEAREST)
    metadata[0] *= float(imageSizes[0]) / ranges[0]
    metadata[1] *= float(imageSizes[1]) / ranges[1]
    metadata[2] = (metadata[2] - mins[0]) * float(imageSizes[0]) / ranges[0]
    metadata[3] = (metadata[3] - mins[1]) * float(imageSizes[1]) / ranges[1]
    metadata[4] = imageSizes[0]
    metadata[5] = imageSizes[1]    
    return image, planes, segmentation, depth, metadata
