from torch.utils.data import Dataset

import numpy as np
import time

from plane_dataset_scannet import PlaneDatasetScanNet
from augmentation import *
from utils import *

## Plane dataset class
class PlaneDataset(Dataset):
    def __init__(self, options, split, random=True):
        self.options = options
        self.split = split
        
        dataset = options.dataset if split == 'train' else options.testingDataset
        self.datasets = []
        if 'scannet' in dataset:
            self.datasets.append(PlaneDatasetScanNet(options, split, random))
            pass

        self.numImages = sum([len(dataset) for dataset in self.datasets])
        numImages = options.numTrainingImages if split == 'train' else options.numTestingImages
        if numImages > 0:
            self.numImages = numImages
            pass
        return
    
    def __len__(self):
        return self.numImages

    def __getitem__(self, index):
        t = int(time.time() * 1000000)
        np.random.seed(((t & 0xff000000) >> 24) +
                       ((t & 0x00ff0000) >> 8) +
                       ((t & 0x0000ff00) << 8) +
                       ((t & 0x000000ff) << 24))
        
        dataset = self.datasets[np.random.randint(len(self.datasets))]
        info = dataset[index]
        
        image = info[0]
        planes = info[1]
        segmentation = info[2]
        depth = info[3]
        metadata = info[4]

        if self.split == 'train':
            if np.random.random() > 0.5:
                image, planes, segmentation, depth, metadata = horizontalFlip(image, planes, segmentation, depth, metadata)
                pass
            pass
        image, planes, segmentation, depth, metadata = cropPatch((np.zeros(2, dtype=np.int32), np.array([image.shape[1], image.shape[0]])), (self.options.outputWidth, self.options.outputHeight), image, planes, segmentation, depth, metadata)
        if len(planes) == 0:
            planes = np.zeros((self.options.numOutputPlanes, 3))
        elif len(planes) < self.options.numOutputPlanes:
            planes = np.concatenate([planes, np.zeros((self.options.numOutputPlanes - len(planes), 3))], axis=0)
        elif len(planes) > self.options.numOutputPlanes:
            planes = planes[:self.options.numOutputPlanes]
            pass
        segmentation[segmentation >= self.options.numOutputPlanes] = self.options.numOutputPlanes
        segmentation[segmentation < 0] = self.options.numOutputPlanes

        numbers = np.array([len(planes)])
        sample = [(image.astype(np.float32) / 255 - MEAN_STD[0]).transpose((2, 0, 1)), planes.astype(np.float32), segmentation.astype(np.int64), depth, metadata.astype(np.float32), numbers]
        #print([[item.shape, item.dtype] for item in sample])
        return sample
