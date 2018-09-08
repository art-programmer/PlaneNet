import numpy as np
from datasets.scannet_scene import ScanNetScene
import os
import glob

## This class maintains a pool of ScanNet scenes to load plane information
class PlaneDatasetScanNet():
    def __init__(self, options, split, random=True):
        self.options = options
        self.random = random
        
        dataFolder = '../../Data/ScanNet/'
        
        self.scenes = []
        self.sceneImageIndices = []
        with open(dataFolder + '/ScanNet/Tasks/Benchmark/scannetv1_' + split + '.txt') as f:
            for line in f:
                scene_id = line.strip()
                scenePath = dataFolder + '/scans/' + scene_id
                if not os.path.exists(scenePath + '/' + scene_id + '.txt') or len(glob.glob(scenePath + '/annotation/segmentation/*')) == 0:
                    continue
                scene = ScanNetScene(options, scenePath, scene_id)
                self.scenes.append(scene)
                self.sceneImageIndices += [[len(self.scenes) - 1, imageIndex] for imageIndex in range(len(scene.imagePaths))]
                continue
            pass
        #np.savetxt(dataFolder + '/image_list_' + split + '.txt', imagePaths, fmt='%s')
        #imagePaths = np.loadtxt(dataFolder + '/image_list_' + split + '.txt', fmt='%s')

        print('num images', len(self.sceneImageIndices))

        np.random.shuffle(self.sceneImageIndices)
        
        numImages = options.numTrainingImages if split == 'train' else options.numTestingImages
        if numImages > 0:
            self.sceneImageIndices = self.sceneImageIndices[:numImages]
            pass
        return

    def __len__(self):
        return len(self.sceneImageIndices)
    
    def __getitem__(self, index):
        if self.random:
            index = np.random.randint(len(self.sceneImageIndices))
        else:
            index = index % len(self.sceneImageIndices)
            pass
        sceneIndex, imageIndex = self.sceneImageIndices[index]
        plane_info = self.scenes[sceneIndex][imageIndex]
        return plane_info
