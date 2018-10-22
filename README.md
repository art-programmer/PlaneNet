# PlaneNet: Piece-wise Planar Reconstruction from a Single RGB Image
By Chen Liu, Jimei Yang, Duygu Ceylan, Ersin Yumer, and Yasutaka Furukawa

## Introduction

This paper presents the first end-to-end neural architecture for piece-wise planar reconstruction from a single RGB image. The proposed network, PlaneNet, learns to directly infer a set of plane parameters and corresponding plane segmentation masks. For more details, please refer to our CVPR 2018 [paper](http://art-programmer.github.io/planenet/paper.pdf) or visit our [project website](http://art-programmer.github.io/planenet.html).

## Updates
We add script for extracting plane information from the original ScanNet dataset and rendering 3D planar segmentation results to 2D views. Please see the README in folder *data_preparation/* for details. Note that we made some modifications to the heuristic-heavy plane fitting algorithms when cleaning up the messy codes developed over time. So the plane fitting results will be slightly different with the training data we used (provided in the *.tfrecords* files).

PyTorch training and testing codes are available now (still experimental and without the CRF module).
## Dependencies
Python 2.7, TensorFlow (>= 1.3), numpy, opencv 3.

## Getting started
### Compilation
Please run the following commands to compile the library for the [crfasrnn module](https://github.com/sadeepj/crfasrnn_keras).
```bash
cd cpp
sh compile.sh
cd ..
```

To train the network, you also need to run the following commands to compile the library for computing the set matching loss. You need Eigen (I am using Eigen 3.2.92) for the compilation. (Please see [here](https://github.com/fanhqme/PointSetGeneration) for details.)
```bash
cd nndistance
make
cd ..
```

### Data preparation
We convert [ScanNet](http://www.scan-net.org/) data to *.tfrecords* files for training and testing. The *.tfrecords* file can be downloaded from [here](https://wustl.box.com/s/d3vmtei5sin40svky6dcbe2aqhh5tmoz) (training) and [here](https://mega.nz/#!IvAixABb!PD3wJtXX_6W3qtfKZQtl_P07mYPLwWst3cwbvuTXlSY) (testing).

For the training data, please run the following command to merge downloaded files into one *.tfrecords* file.

```bash
cat training_data_segments/* > planes_scannet_train.tfrecords
```

### Training
To train the network from the pretrained DeepLab network, please first download the DeepLab model [here](https://github.com/DrSleep/tensorflow-deeplab-resnet) (under the Caffe to TensorFlow conversion), and then run the following command.
```bash
python train_planenet.py --restore=0 --modelPathDeepLab="path to the deep lab model" --dataFolder="folder which contains tfrecords files"
```

### Evaluation
Please first download our trained network from [here](https://mega.nz/#!sjpT2DiQ!Uo-6hxyldmtnPoKk3TTdUHKZADRGy6nIPlmAeVzJs_8) and put the uncompressed folder under ./checkpoint folder.

To evaluate the performance against existing methods, please run:
```bash
python evaluate.py --dataFolder="folder which contains tfrecords files"
```

### Applications
Please first download our trained network (see [Evaluation](### Evaluation) section for details). Script *predict.py* predicts and visualizes custom images (if "customImageFolder" is specified) or ScanNet testing images (if "dataFolder" is specified).

```bash
python predict.py --customImageFolder="folder which contains custom images"
python predict.py --dataFolder="folder which contains tfrecords files" [--startIndex=0] [--numImages=30]
```

This will generate visualization images, a webpage containing all the visualization, as well as cache files under folder "predict/".

Same commands can be used for various applications by providing optional arguments, *applicationType*, *imageIndex*, *textureImageFilename*, and some application-specific arguments. The following commands are used to generate visualizations in the submission. (The TV application needs more manual specification for better visualization.)

```bash
python predict.py --dataFolder=/mnt/vision/Data/PlaneNet/ --textureImageFilename=texture_images/CVPR.jpg --imageIndex=118 --applicationType=logo_texture --startIndex=118 --numImages=1
python predict.py --dataFolder=/mnt/vision/Data/PlaneNet/ --textureImageFilename=texture_images/CVPR.jpg --imageIndex=118 --applicationType=logo_video --startIndex=118 --numImages=1
python predict.py --dataFolder=/mnt/vision/Data/PlaneNet/ --textureImageFilename=texture_images/checkerboard.jpg --imageIndex=72 --applicationType=wall_texture --wallIndices=7,9 --startIndex=72 --numImages=1
python predict.py --dataFolder=/mnt/vision/Data/PlaneNet/ --textureImageFilename=texture_images/checkerboard.jpg --imageIndex=72 --applicationType=wall_video --wallIndices=7,9 --startIndex=72 --numImages=1
python predict.py --customImageFolder=my_images/TV/ --textureImageFilename=texture_images/TV.mp4 --imageIndex=0 --applicationType=TV --wallIndices=2,9
python predict.py --customImageFolder=my_images/ruler --textureImageFilename=texture_images/ruler_36.png --imageIndex=0 --applicationType=ruler --startPixel=950,444 --endPixel=1120,2220
```

Note that, the above script generate image sequences for video applications. Please run the following command under the image sequence folder to generate a video:
```bash
ffmpeg -r 60 -f image2 -s 640x480 -i %04d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p video.mp4
```

To check out the pool ball application, please run the following commands.
```bash
python predict.py --customImageFolder=my_images/pool --imageIndex=0 --applicationType=pool --estimateFocalLength=False
cd pool
python pool.py
```

Use mouse to play:)

## Contact

If you have any questions, please contact me at chenliu@wustl.edu.
