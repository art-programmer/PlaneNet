# PlaneNet

## Dependencies
Python 2.7, TensorFlow (>= 1.0), numpy, opencv 3.

## Getting started
### Compilation
Please run the following commands to compile the library for the [crfasrnn module](https://github.com/sadeepj/crfasrnn_keras).
```bash
cd cpp
sh compile.sh
cd ..
```

To train the network, you also need to run the following commands to compile the library for computing the set matching loss. (See [here](https://github.com/fanhqme/PointSetGeneration) for details.)
```bash
cd nndistance
make
cd ..
```

### Data preparation
We convert [ScanNet](http://www.scan-net.org/) data to *.tfrecords* files for training and testing. The *.tfrecords* file can be downloaded from [here](https://mega.nz/#!IvAixABb!PD3wJtXX_6W3qtfKZQtl_P07mYPLwWst3cwbvuTXlSY).

### Training
To train the network from the pretrained DeepLab network, please first download the DeepLab model [here](https://github.com/DrSleep/tensorflow-deeplab-resnet) (under the Caffe to TensorFlow conversion), and then run the following command.
```bash
python train_planenet.py --restore=0 --modelPathDeepLab="path to the deep lab model" --rootFolder="folder which contains tfrecords files"
```

### Evaluation
Please first download our trained network from [here](https://mega.nz/#!sjpT2DiQ!Uo-6hxyldmtnPoKk3TTdUHKZADRGy6nIPlmAeVzJs_8) and put the uncompressed folder under ./checkpoint folder.

To evaluate the performance against existing methods, please run:
```bash
python evaluate.py --rootFolder="folder which contains tfrecords files"
```

### Applications
Please first download our trained network (see [Evaluation](### Evaluation) section for details). Script *predict.py* and *predict_custom.py* are for ScanNet testing images and custom images respectively. To predict and visualize ScanNet testing images, please run:

```bash
python predict.py --rootFolder="folder which contains tfrecords files" [--startIndex=0] [--numImages=30]
```

This will generate visualization images, a webpage containing all the visualization, as well as cache files under folder "predict". Similarly, the following command can be used to predict and visualize custom images:

```bash
python predict_custom.py --rootFolder="folder which contains custom images"
```

Same commands can be used for various applications by providing arguments, *--imageIndex*, *--suffix*, and *--textureFilename*
