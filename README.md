# PlaneNet

## Dependencies
Python 2.7, TensorFlow (>= 1.0), numpy, opencv 3.

## Getting started
### Compilation
Please run the *make* command under the root folder to compile the library for computing the set matching loss. (See [here](https://github.com/fanhqme/PointSetGeneration) for details.)

### Data preparation
We convert [ScanNet](http://www.scan-net.org/) data to *.tfrecords* files for training and testing. The *.tfrecords* file can be downloaded from [here](https://mega.nz/#!IvAixABb!PD3wJtXX_6W3qtfKZQtl_P07mYPLwWst3cwbvuTXlSY).

### Train the network
To train the network from the pretrained DeepLab network, please first download the DeepLab model [here](https://github.com/DrSleep/tensorflow-deeplab-resnet) (under the Caffe to TensorFlow conversion), and then run the following command.

```bash
python train_planenet.py --restore=0 --modelPathDeepLab="path to the deep lab model" --rootFolder="folder which contains tfrecords files"
```

### Evaluation
Please first download our trained network from [here](https://mega.nz/#!sjpT2DiQ!Uo-6hxyldmtnPoKk3TTdUHKZADRGy6nIPlmAeVzJs_8) and put the uncompressed folder under ./checkpoint
