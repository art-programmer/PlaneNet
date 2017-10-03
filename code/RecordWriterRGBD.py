import tensorflow as tf
import numpy as np
import cv2
import random
import PIL.Image
import glob

HEIGHT=192
WIDTH=256
NUM_PLANES = 20

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def writeExample(writer, imagePath):
    #img = np.array(Image.open(imagePath['image']))
    img = cv2.imread(imagePath['image'])
    img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)

    height = img.shape[0]
    width = img.shape[1]
    img_raw = img.tostring()

    depth = np.array(PIL.Image.open(imagePath['depth'])).astype(np.float32) / 255 * 10
    depth = cv2.resize(depth, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)


    example = tf.train.Example(features=tf.train.Features(feature={
        'image_path': _bytes_feature(imagePath['image']),
        'image_raw': _bytes_feature(img_raw),
        'depth': _float_feature(depth.reshape(-1)),
    }))
    writer.write(example.SerializeToString())
    return

def loadImagePaths():
    image_set_file = '../PythonScripts/SUNCG/image_list_500000.txt'
    with open(image_set_file) as f:
        filenames = [x.strip().replace('plane_global.npy', '') for x in f.readlines()]
        image_paths = [{'image': x + 'mlt.png', 'plane': x + 'plane_global.npy', 'normal': x + 'norm_camera.png', 'depth': x + 'depth.png', 'mask': x + 'valid.png', 'masks': x + 'masks.npy'} for x in filenames]
        pass
    return image_paths


def readRecordFile():
    tfrecords_filename = '../planes.tfrecords'
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

    for string_record in record_iterator:

        example = tf.train.Example()
        example.ParseFromString(string_record)
        height = int(example.features.feature['height']
                     .int64_list
                     .value[0])

        width = int(example.features.feature['width']
                        .int64_list
                        .value[0])

        img_string = (example.features.feature['image_raw']
                      .bytes_list
                      .value[0])

        plane = (example.features.feature['plane_raw']
                 .float_list
                 .value)

        plane_mask = (example.features.feature['plane_mask_raw']
                      .int64_list
                      .value)

        img_1d = np.fromstring(img_string, dtype=np.uint8)
        reconstructed_img = img_1d.reshape((height, width, -1))
        print(np.array(plane).shape)
        print(np.array(plane_mask).shape)
        exit(1)
        continue
    return


def writeRecordFile(tfrecords_filename, imagePaths):
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    for index, imagePath in enumerate(imagePaths):
        if index % 100 == 0:
            print(index)
            pass
        writeExample(writer, imagePath)
        continue
    writer.close()
    return


if __name__=='__main__':
    imagePaths = glob.glob('/home/chenliu/Projects/Data/NYU_RGBD/raw/train/color_*.png')
    imagePaths = [{'image': imagePath, 'depth': imagePath.replace('color', 'depth')} for imagePath in imagePaths]
    print(len(imagePaths))
    #exit(1)
    random.shuffle(imagePaths)
    writeRecordFile('../planes_nyu_rgbd_train.tfrecords', imagePaths)
    #reader.readRecordFile()


    # # The op for initializing the variables.
    # init_op = tf.group(tf.global_variables_initializer(),
    #                    tf.local_variables_initializer())

    # with tf.Session()  as sess:

    #     sess.run(init_op)

    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)

    #     for i in xrange(3):
    #         image, plane, plane_mask = sess.run([image_inp, plane_inp, plane_mask_inp])
    #         print(image.shape)
    #         print(plane)
    #         print(plane_mask.shape)
    #         for index in xrange(image.shape[0]):
    #             print(plane[index])
    #             cv2.imwrite('test/record_' + str(index) + '_image.png', ((image[index] + 0.5) * 255).astype(np.uint8))
    #             cv2.imwrite('test/record_' + str(index) + '_mask.png', (plane_mask[index, :, :, 0] * 255).astype(np.uint8))
    #             continue
    #         exit(1)
    #         continue
    #     pass
    # exit(1)

#func = partial(writeExample, writer, 1)
#pool.map(func, self.imagePaths[self.numTrainingImages:])
#pool.close()
#pool.join()
#writer.close()
