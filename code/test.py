import tensorflow as tf
from RecordReader import *
from RecordReaderRGBD import *
from RecordReader3D import *

def main():
    min_after_dequeue = 1000

    reader_train = RecordReaderRGBD()
    #filename_queue_train = tf.train.string_input_producer(['/mnt/vision/SUNCG_plane/planes_test_450000.tfrecords'], num_epochs=10000)
    filename_queue_train = tf.train.string_input_producer(['../planes_nyu_rgbd_train.tfrecords', '/media/chenliu/My Passport/planes_test_450000.tfrecords'], num_epochs=10000, shuffle=True)
    img_inp_train, global_gt_dict_train, local_gt_dict_train = reader_train.getBatch(filename_queue_train, numOutputPlanes=20, batchSize=8, min_after_dequeue=min_after_dequeue, getLocal=True, random=True)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())    
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for i in xrange(10):
                img = sess.run(img_inp_train)
                image = ((img[0] + 0.5) * 255).astype(np.uint8)
                cv2.imwrite('test/image_' + str(i) + '.png', image)
                continue
            pass
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            pass
        pass
    
    return


if __name__=='__main__':
    main()
