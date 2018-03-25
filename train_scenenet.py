import tensorflow as tf
import numpy as np
np.set_printoptions(precision=2, linewidth=200)
import cv2
import os
import time
import sys
import argparse
import glob
import PIL
import scipy.ndimage as ndimage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
from modules import *


from scenenet import SceneNet
from RecordReader import *

#from SegmentationRefinement import *

#training_flag: toggle dropout and batch normalization mode
#it's true for training and false for validation, testing, prediction
#it also controls which data batch to use (*_train or *_val)


def build_graph(img_inp_train, img_inp_val, training_flag, options):
    with tf.device('/gpu:%d'%options.gpu_id):
        img_inp = tf.cond(training_flag, lambda: img_inp_train, lambda: img_inp_val)

        net = SceneNet({'img_inp': img_inp}, is_training=training_flag, options=options)

        #global predictions
        object_pred = net.layers['object_pred']


        if False:
            object_pred = gt_dict['object']
            pass


        global_pred_dict = {'object': object_pred}

        if options.predictConfidence:
            global_pred_dict['confidence'] = net.layers['object_confidence_pred']
            pass

        #local predictions
        if options.predictLocal:
            local_pred_dict = {'score': net.layers['local_score_pred'], 'plane': net.layers['local_object_pred'], 'mask': net.layers['local_mask_pred']}
        else:
            local_pred_dict = {}
            pass


        #deep supervision
        deep_pred_dicts = []
        for layer in options.deepSupervisionLayers:
            pred_dict = {'object': net.layers[layer+'_object_pred']}
            #if options.predictConfidence:
            #pred_dict['confidence'] = net.layers[layer+'_object_confidence_pred']
            #pass
            deep_pred_dicts.append(pred_dict)
            continue

    return global_pred_dict, local_pred_dict, deep_pred_dicts


def build_loss(img_inp_train, img_inp_val, global_pred_dict, deep_pred_dicts, global_gt_dict_train, global_gt_dict_val, training_flag, options):
    from nndistance import tf_nndistance

    with tf.device('/gpu:%d'%options.gpu_id):
        debug_dict = {}

        img_inp = tf.cond(training_flag, lambda: img_inp_train, lambda: img_inp_val)
        global_gt_dict = {}
        for name in global_gt_dict_train.keys():
            global_gt_dict[name] = tf.cond(training_flag, lambda: global_gt_dict_train[name], lambda: global_gt_dict_val[name])
            continue
        # local_gt_dict = {}
        # for name in local_gt_dict_train.keys():
        #     local_gt_dict[name] = tf.cond(tf.equal(training_flag % 2, 0), lambda: local_gt_dict_train[name], lambda: local_gt_dict_val[name])
        #     continue

        info = global_gt_dict['info'][0]


        validPlaneMask = tf.cast(tf.less(tf.tile(tf.expand_dims(tf.range(options.numOutputObjects), 0), [options.batchSize, 1]), tf.expand_dims(global_gt_dict['num_objects'], -1)), tf.float32)
        backwardLossWeight = options.backwardLossWeight

        #plane loss and segmentation loss (summation over deep supervisions and final supervision)
        all_pred_dicts = deep_pred_dicts + [global_pred_dict]
        object_center_loss = tf.constant(0.0)
        object_size_loss = tf.constant(0.0)
        object_confidence_loss = tf.constant(0.0)

        #keep forward map (segmentation gt) from previous supervision so that we can have same matching for all supervisions (options.sameMatching = 1)
        previous_object_gt = None
        previous_object_confidence_gt = None
        previous_segmentation_gt = None

        for pred_index, pred_dict in enumerate(all_pred_dicts):
            dists_forward, map_forward, dists_backward, _ = tf_nndistance.nn_distance(global_gt_dict['object'][:, :, :3], pred_dict['object'][:, :, :3])
            dists_forward *= validPlaneMask

            dists_forward = tf.reduce_mean(dists_forward)
            dists_backward = tf.reduce_mean(dists_backward)
            object_center_loss += (dists_forward + dists_backward * backwardLossWeight) * 10000

            forward_map = tf.one_hot(map_forward, depth=options.numOutputObjects, axis=-1)
            forward_map *= tf.expand_dims(validPlaneMask, -1)

            objects_gt_shuffled = tf.transpose(tf.matmul(global_gt_dict['object'], forward_map, transpose_a=True), [0, 2, 1])
            object_size_loss += tf.reduce_mean(tf.squared_difference(global_pred_dict['object'][:, :, 3:12], objects_gt_shuffled[:, :, 3:12])) * 10000
            debug_dict['object_gt'] = objects_gt_shuffled
            pass


        #regularization
        l2_losses = tf.add_n([options.l2Weight * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name])

        loss = object_center_loss + object_size_loss + l2_losses

        loss_dict = {'object_center': object_center_loss, 'object_size': object_size_loss}
        pass
    return loss, loss_dict, debug_dict


def main(options):
    if not os.path.exists(options.checkpoint_dir):
        os.system("mkdir -p %s"%options.checkpoint_dir)
        pass
    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass

    min_after_dequeue = 1000


    train_inputs = []
    val_inputs = []
    train_inputs.append('SUNCG_train.tfrecords')
    val_inputs.append('SUNCG_val.tfrecords')

    reader_train = RecordReader()
    filename_queue_train = tf.train.string_input_producer(train_inputs, num_epochs=10000)
    img_inp_train, global_gt_dict_train, local_gt_dict_train = reader_train.getBatch(filename_queue_train, numOutputObjects=options.numOutputObjects, batchSize=options.batchSize, min_after_dequeue=min_after_dequeue, getLocal=True)

    reader_val = RecordReader()
    filename_queue_val = tf.train.string_input_producer(val_inputs, num_epochs=10000)
    img_inp_val, global_gt_dict_val, local_gt_dict_val = reader_val.getBatch(filename_queue_val, numOutputObjects=options.numOutputObjects, batchSize=options.batchSize, min_after_dequeue=min_after_dequeue, getLocal=True)

    training_flag = tf.placeholder(tf.bool, shape=[], name='training_flag')

    #global_pred_dict, local_pred_dict, deep_pred_dicts = build_graph(img_inp_train, img_inp_val, img_inp_rgbd_train, img_inp_rgbd_val, img_inp_3d_train, img_inp_3d_val, training_flag, options)
    global_pred_dict, local_pred_dict, deep_pred_dicts = build_graph(img_inp_train, img_inp_val, training_flag, options)


    #loss, loss_dict, _ = build_loss(global_pred_dict, local_pred_dict, deep_pred_dicts, global_gt_dict_train, local_gt_dict_train, global_gt_dict_val, local_gt_dict_val, training_flag, options)
    #loss_rgbd, loss_dict_rgbd, _ = build_loss_rgbd(global_pred_dict, deep_pred_dicts, global_gt_dict_rgbd_train, global_gt_dict_rgbd_val, training_flag, options)
    loss, loss_dict, debug_dict = build_loss(img_inp_train, img_inp_val, global_pred_dict, deep_pred_dicts, global_gt_dict_train, global_gt_dict_val, training_flag, options)

    #loss = tf.cond(tf.less(training_flag, 2), lambda: loss, lambda: tf.cond(tf.less(training_flag, 4), lambda: loss_rgbd, lambda: loss_3d))


    #train_writer = tf.summary.FileWriter(options.log_dir + '/train')
    #val_writer = tf.summary.FileWriter(options.log_dir + '/val')
    #train_writer_rgbd = tf.summary.FileWriter(options.log_dir + '/train_rgbd')
    #val_writer_rgbd = tf.summary.FileWriter(options.log_dir + '/val_rgbd')
    #writers = [train_writer, val_writer, train_writer_rgbd, val_writer_rgbd]


    with tf.variable_scope('statistics'):
        batchno = tf.Variable(0, dtype=tf.int32, trainable=False, name='batchno')
        batchnoinc=batchno.assign(batchno+1)
        pass


    optimizer = tf.train.AdamOptimizer(options.LR)
    if options.crfrnn >= 0:
        train_op = optimizer.minimize(loss, global_step=batchno)
    else:
        var_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "crfrnn")
        print(var_to_train)
        train_op = optimizer.minimize(loss, global_step=batchno, var_list=var_to_train)
        pass

    var_to_restore = [v for v in tf.global_variables()]

    tf.summary.scalar('loss', loss)
    summary_op = tf.summary.merge_all()


    config=tf.ConfigProto()
    config.allow_soft_placement=True
    #config.gpu_options.allow_growth=True
    config.gpu_options.per_process_gpu_memory_fraction=0.9
    saver=tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if options.restore == 0:
            #fine-tune from DeepLab model
            var_to_restore = [v for v in var_to_restore if 'res5d' not in v.name and 'segmentation' not in v.name and 'plane' not in v.name and 'deep_supervision' not in v.name and 'local' not in v.name and 'boundary' not in v.name and 'degridding' not in v.name and 'res2a_branch2a' not in v.name and 'res2a_branch1' not in v.name and 'Adam' not in v.name and 'beta' not in v.name and 'statistics' not in v.name and 'semantics' not in v.name]
            pretrained_model_loader = tf.train.Saver(var_to_restore)
            pretrained_model_loader.restore(sess, options.modelPathDeepLab)
        elif options.restore == 1:
            #restore the same model from checkpoint
            loader = tf.train.Saver(var_to_restore)
            loader.restore(sess,"%s/checkpoint.ckpt"%(options.checkpoint_dir))
            bno=sess.run(batchno)
            print(bno)
        elif options.restore == 2:
            var_to_restore = [v for v in var_to_restore if 'object' not in v.name]
            loader = tf.train.Saver(var_to_restore)
            loader.restore(sess, '../PlaneNetFinal/checkpoint/sample_np10_hybrid3_bl0_dl0_ds0_crfrnn5_sm0/checkpoint.ckpt')
            sess.run(batchno.assign(1))
            pass
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        MOVING_AVERAGE_DECAY = 0.99
        ema = [0., 0., 0., 0.]
        ema_acc = [1e-10, 1e-10, 1e-10, 1e-10]
        last_snapshot_time = time.time()
        bno=sess.run(batchno)

        try:
            while bno<300000:
                t0 = time.time()

                batchIndexPeriod = bno % 100
                if batchIndexPeriod < len(options.hybrid):
                    #batchType = int(options.hybrid[batchIndexPeriod]) * 2 + 1
                    batchType = 1
                    _, total_loss, losses, summary_str, pred = sess.run([batchnoinc, loss, loss_dict, summary_op, global_pred_dict], feed_dict = {training_flag: batchType == 0})
                else:
                    batchType = 0
                    _, total_loss, losses, summary_str, pred, debug, img, gt = sess.run([train_op, loss, loss_dict, summary_op, global_pred_dict, debug_dict, img_inp_train, global_gt_dict_train], feed_dict = {training_flag: batchType == 0})

                    if bno % (100 + 400 * int(options.crfrnn == 0)) == 50:
                        for batchIndex in xrange(options.batchSize):
                            #print(losses)
                            #print(debug['plane'][batchIndex])
                            continue
                        #exit(1)
                        pass
                    pass


                ema[batchType] = ema[batchType] * MOVING_AVERAGE_DECAY + total_loss
                ema_acc[batchType] = ema_acc[batchType] * MOVING_AVERAGE_DECAY + 1

                bno = sess.run(batchno)
                if time.time()-last_snapshot_time > options.saveInterval:
                    print('save snapshot')
                    saver.save(sess,'%s/checkpoint.ckpt'%options.checkpoint_dir)
                    last_snapshot_time = time.time()
                    pass

                print bno,'train', ema[0] / ema_acc[0], 'val', ema[1] / ema_acc[1], 'train rgbd', ema[2] / ema_acc[2], 'val rgbd', ema[3] / ema_acc[3], 'loss', total_loss, 'time', time.time()-t0

                if np.random.random() < 0.01:
                    print(losses)
                    pass
                continue

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            pass

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()
        pass
    return


def test(options):
    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass

    options.dataset = 'SUNCG'

    options.batchSize = 1
    min_after_dequeue = 1000

    inputs = []
    inputs.append('SUNCG_train.tfrecords')

    reader = RecordReader()
    filename_queue = tf.train.string_input_producer(inputs, num_epochs=10000)
    img_inp, global_gt_dict, local_gt_dict = reader.getBatch(filename_queue, numOutputObjects=options.numOutputObjects, batchSize=options.batchSize, min_after_dequeue=min_after_dequeue, getLocal=True, random=False)


    training_flag = tf.constant(False, tf.bool)

    global_pred_dict, local_pred_dict, deep_pred_dicts = build_graph(img_inp, img_inp, training_flag, options)
    var_to_restore = tf.global_variables()

    loss, loss_dict, debug_dict = build_loss(img_inp, img_inp, global_pred_dict, deep_pred_dicts, global_gt_dict, global_gt_dict, training_flag, options)


    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        #var_to_restore = [v for v in var_to_restore if 'res4b22_relu_non_plane' not in v.name]
        loader = tf.train.Saver(var_to_restore)
        loader.restore(sess, "%s/checkpoint.ckpt"%(options.checkpoint_dir))
        #loader.restore(sess, "%s/checkpoint.ckpt"%('checkpoint/planenet_pb_pp_hybrid1'))
        #loader.restore(sess, options.fineTuningCheckpoint)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)


        try:
            gtDepths = []
            predDepths = []

            imageWidth = WIDTH
            imageHeight = HEIGHT
            focalLength = 517.97
            urange = np.arange(imageWidth).reshape(1, -1).repeat(imageHeight, 0) - imageWidth * 0.5
            vrange = np.arange(imageHeight).reshape(-1, 1).repeat(imageWidth, 1) - imageHeight * 0.5
            ranges = np.array([urange / imageWidth * 640 / focalLength, np.ones(urange.shape), -vrange / imageHeight * 480 / focalLength]).transpose([1, 2, 0])


            for index in xrange(options.numImages):
                print(('image', index))
                t0=time.time()

                img, global_gt, local_gt, global_pred, local_pred, deep_preds, total_loss, losses, debug = sess.run([img_inp, global_gt_dict, local_gt_dict, global_pred_dict, local_pred_dict, deep_pred_dicts, loss, loss_dict, debug_dict])

                print(losses)
                print(total_loss)
                #print(losses)
                #exit(1)
                im = img[0]
                image = ((im + 0.5) * 255).astype(np.uint8)

                gt_d = global_gt['depth'].squeeze()

                if index >= options.visualizeImages:
                    continue

                #cv2.imwrite(options.test_dir + '/' + str(index) + '_image.png', image)
                #cv2.imwrite(options.test_dir + '/' + str(index) + '_depth.png', drawDepthImage(gt_d))
                cv2.imwrite(options.test_dir + '/' + str(index) + '_object_gt.png', drawObjectImage(image, global_gt['object'][0], global_gt['info'][0]))
                cv2.imwrite(options.test_dir + '/' + str(index) + '_object_pred.png', drawObjectImage(image, global_pred['object'][0], global_gt['info'][0]))
                continue

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            pass

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()
        pass
    writeHTML(options)
    return


def writeHTML(options):
    from html import HTML

    h = HTML('html')
    h.p('Results')
    h.br()
    path = '.'
    #methods = ['planenet', 'pixelwise', 'pixelwise+RANSAC', 'GT+RANSAC', 'planenet+crf', 'pixelwise+semantics+RANSAC']
    #methods = ['planenet', 'pixelwise', 'pixelwise+RANSAC', 'GT+RANSAC']

    for index in xrange(options.visualizeImages):
        t = h.table(border='1')
        r_title = t.tr()
        r_title.td('Groundtruth')
        r_title.td('Prediction')

        r = t.tr()
        r.td().img(src=path + '/' + str(index) + '_object_gt.png')
        r.td().img(src=path + '/' + str(index) + '_object_pred.png')
        continue

    html_file = open(options.test_dir + '/index.html', 'w')
    html_file.write(str(h))
    html_file.close()
    return

def writeInfo(options):
    x = (np.arange(11) * 0.1).tolist()
    ys = []
    ys.append(np.load('test/planenet_pixel_IOU.npy').tolist())
    ys.append(np.load('test/pixelwise_pred_pixel_IOU.npy').tolist())
    ys.append(np.load('test/pixelwise_gt_pixel_IOU.npy').tolist())
    plotCurves(x, ys, filename = 'test/object_comparison.png', xlabel='IOU', ylabel='pixel coverage', labels=['planenet', 'pixelwise+RANSAC', 'GT+RANSAC'])

    x = (0.5 - np.arange(11) * 0.05).tolist()
    ys = []
    ys.append(np.load('test/planenet_pixel_diff.npy').tolist())
    ys.append(np.load('test/pixelwise_pred_pixel_diff.npy').tolist())
    ys.append(np.load('test/pixelwise_gt_pixel_diff.npy').tolist())
    plotCurves(x, ys, filename = 'test/object_comparison_diff.png', xlabel='diff', ylabel='pixel coverage', labels=['planenet', 'pixelwise+RANSAC', 'GT+RANSAC'])

    return

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Planenet')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    #task: [train, test, predict]
    parser.add_argument('--task', dest='task',
                        help='task type: [train, test, predict]',
                        default='train', type=str)
    parser.add_argument('--restore', dest='restore',
                        help='how to restore the model',
                        default=0, type=int)
    parser.add_argument('--numOutputObjects', dest='numOutputObjects',
                        help='the number of output planes',
                        default=10, type=int)
    parser.add_argument('--batchSize', dest='batchSize',
                        help='batch size',
                        default=4, type=int)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset name for test/predict',
                        default='', type=str)
    parser.add_argument('--numImages', dest='numImages',
                        help='the number of images to test/predict',
                        default=10, type=int)
    parser.add_argument('--visualizeImages', dest='visualizeImages',
                        help='the number of images to visualize',
                        default=10, type=int)
    parser.add_argument('--boundaryLoss', dest='boundaryLoss',
                        help='use boundary loss: [0, 1]',
                        default=0, type=int)
    parser.add_argument('--diverseLoss', dest='diverseLoss',
                        help='use diverse loss: [0, 1]',
                        default=0, type=int)
    parser.add_argument('--labelLoss', dest='labelLoss',
                        help='use label loss: [0, 1]',
                        default=0, type=int)
    parser.add_argument('--planeLoss', dest='planeLoss',
                        help='use plane loss: [0, 1]',
                        default=1, type=int)
    parser.add_argument('--depthLoss', dest='depthLoss',
                        help='use depth loss: [0, 1, 2]',
                        default=1, type=int)
    parser.add_argument('--deepSupervision', dest='deepSupervision',
                        help='deep supervision level: [0, 1, 2]',
                        default=0, type=int)
    parser.add_argument('--sameMatching', dest='sameMatching',
                        help='use the same matching for all deep supervision layers and the final prediction: [0, 1]',
                        default=0, type=int)
    parser.add_argument('--anchorPlanes', dest='anchorPlanes',
                        help='use anchor planes for all deep supervision layers and the final prediction: [0, 1]',
                        default=0, type=int)
    parser.add_argument('--crf', dest='crf',
                        help='the number of CRF iterations',
                        default=0, type=int)
    parser.add_argument('--crfrnn', dest='crfrnn',
                        help='the number of CRF (as RNN) iterations',
                        default=0, type=int)
    parser.add_argument('--backwardLossWeight', dest='backwardLossWeight',
                        help='backward matching loss',
                        default=0, type=float)
    parser.add_argument('--predictBoundary', dest='predictBoundary',
                        help='whether predict boundary or not: [0, 1]',
                        default=0, type=int)
    parser.add_argument('--predictSemantics', dest='predictSemantics',
                        help='whether predict semantics or not: [0, 1]',
                        default=0, type=int)
    parser.add_argument('--predictLocal', dest='predictLocal',
                        help='whether predict local planes or not: [0, 1]',
                        default=0, type=int)
    parser.add_argument('--predictConfidence', dest='predictConfidence',
                        help='whether predict plane confidence or not: [0, 1]',
                        default=0, type=int)
    parser.add_argument('--predictPixelwise', dest='predictPixelwise',
                        help='whether predict pixelwise depth or not: [0, 1]',
                        default=0, type=int)
    parser.add_argument('--fineTuningCheckpoint', dest='fineTuningCheckpoint',
                        help='specify the model for fine-tuning',
                        default='../PlaneSetGeneration/dump_planenet_diverse/train_planenet_diverse.ckpt', type=str)
    parser.add_argument('--suffix', dest='suffix',
                        help='add a suffix to keyname to distinguish experiments',
                        default='', type=str)
    parser.add_argument('--l2Weight', dest='l2Weight',
                        help='L2 regulation weight',
                        default=5e-4, type=float)
    parser.add_argument('--LR', dest='LR',
                        help='learning rate',
                        default=3e-5, type=float)
    parser.add_argument('--hybrid', dest='hybrid',
                        help='hybrid training',
                        default='3', type=str)
    parser.add_argument('--dataFolder', dest='dataFolder',
                        help='data folder',
                        default='/mnt/vision/PlaneNet/', type=str)
    parser.add_argument('--saveInterval', dest='saveInterval',
                        help='save interval',
                        default=900, type=int)
    parser.add_argument('--modelPathDeepLab', dest='modelPathDeepLab',
                        help='DeepLab model path',
                        default='../PretrainedModels/deeplab_resnet.ckpt', type=str)

    args = parser.parse_args()
    args.keyname = 'scenenet'


    #args.predictSemantics = 0
    #args.predictBoundary = 0

    args.checkpoint_dir = 'checkpoint/' + args.keyname
    args.log_dir = 'log/' + args.keyname
    args.test_dir = 'test/' + args.keyname + '_' + args.dataset
    args.predict_dir = 'predict/' + args.keyname + '_' + args.dataset
    args.dump_dir = 'dump/' + args.keyname

    #layers where deep supervision happens
    args.deepSupervisionLayers = []
    if args.deepSupervision >= 1:
        args.deepSupervisionLayers.append('res4b22_relu')
        pass
    if args.deepSupervision >= 2:
        args.deepSupervisionLayers.append('res4b12_relu')
        pass
    return args


if __name__=='__main__':

    # plane = np.load('temp/plane.npy')
    # depth = np.load('temp/depth.npy')
    # segmentation = np.load('temp/segmentation.npy')
    # info = np.load('temp/info.npy')
    # num_planes = np.load('temp/num_planes.npy')
    # segmentation = np.argmax(segmentation, axis=-1)
    # print(segmentation.shape)
    # planes, segmentation, numPlanes = removeSmallSegments(plane, np.zeros((HEIGHT, WIDTH, 3)), depth.squeeze(), np.zeros((HEIGHT, WIDTH, 3)), segmentation, np.zeros((HEIGHT, WIDTH)), info, num_planes)
    # print(planes)
    # exit(1)

    args = parse_args()

    print "keyname=%s task=%s started"%(args.keyname, args.task)
    try:
        if args.task == "train":
            main(args)
        elif args.task == "test":
            test(args)
        elif args.task == "predict":
            predict(args)
        elif args.task == "fit":
            fitPlanesRGBD(args)
        elif args.task == "write":
            writeInfo(args)
        else:
            assert False,"format wrong"
            pass
    finally:
        pass
