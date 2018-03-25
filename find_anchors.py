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
from yolo import Yolo
from RecordReaderLocal import *

#from SegmentationRefinement import *

#training_flag: toggle dropout and batch normalization mode
#it's true for training and false for validation, testing, prediction
#it also controls which data batch to use (*_train or *_val)


def build_graph(img_inp_train, img_inp_val, training_flag, options):
    with tf.device('/gpu:%d'%options.gpu_id):
        img_inp = tf.cond(training_flag, lambda: img_inp_train, lambda: img_inp_val)

        if options.model == 'yolo':
            net = Yolo(options, {'img_inp': img_inp}, is_training=training_flag)
        else:
            net = SceneNet({'img_inp': img_inp}, is_training=training_flag, options=options)
            pass
        #global predictions

        if False:
            object_pred = gt_dict['object']
            pass

        global_pred_dict = {}

        #local predictions
        if options.predictLocal:
            local_parameters = net.layers['local_object_pred']
            #local_parameters = tf.concat([tf.sigmoid(local_parameters[:, :, :, :2]), tf.exp(local_parameters[:, :, :, 2:4]), tf.sigmoid(local_parameters[:, :, :, 4:6]) * 10, local_parameters[:, :, :, 6:]], axis=-1)
            local_pred_dict = {'score': net.layers['local_score_pred'], 'object': local_parameters, 'class': net.layers['local_class_pred']}
        else:
            local_pred_dict = {}
            pass
        deep_pred_dicts = {}

    return global_pred_dict, local_pred_dict, deep_pred_dicts


def build_loss(img_inp_train, img_inp_val, global_pred_dict, local_pred_dict, deep_pred_dicts, global_gt_dict_train, global_gt_dict_val, local_gt_dict_train, local_gt_dict_val, training_flag, options):
    with tf.device('/gpu:%d'%options.gpu_id):
        debug_dict = {}

        img_inp = tf.cond(training_flag, lambda: img_inp_train, lambda: img_inp_val)
        local_gt_dict = {}
        for name in local_gt_dict_train.keys():
            local_gt_dict[name] = tf.cond(training_flag, lambda: local_gt_dict_train[name], lambda: local_gt_dict_val[name])
            continue
        global_gt_dict = {}
        for name in global_gt_dict_train.keys():
            global_gt_dict[name] = tf.cond(training_flag, lambda: global_gt_dict_train[name], lambda: global_gt_dict_val[name])
            continue

        info = local_gt_dict['info'][0]

        #local_pred_dict['object'] = local_gt_dict['object']
        #local_pred_dict['score'] = (local_gt_dict['score'] - 0.5) * 100


        local_score_loss = tf.losses.sigmoid_cross_entropy(logits=local_pred_dict['score'], multi_class_labels=local_gt_dict['score'], weights = tf.maximum(local_gt_dict['score'] * 5, 1))
        #local_score_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=local_pred_dict['score'], labels=local_gt_dict['score'])
        #local_score_loss = tf.reduce_mean(tf.square_difference(local_pred_dict['score'], local_gt_dict['score']))

        if options.axisAligned:
            if options.useAnchor:
                parameters_gt = local_gt_dict['object'][:, :, :, :6]
                parameters_gt = tf.concat([-tf.log(tf.maximum(1 / tf.maximum(parameters_gt[:, :, :, :2], 1e-4) - 1, 1e-4)), tf.log(tf.maximum(parameters_gt[:, :, :, 2:4], 1e-4)), parameters_gt[:, :, :, 4:]], axis=-1)
                local_object_loss = tf.reduce_mean(tf.squared_difference(local_pred_dict['object'][:, :, :, :6], parameters_gt) * local_gt_dict['score'])
                parameters_pred = local_pred_dict['object']
                local_pred_dict['object'] = tf.concat([tf.sigmoid(parameters_pred[:, :, :, :2]), tf.exp(parameters_pred[:, :, :, 2:4]), parameters_pred[:, :, :, 4:]], axis=-1)
            else:
                local_object_loss = tf.reduce_mean(tf.squared_difference(local_pred_dict['object'] * local_gt_dict['score'], local_gt_dict['object'])[:, :, :, :6])
                pass
        else:
            local_object_loss = tf.reduce_mean(tf.squared_difference(local_pred_dict['object'] * local_gt_dict['score'], local_gt_dict['object']))
            pass

        if options.considerClass:
            local_class_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=local_pred_dict['class'], labels=local_gt_dict['class']) * local_gt_dict['score'])
            #local_class_loss = tf.reduce_mean(tf.squared_difference(local_pred_dict['class'] * local_gt_dict['score'], local_gt_dict['class'])) * 100
        else:
            local_class_loss = 0
            pass

        #regularization
        l2_losses = tf.add_n([options.l2Weight * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]) * 0

        loss = local_score_loss + local_object_loss + local_class_loss + l2_losses

        loss_dict = {'score': local_score_loss, 'object': local_object_loss, 'class': local_class_loss, 'l2': l2_losses}
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
    img_inp_train, global_gt_dict_train, local_gt_dict_train = reader_train.getBatch(filename_queue_train, options, min_after_dequeue=min_after_dequeue)

    reader_val = RecordReader()
    filename_queue_val = tf.train.string_input_producer(val_inputs, num_epochs=10000)
    img_inp_val, global_gt_dict_val, local_gt_dict_val = reader_val.getBatch(filename_queue_val, options, min_after_dequeue=min_after_dequeue)

    training_flag = tf.placeholder(tf.bool, shape=[], name='training_flag')

    #global_pred_dict, local_pred_dict, deep_pred_dicts = build_graph(img_inp_train, img_inp_val, img_inp_rgbd_train, img_inp_rgbd_val, img_inp_3d_train, img_inp_3d_val, training_flag, options)
    global_pred_dict, local_pred_dict, deep_pred_dicts = build_graph(img_inp_train, img_inp_val, training_flag, options)


    loss, loss_dict, debug_dict = build_loss(img_inp_train, img_inp_val, global_pred_dict, local_pred_dict, deep_pred_dicts, global_gt_dict_train, global_gt_dict_val, local_gt_dict_train, local_gt_dict_val, training_flag, options)


    train_writer = tf.summary.FileWriter(options.log_dir + '/train')
    val_writer = tf.summary.FileWriter(options.log_dir + '/val')
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
    tf.summary.scalar('score loss', loss_dict['score'])
    tf.summary.scalar('object loss', loss_dict['object'])
    tf.summary.scalar('class loss', loss_dict['class'])
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
            if options.model == 'yolo':
                var_to_restore = [v for v in var_to_restore if 'pred' not in v.name]
                pretrained_model_loader = tf.train.Saver(var_to_restore)
                pretrained_model_loader.restore(sess, '../PretrainedModels/Darknet19/checkpoint.ckpt')
            else:
                var_to_restore = [v for v in var_to_restore if 'res5d' not in v.name and 'segmentation' not in v.name and 'plane' not in v.name and 'deep_supervision' not in v.name and 'local' not in v.name and 'boundary' not in v.name and 'degridding' not in v.name and 'res2a_branch2a' not in v.name and 'res2a_branch1' not in v.name and 'Adam' not in v.name and 'beta' not in v.name and 'statistics' not in v.name and 'semantics' not in v.name and 'object' not in v.name]
                pretrained_model_loader = tf.train.Saver(var_to_restore)
                pretrained_model_loader.restore(sess, options.modelPathDeepLab)
                pass
        elif options.restore == 1:
            #restore the same model from checkpoint
            loader = tf.train.Saver(var_to_restore)
            if options.startIteration == 0:
                loader.restore(sess,"%s/checkpoint.ckpt"%(options.checkpoint_dir))
            else:
                loader.restore(sess,"%s/checkpoint_%d.ckpt"%(options.checkpoint_dir, options.startSteration))
                pass
            bno=sess.run(batchno)
            print(bno)
        elif options.restore == 2:
            var_to_restore = [v for v in var_to_restore if 'object' not in v.name and 'local' not in v.name]
            loader = tf.train.Saver(var_to_restore)
            loader.restore(sess, '../PlaneNetFinal/checkpoint/sample_np10_hybrid3_bl0_dl0_ds0_crfrnn5_sm0/checkpoint.ckpt')
            sess.run(batchno.assign(0))
        elif options.restore == 3:
            var_to_restore = [v for v in var_to_restore if 'local' not in v.name]
            loader = tf.train.Saver(var_to_restore)
            loader.restore(sess,"%s/checkpoint.ckpt"%(options.checkpoint_dir))
            #bno=sess.run(batchno)
            sess.run(batchno.assign(0))
            pass
        elif options.restore == 4:
            loader = tf.train.Saver(var_to_restore)
            loader.restore(sess,"checkpoint/detection/checkpoint.ckpt")
            #bno=sess.run(batchno)
            sess.run(batchno.assign(0))
            pass

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        MOVING_AVERAGE_DECAY = 0.99
        ema = [0., 0., 0., 0.]
        ema_acc = [1e-10, 1e-10, 1e-10, 1e-10]
        last_snapshot_time = time.time()
        bno=sess.run(batchno)

        while bno < 300000:
            t0 = time.time()
            #try:
            if True:
                for iteration in xrange(1000):
                    if iteration == 500:
                        _, total_loss, losses, summary_str, img, gt, pred = sess.run([train_op, loss, loss_dict, summary_op, img_inp_train, local_gt_dict_train, local_pred_dict], feed_dict = {training_flag: True})
                        visualizeBatch(options, img, gt, pred, 'train')
                    else:
                        _, total_loss, losses, summary_str = sess.run([train_op, loss, loss_dict, summary_op], feed_dict = {training_flag: True})
                        pass
                    ema[0] = ema[0] * MOVING_AVERAGE_DECAY + total_loss
                    ema_acc[0] = ema_acc[0] * MOVING_AVERAGE_DECAY + 1

                    print(bno + iteration, 'train', ema[0] / ema_acc[0], 'val', ema[1] / ema_acc[1], 'losses', losses['object'], losses['score'], losses['class'])
                    train_writer.add_summary(summary_str, bno)
                    continue

                for iteration in xrange(10):
                    if iteration == 5:
                        total_loss, losses, summary_str, img, gt, pred = sess.run([loss, loss_dict, summary_op, img_inp_val, local_gt_dict_val, local_pred_dict], feed_dict = {training_flag: False})
                        visualizeBatch(options, img, gt, pred, 'val')
                    else:
                        total_loss, losses, summary_str = sess.run([loss, loss_dict, summary_op], feed_dict = {training_flag: False})
                        pass
                    ema[1] = ema[1] * MOVING_AVERAGE_DECAY + total_loss
                    ema_acc[1] = ema_acc[1] * MOVING_AVERAGE_DECAY + 1

                    print(bno + iteration, 'train', ema[0] / ema_acc[0], 'val', ema[1] / ema_acc[1], 'losses', losses['object'], losses['score'])
                    val_writer.add_summary(summary_str, bno)
                    continue

                bno = sess.run(batchno)
                print('save snapshot')
                saver.save(sess,'%s/checkpoint.ckpt'%options.checkpoint_dir)
                if bno % 50000 <= 1:
                    saver.save(sess,'%s/checkpoint_%d.ckpt'%(options.checkpoint_dir, bno))
                    pass
                continue

            # except tf.errors.OutOfRangeError:
            #     print('Done training -- epoch limit reached')
            # finally:
            #     # When done, ask the threads to stop.
            #     coord.request_stop()
            #     pass
            continue

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
    inputs.append('SUNCG_val.tfrecords')

    reader = RecordReader()
    filename_queue = tf.train.string_input_producer(inputs, num_epochs=10000)
    img_inp, global_gt_dict, local_gt_dict = reader.getBatch(filename_queue, options, min_after_dequeue=min_after_dequeue, random=False)


    training_flag = tf.constant(False, tf.bool)

    global_pred_dict, local_pred_dict, deep_pred_dicts = build_graph(img_inp, img_inp, training_flag, options)
    var_to_restore = tf.global_variables()

    loss, loss_dict, debug_dict = build_loss(img_inp, img_inp, global_pred_dict, local_pred_dict, deep_pred_dicts, global_gt_dict, global_gt_dict, local_gt_dict, local_gt_dict, training_flag, options)


    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        #var_to_restore = [v for v in var_to_restore if 'res4b22_relu_non_plane' not in v.name]
        loader = tf.train.Saver(var_to_restore)
        #loader.restore(sess, '%s/checkpoint.ckpt'%(options.checkpoint_dir))
        if options.startIteration == 0:
            loader.restore(sess,"%s/checkpoint.ckpt"%(options.checkpoint_dir))
        else:
            loader.restore(sess,"%s/checkpoint_%d.ckpt"%(options.checkpoint_dir, options.startSteration))
            pass
        #loader.restore(sess, "%s/checkpoint.ckpt"%('checkpoint/planenet_pb_pp_hybrid1'))
        #loader.restore(sess, options.fineTuningCheckpoint)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)


        for index in xrange(options.numImages):
            #print(('image', index))
            t0=time.time()

            img, local_gt, local_pred, total_loss, losses = sess.run([img_inp, local_gt_dict, local_pred_dict, loss, loss_dict])

            #if index != 152:
            #continue

            print(losses, total_loss)

            # if total_loss > 100:

            #     print(local_gt['object'][0].reshape((-1, 12))[local_gt['score'][0].reshape(-1) > 0.5])
            #     print(local_gt['image_path'])
            #     exit(1)
            #     pass

            # if local_gt['score'].min() < 0 or local_gt['score'].max() > 1:
            #     print('invalid score', local_gt['score'], local_gt['image_path'])
            #     exit(1)
            #     pass
            # if local_gt['class'].min() < 0 or local_gt['class'].max() > 1:
            #     print('invalid class', local_gt['class'], local_gt['image_path'])
            #     exit(1)
            #     pass
            # if local_gt['object'].min() < 0 or local_gt['object'][:, :, :, :2].max() > 1 or local_gt['object'][:, :, :, 4:6].max() > 10:
            #     print('invalid object', local_gt['object'], local_gt['image_path'])
            #     exit(1)
            #     pass

            visualizeBatch(options, img, local_gt, local_pred, 'test', index)
            continue

        # except tf.errors.OutOfRangeError:
        #     print('Done training -- epoch limit reached')
        # finally:
        #     # When done, ask the threads to stop.
        #     coord.request_stop()
        #     pass

        coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()
        pass
    #writeHTML(options)
    return

def visualizeBatch(options, image, gt_dict, pred_dict, prefix, indexOffset=0):
    image = ((image + 0.5) * 255).astype(np.uint8)

    outputWidth = WIDTH / options.outputStride
    outputHeight = HEIGHT / options.outputStride
    anchorW = np.full((outputHeight, outputWidth, 1), 1.0 / outputWidth)
    anchorH = np.full((outputHeight, outputWidth, 1), 1.0 / outputHeight)
    anchors = np.stack([np.tile(np.expand_dims(np.arange(outputWidth, dtype=np.float32), 0), (outputHeight, 1)) / outputWidth, np.tile(np.expand_dims(np.arange(outputHeight, dtype=np.float32), 1), (1, outputWidth)) / outputHeight], axis=-1)
    anchors = np.concatenate([anchors, anchorW, anchorH], axis=-1).reshape((-1, 4))

    for batchIndex in xrange(options.batchSize):
        info = gt_dict['info'][batchIndex]

        #print('num objects', gt_dict['num_objects'][batchIndex])
        object_gt = gt_dict['object'][batchIndex].reshape((-1, 12))
        print(object_gt.min(0), object_gt.max(0))
        if options.useAnchor:
            object_gt[:, :2] = object_gt[:, :2] * anchors[:, 2:4] + anchors[:, :2]
            object_gt[:, 2:4] *= anchors[:, 2:4]
            centerZ = (object_gt[:, 0] * info[16] - info[2]) / info[0] * object_gt[:, 4]
            centerY = -(object_gt[:, 1] * info[17] - info[6]) / info[5] * object_gt[:, 4]
            sizeZ = object_gt[:, 2] * info[16] / info[0] * object_gt[:, 4]
            sizeY = object_gt[:, 3] * info[17] / info[5] * object_gt[:, 4]
            centers = np.stack([(object_gt[:, 4] + object_gt[:, 5]) / 2, centerY, centerZ], axis=-1)
            sizes = np.stack([object_gt[:, 5] - object_gt[:, 4], sizeY, sizeZ], axis=-1)
            object_gt = np.concatenate([centers, sizes], axis=-1)
            object_gt = np.concatenate([object_gt, np.zeros(object_gt.shape)], axis=-1)
            pass
        object_gt = np.concatenate([object_gt, np.expand_dims(np.argmax(gt_dict['class'][batchIndex].reshape((-1, options.numClasses)), axis=-1), -1)], axis=-1)
        object_gt = object_gt[gt_dict['score'][batchIndex].reshape(-1) > 0.5]
        # anchors = anchors[gt_dict['score'][batchIndex].reshape(-1) > 0.5]
        # print(object_gt)
        # print(anchors)

        # centers = object_gt[:, :3] - object_gt[:, 3:6] / 2 * np.expand_dims(np.array([1, 0, 0], dtype=np.float32), 0)
        # U = np.clip((centers[:, 2] / centers[:, 0] * info[0] + info[2]) / info[16], 0, 1)
        # V = np.clip((-centers[:, 1] / centers[:, 0] * info[5] + info[6]) / info[17], 0, 1)
        # print(U, V)
        # gridU = np.minimum(np.maximum((U * outputWidth).astype(np.int32), 0), outputWidth - 1)
        # gridV = np.minimum(np.maximum((V * outputHeight).astype(np.int32), 0), outputHeight - 1)
        # print(gridU, gridV)

        # mins = object_gt[:, :3] - object_gt[:, 3:6] / 2 * np.expand_dims(np.array([1, 1, -1], dtype=np.float32), 0)
        # maxs = object_gt[:, :3] - object_gt[:, 3:6] / 2 * np.expand_dims(np.array([1, -1, 1], dtype=np.float32), 0)
        # minU = (mins[:, 2] / mins[:, 0] * info[0] + info[2]) / info[16]
        # minV = (-mins[:, 1] / mins[:, 0] * info[5] + info[6]) / info[17]
        # maxU = (maxs[:, 2] / maxs[:, 0] * info[0] + info[2]) / info[16]
        # maxV = (-maxs[:, 1] / maxs[:, 0] * info[5] + info[6]) / info[17]
        # boxes = np.stack([U, V, maxU - minU, maxV - minV], axis=-1)

        # print((boxes[:, :2] - anchors[:, :2]) / anchors[:, 2:4])
        # print(boxes[:, 2:4] / anchors[:, 2:4])

        #cv2.imwrite(options.test_dir + '/' + str(index) + '_object_gt.png', drawObjectImage(image, dict_dict['object'][batchIndex], gt_dict['info'][batchIndex]))
        cv2.imwrite(options.test_dir + '/' + prefix + '_' + str(indexOffset + batchIndex) + '_object_gt.png', drawObjectImage(image[batchIndex], object_gt, gt_dict['info'][batchIndex], axis_aligned=options.axisAligned))
        #exit(1)

        object_pred = pred_dict['object'][batchIndex].reshape((-1, 12))
        if options.useAnchor:
            object_pred[:, :2] = object_pred[:, :2] * anchors[:, 2:4] + anchors[:, :2]
            object_pred[:, 2:4] *= anchors[:, 2:4]
            centerZ = (object_pred[:, 0] * info[16] - info[2]) / info[0] * object_pred[:, 4]
            centerY = -(object_pred[:, 1] * info[17] - info[6]) / info[5] * object_pred[:, 4]
            sizeZ = object_pred[:, 2] * info[16] / info[0] * object_pred[:, 4]
            sizeY = object_pred[:, 3] * info[17] / info[5] * object_pred[:, 4]
            centers = np.stack([(object_pred[:, 4] + object_pred[:, 5]) / 2, centerY, centerZ], axis=-1)
            sizes = np.stack([object_pred[:, 5] - object_pred[:, 4], sizeY, sizeZ], axis=-1)
            object_pred = np.concatenate([centers, sizes], axis=-1)
            object_pred = np.concatenate([object_pred, np.zeros(object_pred.shape)], axis=-1)
            pass
        object_pred = np.concatenate([object_pred, np.expand_dims(np.argmax(pred_dict['class'][batchIndex].reshape((-1, options.numClasses)), axis=-1), axis=-1)], axis=-1)

        object_pred = object_pred[pred_dict['score'][batchIndex].reshape(-1) > 0.5]
        cv2.imwrite(options.test_dir + '/' + prefix + '_' + str(indexOffset + batchIndex) + '_object_pred.png', drawObjectImage(image[batchIndex], object_pred, gt_dict['info'][batchIndex], axis_aligned=options.axisAligned))

        mask_gt = drawMaskImage(gt_dict['score'][batchIndex].squeeze())
        #mask_gt = cv2.dilate(mask_gt, np.ones((3, 3)))
        mask_gt = cv2.resize(mask_gt, (WIDTH, HEIGHT))
        cv2.imwrite(options.test_dir + '/' + prefix + '_' + str(indexOffset + batchIndex) + '_score_gt.png', mask_gt)
        mask_pred = drawMaskImage(sigmoid(pred_dict['score'][batchIndex].squeeze()))
        #mask_pred = cv2.dilate(mask_pred, np.ones((3, 3)))
        mask_pred = cv2.resize(mask_pred, (WIDTH, HEIGHT))
        cv2.imwrite(options.test_dir + '/' + prefix + '_' + str(indexOffset + batchIndex) + '_score_pred.png', mask_pred)
        continue
    return
    #exit(1)

def writeHTML(options, prefix):
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
        r.td().img(src=path + '/' + prefix + '_' + str(index) + '_object_gt.png')
        r.td().img(src=path + '/' + prefix + '_' + str(index) + '_object_pred.png')
        r.td().img(src=path + '/' + prefix + '_' + str(index) + '_score_gt.png')
        r.td().img(src=path + '/' + prefix + '_' + str(index) + '_score_pred.png')
        continue

    html_file = open(options.test_dir + '/index_' + prefix + '.html', 'w')
    html_file.write(str(h))
    html_file.close()
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
                        default=8, type=int)
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
                        default=1, type=int)
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
    parser.add_argument('--axisAligned', dest='axisAligned',
                        help='axis aligned',
                        action='store_true')
    parser.add_argument('--considerClass', dest='considerClass',
                        help='consider class',
                        action='store_false')
    parser.add_argument('--numClasses', dest='numClasses',
                        help='the number of classes',
                        default=40, type=int)
    parser.add_argument('--outputStride', dest='outputStride',
                        help='ouput stride',
                        default=8, type=int)
    parser.add_argument('--startIteration', dest='startIteration',
                        help='start iteration',
                        default=0, type=int)
    parser.add_argument('--useAnchor', dest='useAnchor',
                        help='use anchor',
                        action='store_false')
    parser.add_argument('--model', dest='model',
                        help='model',
                        default='detection', type=str)

    args = parser.parse_args()
    args.keyname = args.model

    if args.useAnchor:
        args.keyname += '_a'
        args.axisAligned = True
    elif args.axisAligned:
        args.keyname += '_aa'
        pass

    if args.outputStride != 8:
        args.keyname += '_' + str(args.outputStride)
        pass
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

    # writeHTML(args, 'train')
    # writeHTML(args, 'test')
    # writeHTML(args, 'val')
    # exit(1)

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
