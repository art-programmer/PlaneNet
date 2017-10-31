import tensorflow as tf
import numpy as np
np.set_printoptions(precision=2, linewidth=200)
import cv2
import os
import time
import sys
import tf_nndistance
import argparse
import glob
import PIL
import scipy.ndimage as ndimage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
from plane_utils import *
from modules import *


from planenet import PlaneNet
from RecordReader import *
from RecordReaderRGBD import *
from RecordReader3D import *
from RecordReaderAll import *
#from SegmentationRefinement import *

#training_flag: toggle dropout and batch normalization mode
#it's true for training and false for validation, testing, prediction
#it also controls which data batch to use (*_train or *_val)


def build_graph(img_inp_train, img_inp_val, training_flag, options):
    with tf.device('/gpu:%d'%options.gpu_id):
        img_inp = tf.cond(training_flag, lambda: img_inp_train, lambda: img_inp_val)
        
        net = PlaneNet({'img_inp': img_inp}, is_training=training_flag, options=options)

        #global predictions
        plane_pred = net.layers['plane_pred']
        
        segmentation_pred = net.layers['segmentation_pred']
        non_plane_mask_pred = net.layers['non_plane_mask_pred']
        non_plane_depth_pred = net.layers['non_plane_depth_pred']
        non_plane_normal_pred = net.layers['non_plane_normal_pred']
        non_plane_normal_pred = tf.nn.l2_normalize(non_plane_normal_pred, dim=-1)


        if False:
            plane_pred = gt_dict['plane']
            non_plane_mask_pred = gt_dict['non_plane_mask'] * 10
            non_plane_depth_pred = gt_dict['depth']
            non_plane_normal_pred = gt_dict['normal']            
            segmentation_pred = gt_dict['segmentation'][:, :, :, :20] * 10
            pass
        
        
        global_pred_dict = {'plane': plane_pred, 'segmentation': segmentation_pred, 'non_plane_mask': non_plane_mask_pred, 'non_plane_depth': non_plane_depth_pred, 'non_plane_normal': non_plane_normal_pred}

        if options.predictBoundary:
            global_pred_dict['boundary'] = net.layers['boundary_pred']
            pass
        if options.predictConfidence:
            global_pred_dict['confidence'] = net.layers['plane_confidence_pred']
            pass
        if options.predictSemantics:
            global_pred_dict['semantics'] = net.layers['semantics_pred']
            pass

        #local predictions
        if options.predictLocal:
            local_pred_dict = {'score': net.layers['local_score_pred'], 'plane': net.layers['local_plane_pred'], 'mask': net.layers['local_mask_pred']}
        else:
            local_pred_dict = {}
            pass

        
        #deep supervision
        deep_pred_dicts = []
        for layer in options.deepSupervisionLayers:
            pred_dict = {'plane': net.layers[layer+'_plane_pred'], 'segmentation': net.layers[layer+'_segmentation_pred'], 'non_plane_mask': net.layers[layer+'_non_plane_mask_pred']}
            #if options.predictConfidence:
            #pred_dict['confidence'] = net.layers[layer+'_plane_confidence_pred']
            #pass
            deep_pred_dicts.append(pred_dict)
            continue

        
        if options.anchorPlanes:
            anchors_np = np.load('dump/anchors_' + options.hybrid + '.npy')
            anchors = tf.reshape(tf.constant(anchors_np.reshape(-1)), anchors_np.shape)
            anchors = tf.tile(tf.expand_dims(anchors, 0), [options.batchSize, 1, 1])
            all_pred_dicts = deep_pred_dicts + [global_pred_dict]            
            for pred_index, pred_dict in enumerate(all_pred_dicts):
                all_pred_dicts[pred_index]['plane'] += anchors
                continue
            pass

        pass
    
    return global_pred_dict, local_pred_dict, deep_pred_dicts


def build_loss(img_inp_train, img_inp_val, global_pred_dict, deep_pred_dicts, global_gt_dict_train, global_gt_dict_val, training_flag, options):
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

        plane_parameters = tf.reshape(global_pred_dict['plane'], (-1, 3))
        info = global_gt_dict['info'][0]
        plane_depths = planeDepthsModule(plane_parameters, WIDTH, HEIGHT, info)
        plane_depths = tf.transpose(tf.reshape(plane_depths, [HEIGHT, WIDTH, -1, options.numOutputPlanes]), [2, 0, 1, 3])

        non_plane_depth = global_pred_dict['non_plane_depth']
        all_depths = tf.concat([plane_depths, non_plane_depth], axis=3)

        

        #if options.predictPixelwise == 1:
        depth_diff = global_pred_dict['non_plane_depth'] - global_gt_dict['depth']
        depth_diff_gx = depth_diff - tf.concat([tf.ones([options.batchSize, HEIGHT, 1, 1]), depth_diff[:, :, :WIDTH - 1]], axis=2)
        depth_diff_gy = depth_diff - tf.concat([tf.ones([options.batchSize, 1, WIDTH, 1]), depth_diff[:, :HEIGHT - 1]], axis=1)

        numValidPixels = tf.reduce_sum(validDepthMask, axis=[1, 2, 3])
        depth_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(depth_diff * validDepthMask, 2), axis=[1, 2, 3]) / numValidPixels - 0.5 * tf.pow(tf.reduce_sum(depth_diff * validDepthMask, axis=[1, 2, 3]) / numValidPixels, 2) + tf.reduce_sum((tf.pow(depth_diff_gx, 2) + tf.pow(depth_diff_gy, 2)) * validDepthMask, axis=[1, 2, 3]) / numValidPixels) * 1000
        
        #depth_loss += tf.reduce_mean(tf.squared_difference(global_pred_dict['non_plane_depth'], global_gt_dict['depth']) * validDepthMask) * 10000


        valid_normal_mask = tf.squeeze(tf.cast(tf.less(tf.slice(global_gt_dict['info'], [0, 19], [options.batchSize, 1]), 2), tf.float32))
        normal_gt = tf.nn.l2_normalize(global_gt_dict['normal'], dim=-1)
        normal_loss = tf.reduce_mean(tf.reduce_sum(-global_pred_dict['non_plane_normal'] * normal_gt * validDepthMask, axis=[1, 2, 3]) / numValidPixels * valid_normal_mask) * 1000
        #normal_loss = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(global_pred_dict['non_plane_normal'], global_gt_dict['normal']) * validDepthMask, axis=[1, 2, 3]) * valid_normal_mask) * 1000
        #normal_loss = tf.constant(0.0)

        valid_semantics_mask = tf.squeeze(tf.cast(tf.not_equal(tf.slice(global_gt_dict['info'], [0, 19], [options.batchSize, 1]), 1), tf.float32))
        semantics_loss = tf.reduce_mean(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=global_pred_dict['semantics'], labels=global_gt_dict['semantics']), axis=[1, 2]) * valid_semantics_mask) * 1000
        

        l2_losses = tf.add_n([options.l2Weight * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name])


        loss = depth_loss + normal_loss + semantics_loss + l2_losses

        #if options.pixelwiseLoss:
        #normal_loss = tf.reduce_mean(tf.squared_difference(global_pred_dict['non_plane_normal'], global_gt_dict['normal'])) * 1000
        #depth_loss = tf.reduce_mean(tf.squared_difference(global_pred_dict['non_plane_depth'], global_gt_dict['depth']) * validDepthMask) * 1000
        #pass

        loss_dict = {'depth': depth_loss, 'normal': normal_loss, 'semantics': semantics_loss}
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
    if '0' in options.hybrid:
        train_inputs.append(options.rootFolder + '/planes_SUNCG_train.tfrecords')
        val_inputs.append(options.rootFolder + '/planes_SUNCG_val.tfrecords')        
        pass
    if '1' in options.hybrid:
        for _ in xrange(10):
            train_inputs.append(options.rootFolder + '/planes_nyu_rgbd_train.tfrecords')
            train_inputs.append(options.rootFolder + '/planes_nyu_rgbd_labeled_train.tfrecords')
            val_inputs.append(options.rootFolder + '/planes_nyu_rgbd_val.tfrecords')
            continue
        pass
    if '2' in options.hybrid:
        train_inputs.append(options.rootFolder + '/planes_matterport_train.tfrecords')
        val_inputs.append(options.rootFolder + '/planes_matterport_val.tfrecords')
        pass
    if '3' in options.hybrid:
        train_inputs.append(options.rootFolder + '/planes_scannet_train.tfrecords')
        val_inputs.append(options.rootFolder + '/planes_scannet_val.tfrecords')
        pass
    
    reader_train = RecordReaderAll()
    filename_queue_train = tf.train.string_input_producer(train_inputs, num_epochs=10000)    
    img_inp_train, global_gt_dict_train, local_gt_dict_train = reader_train.getBatch(filename_queue_train, numOutputPlanes=options.numOutputPlanes, batchSize=options.batchSize, min_after_dequeue=min_after_dequeue, getLocal=True)

    reader_val = RecordReaderAll()
    filename_queue_val = tf.train.string_input_producer(val_inputs, num_epochs=10000)
    img_inp_val, global_gt_dict_val, local_gt_dict_val = reader_val.getBatch(filename_queue_val, numOutputPlanes=options.numOutputPlanes, batchSize=options.batchSize, min_after_dequeue=min_after_dequeue, getLocal=True)
    
    training_flag = tf.placeholder(tf.bool, shape=[], name='training_flag')
    
    #global_pred_dict, local_pred_dict, deep_pred_dicts = build_graph(img_inp_train, img_inp_val, img_inp_rgbd_train, img_inp_rgbd_val, img_inp_3d_train, img_inp_3d_val, training_flag, options)
    global_pred_dict, local_pred_dict, deep_pred_dicts = build_graph(img_inp_train, img_inp_val, training_flag, options)
    
    
    #loss, loss_dict, _ = build_loss(global_pred_dict, local_pred_dict, deep_pred_dicts, global_gt_dict_train, local_gt_dict_train, global_gt_dict_val, local_gt_dict_val, training_flag, options)
    #loss_rgbd, loss_dict_rgbd, _ = build_loss_rgbd(global_pred_dict, deep_pred_dicts, global_gt_dict_rgbd_train, global_gt_dict_rgbd_val, training_flag, options)
    loss, loss_dict, _ = build_loss(img_inp_train, img_inp_val, global_pred_dict, deep_pred_dicts, global_gt_dict_train, global_gt_dict_val, training_flag, options)    
        
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
    train_op = optimizer.minimize(loss, global_step=batchno)

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
            var_to_restore = [v for v in var_to_restore if 'res5d' not in v.name and 'segmentation' not in v.name and 'plane' not in v.name and 'deep_supervision' not in v.name and 'local' not in v.name and 'boundary' not in v.name and 'degridding' not in v.name and 'res2a_branch2a' not in v.name and 'res2a_branch1' not in v.name]
            pretrained_model_loader = tf.train.Saver(var_to_restore)
            pretrained_model_loader.restore(sess,"../pretrained_models/deeplab_resnet.ckpt")
        elif options.restore == 1:
            #restore the same model from checkpoint
            loader = tf.train.Saver(var_to_restore)
            loader.restore(sess,"%s/checkpoint.ckpt"%(options.checkpoint_dir))
            bno=sess.run(batchno)
            print(bno)
        elif options.restore == 2:            
            #restore the same model from checkpoint but reset batchno to 1
            loader = tf.train.Saver(var_to_restore)
            loader.restore(sess,"%s/checkpoint.ckpt"%(options.checkpoint_dir))
            sess.run(batchno.assign(1))
        elif options.restore == 3:            
            #restore the same model from standard training
            # if options.predictBoundary == 1:
            #     var_to_restore = [v for v in var_to_restore if 'boundary' not in v.name]
            #     pass            
            # if options.predictConfidence == 1:
            #     var_to_restore = [v for v in var_to_restore if 'confidence' not in v.name]
            #     pass
            if options.predictSemantics == 1:
                var_to_restore = [v for v in var_to_restore if 'semantics' not in v.name]
                pass
            
            loader = tf.train.Saver(var_to_restore)
            if len(options.hybrid) == 1:
                hybrid = options.hybrid
            else:
                hybrid = str(3)
                pass
            #loader.restore(sess, options.rootFolder + '/checkpoint/planenet_hybrid' + hybrid + '_bl0_ll1_bw0.5_pb_pp_ps_sm0/checkpoint.ckpt')
            loader.restore(sess, options.rootFolder + '/checkpoint/planenet_hybrid' + hybrid + '_bl0_dl0_ll1_bw0.5_pb_pp/checkpoint.ckpt')            
            #loader.restore(sess,"checkpoint/planenet/checkpoint.ckpt")
            sess.run(batchno.assign(1))
        elif options.restore == 4:
            #fine-tune another model
            #var_to_restore = [v for v in var_to_restore if 'res4b22_relu_non_plane' not in v.name]
            loader = tf.train.Saver(var_to_restore)
            loader.restore(sess, options.fineTuningCheckpoint)
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
                    _, total_loss, losses, summary_str, gt_dict = sess.run([batchnoinc, loss, loss_dict, summary_op, global_pred_dict], feed_dict = {training_flag: batchType == 0})
                    
                else:
                    batchType = 0
                    _, total_loss, losses, summary_str, gt_dict = sess.run([train_op, loss, loss_dict, summary_op, global_pred_dict], feed_dict = {training_flag: batchType == 0})
                    pass

                # for batchIndex in xrange(options.batchSize):
                #     if np.isnan(global_gt['plane'][batchIndex]).any():
                #         #print(losses)
                #         #print(global_gt['plane'][batchIndex])
                #         print(global_gt['num_planes'][batchIndex])
                #         for planeIndex in xrange(global_gt['num_planes'][batchIndex]):
                #             cv2.imwrite('test/mask_' + str(planeIndex) + '.png', drawMaskImage(global_gt['segmentation'][batchIndex, :, :, planeIndex]))
                #             continue
                #         np.save('temp/plane.npy', global_gt['plane'][batchIndex])                        
                #         np.save('temp/depth.npy', global_gt['depth'][batchIndex])
                #         np.save('temp/segmentation.npy', global_gt['segmentation'][batchIndex])
                #         np.save('temp/info.npy', global_gt['info'][batchIndex])
                #         np.save('temp/num_planes.npy', global_gt['num_planes'][batchIndex])
                #         planes, segmentation, numPlanes = removeSmallSegments(global_gt['plane'][batchIndex], np.zeros((HEIGHT, WIDTH, 3)), global_gt['depth'][batchIndex].squeeze(), np.zeros((HEIGHT, WIDTH, 3)), np.argmax(global_gt['segmentation'][batchIndex], axis=-1), global_gt['semantics'][batchIndex], global_gt['info'][batchIndex], global_gt['num_planes'][batchIndex])
                #         print(planes)
                #         exit(1)
                #         pass
                #     continue
                #writers[batchType].add_summary(summary_str, bno)
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

    if options.dataset == '':
        assert(len(options.hybrid) == 1)
        if options.hybrid == '0':
            options.dataset = 'SUNCG'
        elif options.hybrid == '1':
            options.dataset = 'NYU_RGBD'
        elif options.hybrid == '2':
            options.dataset = 'matterport'
        elif options.hybrid == '3':
            options.dataset = 'ScanNet'
            
        options.dataset
    options.batchSize = 1
    min_after_dequeue = 1000

    reader = RecordReaderAll()
    if options.dataset == 'SUNCG':
        filename_queue = tf.train.string_input_producer([options.rootFolder + '/planes_SUNCG_val.tfrecords'], num_epochs=10000)
    elif options.dataset == 'NYU_RGBD':
        filename_queue = tf.train.string_input_producer([options.rootFolder + '/planes_nyu_rgbd_val.tfrecords'], num_epochs=1)
        options.deepSupervision = 0
        options.predictLocal = 0
    elif options.dataset == 'matterport':
        filename_queue = tf.train.string_input_producer([options.rootFolder + '/planes_matterport_val.tfrecords'], num_epochs=1)
    else:
        filename_queue = tf.train.string_input_producer([options.rootFolder + '/planes_scannet_train.tfrecords'], num_epochs=1)
        pass
    img_inp, global_gt_dict, local_gt_dict = reader.getBatch(filename_queue, numOutputPlanes=options.numOutputPlanes, batchSize=options.batchSize, min_after_dequeue=min_after_dequeue, getLocal=True, random=False)

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
            planeMasks = []
            #predMasks = []
            gtPlanes = []
            predPlanes = []
            gtSegmentations = []
            predSegmentations = []
            gtNumPlanes = []
            
            imageWidth = WIDTH
            imageHeight = HEIGHT
            focalLength = 517.97
            urange = np.arange(imageWidth).reshape(1, -1).repeat(imageHeight, 0) - imageWidth * 0.5
            vrange = np.arange(imageHeight).reshape(-1, 1).repeat(imageWidth, 1) - imageHeight * 0.5
            ranges = np.array([urange / imageWidth * 640 / focalLength, np.ones(urange.shape), -vrange / imageHeight * 480 / focalLength]).transpose([1, 2, 0])


            for index in xrange(10):
                print(('image', index))
                t0=time.time()

                img, global_gt, local_gt, global_pred, local_pred, deep_preds, total_loss, losses, debug = sess.run([img_inp, global_gt_dict, local_gt_dict, global_pred_dict, local_pred_dict, deep_pred_dicts, loss, loss_dict, debug_dict])

                # print(options.test_dir)
                # cv2.imwrite(options.test_dir + '/depth.png', drawDepthImage(debug['depth'][0]))
                # cv2.imwrite(options.test_dir + '/normal.png', drawNormalImage(debug['normal'][0]))
                # boundary = debug['boundary'][0]
                # boundary = np.concatenate([boundary, np.zeros((HEIGHT, WIDTH, 1))], axis=2)
                # cv2.imwrite(options.test_dir + '/boundary.png', drawMaskImage(boundary))
                # cv2.imwrite(options.test_dir + '/depth_gt.png', drawDepthImage(debug['depth_gt'][0].squeeze()))
                # exit(1)

                if 'pixelwise' in options.suffix:
                    image = ((img[0] + 0.5) * 255).astype(np.uint8)
                    gt_d = global_gt['depth'].squeeze()
                    pred_d = global_pred['non_plane_depth'].squeeze()
                    #depth = global_gt['depth'].squeeze()
                    if '_2' in options.suffix:
                        pred_p, pred_s, pred_d = fitPlanes(pred_d, numPlanes=20, planeAreaThreshold=3*4, numIterations=100, distanceThreshold=0.05, local=0.2)
                    elif '_3' in options.suffix:
                        pred_p, pred_s, pred_d = fitPlanes(gt_d, numPlanes=20, planeAreaThreshold=3*4, numIterations=100, distanceThreshold=0.05, local=0.2)
                        pass

                    # gt_p = global_gt['plane'][0]
                    # pred_p = planes
                    # valid_mask = (np.linalg.norm(gt_p, axis=1) > 0).astype(np.float32)
                    # diff = np.min(np.linalg.norm(np.expand_dims(gt_p, 1) - np.expand_dims(pred_p, 0), axis=2), 1)
                    # num += valid_mask.sum()
                    # lossSum += (diff * valid_mask).sum()
                    if options.dataset == 'SUNCG':
                        planeMask = np.squeeze(debug['segmentation']).sum(axis=2)
                    else:
                        planeMask = np.ones((HEIGHT, WIDTH))
                        if '_2' in options.suffix or '_3' in options.suffix:
                            planeMask *= (pred_s < 20).astype(np.float32)
                        pass
                    
                    if index < 10:
                        cv2.imwrite(options.test_dir + '/' + str(index) + '_image.png', image)
                        cv2.imwrite(options.test_dir + '/' + str(index) + '_depth.png', drawDepthImage(gt_d))
                        cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_pred.png', drawDepthImage(pred_d))
                        if '_2' in options.suffix or '_3' in options.suffix:
                            cv2.imwrite(options.test_dir + '/' + str(index) + '_segmentation_pred.png', drawSegmentationImage(pred_s))
                            pass
                        #cv2.imwrite(options.test_dir + '/' + str(index) + '_plane_mask.png', drawMaskImage(planeMask))                
                        pass

                        
                    gtDepths.append(gt_d)
                    predDepths.append(pred_d)
                    planeMasks.append(planeMask)


                    if options.dataset != 'NYU_RGBD' and ('_2' in options.suffix or '_3' in options.suffix):
                        gt_p = global_gt['plane'][0]
                        gt_s = global_gt['segmentation'][0]
                        gt_num_p = global_gt['num_planes'][0]

                        pred_s = (np.expand_dims(pred_s, -1) == np.reshape(np.arange(options.numOutputPlanes), [1, 1, -1])).astype(np.float32)
                        gtPlanes.append(gt_p)
                        predPlanes.append(pred_p)
                        gtSegmentations.append(gt_s)
                        gtNumPlanes.append(gt_num_p)
                        predSegmentations.append(pred_s)
                        pass
                    
                    #planeMasks.append((planeSegmentation < 20).astype(np.float32))
                    continue
                
                print(losses)
                print(total_loss)
                #print(losses)
                #exit(1)
                im = img[0]
                image = ((im + 0.5) * 255).astype(np.uint8)                

                gt_d = global_gt['depth'].squeeze()
                                

                if options.predictLocal:
                    pred_local_s = 1 / (1 + np.exp(-local_pred['score'][0]))
                    pred_local_p = local_pred['plane'][0]
                    pred_local_m = local_pred['mask'][0]
                  
                    gt_local_s = local_gt['score'][0]
                    gt_local_m = local_gt['mask'][0]

                    #visualize local plane prediction
                    stride = 8
                    boxSize = 64
                    xs = np.arange(WIDTH / stride) * stride + stride / 2
                    ys = np.arange(HEIGHT / stride) * stride + stride / 2
                    padding = boxSize / 2 + 1
                    maskImage = np.zeros((HEIGHT + padding * 2, WIDTH + padding * 2, 3), dtype=np.uint8)
                    maskImage[padding:padding + HEIGHT, padding:padding + WIDTH, :] = image / 2
                    for gridY, y in enumerate(ys):
                        for gridX, x in enumerate(xs):
                            score = pred_local_s[gridY][gridX]
                            if score < 0.5:
                                continue                              
                            mask = pred_local_m[gridY][gridX].reshape([16, 16])        
                            mask = cv2.resize(mask, (boxSize, boxSize))
                            maskImage[y - boxSize / 2 + padding:y + boxSize / 2 + padding, x - boxSize / 2 + padding:x + boxSize / 2 + padding, 0][mask > 0.5] = 255
                            continue
                        continue
                    for gridY, y in enumerate(ys):
                        for gridX, x in enumerate(xs):
                            score = gt_local_s[gridY][gridX]
                            if score < 0.5:
                                continue                              
                            mask = gt_local_m[gridY][gridX].reshape([16, 16])        
                            mask = cv2.resize(mask, (boxSize, boxSize))
                            maskImage[y - boxSize / 2 + padding:y + boxSize / 2 + padding, x - boxSize / 2 + padding:x + boxSize / 2 + padding, 2][mask > 0.5] = 255
                            continue
                        continue
                    pass

                pred_p = global_pred['plane'][0]
                pred_s = global_pred['segmentation'][0]

                pred_np_m = global_pred['non_plane_mask'][0]
                pred_np_d = global_pred['non_plane_depth'][0]
                pred_np_n = global_pred['non_plane_normal'][0]
                
                planeMask = 1 - global_gt['non_plane_mask'][0]
                info = global_gt['info'][0]

                
                all_segmentations = np.concatenate([pred_s, pred_np_m], axis=2)
                all_segmentations_softmax = softmax(all_segmentations)
                segmentation = np.argmax(all_segmentations, 2)
                
                
                #pred_p, segmentation, numPlanes = mergePlanes(global_gt['plane'][0], np.concatenate([global_gt['segmentation'][0], global_gt['non_plane_mask'][0]], axis=2), global_gt['depth'][0].squeeze(), global_gt['info'][0], np.concatenate([pred_s, pred_np_m], axis=2))
                

                plane_depths = calcPlaneDepths(pred_p, WIDTH, HEIGHT, info)
                all_depths = np.concatenate([plane_depths, pred_np_d], axis=2)
                
                pred_d = all_depths.reshape(-1, options.numOutputPlanes + 1)[np.arange(WIDTH * HEIGHT), segmentation.reshape(-1)].reshape(HEIGHT, WIDTH)

                plane_normals = calcPlaneNormals(pred_p, WIDTH, HEIGHT)
                all_normals = np.concatenate([plane_normals, np.expand_dims(pred_np_n, 2)], axis=2)
                pred_n = np.sum(all_normals * np.expand_dims(one_hot(segmentation, options.numOutputPlanes+1), -1), 2)
                #pred_n = all_normals.reshape(-1, options.numOutputPlanes + 1, 3)[np.arange(WIDTH * HEIGHT), segmentation.reshape(-1)].reshape((HEIGHT, WIDTH, 3))
                

                if False:
                    gt_s = global_gt['segmentation'][0]
                    all_segmentations = np.concatenate([gt_s, 1 - planeMask], axis=2)
                    gt_p = global_gt['plane'][0]

                    # valid_mask = (np.linalg.norm(gt_p, axis=1) > 0).astype(np.float32)
                    # diff = np.min(np.linalg.norm(np.expand_dims(gt_p, 1) - np.expand_dims(pred_p, 0), axis=2), 1)
                    # num += valid_mask.sum()
                    # lossSum += (diff * valid_mask).sum()
                    #gt_p = np.stack([-gt_p[:, 0], -gt_p[:, 2], -gt_p[:, 1]], axis=1)
                    
                    plane_depths = calcPlaneDepths(gt_p, WIDTH, HEIGHT, info)
                    all_depths = np.concatenate([plane_depths, pred_np_d], axis=2)

                    segmentation = np.argmax(all_segmentations, 2)
                    pred_d = all_depths.reshape(-1, options.numOutputPlanes + 1)[np.arange(WIDTH * HEIGHT), segmentation.reshape(-1)].reshape(HEIGHT, WIDTH)
                    # print(gt_p)
                    # for segmentIndex in xrange(options.numOutputPlanes):
                    #     cv2.imwrite(options.test_dir + '/' + str(index) + '_mask_' + str(segmentIndex) + '.png', drawMaskImage(segmentation == segmentIndex))
                    #     continue
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_pred.png', drawDepthImage(pred_d))
                    #exit(1)
                    pass

                

                gt_p = global_gt['plane'][0]
                gt_s = global_gt['segmentation'][0]
                gt_num_p = global_gt['num_planes'][0]
                gtPlanes.append(gt_p)
                predPlanes.append(pred_p)
                gtSegmentations.append(gt_s)
                gtNumPlanes.append(gt_num_p)
                predSegmentations.append(pred_s)
                
            
                gtDepths.append(gt_d)
                planeMasks.append(planeMask.squeeze())
                predDepths.append(pred_d)
                evaluateDepths(predDepths[-1], gtDepths[-1], gtDepths[-1] > 0, planeMasks[-1])
                
                
                if index >= 10:
                    continue

                if options.predictSemantics:
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_semantics_pred.png', drawSegmentationImage(global_pred['semantics'][0], blackIndex=0))
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_semantics_gt.png', drawSegmentationImage(global_gt['semantics'][0], blackIndex=0))
                    pass

                if 'cost_mask' in debug:
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_cost_mask.png', drawMaskImage(np.sum(debug['cost_mask'][0], axis=-1)))

                    for planeIndex in xrange(options.numOutputPlanes + 1):
                        cv2.imwrite(options.test_dir + '/' + str(index) + '_cost_mask_' + str(planeIndex) + '.png', drawMaskImage(debug['cost_mask'][0, :, :, planeIndex]))
                        continue
                    all_segmentations = np.concatenate([pred_s, pred_np_m], axis=2)
                    for planeIndex in xrange(options.numOutputPlanes + 1):
                        cv2.imwrite(options.test_dir + '/' + str(index) + '_segmentation_pred_' + str(planeIndex) + '.png', drawMaskImage(all_segmentations[:, :, planeIndex]))
                        continue
                    exit(1)
                    pass
                
                if 'normal' in global_gt:
                    gt_n = global_gt['normal'][0]
                    norm = np.linalg.norm(gt_n, axis=-1, keepdims=True)
                    gt_n /= np.maximum(norm, 1e-4)
                    
                    #gt_n = np.stack([-gt_n[:, :, 0], -gt_n[:, :, 2], -gt_n[:, :, 1]], axis=2)
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_normal_gt.png', drawNormalImage(gt_n))
                    #cv2.imwrite(options.test_dir + '/' + str(index) + '_normal_pred.png', drawNormalImage(pred_np_n))
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_normal_pred.png', drawNormalImage(pred_n))
                    pass
                
                if 'segmentation' in global_gt:
                    gt_s = global_gt['segmentation'][0]
                    gt_p = global_gt['plane'][0]

                    
                    #for planeIndex in xrange(20):
                    #cv2.imwrite('test/mask_' + str(planeIndex) + '.png', drawMaskImage(gt_s[:, :, planeIndex]))
                    #continue
                    
                    
                    # print(gt_p)
                    # print(gt_n[109][129])
                    # print(gt_n[166][245])
                    # print(gt_s[109][129])
                    # print(gt_s[166][245])
                    # print(plane_normals[109][129])
                    # print(plane_normals[166][245])                    
                    # for planeIndex in xrange(20):
                    #     cv2.imwrite('test/mask_' + str(planeIndex) + '.png', drawMaskImage(gt_s[:, :, planeIndex]))
                    #     continue
                    # exit(1)
                    
                    gt_s, gt_p = sortSegmentations(gt_s, gt_p, pred_p)
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_plane_mask_gt.png', drawMaskImage(planeMask))  
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_segmentation_gt.png', drawSegmentationImage(np.concatenate([gt_s, 1 - planeMask], axis=2), blackIndex=options.numOutputPlanes)
)
                    #cv2.imwrite(options.test_dir + '/' + str(index) + '_test.png', drawMaskImage(np.sum(np.concatenate([gt_s, 1 - planeMask], axis=2), axis=2)))
                    #exit(1)
                    #exit(1)
                    pass


                if options.predictConfidence == 1 and options.dataset == 'SUNCG':
                    assert(False)
                    pred_p_c = global_pred['confidence'][0]
                    pred_p_c = 1 / (1 + np.exp(-pred_p_c))
                    #print(pred_p_c)
                    # print(losses)
                    # print(debug['plane'][0])
                    # print(pred_p)
                    # exit(1)
                    numPlanes = global_gt['num_planes'][0]
                    print((numPlanes, (pred_p_c > 0.5).sum()))
                    
                    pred_p_c = (pred_p_c > 0.5).astype(np.float32)
                    pred_p *= pred_p_c
                    pred_s -= (1 - pred_p_c.reshape([1, 1, options.numOutputPlanes])) * 10
                    pass
                  
                if False:
                    #dump results for post processing
                    if index >= 10:
                        break
                    np.save(options.dump_dir + '/planes_' + str(index) + '.npy', pred_p)
                    np.save(options.dump_dir + '/segmentations_' + str(index) + '.npy', pred_s)
                    np.save(options.dump_dir + '/segmentations_gt_' + str(index) + '.npy', gt_s)
                    np.save(options.dump_dir + '/non_plane_depth_' + str(index) + '.npy', pred_np_d)
                    np.save(options.dump_dir + '/non_plane_segmentation_' + str(index) + '.npy', pred_np_m)
                    boundary = pred_boundary[0]
                    boundary = 1 / (1 + np.exp(-boundary))
                    boundary = np.concatenate([boundary, np.zeros((HEIGHT, WIDTH, 1))], axis=2)                    
                    cv2.imwrite(options.dump_dir + '/boundary_' + str(index) + '.png', drawMaskImage(boundary))
                    cv2.imwrite(options.dump_dir + '/image_' + str(index) + '.png', cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR))
                    np.save(options.dump_dir + '/depth_' + str(index) + '.npy', gt_d)
                    continue                

                cv2.imwrite(options.test_dir + '/' + str(index) + '_image.png', image)
                cv2.imwrite(options.test_dir + '/' + str(index) + '_depth.png', drawDepthImage(gt_d))
                #cv2.imwrite(options.test_dir + '/' + str(index) + '_overlay.png', drawDepthImageOverlay(image, gt_d))
                
                if options.predictBoundary:
                    pred_boundary = global_pred['boundary'][0]
                    pred_boundary = 1 / (1 + np.exp(-pred_boundary))
                    boundary = np.concatenate([pred_boundary, np.zeros((HEIGHT, WIDTH, 1))], axis=2)
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_boundary_pred.png', drawMaskImage(boundary))
                    pass

                if 'boundary' in global_gt:
                    gt_boundary = global_gt['boundary'][0]
                    boundary = np.concatenate([gt_boundary, np.zeros((HEIGHT, WIDTH, 1))], axis=2)                
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_boundary_gt.png', drawMaskImage(boundary))
                    pass

                for layerIndex, layer in enumerate(options.deepSupervisionLayers):
                    segmentation_deep = np.argmax(deep_preds[layerIndex]['segmentation'][0], 2)
                    segmentation_deep[segmentation_deep == options.numOutputPlanes] = -1
                    segmentation_deep += 1
                
                    plane_depths_deep = calcPlaneDepths(deep_preds[layerIndex]['plane'][0], WIDTH, HEIGHT, info)
                    all_depths_deep = np.concatenate([pred_np_d, plane_depths_deep], axis=2)
                    pred_d_deep = all_depths_deep.reshape(-1, options.numOutputPlanes + 1)[np.arange(WIDTH * HEIGHT), segmentation_deep.reshape(-1)].reshape(HEIGHT, WIDTH)
                
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_segmentation_pred_' + str(layerIndex) + '.png', drawSegmentationImage(deep_preds[layerIndex]['segmentation'][0]))
                    cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_pred_' + str(layerIndex) + '.png', drawDepthImage(pred_d_deep))
                    pass


                #print(pred_np_m)
                #print(pred_s)
                #print(global_gt['plane'][0])
                #print(pred_p)
                #exit(1)

                print('depth diff', np.abs(pred_d - gt_d).mean())
                
                cv2.imwrite(options.test_dir + '/' + str(index) + '_segmentation_pred.png', drawSegmentationImage(all_segmentations, blackIndex=options.numOutputPlanes))
                #exit(1)
                cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_pred.png', drawDepthImage(pred_d))
                cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_diff.png', drawMaskImage(np.abs(pred_d - gt_d) * planeMask.squeeze() / 0.2))
                #exit(1)
                
                
                #cv2.imwrite(options.test_dir + '/' + str(index) + '_plane_mask.png', drawMaskImage(planeMask))           

                #print(np.concatenate([np.arange(options.numOutputPlanes).reshape(-1, 1), planes, pred_p, preds_p[0][0]], axis=1))

                #print(np.concatenate([distance[0], preds_p[0][0], pred_p], axis=1))
                            
                #segmentation = np.argmax(pred_s, 2)
                #writePLYFile(options.test_dir, index, image, pred_d, segmentation, np.zeros(pred_boundary[0].shape))
                #writePLYFileParts(options.test_dir, index, image, pred_d, segmentation)
                #gt_s_ori = gt_s_ori[0]


                # if 'depth' in debug:
                #     cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_pred_1.png', drawDepthImage(debug['depth'][0]))
                #     gt_s_ori = debug['segmentation'][0]
                #     cv2.imwrite(options.test_dir + '/' + str(index) + '_segmentation_gt.png', drawSegmentationImage(gt_s_ori, blackIndex=options.numOutputPlanes))
                #     total_loss = 0
                #     for planeIndex in xrange(gt_s_ori.shape[-1]):
                #         mask_pred = all_segmentations_softmax[:, :, planeIndex]
                #         loss = -(gt_s_ori[:, :, planeIndex] * np.log(np.maximum(all_segmentations_softmax[:, :, planeIndex], 1e-31))).mean() * 1000
                #         print(planeIndex, loss)
                #         total_loss += loss
                #         cv2.imwrite(options.test_dir + '/mask_pred_' + str(planeIndex) + '.png', drawMaskImage(mask_pred))
                #         cv2.imwrite(options.test_dir + '/mask_gt_' + str(planeIndex) + '.png', drawMaskImage(gt_s_ori[:, :, planeIndex]))
                #         continue
                #     print(total_loss)
                #     print(gt_s_ori.sum(2).max())                    
                #     exit(1)
                #     pass

                
                if False:
                    #distance = distance[0]
                    print(distance)
                    diff = np.linalg.norm(np.expand_dims(planes, 1) - np.expand_dims(pred_p, 0), axis=2)
                    print(np.concatenate([planes, pred_p, pow(np.min(diff, axis=1, keepdims=True), 2), np.expand_dims(np.argmin(diff, axis=1), -1)], axis=1))
                    print(pred_p_c)
                    
                    #print(distance[:, 6:8].sum(0))
                    #print(pow(np.linalg.norm(distance[:, :3] - distance[:, 3:6], 2, 1), 2) * 100)
                    #print(test)
                    segmentation = np.argmax(all_segmentations, 2) - 1
                    for planeIndex in xrange(options.numOutputPlanes):
                        cv2.imwrite(options.test_dir + '/segmentation_' + str(planeIndex) + '.png', drawMaskImage(segmentation == planeIndex))
                        cv2.imwrite(options.test_dir + '/segmentation_' + str(planeIndex) + '_gt.png', drawMaskImage(gt_s[:, :, planeIndex]))
                        cv2.imwrite(options.test_dir + '/segmentation_' + str(planeIndex) + '_gt_ori.png', drawMaskImage(gt_s_ori[:, :, planeIndex]))
                        continue
                    exit(1)
                continue

            # if options.dataset == 'SUNCG':
            #     if 'pixelwise' not in options.suffix:
            #         evaluatePlaneSegmentation(np.array(predPlanes), np.array(predSegmentations), np.array(gtPlanes), np.array(gtSegmentations), np.array(gtNumPlanes), planeDistanceThreshold = 0.3, IOUThreshold = 0.5, prefix='test/planenet_')
            #     elif '_2' in options.suffix:
            #         evaluatePlaneSegmentation(np.array(predPlanes), np.array(predSegmentations), np.array(gtPlanes), np.array(gtSegmentations), np.array(gtNumPlanes), planeDistanceThreshold = 0.3, IOUThreshold = 0.5, prefix='test/pixelwise_pred_')
            #     elif '_3' in options.suffix:
            #         evaluatePlaneSegmentation(np.array(predPlanes), np.array(predSegmentations), np.array(gtPlanes), np.array(gtSegmentations), np.array(gtNumPlanes), planeDistanceThreshold = 0.3, IOUThreshold = 0.5, prefix='test/pixelwise_gt_')
            #         pass
            #     pass
            
            predDepths = np.array(predDepths)
            gtDepths = np.array(gtDepths)
            planeMasks = np.array(planeMasks)
            #predMasks = np.array(predMasks)
            evaluateDepths(predDepths, gtDepths, gtDepths > 0, planeMasks)
            exit(1)

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


# def predict(options):
#     options.test_dir += '_predict'
#     if not os.path.exists(options.test_dir):
#         os.system("mkdir -p %s"%options.test_dir)
#         pass

#     batchSize = 1
#     img_inp = tf.placeholder(tf.float32,shape=(batchSize,HEIGHT,WIDTH,3),name='img_inp')
#     plane_gt=tf.placeholder(tf.float32,shape=(batchSize,options.numOutputPlanes, 3),name='plane_inp')
#     validating_inp = tf.constant(0, tf.int32)

#     global_pred_dict, local_pred_dict, deep_pred_dicts = build_graph(img_inp_train, img_inp_val, img_inp_rgbd_train, img_inp_rgbd_val, training_flag, options)

#     var_to_restore = tf.global_variables()
    
 
#     config=tf.ConfigProto()
#     config.gpu_options.allow_growth=True
#     config.allow_soft_placement=True


#     if dataset == 'SUNCG':
#         image_list_file = os.path.join('../PythonScripts/SUNCG/image_list_100_tail_500000.txt')
#         with open(image_list_file) as f:
#             im_names = [{'image': im_name.strip().replace('plane_global.npy', 'mlt.png'), 'depth': im_name.strip().replace('plane_global.npy', 'depth.png'), 'normal': im_name.strip().replace('plane_global.npy', 'norm_camera.png'), 'valid': im_name.strip().replace('plane_global.npy', 'valid.png'), 'plane': im_name.strip()} for im_name in f.readlines()]
#             pass
#     else:
#         im_names = glob.glob('../../Data/NYU_RGBD/*_color.png')
#         im_names = [{'image': im_name, 'depth': im_name.replace('color.png', 'depth.png'), 'normal': im_name.replace('color.png', 'norm_camera.png'), 'invalid_mask': im_name.replace('color.png', 'valid.png')} for im_name in im_names]
#         pass
      
#     if numImages > 0:
#         im_names = im_names[:numImages]
#         pass

#     #if args.imageIndex > 0:
#     #im_names = im_names[args.imageIndex:args.imageIndex + 1]
#     #pass    

#     init_op = tf.group(tf.global_variables_initializer(),
#                        tf.local_variables_initializer())


#     with tf.Session(config=config) as sess:
#         saver = tf.train.Saver()
#         #sess.run(tf.global_variables_initializer())
#         saver.restore(sess,"%s/%s.ckpt"%(options.checkpoint_dir,keyname))

#         gtDepths = []
#         predDepths = []
#         segmentationDepths = []
#         predDepthsOneHot = []
#         planeMasks = []
#         predMasks = []

#         imageWidth = WIDTH
#         imageHeight = HEIGHT
#         focalLength = 517.97
#         urange = np.arange(imageWidth).reshape(1, -1).repeat(imageHeight, 0) - imageWidth * 0.5
#         vrange = np.arange(imageHeight).reshape(-1, 1).repeat(imageWidth, 1) - imageHeight * 0.5
#         ranges = np.array([urange / focalLength, np.ones(urange.shape), -vrange / focalLength]).transpose([1, 2, 0])
        
#         cv2.imwrite(options.test_dir + '/one.png', np.ones((HEIGHT, WIDTH), dtype=np.uint8) * 255)
#         cv2.imwrite(options.test_dir + '/zero.png', np.zeros((HEIGHT, WIDTH), dtype=np.uint8) * 255)
#         for index, im_name in enumerate(im_names):
#             if index <= -1:
#                 continue
#             print(im_name['image'])
#             im = cv2.imread(im_name['image'])
#             image = im.astype(np.float32, copy=False)
#             image = image / 255 - 0.5
#             image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)

#             #planes = np.load(im_name['plane'])
#             # numPlanes = planes.shape[0]
#             # if numPlanes > options.numOutputPlanes:
#             #     planeAreas = planes[:, 3:].sum(1)
#             #     sortInds = np.argsort(planeAreas)[::-1]
#             #     planes = planes[sortInds[:options.numOutputPlanes]]
#             #     pass
#             # gt_p = np.zeros((1, options.numOutputPlanes, 3))
#             # gt_p[0, :numPlanes] = planes[:numPlanes, :3]

#             normal = np.array(PIL.Image.open(im_name['normal'])).astype(np.float) / 255 * 2 - 1
#             norm = np.linalg.norm(normal, 2, 2)
#             for c in xrange(3):
#                 normal[:, :, c] /= norm
#                 continue
#             normal = cv2.resize(normal, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
            
#             depth = np.array(PIL.Image.open(im_name['depth'])).astype(np.float) / 1000
#             depth = cv2.resize(depth, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)

#             invalid_mask = cv2.resize(cv2.imread(im_name['invalid_mask'], 0), (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR) > 128

#             gtDepths.append(depth)

            
#             pred_p, pred_d, pred_n, pred_s, pred_np_m, pred_np_d, pred_np_n, pred_boundary, pred_local_score, pred_local_p, pred_local_mask = sess.run([plane_pred, depth_pred, normal_pred, segmentation_pred, non_plane_mask_pred, non_plane_depth_pred, non_plane_normal_pred, boundary_pred, local_score_pred, local_p_pred, local_mask_pred], feed_dict = {img_inp:np.expand_dims(image, 0), plane_gt: np.zeros((batchSize, options.numOutputPlanes, 3))})


#             pred_s = pred_s[0] 
#             pred_p = pred_p[0]
#             pred_np_m = pred_np_m[0]
#             pred_np_d = pred_np_d[0]
#             pred_np_n = pred_np_n[0]
#             #pred_s = 1 / (1 + np.exp(-pred_s))

#             plane_depths = calcPlaneDepths(pred_p, WIDTH, HEIGHT)
#             all_depths = np.concatenate([pred_np_d, plane_depths], axis=2)

#             all_segmentations = np.concatenate([pred_np_m, pred_s], axis=2)
#             segmentation = np.argmax(all_segmentations, 2)
#             if suffix != 'pixelwise':
#                 pred_d = all_depths.reshape(-1, options.numOutputPlanes + 1)[np.arange(WIDTH * HEIGHT), segmentation.reshape(-1)].reshape(HEIGHT, WIDTH)
#             else:
#                 pred_d = np.squeeze(pred_np_d)
#                 pass
#             predDepths.append(pred_d)
#             predMasks.append(segmentation != 0)
#             planeMasks.append(invalid_mask)

#             #depthError, normalError, occupancy, segmentationTest, reconstructedDepth, occupancyMask = evaluatePlanes(pred_p, im_name['image'])
#             #reconstructedDepth = cv2.resize(reconstructedDepth, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
            
#             #evaluatePlanes(pred_p[0], im_name, options.test_dir, index)
#             #print(pred_p)
#             #print(gt_p)
#             #print((pow(pred_d[0, :, :, 0] - depth, 2) * (gt_s.max(2) > 0.5)).mean())
#             #print((depthError, normalError, occupancy))
            
#             evaluateDepths(predDepths[index], gtDepths[index], np.ones(planeMasks[index].shape), planeMasks[index])

#             if index >= 10:
#                 continue
#             cv2.imwrite(options.test_dir + '/' + str(index) + '_image.png', cv2.resize(im, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR))
#             #cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_gt.png', (minDepth / np.clip(depth, minDepth, 20) * 255).astype(np.uint8))
#             #cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_pred.png', (minDepth / np.clip(pred_d[0, :, :, 0], minDepth, 20) * 255).astype(np.uint8))
#             #cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_plane.png', (minDepth / np.clip(reconstructedDepth, minDepth, 20) * 255).astype(np.uint8))

#             pred_boundary = pred_boundary[0]
#             boundary = (1 / (1 + np.exp(-pred_boundary)) * 255).astype(np.uint8)
#             boundary = np.concatenate([boundary, np.zeros((HEIGHT, WIDTH, 1))], axis=2)
#             cv2.imwrite(options.test_dir + '/' + str(index) + '_boundary_pred.png', boundary)
            
#             cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_inp.png', drawDepthImage(depth))
#             cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_pred.png', drawDepthImage(pred_d))
#             #cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_plane.png', drawDepthImage(reconstructedDepth))
#             cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_pred_diff.png', drawDiffImage(pred_d, depth, 0.5))
#             #cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_plane_diff.png', np.minimum(np.abs(reconstructedDepth - depth) / 0.5 * 255, 255).astype(np.uint8))
#             cv2.imwrite(options.test_dir + '/' + str(index) + '_normal_inp.png', drawNormalImage(normal))
#             cv2.imwrite(options.test_dir + '/' + str(index) + '_normal_pred.png', drawNormalImage(pred_np_n))
#             cv2.imwrite(options.test_dir + '/' + str(index) + '_segmentation_pred.png', drawSegmentationImage(all_segmentations, black=True))

#             segmentation = np.argmax(pred_s, 2)
#             #writePLYFile(options.test_dir, index, image, pred_p, segmentation)

#             if index < 0:
#                 for planeIndex in xrange(options.numOutputPlanes):
#                     cv2.imwrite(options.test_dir + '/' + str(index) + '_segmentation_' + str(planeIndex) + '.png', drawMaskImage(pred_s[:, :, planeIndex]))
#                     #cv2.imwrite(options.test_dir + '/' + str(index) + '_segmentation_' + str(planeIndex) + '_gt.png', drawMaskImage(gt_s[:, :, planeIndex]))
#                     continue
#                 pass
#             continue
#         predDepths = np.array(predDepths)
#         gtDepths = np.array(gtDepths)
#         planeMasks = np.array(planeMasks)
#         predMasks = np.array(predMasks)
#         #evaluateDepths(predDepths, gtDepths, planeMasks, predMasks)
#         evaluateDepths(predDepths, gtDepths, planeMasks, planeMasks)
#         #exit(1)
#         pass
#     return


# def fitPlanesRGBD(options):
#     writeHTMLRGBD('../results/RANSAC_RGBD/index.html', 10)
#     exit(1)
#     if not os.path.exists(options.checkpoint_dir):
#         os.system("mkdir -p %s"%options.checkpoint_dir)
#         pass
#     if not os.path.exists(options.test_dir):
#         os.system("mkdir -p %s"%options.test_dir)
#         pass
    
#     min_after_dequeue = 1000

#     reader_rgbd = RecordReaderRGBD()
#     filename_queue_rgbd = tf.train.string_input_producer(['../planes_nyu_rgbd_train.tfrecords'], num_epochs=1)
#     img_inp_rgbd, global_gt_dict_rgbd, local_gt_dict_rgbd = reader_rgbd.getBatch(filename_queue_rgbd, numOutputPlanes=options.numOutputPlanes, batchSize=1, min_after_dequeue=min_after_dequeue, getLocal=True)

#     config=tf.ConfigProto()
#     config.gpu_options.allow_growth=True
#     config.allow_soft_placement=True

#     init_op = tf.group(tf.global_variables_initializer(),
#                        tf.local_variables_initializer())

#     with tf.Session(config=config) as sess:
#         sess.run(init_op)

#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(sess=sess, coord=coord)

#         try:
#             gtDepths = []
#             predDepths = []
#             planeMasks = []
#             for index in xrange(10):
#                 image, depth, path = sess.run([img_inp_rgbd, global_gt_dict_rgbd['depth'], global_gt_dict_rgbd['path']])
#                 image = ((image[0] + 0.5) * 255).astype(np.uint8)
#                 depth = depth.squeeze()
                
#                 cv2.imwrite(options.test_dir + '/' + str(index) + '_image.png', image)
#                 cv2.imwrite(options.test_dir + '/' + str(index) + '_depth.png', drawDepthImage(depth))
#                 #cv2.imwrite(options.test_dir + '/' + str(index) + '_mask.png', drawMaskImage(depth == 0))
#                 planes, planeSegmentation, depthPred = fitPlanes(depth, numPlanes=20)                
#                 cv2.imwrite(options.test_dir + '/' + str(index) + '_segmentation_pred.png', drawSegmentationImage(planeSegmentation))
#                 cv2.imwrite(options.test_dir + '/' + str(index) + '_depth_pred.png', drawDepthImage(depthPred))

#                 gtDepths.append(depth)
#                 predDepths.append(depthPred)
#                 planeMasks.append((planeSegmentation < 20).astype(np.float32))
#                 continue
#             predDepths = np.array(predDepths)
#             gtDepths = np.array(gtDepths)
#             planeMasks = np.array(planeMasks)
#             evaluateDepths(predDepths, gtDepths, np.ones(planeMasks.shape, dtype=np.bool), planeMasks)            
#         except tf.errors.OutOfRangeError:
#             print('done fitting')
#         finally:
#             # When done, ask the threads to stop.
#             coord.request_stop()
#             pass
#     return

def writeInfo(options):
    x = (np.arange(11) * 0.1).tolist()
    ys = []
    ys.append(np.load('test/planenet_pixel_IOU.npy').tolist())
    ys.append(np.load('test/pixelwise_pred_pixel_IOU.npy').tolist())
    ys.append(np.load('test/pixelwise_gt_pixel_IOU.npy').tolist())
    plotCurves(x, ys, filename = 'test/plane_comparison.png', xlabel='IOU', ylabel='pixel coverage', labels=['planenet', 'pixelwise+RANSAC', 'GT+RANSAC'])

    x = (0.5 - np.arange(11) * 0.05).tolist()
    ys = []
    ys.append(np.load('test/planenet_pixel_diff.npy').tolist())
    ys.append(np.load('test/pixelwise_pred_pixel_diff.npy').tolist())
    ys.append(np.load('test/pixelwise_gt_pixel_diff.npy').tolist())
    plotCurves(x, ys, filename = 'test/plane_comparison_diff.png', xlabel='diff', ylabel='pixel coverage', labels=['planenet', 'pixelwise+RANSAC', 'GT+RANSAC'])
    
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
    parser.add_argument('--numOutputPlanes', dest='numOutputPlanes',
                        help='the number of output planes',
                        default=20, type=int)
    parser.add_argument('--batchSize', dest='batchSize',
                        help='batch size',
                        default=16, type=int)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset name for test/predict',
                        default='', type=str)
    parser.add_argument('--numImages', dest='numImages',
                        help='the number of images to test/predict',
                        default=10, type=int)
    parser.add_argument('--boundaryLoss', dest='boundaryLoss',
                        help='use boundary loss: [0, 1]',
                        default=0, type=int)
    parser.add_argument('--diverseLoss', dest='diverseLoss',
                        help='use diverse loss: [0, 1]',
                        default=1, type=int)
    parser.add_argument('--labelLoss', dest='labelLoss',
                        help='use label loss: [0, 1]',
                        default=1, type=int)    
    parser.add_argument('--deepSupervision', dest='deepSupervision',
                        help='deep supervision level: [0, 1, 2]',
                        default=1, type=int)
    parser.add_argument('--sameMatching', dest='sameMatching',
                        help='use the same matching for all deep supervision layers and the final prediction: [0, 1]',
                        default=0, type=int)    
    parser.add_argument('--anchorPlanes', dest='anchorPlanes',
                        help='use anchor planes for all deep supervision layers and the final prediction: [0, 1]',
                        default=0, type=int) 
    parser.add_argument('--crf', dest='crf',
                        help='the number of CRF iterations',
                        default=0, type=int)
    parser.add_argument('--backwardLossWeight', dest='backwardLossWeight',
                        help='backward matching loss',
                        default=0.5, type=float)
    parser.add_argument('--predictBoundary', dest='predictBoundary',
                        help='whether predict boundary or not: [0, 1]',
                        default=0, type=int)
    parser.add_argument('--predictSemantics', dest='predictSemantics',
                        help='whether predict semantics or not: [0, 1]',
                        default=1, type=int)    
    parser.add_argument('--predictLocal', dest='predictLocal',
                        help='whether predict local planes or not: [0, 1]',
                        default=0, type=int)
    parser.add_argument('--predictConfidence', dest='predictConfidence',
                        help='whether predict plane confidence or not: [0, 1]',
                        default=0, type=int)
    parser.add_argument('--predictPixelwise', dest='predictPixelwise',
                        help='whether predict pixelwise depth or not: [0, 1]',
                        default=1, type=int)    
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
    parser.add_argument('--rootFolder', dest='rootFolder',
                        help='root folder',
                        default='/mnt/vision/PlaneNet/', type=str)
    parser.add_argument('--dataFolder', dest='dataFolder',
                        help='data folder',
                        default='/home/chenliu/Projects/PlaneNet/', type=str)
    parser.add_argument('--saveInterval', dest='saveInterval',
                        help='save interval',
                        default=900, type=int)
    

    args = parser.parse_args()
    args.keyname = os.path.basename(__file__).rstrip('.py')
    args.keyname = args.keyname.replace('train_', '')

    if args.numOutputPlanes != 20:
        args.keyname += '_np' + str(args.numOutputPlanes)
        pass
    args.keyname += '_hybrid' + args.hybrid
    
    if args.boundaryLoss != 1:
        args.keyname += '_bl' + str(args.boundaryLoss)
        pass
    if args.diverseLoss == 0:
        args.keyname += '_dl0'
        pass
    if args.labelLoss == 1:
        args.keyname += '_ll1'
        pass    
    if args.deepSupervision != 1:
        args.keyname += '_ds' + str(args.deepSupervision)
        pass
    if args.crf > 0:
        args.keyname += '_crf' + str(args.crf)
        pass
    if args.backwardLossWeight > 0:
        args.keyname += '_bw' + str(args.backwardLossWeight)
        pass
    if args.predictBoundary == 1:
        args.keyname += '_pb'
        pass
    if args.predictConfidence == 1:
        args.keyname += '_pc'
        pass        
    if args.predictLocal == 1:
        args.keyname += '_pl'
        pass
    if args.predictPixelwise == 1:
        args.keyname += '_pp'
        pass
    if args.predictSemantics == 1:
        args.keyname += '_ps'
        pass    
    if args.sameMatching == 0:
        args.keyname += '_sm0'
        pass
    if args.anchorPlanes == 1:
        args.keyname += '_ap1'
        pass

    #args.predictSemantics = 0
    #args.predictBoundary = 0
    
    args.checkpoint_dir = args.rootFolder + '/checkpoint/' + args.keyname
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

