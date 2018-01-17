import pdb
import argparse
import cv2
import glob
import numpy as np
import os
import random
import socket
import struct
import sys
import tensorflow as tf
import time
import threading

try:
    import cPickle as pickle
except ImportError:
    # Python 3 compatibility
    import pickle

from io import BytesIO

import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))

import get_datasets
from tracker import network
from re3_utils.util import bb_util
from re3_utils.util import im_util
from re3_utils.tensorflow_util import tf_util
from re3_utils.tensorflow_util import tf_queue
from re3_utils.util import drawing
from re3_utils.util import IOU
from re3_utils.simulator import simulator

from constants import CROP_PAD
from constants import CROP_SIZE
from constants import LSTM_SIZE
from constants import GPU_ID
from constants import LOG_DIR
from constants import OUTPUT_WIDTH
from constants import OUTPUT_HEIGHT

HOST = 'localhost'
NUM_ITERATIONS = int(1e6)
PORT = 9997
REPLAY_BUFFER_SIZE = 1024
PARALLEL_SIZE = 1
ENQUEUE_BATCH_SIZE = 1

SIMULATION_WIDTH = simulator.IMAGE_WIDTH
SIMULATION_HEIGHT = simulator.IMAGE_HEIGHT
simulator.NUM_DISTRACTORS = 20

USE_SIMULATOR = 0.5
USE_NETWORK_PROB = 0.8
REAL_MOTION_PROB = 1.0 / 8
AREA_CUTOFF = 0.25

# Get a sequence from the batch cache.
def getData():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    sock.sendall(('Ready' + '\n').encode())
    receivedBytes = sock.recv(4)
    messageLength = struct.unpack('>I', receivedBytes)[0]
    key = sock.recv(messageLength)
    key = pickle.loads(key)

    images = [None] * delta
    for nn in range(delta):
        image = BytesIO()
        # Connect to server and send data.

        # Get the array.
        received = 0
        receivedBytes = sock.recv(4)
        messageLength = struct.unpack('>I', receivedBytes)[0]
        while received < messageLength:
            receivedBytes = sock.recv(min(1024, messageLength - received))
            image.write(receivedBytes)
            received += len(receivedBytes)

        imageArray = np.fromstring(image.getvalue(), dtype=np.uint8)
        image.close()

        # Get shape.
        receivedBytes = sock.recv(4)
        messageLength = struct.unpack('>I', receivedBytes)[0]
        shape = sock.recv(messageLength)
        shape = pickle.loads(shape)
        imageArray = imageArray.reshape(shape)
        if len(imageArray.shape) < 3:
            imageArray = np.tile(imageArray[:,:,np.newaxis], (1,1,3))
        images[nn] = imageArray
    sock.close()
    return (key, images)


# Randomly jitter the box for a bit of noise.
def add_noise(bbox, prevBBox, imageWidth, imageHeight):
    numTries = 0
    bboxXYWHInit = bb_util.xyxy_to_xywh(bbox)
    while numTries < 10:
        bboxXYWH = bboxXYWHInit.copy()
        centerNoise = np.random.laplace(0,1.0/5,2) * bboxXYWH[[2,3]]
        sizeNoise = np.clip(np.random.laplace(1,1.0/15,2), .6, 1.4)
        bboxXYWH[[2,3]] *= sizeNoise
        bboxXYWH[[0,1]] = bboxXYWH[[0,1]] + centerNoise
        if not (bboxXYWH[0] < prevBBox[0] or bboxXYWH[1] < prevBBox[1] or
            bboxXYWH[0] > prevBBox[2] or bboxXYWH[1] > prevBBox[3] or
            bboxXYWH[0] < 0 or bboxXYWH[1] < 0 or
            bboxXYWH[0] > imageWidth or bboxXYWH[1] > imageHeight):
            numTries = 10
        else:
            numTries += 1

    return fix_bbox_intersection(bb_util.xywh_to_xyxy(bboxXYWH), prevBBox, imageWidth, imageHeight)

# Make sure there is a minimum intersection with the ground truth box and the visible crop.
def fix_bbox_intersection(bbox, gtBox, imageWidth, imageHeight):
    if type(bbox) == list:
        bbox = np.array(bbox)
    if type(gtBox) == list:
        gtBox = np.array(gtBox)

    gtBoxArea = float((gtBox[3] - gtBox[1]) * (gtBox[2] - gtBox[0]))
    bboxLarge = bb_util.scale_bbox(bbox, CROP_PAD)
    while IOU.intersection(bboxLarge, gtBox) / gtBoxArea < AREA_CUTOFF:
        bbox = bbox * .9 + gtBox * .1
        bboxLarge = bb_util.scale_bbox(bbox, CROP_PAD)
    return bbox


def main(FLAGS):
    global PORT, delta, REPLAY_BUFFER_SIZE

    simulator.make_paths()
    delta = FLAGS.delta
    batchSize = FLAGS.batch_size
    timing = FLAGS.timing
    debug = FLAGS.debug or FLAGS.output
    PORT = FLAGS.port

    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.cuda_visible_devices)
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)

    # Read in and format GT.
    # dict from (dataset_ind, video_id, track_id, image_id) to line in labels array
    key_lookup = dict()
    datasets = []

    def add_dataset(dataset_name):
        dataset_ind = len(datasets)
        dataset_gt = get_datasets.get_data_for_dataset(dataset_name, 'train')['gt']
        for xx in range(dataset_gt.shape[0]):
            line = dataset_gt[xx,:].astype(int)
            key_lookup[(dataset_ind, line[4], line[5], line[6])] = xx
        datasets.append(dataset_gt)

    add_dataset('imagenet_video')

    # Tensorflow setup
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(LOG_DIR + '/checkpoints'):
        os.makedirs(LOG_DIR + '/checkpoints')

    tf.Graph().as_default()
    tf.logging.set_verbosity(tf.logging.INFO)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

    imagePlaceholder = tf.placeholder(tf.uint8, shape=(ENQUEUE_BATCH_SIZE, delta * 2, CROP_SIZE, CROP_SIZE, 3))
    labelPlaceholder = tf.placeholder(tf.float32, shape=(ENQUEUE_BATCH_SIZE, delta, 4))
    learningRate = tf.placeholder(tf.float32)

    # Set up the data queue for holding images and retrieving them from RAM rather than disk.
    queue = tf_queue.TFQueue(sess,
            placeholders=[imagePlaceholder, labelPlaceholder],
            max_queue_size=REPLAY_BUFFER_SIZE,
            max_queue_uses=1,
            use_random_order=False,
            batch_size=batchSize)

    imageBatch = tf.reshape(queue.placeholder_outs[imagePlaceholder],
            (batchSize * delta * 2, CROP_SIZE, CROP_SIZE, 3))
    labelsBatch = tf.reshape(queue.placeholder_outs[labelPlaceholder], (batchSize * delta, -1))
    if ',' in FLAGS.cuda_visible_devices:
        with tf.device('/gpu:0'):
            tfOutputs = network.inference(imageBatch, num_unrolls=delta, train=True, reuse=False)
            tfLossFull, tfLoss = network.loss(tfOutputs, labelsBatch)
            train_op = network.training(tfLossFull, learningRate)
    else:
        tfOutputs = network.inference(imageBatch, num_unrolls=delta, train=True, reuse=False)
        tfLossFull, tfLoss = network.loss(tfOutputs, labelsBatch)
        train_op = network.training(tfLossFull, learningRate)

    loss_summary_op = tf.summary.merge([
        tf.summary.scalar('loss', tfLoss),
        tf.summary.scalar('l2_regularizer', tfLossFull - tfLoss),
        ])

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    longSaver = tf.train.Saver()

    # Create the nodes for single image forward passes for learning to fix mistakes.
    # Parameters here are shared with the learned network.
    if ',' in FLAGS.cuda_visible_devices:
        with tf.device('/gpu:1'):
            forwardNetworkImagePlaceholder = tf.placeholder(tf.uint8, shape=(2, CROP_SIZE, CROP_SIZE, 3))
            prevLstmState = tuple([tf.placeholder(tf.float32, shape=(1, LSTM_SIZE)) for _ in range(4)])
            initialLstmState = tuple([np.zeros((1, LSTM_SIZE)) for _ in range(4)])
            networkOutputs, state1, state2 = network.inference(
                    forwardNetworkImagePlaceholder, num_unrolls=1, train=False,
                    prevLstmState=prevLstmState, reuse=True)
    else:
        forwardNetworkImagePlaceholder = tf.placeholder(tf.uint8, shape=(2, CROP_SIZE, CROP_SIZE, 3))
        prevLstmState = tuple([tf.placeholder(tf.float32, shape=(1, LSTM_SIZE)) for _ in range(4)])
        initialLstmState = tuple([np.zeros((1, LSTM_SIZE)) for _ in range(4)])
        networkOutputs, state1, state2 = network.inference(
                forwardNetworkImagePlaceholder, num_unrolls=1, train=False,
                prevLstmState=prevLstmState, reuse=True)

    # Initialize the network and load saved parameters.
    sess.run(init)
    startIter = 0
    if FLAGS.restore:
        print('Restoring')
        ckpt = tf.train.get_checkpoint_state(LOG_DIR + '/checkpoints')
        if ckpt and ckpt.model_checkpoint_path:
            tf_util.restore(sess, ckpt.model_checkpoint_path)
            startIter = int(ckpt.model_checkpoint_path.split('-')[-1])
            print('Restored', startIter)
    if not debug:
        tt = time.localtime()
        time_str = ('%04d_%02d_%02d_%02d_%02d_%02d' %
                (tt.tm_year, tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))
        summary_writer = tf.summary.FileWriter(LOG_DIR + '/train/' + time_str +
                '_n_' + str(delta) + '_b_' + str(batchSize), sess.graph)
        summary_full = tf.summary.merge_all()
        conv_var_list = [v for v in tf.trainable_variables() if 'conv' in v.name and 'weight' in v.name and
                (v.get_shape().as_list()[0] != 1 or v.get_shape().as_list()[1] != 1)]
        for var in conv_var_list:
            tf_util.conv_variable_summaries(var, scope=var.name.replace('/', '_')[:-2])
        summary_with_images = tf.summary.merge_all()

    # Logging stuff
    robustness_ph = tf.placeholder(tf.float32, shape=[])
    lost_targets_ph = tf.placeholder(tf.float32, shape=[])
    mean_iou_ph = tf.placeholder(tf.float32, shape=[])
    avg_ph = tf.placeholder(tf.float32, shape=[])
    with tf.name_scope('test'):
        test_summary_op = tf.summary.merge([
            tf.summary.scalar('robustness', robustness_ph),
            tf.summary.scalar('lost_targets', lost_targets_ph),
            tf.summary.scalar('mean_iou', mean_iou_ph),
            tf.summary.scalar('avg_iou_robustness', avg_ph),
            ])

    if debug:
        cv2.namedWindow('debug', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('debug', OUTPUT_WIDTH, OUTPUT_HEIGHT)

    sess.graph.finalize()

    # Read a sequence from the batch cache or simulate one and get the ground truth and crops for it.
    def get_data_sequence():
        # Preallocate the space for the images and labels.
        tImage = np.zeros((delta, 2, CROP_SIZE, CROP_SIZE, 3),
                dtype=np.uint8)
        xywhLabels = np.zeros((delta, 4))

        mirrored = random.random() < 0.5
        useSimulator = random.random() < USE_SIMULATOR
        gtType = random.random()
        realMotion = random.random() < REAL_MOTION_PROB

        # Initialize first frame (give the network context).

        if useSimulator:
            # Initialize the simulation and run through a few frames.
            trackingObj, trackedObjects, background = simulator.create_new_track()
            for _ in range(random.randint(0,200)):
                simulator.step(trackedObjects)
                bbox = trackingObj.get_object_box()
                occlusion = simulator.measure_occlusion(bbox, trackingObj.occluder_boxes, cropPad=1)
                if occlusion > .2:
                    break
            for _ in range(1000):
                bbox = trackingObj.get_object_box()
                occlusion = simulator.measure_occlusion(bbox, trackingObj.occluder_boxes, cropPad=1)
                if occlusion < 0.01:
                    break
                simulator.step(trackedObjects)
            initBox = trackingObj.get_object_box()
            if debug:
                images = [simulator.get_image_for_frame(trackedObjects, background)]
            else:
                images = [np.zeros((SIMULATION_HEIGHT, SIMULATION_WIDTH))]

        else:
            # Read a new data sequence from batch cache and get the ground truth.
            (batchKey, images) = getData()
            gtKey = batchKey
            imageIndex = key_lookup[gtKey]
            initBox = datasets[gtKey[0]][imageIndex, :4].copy()
        if debug:
            bboxes = []
            cropBBoxes = []

        # bboxPrev starts at the initial box and is the best guess (or gt) for the image0 location.
        # noisyBox holds the bboxPrev estimate plus some noise.
        bboxPrev = initBox
        lstmState = None

        for dd in range(delta):
            # bboxOn is the gt location in image1
            if useSimulator:
                bboxOn = trackingObj.get_object_box()
            else:
                newKey = list(gtKey)
                newKey[3] += dd
                newKey = tuple(newKey)
                imageIndex = key_lookup[newKey]
                bboxOn = datasets[newKey[0]][imageIndex, :4].copy()
            if dd == 0:
                noisyBox = bboxOn.copy()
            elif not realMotion and not useSimulator and gtType >= USE_NETWORK_PROB:
                noisyBox = add_noise(bboxOn, bboxOn, images[0].shape[1], images[0].shape[0])
            else:
                noisyBox = fix_bbox_intersection(bboxPrev, bboxOn, images[0].shape[1], images[0].shape[0])

            if useSimulator:
                patch = simulator.render_patch(bboxPrev, background, trackedObjects)
                tImage[dd,0,...] = patch
                if dd > 0:
                    simulator.step(trackedObjects)
                    bboxOn = trackingObj.get_object_box()
                    noisyBox = fix_bbox_intersection(bboxPrev, bboxOn, images[0].shape[1], images[0].shape[0])
            else:
                tImage[dd,0,...] = im_util.get_cropped_input(
                        images[max(dd-1, 0)], bboxPrev, CROP_PAD, CROP_SIZE)[0]

            if useSimulator:
                patch = simulator.render_patch(noisyBox, background, trackedObjects)
                tImage[dd,1,...] = patch
                if debug:
                    images.append(simulator.get_image_for_frame(trackedObjects, background))
            else:
                tImage[dd,1,...] = im_util.get_cropped_input(
                        images[dd], noisyBox, CROP_PAD, CROP_SIZE)[0]

            shiftedBBox = bb_util.to_crop_coordinate_system(bboxOn, noisyBox, CROP_PAD, 1)
            shiftedBBoxXYWH = bb_util.xyxy_to_xywh(shiftedBBox)
            xywhLabels[dd,:] = shiftedBBoxXYWH


            if gtType < USE_NETWORK_PROB:
                # Run through a single forward pass to get the next box estimate.
                if dd < delta - 1:
                    if dd == 0:
                        lstmState = initialLstmState

                    feed_dict = {
                            forwardNetworkImagePlaceholder : tImage[dd,...],
                            prevLstmState : lstmState
                            }
                    networkOuts, s1, s2 = sess.run([networkOutputs, state1, state2], feed_dict=feed_dict)
                    lstmState = (s1[0], s1[1], s2[0], s2[1])

                    xyxyPred = networkOuts.squeeze() / 10
                    outputBox = bb_util.from_crop_coordinate_system(xyxyPred, noisyBox, CROP_PAD, 1)

                    bboxPrev = outputBox
                    if debug:
                        bboxes.append(outputBox)
                        cropBBoxes.append(xyxyPred)
            else:
                bboxPrev = bboxOn

            if FLAGS.debug:
                # Look at the inputs to make sure they are correct.
                image0 = tImage[dd,0,...].copy()
                image1 = tImage[dd,1,...].copy()

                xyxyLabel = bb_util.xywh_to_xyxy(xywhLabels[dd,:].squeeze())
                print('xyxy raw', xyxyLabel, 'actual', xyxyLabel * CROP_PAD)
                label = np.zeros((CROP_PAD, CROP_PAD))
                drawing.drawRect(label,  xyxyLabel * CROP_PAD, 0, 1)
                drawing.drawRect(image0, bb_util.xywh_to_xyxy(np.full((4,1), .5) * CROP_SIZE), 2, [255,0,0])
                bigImage0 = images[max(dd-1,0)].copy()
                bigImage1 = images[dd].copy()
                if dd < len(cropBBoxes):
                    drawing.drawRect(bigImage1, bboxes[dd], 5, [255,0,0])
                    drawing.drawRect(image1, cropBBoxes[dd] * CROP_SIZE, 1, [0,255,0])
                    print('pred raw', cropBBoxes[dd], 'actual', cropBBoxes[dd] * CROP_PAD)
                print('\n')

                label[0,0] = 1
                label[0,1] = 0
                plots = [bigImage0, bigImage1, image0, image1, label]
                subplot = drawing.subplot(plots, 3, 2, outputWidth=OUTPUT_WIDTH, outputHeight=OUTPUT_HEIGHT, border=5)
                cv2.imshow('debug', subplot[:,:,::-1])
                cv2.waitKey(0)

        if mirrored:
            tImage = np.fliplr(
                    tImage.transpose(2,3,4,0,1)).transpose(3,4,0,1,2)
            xywhLabels[...,0] = 1 - xywhLabels[...,0]

        tImage = tImage.reshape([delta * 2] + list(tImage.shape[2:]))
        xyxyLabels = bb_util.xywh_to_xyxy(xywhLabels.T).T * 10
        return {imagePlaceholder: tImage, labelPlaceholder: xyxyLabels}


    def load_data():
        while True:
            new_data = get_data_sequence()
            queue.enqueue(new_data)

    if FLAGS.debug:
        new_data = get_data_sequence()
        for _ in range(10):
            queue.enqueue(new_data)
    else:
        # Start some data loading threads.
        for i in range(PARALLEL_SIZE):
            load_data_thread = threading.Thread(target=load_data)
            load_data_thread.daemon = True
            load_data_thread.start()
            time.sleep(1)

    try:
        timeTotal = 0.000001
        numIters = 0
        iteration = startIter
        # Run training iterations in the main thread.
        while iteration < FLAGS.max_steps:
            if (iteration - 1) % 10 == 0:
                currentTimeStart = time.time()

            startSolver = time.time()
            if debug:
                _, outputs, lossValue, images, labels, = sess.run([
                    train_op, tfOutputs, tfLoss, imageBatch, labelsBatch],
                    feed_dict={learningRate : 1e-5 if iteration < 10000 else 1e-6})
                debug_feed_dict = {
                        imagePlaceholder : images,
                        labelPlaceholder : labels,
                        }
            else:
                if iteration % 10 == 0:
                    _, lossValue, loss_summary = sess.run([
                            train_op, tfLoss, loss_summary_op],
                            feed_dict={learningRate : 1e-5 if iteration < 10000 else 1e-6})
                    summary_writer.add_summary(loss_summary, iteration)
                else:
                    _, lossValue = sess.run([train_op, tfLoss],
                            feed_dict={learningRate : 1e-5 if iteration < 10000 else 1e-6})
            endSolver = time.time()

            numIters += 1
            iteration += 1

            timeTotal += (endSolver - startSolver)
            if timing and (iteration - 1) % 10 == 0:
                print('Iteration:       %d' % (iteration - 1))
                print('Queue Size:      %d' % queue.size.eval(session=sess))
                print('Loss:            %.3f' % lossValue)
                print('Average Time:    %.3f' % (timeTotal / numIters))
                print('Current Time:    %.3f' % (endSolver - startSolver))
                if numIters > 20:
                    print('Current Average: %.3f' % ((time.time() - currentTimeStart) / 10))
                print('')

            # Save a checkpoint and remove old ones.
            if iteration % 500 == 0 or iteration == FLAGS.max_steps:
                checkpoint_file = os.path.join(LOG_DIR, 'checkpoints', 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=iteration)
                if FLAGS.clearSnapshots:
                    files = glob.glob(LOG_DIR + '/checkpoints/*')
                    for file in files:
                        basename = os.path.basename(file)
                        if os.path.isfile(file) and str(iteration) not in file and 'checkpoint' not in basename:
                            os.remove(file)
            # Every once in a while save a checkpoint that isn't ever removed except by hand.
            if iteration % 10000 == 0 or iteration == FLAGS.max_steps:
                if not os.path.exists(LOG_DIR + '/checkpoints/long_checkpoints'):
                    os.makedirs(LOG_DIR + '/checkpoints/long_checkpoints')
                checkpoint_file = os.path.join(LOG_DIR, 'checkpoints/long_checkpoints', 'model.ckpt')
                longSaver.save(sess, checkpoint_file, global_step=iteration)
            if not debug:
                if (numIters == 1 or
                    iteration % 100 == 0 or
                    iteration == FLAGS.max_steps):
                    # Write out the full graph sometimes.
                    if (numIters == 1 or
                        iteration == FLAGS.max_steps):
                        print('Running detailed summary')
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        _, summary_str = sess.run([train_op, summary_with_images],
                                              options=run_options,
                                              run_metadata=run_metadata,
                                              feed_dict={learningRate : 1e-5 if iteration < 10000 else 1e-6})
                        summary_writer.add_run_metadata(run_metadata, 'step_%07d' % iteration)
                    elif iteration % 1000 == 0:
                        _, summary_str = sess.run([train_op, summary_with_images],
                            feed_dict={learningRate : 1e-5 if iteration < 10000 else 1e-6})
                        print('Running image summary')
                    else:
                        print('Running summary')
                        _, summary_str = sess.run([train_op, summary_full],
                            feed_dict={learningRate : 1e-5 if iteration < 10000 else 1e-6})
                    summary_writer.add_summary(summary_str, iteration)
                    summary_writer.flush()
                if (FLAGS.run_val and (numIters == 1 or iteration % 500 == 0)):
                    # Run a validation set eval in a separate process.
                    def test_func():
                        test_iter_on = iteration
                        print('Staring test iter', test_iter_on)
                        import subprocess
                        import json
                        command = ['python', 'test_net.py', '--video_sample_rate', str(10), '--no-display', '-v', str(FLAGS.val_device)]
                        subprocess.call(command)
                        result = json.load(open('results.json', 'r'))
                        summary_str = sess.run(test_summary_op, feed_dict={
                            robustness_ph : result['robustness'],
                            lost_targets_ph : result['lostTarget'],
                            mean_iou_ph : result['meanIou'],
                            avg_ph : (result['meanIou'] + result['robustness']) / 2,
                            })
                        summary_writer.add_summary(summary_str, test_iter_on)
                        os.remove('results.json')
                        print('Ending test iter', test_iter_on)
                    test_thread = threading.Thread(target=test_func)
                    test_thread.daemon = True
                    test_thread.start()
            if FLAGS.output:
                # Look at some of the outputs.
                print('new batch')
                queue.lock.acquire()
                images = debug_feed_dict[imagePlaceholder].astype(np.uint8).reshape(
                        (batchSize, delta, 2, CROP_SIZE, CROP_SIZE, 3))
                labels = debug_feed_dict[labelPlaceholder].reshape(
                        (batchSize, delta, 4))
                outputs = outputs.reshape((batchSize, delta, 4))
                for bb in range(batchSize):
                    for dd in range(delta):
                        image0 = images[bb,dd,0,...]
                        image1 = images[bb,dd,1,...]

                        label = labels[bb,dd,:]
                        xyxyLabel = label / 10
                        labelBox = xyxyLabel * CROP_PAD

                        output = outputs[bb,dd,...]
                        xyxyPred = output / 10
                        outputBox = xyxyPred * CROP_PAD

                        drawing.drawRect(image0, bb_util.xywh_to_xyxy(np.full((4,1), .5) * CROP_SIZE), 2, [255,0,0])
                        drawing.drawRect(image1, xyxyLabel * CROP_SIZE, 2, [0,255,0])
                        drawing.drawRect(image1, xyxyPred * CROP_SIZE, 2, [255,0,0])

                        plots = [image0, image1]
                        subplot = drawing.subplot(plots, 1, 2, outputWidth=OUTPUT_WIDTH, outputHeight=OUTPUT_HEIGHT, border=5)
                        cv2.imshow('debug', subplot[:,:,::-1])
                        cv2.waitKey(0)
                queue.lock.release()
    except:
        # Save if error or killed by ctrl-c.
        if not debug:
            print('Saving...')
            checkpoint_file = os.path.join(LOG_DIR, 'checkpoints', 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=iteration)
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training for Re3.')
    parser.add_argument('-n', '--num_unrolls', action='store', default=2, dest='delta', type=int)
    parser.add_argument('-b', '--batch_size', action='store', default=64, type=int)
    parser.add_argument('-v', '--cuda_visible_devices', type=str, default=str(GPU_ID), help='Device number or string')
    parser.add_argument('-r', '--restore', action='store_true', default=False)
    parser.add_argument('-d', '--debug', action='store_true', default=False)
    parser.add_argument('-t', '--timing', action='store_true', default=False)
    parser.add_argument('-o', '--output', action='store_true', default=False)
    parser.add_argument('-c', '--clear_snapshots', action='store_true', default=False, dest='clearSnapshots')
    parser.add_argument('-p', '--port', action='store', default=9987, dest='port', type=int)
    parser.add_argument('--run_val', action='store_true', default=False)
    parser.add_argument('--val_device', type=str, default='0', help='Device number or string for val process to use.')
    parser.add_argument('-m', '--max_steps', type=int, default=NUM_ITERATIONS, help='Number of steps to run trainer.')
    FLAGS = parser.parse_args()
    main(FLAGS)

