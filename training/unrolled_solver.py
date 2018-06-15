import pdb
import argparse
import cv2
import glob
import numpy as np
import os
import random
import struct
import sys
import tensorflow as tf
import time
import threading

from io import BytesIO

import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))

import tf_dataset
import test_net
from tracker import network
from tracker import re3_tracker
from re3_utils.util import bb_util
from re3_utils.util import im_util
from re3_utils.tensorflow_util import tf_util
from re3_utils.util import drawing
from re3_utils.util import IOU


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

LEARNING_RATE = 1e-5

def main(FLAGS):
    global PORT, delta, REPLAY_BUFFER_SIZE
    delta = FLAGS.delta
    batchSize = FLAGS.batch_size
    timing = FLAGS.timing
    debug = FLAGS.debug or FLAGS.output
    PORT = FLAGS.port

    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.cuda_visible_devices)
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)

    # Tensorflow setup
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(LOG_DIR + '/checkpoints'):
        os.makedirs(LOG_DIR + '/checkpoints')

    tf.Graph().as_default()
    tf.logging.set_verbosity(tf.logging.INFO)

    sess = tf_util.Session()

    # Create the nodes for single image forward passes for learning to fix mistakes.
    # Parameters here are shared with the learned network.
    if ',' in FLAGS.cuda_visible_devices:
        with tf.device('/gpu:1'):
            forwardNetworkImagePlaceholder = tf.placeholder(tf.uint8, shape=(2, CROP_SIZE, CROP_SIZE, 3))
            prevLstmState = tuple([tf.placeholder(tf.float32, shape=(1, LSTM_SIZE)) for _ in range(4)])
            initialLstmState = tuple([np.zeros((1, LSTM_SIZE)) for _ in range(4)])
            networkOutputs, state1, state2 = network.inference(
                    forwardNetworkImagePlaceholder, num_unrolls=1, train=False,
                    prevLstmState=prevLstmState, reuse=False)
    else:
        forwardNetworkImagePlaceholder = tf.placeholder(tf.uint8, shape=(2, CROP_SIZE, CROP_SIZE, 3))
        prevLstmState = tuple([tf.placeholder(tf.float32, shape=(1, LSTM_SIZE)) for _ in range(4)])
        initialLstmState = tuple([np.zeros((1, LSTM_SIZE)) for _ in range(4)])
        networkOutputs, state1, state2 = network.inference(
                forwardNetworkImagePlaceholder, num_unrolls=1, train=False,
                prevLstmState=prevLstmState, reuse=False)

    tf_dataset_obj = tf_dataset.Dataset(sess, delta, batchSize * 2, PORT,
            debug=FLAGS.debug)
    tf_dataset_obj.initialize_tf_placeholders(
            forwardNetworkImagePlaceholder, prevLstmState, networkOutputs, state1, state2)


    tf_dataset_iterator = tf_dataset_obj.get_dataset(batchSize)
    imageBatch, labelsBatch = tf_dataset_iterator.get_next()
    imageBatch = tf.reshape(imageBatch, (batchSize * delta * 2, CROP_SIZE, CROP_SIZE, 3))
    labelsBatch = tf.reshape(labelsBatch, (batchSize * delta, -1))

    learningRate = tf.placeholder(tf.float32)
    imagePlaceholder = tf.placeholder(tf.uint8, shape=(batchSize, delta * 2, CROP_SIZE, CROP_SIZE, 3))
    labelPlaceholder = tf.placeholder(tf.float32, shape=(batchSize, delta, 4))

    if ',' in FLAGS.cuda_visible_devices:
        with tf.device('/gpu:0'):
            tfOutputs = network.inference(imageBatch, num_unrolls=delta, train=True, reuse=True)
            tfLossFull, tfLoss = network.loss(tfOutputs, labelsBatch)
            train_op = network.training(tfLossFull, learningRate)
    else:
        tfOutputs = network.inference(imageBatch, num_unrolls=delta, train=True, reuse=True)
        tfLossFull, tfLoss = network.loss(tfOutputs, labelsBatch)
        train_op = network.training(tfLossFull, learningRate)

    loss_summary_op = tf.summary.merge([
        tf.summary.scalar('loss', tfLoss),
        tf.summary.scalar('l2_regularizer', tfLossFull - tfLoss),
        ])

    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    longSaver = tf.train.Saver()

    # Initialize the network and load saved parameters.
    sess.run(init)
    startIter = 0
    if FLAGS.restore:
        print('Restoring')
        startIter = tf_util.restore_from_dir(sess, os.path.join(LOG_DIR, 'checkpoints'))
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
    if FLAGS.run_val:
        val_gpu = None if FLAGS.val_device == '0' else FLAGS.val_device
        test_tracker = re3_tracker.CopiedRe3Tracker(sess, train_vars, val_gpu)
        test_runner = test_net.TestTrackerRunner(test_tracker)
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
                    feed_dict={learningRate : LEARNING_RATE})
                debug_feed_dict = {
                        imagePlaceholder : images,
                        labelPlaceholder : labels,
                        }
            else:
                if iteration % 10 == 0:
                    _, lossValue, loss_summary = sess.run([
                            train_op, tfLoss, loss_summary_op],
                            feed_dict={learningRate : LEARNING_RATE})
                    summary_writer.add_summary(loss_summary, iteration)
                else:
                    _, lossValue = sess.run([train_op, tfLoss],
                            feed_dict={learningRate : LEARNING_RATE})
            endSolver = time.time()

            numIters += 1
            iteration += 1

            timeTotal += (endSolver - startSolver)
            if timing and (iteration - 1) % 10 == 0:
                print('Iteration:       %d' % (iteration - 1))
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
                                              feed_dict={learningRate : LEARNING_RATE})
                        summary_writer.add_run_metadata(run_metadata, 'step_%07d' % iteration)
                    elif iteration % 1000 == 0:
                        _, summary_str = sess.run([train_op, summary_with_images],
                            feed_dict={learningRate : LEARNING_RATE})
                        print('Running image summary')
                    else:
                        print('Running summary')
                        _, summary_str = sess.run([train_op, summary_full],
                            feed_dict={learningRate : LEARNING_RATE})
                    summary_writer.add_summary(summary_str, iteration)
                    summary_writer.flush()
                if (FLAGS.run_val and (numIters == 1 or iteration % 500 == 0)):
                    # Run a validation set eval in a separate thread.
                    def test_func(test_iter_on):
                        print('Starting test iter', test_iter_on)
                        test_runner.reset()
                        result = test_runner.run_test(dataset=FLAGS.val_dataset, display=False)
                        summary_str = sess.run(test_summary_op, feed_dict={
                            robustness_ph : result['robustness'],
                            lost_targets_ph : result['lostTarget'],
                            mean_iou_ph : result['meanIou'],
                            avg_ph : (result['meanIou'] + result['robustness']) / 2,
                            })
                        summary_writer.add_summary(summary_str, test_iter_on)
                        os.remove('results.json')
                        print('Ending test iter', test_iter_on)
                    test_thread = threading.Thread(target=test_func, args=(iteration,))
                    test_thread.start()
            if FLAGS.output:
                # Look at some of the outputs.
                print('new batch')
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
    parser.add_argument('--val_dataset', type=str, default='vot', help='Dataset to test on.')
    parser.add_argument('--val_device', type=str, default='0', help='Device number or string for val process to use.')
    parser.add_argument('-m', '--max_steps', type=int, default=NUM_ITERATIONS, help='Number of steps to run trainer.')
    FLAGS = parser.parse_args()
    main(FLAGS)

