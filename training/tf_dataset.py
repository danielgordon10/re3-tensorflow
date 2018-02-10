import tensorflow as tf
import cv2
import numpy as np
import socket
import struct
import time
import multiprocessing
import random

try:
    import cPickle as pickle
except:
    # Python 3
    import pickle

import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))

import get_datasets

from io import BytesIO

from tracker import network
from re3_utils.util import bb_util
from re3_utils.util import im_util
from re3_utils.util import drawing
from re3_utils.util import IOU
from re3_utils.simulator import simulator
from re3_utils.tensorflow_util import tf_util

from constants import CROP_PAD
from constants import CROP_SIZE
from constants import LSTM_SIZE
from constants import OUTPUT_WIDTH
from constants import OUTPUT_HEIGHT
from constants import LOG_DIR

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)

HOST = 'localhost'

SIMULATION_WIDTH = simulator.IMAGE_WIDTH
SIMULATION_HEIGHT = simulator.IMAGE_HEIGHT
simulator.NUM_DISTRACTORS = 20

USE_SIMULATOR = 0.5
USE_NETWORK_PROB = 0.8
REAL_MOTION_PROB = 1.0 / 8
AREA_CUTOFF = 0.7
PARALLEL_SIZE = 4

class Dataset(object):
    def __init__(self, sess, delta, prefetch_size, port, debug=False):
        self.sess = sess
        self.delta = delta
        self.prefetch_size = prefetch_size
        self.port = port
        self.debug = debug

        self.key_lookup = dict()
        self.datasets = []
        self.add_dataset('imagenet_video')
        simulator.make_paths()

    def initialize_tf_placeholders(self, forwardNetworkImagePlaceholder, prevLstmState, networkOutputs, state1, state2):
        self.forwardNetworkImagePlaceholder = forwardNetworkImagePlaceholder
        self.prevLstmState = prevLstmState
        self.networkOutputs = networkOutputs
        self.state1 = state1
        self.state2 = state2
        self.initialLstmState = tuple([np.zeros((1, LSTM_SIZE)) for _ in range(4)])

    def add_dataset(self, dataset_name):
        dataset_ind = len(self.datasets)
        dataset_gt = get_datasets.get_data_for_dataset(dataset_name, 'train')['gt']
        for xx in range(dataset_gt.shape[0]):
            line = dataset_gt[xx,:].astype(int)
            self.key_lookup[(dataset_ind, line[4], line[5], line[6])] = xx
        self.datasets.append(dataset_gt)

    def getData(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, self.port))
        sock.sendall(('Ready' + '\n').encode('utf-8'))
        receivedBytes = sock.recv(4)
        messageLength = struct.unpack('>I', receivedBytes)[0]
        key = sock.recv(messageLength)
        key = pickle.loads(key)

        images = [None] * self.delta
        for nn in range(self.delta):
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
    def add_noise(self, bbox, prevBBox, imageWidth, imageHeight):
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

        return self.fix_bbox_intersection(bb_util.xywh_to_xyxy(bboxXYWH), prevBBox, imageWidth, imageHeight)

    # Make sure there is a minimum intersection with the ground truth box and the visible crop.
    def fix_bbox_intersection(self, bbox, gtBox, imageWidth, imageHeight):
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

    def get_data_sequence(self):
        try:
            # Preallocate the space for the images and labels.
            tImage = np.zeros((self.delta, 2, CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
            xywhLabels = np.zeros((self.delta, 4), dtype=np.float32)

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
                if self.debug:
                    images = [simulator.get_image_for_frame(trackedObjects, background)]
                else:
                    images = [np.zeros((SIMULATION_HEIGHT, SIMULATION_WIDTH))]

            else:
                # Read a new data sequence from batch cache and get the ground truth.
                (batchKey, images) = self.getData()
                gtKey = batchKey
                imageIndex = self.key_lookup[gtKey]
                initBox = self.datasets[gtKey[0]][imageIndex, :4].copy()
            if self.debug:
                bboxes = []
                cropBBoxes = []

            # bboxPrev starts at the initial box and is the best guess (or gt) for the image0 location.
            # noisyBox holds the bboxPrev estimate plus some noise.
            bboxPrev = initBox
            lstmState = None

            for dd in range(self.delta):
                # bboxOn is the gt location in image1
                if useSimulator:
                    bboxOn = trackingObj.get_object_box()
                else:
                    newKey = list(gtKey)
                    newKey[3] += dd
                    newKey = tuple(newKey)
                    imageIndex = self.key_lookup[newKey]
                    bboxOn = self.datasets[newKey[0]][imageIndex, :4].copy()
                if dd == 0:
                    noisyBox = bboxOn.copy()
                elif not realMotion and not useSimulator and gtType >= USE_NETWORK_PROB:
                    noisyBox = self.add_noise(bboxOn, bboxOn, images[0].shape[1], images[0].shape[0])
                else:
                    noisyBox = self.fix_bbox_intersection(bboxPrev, bboxOn, images[0].shape[1], images[0].shape[0])

                if useSimulator:
                    patch = simulator.render_patch(bboxPrev, background, trackedObjects)
                    tImage[dd,0,...] = patch
                    if dd > 0:
                        simulator.step(trackedObjects)
                        bboxOn = trackingObj.get_object_box()
                        noisyBox = self.fix_bbox_intersection(bboxPrev, bboxOn, images[0].shape[1], images[0].shape[0])
                else:
                    tImage[dd,0,...] = im_util.get_cropped_input(
                            images[max(dd-1, 0)], bboxPrev, CROP_PAD, CROP_SIZE)[0]

                if useSimulator:
                    patch = simulator.render_patch(noisyBox, background, trackedObjects)
                    tImage[dd,1,...] = patch
                    if self.debug:
                        images.append(simulator.get_image_for_frame(trackedObjects, background))
                else:
                    tImage[dd,1,...] = im_util.get_cropped_input(
                            images[dd], noisyBox, CROP_PAD, CROP_SIZE)[0]

                shiftedBBox = bb_util.to_crop_coordinate_system(bboxOn, noisyBox, CROP_PAD, 1)
                shiftedBBoxXYWH = bb_util.xyxy_to_xywh(shiftedBBox)
                xywhLabels[dd,:] = shiftedBBoxXYWH


                if gtType < USE_NETWORK_PROB:
                    # Run through a single forward pass to get the next box estimate.
                    if dd < self.delta - 1:
                        if dd == 0:
                            lstmState = self.initialLstmState

                        feed_dict = {
                                self.forwardNetworkImagePlaceholder : tImage[dd,...],
                                self.prevLstmState : lstmState
                                }
                        networkOuts, s1, s2 = self.sess.run([self.networkOutputs, self.state1, self.state2], feed_dict=feed_dict)
                        lstmState = (s1[0], s1[1], s2[0], s2[1])

                        xyxyPred = networkOuts.squeeze() / 10
                        outputBox = bb_util.from_crop_coordinate_system(xyxyPred, noisyBox, CROP_PAD, 1)

                        bboxPrev = outputBox
                        if self.debug:
                            bboxes.append(outputBox)
                            cropBBoxes.append(xyxyPred)
                else:
                    bboxPrev = bboxOn

                if self.debug:
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
                    plots = [bigImage0, bigImage1, image0, image1]
                    subplot = drawing.subplot(plots, 2, 2, outputWidth=OUTPUT_WIDTH, outputHeight=OUTPUT_HEIGHT, border=5)
                    cv2.imshow('debug', subplot[:,:,::-1])
                    cv2.waitKey(0)

            if mirrored:
                tImage = np.fliplr(
                        tImage.transpose(2,3,4,0,1)).transpose(3,4,0,1,2)
                xywhLabels[...,0] = 1 - xywhLabels[...,0]

            tImage = tImage.reshape([self.delta * 2] + list(tImage.shape[2:]))
            xyxyLabels = bb_util.xywh_to_xyxy(xywhLabels.T).T * 10
            xyxyLabels = xyxyLabels.astype(np.float32)
            return tImage, xyxyLabels
        except Exception as e:
            import traceback
            traceback.print_exc()
            import pdb
            pdb.set_trace()
            print('exception')


    def generator(self):
        while True:
            yield self.get_data_sequence()


    def get_dataset(self, batch_size):
        def get_data_generator(ind):
            dataset = tf.data.Dataset.from_generator(self.generator, (tf.uint8, tf.float32))
            dataset = dataset.prefetch(int(np.ceil(self.prefetch_size * 1.0 / PARALLEL_SIZE)))
            return dataset

        dataset = tf.data.Dataset.from_tensor_slices(list(range(PARALLEL_SIZE))).interleave(
                get_data_generator, cycle_length=PARALLEL_SIZE)

        dataset = dataset.batch(batch_size)
        dataset_iterator = dataset.make_one_shot_iterator()
        return dataset_iterator



if __name__ == '__main__':
    port = 9997
    delta = 2
    debug = False
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    dataset = Dataset(sess, delta, 1, port, debug)
    forwardNetworkImagePlaceholder = tf.placeholder(tf.uint8, shape=(2, CROP_SIZE, CROP_SIZE, 3))
    prevLstmState = tuple([tf.placeholder(tf.float32, shape=(1, LSTM_SIZE)) for _ in range(4)])
    initialLstmState = tuple([np.zeros((1, LSTM_SIZE)) for _ in range(4)])
    networkOutputs, state1, state2 = network.inference(
            forwardNetworkImagePlaceholder, num_unrolls=1, train=False,
            prevLstmState=prevLstmState, reuse=False)
    dataset.initialize_tf_placeholders(
            forwardNetworkImagePlaceholder, prevLstmState, networkOutputs, state1, state2)
    init = tf.global_variables_initializer()
    sess.run(init)
    ckpt = tf.train.get_checkpoint_state(LOG_DIR + '/checkpoints')
    if ckpt and ckpt.model_checkpoint_path:
        tf_util.restore(sess, ckpt.model_checkpoint_path)
        startIter = int(ckpt.model_checkpoint_path.split('-')[-1])
        print('Restored', startIter)
    iteration = 0
    while True:
        iteration += 1
        print('iteration', iteration)
        dataset.get_data_sequence()

