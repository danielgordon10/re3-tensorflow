import argparse
import cv2
import numpy as np
import os
import time
import json

import sys
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))

from re3_utils.util import bb_util
from re3_utils.util import im_util
from re3_utils.util.drawing import *
from re3_utils.util.IOU import *

import get_datasets
from tracker import re3_tracker

from constants import CROP_PAD
from constants import CROP_SIZE
from constants import MAX_TRACK_LENGTH

# Display constants
from constants import OUTPUT_WIDTH
from constants import OUTPUT_HEIGHT
from constants import PADDING
from constants import GPU_ID

NUM_COLS = 1
NUM_ROWS = 1
BORDER = 0
FPS = 30
PRINT = True

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

class TestTrackerRunner(object):
    def __init__(self, tracker):
        self.tracker = tracker
        self.imageNames = None
        self.totalIou = 0
        self.numFrames = 0
        self.ignoreFrames = 0
        self.initializeFrames = 0
        self.lostTarget = 0
        self.initialize = True
        self.gt = None
        self.display = False

    def reset(self):
        self.tracker.reset()
        self.imageNames = None
        self.totalIou = 0
        self.numFrames = 0
        self.ignoreFrames = 0
        self.initializeFrames = 0
        self.lostTarget = 0
        self.initialize = True
        self.gt = None
        self.display = False

    def run_test(self, record=False, fancy_text=False, maxCount=-1, skipCount=0, mode='val',
            dataset='imagenet_video', video_sample_rate=1, display=True):
        data = get_datasets.get_data_for_dataset(dataset, mode)
        self.gt = data['gt']
        self.display = display
        self.imageNames = data['image_paths']
        sample_inds = np.where(self.gt[:,4] % video_sample_rate == 0)[0]
        self.gt = self.gt[sample_inds, :]
        numImages = self.gt.shape[0]
        imageNums = self.gt[:, 6].astype(int)

        if maxCount == -1:
            maxCount = numImages - skipCount

        print('Testing', numImages, 'frames')
        # Set up global data holders
        imOn = skipCount

        if record:
            tt = time.localtime()
            import imageio
            writer = imageio.get_writer('./video_%02d_%02d_%02d_%02d_%02d.mp4' %
                    (tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec), fps=FPS)

        if self.display:
            cv2.namedWindow('Output')

        maxIter =  min(maxCount + skipCount, numImages)
        for imOn in range(skipCount, maxIter):
            if self.display:
                plots, titles, results = self.runFrame(imageNums[int(imOn)], int(imOn))
                im = subplot(plots, NUM_ROWS, NUM_COLS, titles=titles,
                        outputWidth=OUTPUT_WIDTH, outputHeight=OUTPUT_HEIGHT,
                        border=BORDER, fancy_text=fancy_text)
                cv2.imshow('Output', im)
                waitKey = cv2.waitKey(1)
                if record:
                    if imOn % 100 == 0:
                        print(imOn)
                    writer.append_data(im[:,:,::-1])
            else:
                results = self.runFrame(imageNums[int(imOn)], int(imOn))
            if PRINT and imOn % 1000 == 0:
                print('Results: ' + str([key + ' : ' + str(results[key]) for key in sorted(results.keys())]))

        print('Final Results: ' + str([key + ' : ' + str(results[key]) for key in sorted(results.keys())]))

        if record:
            writer.close()

        with open('results.json', 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def runFrame(self, imOn, gtOn):
        titles = []

        if (gtOn == 0 or not (
            self.gt[gtOn, 4] == self.gt[gtOn - 1, 4] and
            self.gt[gtOn, 5] == self.gt[gtOn - 1, 5] and
            self.gt[gtOn, 6] - 1 == self.gt[gtOn - 1, 6])):
            if PRINT:
                print('beginning sequence', self.gt[gtOn, [5, 6]])
            # Clear the state if a new sequence has started.
            self.initialize = True
            self.ignoreFrames = 0
            self.initializeFrames = 0

        iou = 1
        robustness = 1

        gtBox = self.gt[gtOn, :4].copy()

        if self.display:
            inputImageBGR = cv2.imread(self.imageNames[imOn])
            inputImage = inputImageBGR[:,:,::-1]
            imageToDraw = inputImageBGR.copy()
            drawRect(imageToDraw, gtBox, PADDING * 2, [0, 255, 0])
        else:
            inputImage = self.imageNames[imOn]


        if self.ignoreFrames > 0:
            self.ignoreFrames -= 1
        else:
            if self.initialize:
                outputBox = self.tracker.track('test_track', inputImage, gtBox)
                self.initialize = False
            else:
                outputBox = self.tracker.track('test_track', inputImage)

            if self.display:
                drawRect(imageToDraw, outputBox, PADDING, [0, 0, 255])

            if self.initializeFrames == 0:
                iou = IOU(outputBox, gtBox)
                self.totalIou += iou
                if iou == 0:
                    self.ignoreFrames = 5
                    self.initializeFrames = 10
                    self.lostTarget += 1
                    self.initialize = True
                self.numFrames += 1
                robustness = np.exp(-30.0 * self.lostTarget / self.numFrames)
            else:
                self.initializeFrames -= 1


        meanIou = self.totalIou * 1.0 / max(self.numFrames, 1)

        if self.display:
            imageToDraw[0,0] = 255
            imageToDraw[0,1] = 0
            titles.append(
                    'Frame %d, IOU %.2f, Mean IOU %.2f, Robustness %.2f, Dropped %d' %
                    (gtOn, iou, meanIou, robustness, self.lostTarget))
            imPlots = [imageToDraw]

        results = {
                'gtOn' : gtOn,
                'meanIou' : meanIou,
                'robustness' : robustness,
                'lostTarget' : self.lostTarget,
                }

        if self.display:
            return (imPlots, titles, results)
        else:
            return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show the Network Results.')
    parser.add_argument('-r', '--record', action='store_true', default=False)
    parser.add_argument('-f', '--fancy_text', action='store_true', default=False,
            help='Use a fancier font than OpenCVs, but takes longer to render an image.'
                 'This should be used for making higher-quality videos.')
    parser.add_argument('-n', '--max_images', action='store', default=-1, dest='maxCount', type=int)
    parser.add_argument('-s', '--num_images_to_skip', action='store', default=0, dest='skipCount', type=int)
    parser.add_argument('-m', '--mode', action='store', default='val', type=str, help='train or val')
    parser.add_argument('--dataset', default='imagenet_video', type=str, help='name of the dataset')
    parser.add_argument('--video_sample_rate', default=1, type=int,
            help='One of every n videos will be run. Useful for testing portions of larger datasets.')
    parser.add_argument('-v', '--cuda_visible_devices', type=str, default=str(GPU_ID), help='Device number or string')
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--display', dest='display', action='store_true')
    feature_parser.add_argument('--no-display', dest='display', action='store_false')
    parser.set_defaults(display=True)

    FLAGS = parser.parse_args()

    tracker = re3_tracker.Re3Tracker(FLAGS.cuda_visible_devices)
    test_track_runner = TestTrackerRunner(tracker)
    test_track_runner.run_test(record=FLAGS.record, fancy_text=FLAGS.fancy_text,
            maxCount=FLAGS.maxCount, skipCount=FLAGS.skipCount, mode=FLAGS.mode,
            dataset=FLAGS.dataset, video_sample_rate=FLAGS.video_sample_rate, display=FLAGS.display)

