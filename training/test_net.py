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
DISPLAY = True

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

def runFrame(imOn, gtOn):
    # Necessary for linking up global variables.
    global tracker
    global totalIou, numFrames
    global initialize
    global imageNames
    global ignoreFrames, initializeFrames, lostTarget

    titles = []

    if (gtOn == 0 or not (
        gt[gtOn, 4] == gt[gtOn - 1, 4] and
        gt[gtOn, 5] == gt[gtOn - 1, 5] and
        gt[gtOn, 6] - 1 == gt[gtOn - 1, 6])):
        print('beginning sequence', gt[gtOn, [5, 6]])
        # Clear the state if a new sequence has started.
        initialize = True
        ignoreFrames = 0
        initializeFrames = 0

    iou = 1
    robustness = 1

    gtBox = gt[gtOn, :4].copy()

    if DISPLAY:
        inputImageBGR = cv2.imread(imageNames[imOn])
        inputImage = inputImageBGR[:,:,::-1]
        imageToDraw = inputImageBGR.copy()
        drawRect(imageToDraw, gtBox, PADDING * 2, [0, 255, 0])
    else:
        inputImage = imageNames[imOn]


    if ignoreFrames > 0:
        ignoreFrames -= 1
    else:
        if initialize:
            outputBox = tracker.track('test_track', inputImage, gtBox)
            initialize = False
        else:
            outputBox = tracker.track('test_track', inputImage)

        if DISPLAY:
            drawRect(imageToDraw, outputBox, PADDING, [0, 0, 255])

        if initializeFrames == 0:
            iou = IOU(outputBox, gtBox)
            totalIou += iou
            if iou == 0:
                ignoreFrames = 5
                initializeFrames = 10
                lostTarget += 1
                initialize = True
            numFrames += 1
            robustness = np.exp(-30.0 * lostTarget / numFrames)
        else:
            initializeFrames -= 1


    meanIou = totalIou * 1.0 / max(numFrames, 1)

    if DISPLAY:
        imageToDraw[0,0] = 255
        imageToDraw[0,1] = 0
        titles.append(
                'Frame %d, IOU %.2f, Mean IOU %.2f, Robustness %.2f, Dropped %d' %
                (gtOn, iou, meanIou, robustness, lostTarget))
        imPlots = [imageToDraw]

    results = {
            'gtOn' : gtOn,
            'meanIou' : meanIou,
            'robustness' : robustness,
            'lostTarget' : lostTarget,
            }

    if DISPLAY:
        return (imPlots, titles, results)
    else:
        return results


# Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show the Network Results.')
    parser.add_argument('-d', '--debug', action='store_true', default=False)
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

    DISPLAY = FLAGS.display

    data = get_datasets.get_data_for_dataset(FLAGS.dataset, FLAGS.mode)
    gt = data['gt']
    imageNames = data['image_paths']
    sample_inds = np.where(gt[:,4] % FLAGS.video_sample_rate == 0)[0]
    gt = gt[sample_inds, :]
    numImages = gt.shape[0]
    imageNums = gt[:, 6].astype(int)

    if FLAGS.maxCount == -1:
        FLAGS.maxCount = numImages - FLAGS.skipCount

    tracker = re3_tracker.Re3Tracker(FLAGS.cuda_visible_devices)

    print('Testing', numImages, 'frames')

    # Set up global data holders
    imOn = FLAGS.skipCount
    numFrames = 0
    ignoreFrames = 0
    initializeFrames = 0
    lostTarget = 0
    currentTrackLength = 0
    initialize = True

    totalIou = 0

    if FLAGS.record:
        tt = time.localtime()
        import imageio
        writer = imageio.get_writer('./video_%02d_%02d_%02d_%02d_%02d.mp4' %
                (tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec), fps=FPS)

    if DISPLAY:
        cv2.namedWindow('Output')

    maxIter =  min(FLAGS.maxCount + FLAGS.skipCount, numImages)
    for imOn in range(FLAGS.skipCount, maxIter):
        if DISPLAY:
            plots, titles, results = runFrame(imageNums[int(imOn)], int(imOn))
            im = subplot(plots, NUM_ROWS, NUM_COLS, titles=titles,
                    outputWidth=OUTPUT_WIDTH, outputHeight=OUTPUT_HEIGHT,
                    border=BORDER, fancy_text=FLAGS.fancy_text)
            cv2.imshow('Output', im)
            waitKey = cv2.waitKey(1)
            if FLAGS.record:
                if imOn % 100 == 0:
                    print(imOn)
                writer.append_data(im[:,:,::-1])
        else:
            results = runFrame(imageNums[int(imOn)], int(imOn))
        if imOn % 100 == 0 or (imOn + 1) == maxIter:
            print('Results: ' + str([key + ' : ' + str(results[key]) for key in sorted(results.keys())]))

    if FLAGS.record:
        writer.close()

    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
