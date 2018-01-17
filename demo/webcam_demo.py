import cv2
import argparse
import glob
import numpy as np
import os
import time
import sys

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
from tracker import re3_tracker

from re3_utils.util import drawing
from re3_utils.util import bb_util
from re3_utils.util import im_util

from constants import OUTPUT_WIDTH
from constants import OUTPUT_HEIGHT
from constants import PADDING

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

drawnBox = np.zeros(4)
boxToDraw = np.zeros(4)
mousedown = False
mouseupdown = False
initialize = False
def on_mouse(event, x, y, flags, params):
    global mousedown, mouseupdown, drawnBox, boxToDraw, initialize
    if event == cv2.EVENT_LBUTTONDOWN:
        drawnBox[[0,2]] = x
        drawnBox[[1,3]] = y
        mousedown = True
        mouseupdown = False
    elif mousedown and event == cv2.EVENT_MOUSEMOVE:
        drawnBox[2] = x
        drawnBox[3] = y
    elif event == cv2.EVENT_LBUTTONUP:
        drawnBox[2] = x
        drawnBox[3] = y
        mousedown = False
        mouseupdown = True
        initialize = True
    boxToDraw = drawnBox.copy()
    boxToDraw[[0,2]] = np.sort(boxToDraw[[0,2]])
    boxToDraw[[1,3]] = np.sort(boxToDraw[[1,3]])


def show_webcam(mirror=False):
    global tracker, initialize
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Webcam', OUTPUT_WIDTH, OUTPUT_HEIGHT)
    cv2.setMouseCallback('Webcam', on_mouse, 0)
    frameNum = 0
    outputDir = None
    outputBoxToDraw = None
    if RECORD:
        print('saving')
        if not os.path.exists('outputs'):
            os.mkdir('outputs')
        tt = time.localtime()
        outputDir = ('outputs/%02d_%02d_%02d_%02d_%02d/' %
                (tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))
        os.mkdir(outputDir)
        labels = open(outputDir + 'labels.txt', 'w')
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        origImg = img.copy()
        if mousedown:
            cv2.rectangle(img,
                    (int(boxToDraw[0]), int(boxToDraw[1])),
                    (int(boxToDraw[2]), int(boxToDraw[3])),
                    [0,0,255], PADDING)
            if RECORD:
                cv2.circle(img, (int(drawnBox[2]), int(drawnBox[3])), 10, [255,0,0], 4)
        elif mouseupdown:
            if initialize:
                outputBoxToDraw = tracker.track('webcam', img[:,:,::-1], boxToDraw)
                initialize = False
            else:
                outputBoxToDraw = tracker.track('webcam', img[:,:,::-1])
            cv2.rectangle(img,
                    (int(outputBoxToDraw[0]), int(outputBoxToDraw[1])),
                    (int(outputBoxToDraw[2]), int(outputBoxToDraw[3])),
                    [0,0,255], PADDING)
        cv2.imshow('Webcam', img)
        if RECORD:
            if outputBoxToDraw is not None:
                labels.write('%d %.2f %.2f %.2f %.2f\n' %
                        (frameNum, outputBoxToDraw[0], outputBoxToDraw[1],
                            outputBoxToDraw[2], outputBoxToDraw[3]))
            cv2.imwrite('%s%08d.jpg' % (outputDir, frameNum), origImg)
            print('saving')
        keyPressed = cv2.waitKey(1)
        if keyPressed == 27 or keyPressed == 1048603:
            break  # esc to quit
        frameNum += 1
    cv2.destroyAllWindows()



# Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Show the Webcam demo.')
    parser.add_argument('-r', '--record', action='store_true', default=False)
    args = parser.parse_args()
    RECORD = args.record

    tracker = re3_tracker.Re3Tracker()

    show_webcam(mirror=True)


