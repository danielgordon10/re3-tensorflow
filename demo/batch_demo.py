import cv2
import glob
import numpy as np
import sys
import os.path

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
from tracker import re3_tracker

if not os.path.exists(os.path.join(basedir, 'data')):
    import tarfile
    tar = tarfile.open(os.path.join(basedir, 'data.tar.gz'))
    tar.extractall(path=basedir)

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 640, 480)
tracker = re3_tracker.Re3Tracker()
image_paths = sorted(glob.glob(os.path.join(
    os.path.dirname(__file__), 'data', '*.jpg')))
initial_bbox = [190, 158, 249, 215]
# Provide a unique id, an image/path, and a bounding box.
tracker.track('ball', image_paths[0], initial_bbox)
print('ball track started')
for ii,image_path in enumerate(image_paths):
    image = cv2.imread(image_path)
    # Tracker expects RGB, but opencv loads BGR.
    imageRGB = image[:,:,::-1]
    if ii < 100:
        # The track alread exists, so all that is needed is the unique id and the image.
        bbox = tracker.track('ball', imageRGB)
        color = cv2.cvtColor(np.uint8([[[0, 128, 200]]]),
            cv2.COLOR_HSV2RGB).squeeze().tolist()
        cv2.rectangle(image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color, 2)
    elif ii == 100:
        # Start a new track, but continue the first as well. Only the new track needs an initial bounding box.
        bboxes = tracker.multi_track(['ball', 'logo'], imageRGB, {'logo' : [399, 20, 428, 45]})
        print('logo track started')
    else:
        # Both tracks are started, neither needs bounding boxes.
        bboxes = tracker.multi_track(['ball', 'logo'], imageRGB)
    if ii >= 100:
        for bb,bbox in enumerate(bboxes):
            color = cv2.cvtColor(np.uint8([[[bb * 255 / len(bboxes), 128, 200]]]),
                cv2.COLOR_HSV2RGB).squeeze().tolist()
            cv2.rectangle(image,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    color, 2)
    cv2.imshow('Image', image)
    cv2.waitKey(1)
