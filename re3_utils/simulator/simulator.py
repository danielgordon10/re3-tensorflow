import cv2
import glob
import numpy as np
import random
import time


import sys
import os
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.path.pardir,
    os.path.pardir
    )))

from re3_utils.util.bb_util import *
from re3_utils.util.IOU import *
from re3_utils.simulator import TrackedObject
from constants import CROP_SIZE
from constants import CROP_PAD

OBJECT_PAD = 4
MAX_NEG_IOU_THRESH = .3

IMAGE_WIDTH = 400
IMAGE_HEIGHT = 400
NUM_DISTRACTORS = 20

BOXES = None
IMAGE_NAMES = None

BLEND_WEIGHT = None
FEATHER_WEIGHT_ARRAY = None

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def set_speed(speed):
    TrackedObject.SPEED = speed

def set_crop_size(crop_size):
    global CROP_SIZE, BLEND_WEIGHT, FEATHER_WEIGHT_ARRAY
    CROP_SIZE = crop_size
    BLEND_WEIGHT = np.arange(0,1, 4.0 / CROP_SIZE)
    FEATHER_WEIGHT_ARRAY = np.concatenate((
        BLEND_WEIGHT, np.full((int(CROP_SIZE - 2 * len(BLEND_WEIGHT))), 1),
        BLEND_WEIGHT[::-1]))

    FEATHER_WEIGHT_ARRAY = np.tile(FEATHER_WEIGHT_ARRAY[:,np.newaxis,np.newaxis], (1, len(FEATHER_WEIGHT_ARRAY), 3))
    FEATHER_WEIGHT_ARRAY = np.minimum(FEATHER_WEIGHT_ARRAY,
            FEATHER_WEIGHT_ARRAY.transpose(1,0,2))
    FEATHER_WEIGHT_ARRAY *= 255
    FEATHER_WEIGHT_ARRAY = FEATHER_WEIGHT_ARRAY.astype(np.uint8)

set_crop_size(CROP_SIZE)

def make_paths(train=True):
    global IMAGE_NAMES, BOXES
    base_path = os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            os.path.pardir,
            'training',
            'datasets',
            'imagenet_detection')
    image_datadir = os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            os.path.pardir,
            'training',
            'datasets',
            'imagenet_detection')
    dataType = 'train' if train else 'val'
    IMAGE_NAMES = [image_datadir + '/' + line.strip() for line in open(base_path + '/labels/' + dataType + '/image_names.txt')]
    BOXES = np.load(base_path + '/labels/' + dataType + '/labels.npy')

def get_random_image():
    randInd = random.randint(0, BOXES.shape[0] - 1)
    nameInd = BOXES[randInd, 4]
    image = cv2.imread(IMAGE_NAMES[nameInd])[:,:,::-1]
    if len(image.shape) < 3:
        image = np.tile(image[:,:,np.newaxis], (1,1,3))
    if image.shape[2] > 3:
        image = image[:,:,:3]
    return image, randInd

def get_image_crop(inputImage, bbox, padScale=OBJECT_PAD):
    bbox = np.array(bbox)
    boxOn = scale_bbox(bbox, padScale, round=True)

    bboxClip = boxOn.copy()
    bboxClip[[0,2]] = np.clip(bboxClip[[0,2]], 0, inputImage.shape[1])
    bboxClip[[1,3]] = np.clip(bboxClip[[1,3]], 0, inputImage.shape[0])
    objectBox = bbox - bboxClip[[0,1,0,1]]
    imagePatch = inputImage[bboxClip[1]:bboxClip[3], bboxClip[0]:bboxClip[2],:]
    if imagePatch.shape[0] == 0 or imagePatch.shape[1] == 0:
        imagePatch = np.zeros((3,3,3))
    elif len(imagePatch.shape) < 3:
        imagePatch = np.tile(imagePatch[:,:,np.newaxis], (1,1,3))
    imagePatch = imagePatch.squeeze()
    return imagePatch, objectBox

def get_distractor_crop(inputImage, bbox):
    # Take a random crop the same dimensionality as the bbox
    bboxOnXYWH = xyxy_to_xywh(bbox)
    randRect = np.zeros(4)
    minIntersection = 1
    minIntersectionRect = np.zeros(4)
    bboxArea = bboxOnXYWH[2] * bboxOnXYWH[3]
    for _ in range(10000):
        randW = np.random.randint(10, inputImage.shape[1] * .5)
        randH = np.random.randint(10, inputImage.shape[0] * .5)
        randX = np.random.randint(int(randW / 2), int(inputImage.shape[1] - randW / 2))
        randY = np.random.randint(int(randH / 2), int(inputImage.shape[0] - randH / 2))
        randRect = xywh_to_xyxy([randX, randY, randW, randH], round=True)
        iou = intersection(randRect, bbox) / bboxArea
        if iou < MAX_NEG_IOU_THRESH:
            break
        elif iou < minIntersection:
            minIntersection = iou
            minIntersectionRect = randRect.copy()
    if iou >= MAX_NEG_IOU_THRESH:
        randRect = minIntersectionRect
    randPatch = inputImage[randRect[1]:randRect[3], randRect[0]:randRect[2], :]
    randRect -= randRect[[0,1,0,1]]
    return randPatch, randRect


def create_new_track():
    global NUM_DISTRACTORS
    image, randInd = get_random_image()
    bbox = BOXES[randInd, :4].copy()
    if random.random() < .5:
        image = np.fliplr(image)
        bbox[[0,2]] = image.shape[1] - bbox[[2,0]]
    patch, objectBox = get_image_crop(image, bbox)

    trackObjects = []

    newObj = TrackedObject.TrackedObject(IMAGE_WIDTH, IMAGE_HEIGHT, patch, objectBox)
    maxDistractorSize = min(max(newObj.size) / 2.0, max(newObj.size) * 30.0 / (NUM_DISTRACTORS))

    distractorImage = image
    fakeBox = np.array([1,1,2,2])

    for dd in range(int(NUM_DISTRACTORS / 2)):
        distractorPatch, distractorRect = get_distractor_crop(distractorImage, fakeBox)
        distractorObj = TrackedObject.TrackedObject(IMAGE_WIDTH, IMAGE_HEIGHT,
                distractorPatch, distractorRect, maxDistractorSize)
        trackObjects.append(distractorObj)

    trackObjects.append(newObj)

    for dd in range(int((NUM_DISTRACTORS + 1) / 2)):
        distractorPatch, distractorRect = get_distractor_crop(distractorImage, fakeBox)
        distractorObj = TrackedObject.TrackedObject(IMAGE_WIDTH, IMAGE_HEIGHT,
                distractorPatch, distractorRect, maxDistractorSize)
        trackObjects.append(distractorObj)

    for oo, trackObj in enumerate(trackObjects):
        trackObj.occluder_boxes = trackObjects[oo + 1:]
    # Try once more to get the tracking object unoccluded.
    newObj.bbox_init()
    backgroundImage = image
    background = cv2.resize(backgroundImage, (IMAGE_WIDTH, IMAGE_HEIGHT))
    return newObj, trackObjects, background


def step(trackedObjects):
    for tObject in trackedObjects:
        tObject.step()

def step_back(trackedObjects, back_ind):
    for tObject in trackedObjects:
        tObject.step_back(back_ind)

def reset_step(trackedObjects):
    for tObject in trackedObjects:
        tObject.reset_step()


def get_image_for_frame(trackedObjects, backgroundImage=None):
    if backgroundImage is None:
        new_image = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), -1, dtype=np.uint8)
    else:
        new_image = cv2.resize(backgroundImage, (IMAGE_WIDTH, IMAGE_HEIGHT))
    for tObject in trackedObjects:
        bbox = tObject.get_bounded_bbox()
        new_image[bbox[1]:bbox[3], bbox[0]:bbox[2]] = cv2.resize(
                tObject.texture, (bbox[2]-bbox[0], bbox[3]-bbox[1]))
    return new_image

def render_patch(bbox, background, trackedObjects, cropSize=CROP_SIZE, cropPad=CROP_PAD):
    bboxXYWH = xyxy_to_xywh(bbox)
    image = np.zeros((int(cropSize), int(cropSize), 3), dtype=np.uint8)
    fullBBoxXYWH = bboxXYWH.copy()
    fullBBoxXYWH[[2,3]] *= cropPad
    fullBBox = xywh_to_xyxy(fullBBoxXYWH)
    fullBBoxXYWH = fullBBoxXYWH
    fullBBoxXYWH[[2,3]] = np.maximum(fullBBoxXYWH[[2,3]], 1)
    # First do background
    boxPos = np.array([0, 0, background.shape[1], background.shape[0]])
    boxPosXYWH = xyxy_to_xywh(boxPos)
    cropCoords = np.clip(boxPos - fullBBox[[0,1,0,1]], 0, fullBBoxXYWH[[2,3,2,3]])
    cropCoords *= (cropSize) * 1.0 / fullBBoxXYWH[[2,3,2,3]]
    cropCoords = np.clip(np.round(cropCoords), 0, cropSize).astype(int)

    textureCrop = np.zeros(4)
    textureCrop[0] = int(max(fullBBox[0] - boxPos[0], 0) * background.shape[1] * 1.0 / boxPosXYWH[2])
    textureCrop[1] = int(max(fullBBox[1] - boxPos[1], 0) * background.shape[0] * 1.0 / boxPosXYWH[3])
    textureCrop[2] = int(min((fullBBox[2] - boxPos[0]) * 1.0 / boxPosXYWH[2], 1) * background.shape[1])
    textureCrop[3] = int(min((fullBBox[3] - boxPos[1]) * 1.0 / boxPosXYWH[3], 1) * background.shape[0])
    if (textureCrop[2] - textureCrop[0] < 1 or
            textureCrop[3] - textureCrop[1] < 1):
        textureCrop = [0,0,1,1]
    textureCrop = np.round(textureCrop).astype(int)
    textureCrop[[0,2]] = np.clip(textureCrop[[0,2]], 0, background.shape[1])
    textureCrop[[1,3]] = np.clip(textureCrop[[1,3]], 0, background.shape[0])

    if cropCoords[3] > cropCoords[1] + 1 and cropCoords[2] > cropCoords[0] + 1:
        image[cropCoords[1]:cropCoords[3], cropCoords[0]:cropCoords[2],:] = (
                cv2.resize(
                    background[textureCrop[1]:textureCrop[3],
                        textureCrop[0]:textureCrop[2], :],
                    (cropCoords[2]-cropCoords[0], cropCoords[3]-cropCoords[1])))

    # Now do all objects
    for obj in trackedObjects:
        boxPos = obj.get_bbox()
        boxPosXYWH = xyxy_to_xywh(boxPos)
        if IOU(boxPos, fullBBox) < 0.001:
            continue
        cropCoords = np.zeros(4)
        cropCoords = np.clip(boxPos - fullBBox[[0,1,0,1]], 0, fullBBoxXYWH[[2,3,2,3]])
        cropCoords *= cropSize * 1.0 / fullBBoxXYWH[[2,3,2,3]]
        cropCoords = np.clip(np.round(cropCoords), 0, cropSize).astype(int)
        if (cropCoords[2] - cropCoords[0] < 1 or
                cropCoords[3] - cropCoords[1] < 1):
            cropCoords[[0,1]] = np.clip(cropCoords[[0,1]] - 1, 0, cropSize).astype(int)
            cropCoords[[2,3]] = np.clip(cropCoords[[2,3]] + 1, 0, cropSize).astype(int)
        textureCrop = np.zeros(4, dtype=int)
        textureCrop[0] = int(max(fullBBox[0] - boxPos[0], 0) * obj.texture.shape[1] * 1.0 / boxPosXYWH[2])
        textureCrop[1] = int(max(fullBBox[1] - boxPos[1], 0) * obj.texture.shape[0] * 1.0 / boxPosXYWH[3])
        textureCrop[2] = int(min((fullBBox[2] - boxPos[0]) * 1.0 / boxPosXYWH[2], 1) * obj.texture.shape[1])
        textureCrop[3] = int(min((fullBBox[3] - boxPos[1]) * 1.0 / boxPosXYWH[3], 1) * obj.texture.shape[0])
        if (textureCrop[2] - textureCrop[0] < 1 or
                textureCrop[3] - textureCrop[1] < 1):
            textureCrop = [0,0,2,2]
        textureCrop = np.round(textureCrop).astype(int)
        textureCrop[[0,2]] = np.clip(textureCrop[[0,2]], 0, obj.texture.shape[1])
        textureCrop[[1,3]] = np.clip(textureCrop[[1,3]], 0, obj.texture.shape[0])

        # Feathering
        currentIm = image[cropCoords[1]:cropCoords[3], cropCoords[0]:cropCoords[2],:].astype(np.float32)
        newIm = cv2.resize(
                    obj.texture[textureCrop[1]:textureCrop[3],
                        textureCrop[0]:textureCrop[2], :],
                    (cropCoords[2]-cropCoords[0], cropCoords[3]-cropCoords[1]),
                    ).astype(np.float32)

        if (cropCoords[2] - cropCoords[0] < 1 or
                cropCoords[3] - cropCoords[1] < 1):
            featherWeightOn = 0
        else:
            featherCrop = np.zeros(4)
            featherCrop[0] = int(max(fullBBox[0] - boxPos[0], 0) * FEATHER_WEIGHT_ARRAY.shape[1] * 1.0 / boxPosXYWH[2])
            featherCrop[1] = int(max(fullBBox[1] - boxPos[1], 0) * FEATHER_WEIGHT_ARRAY.shape[0] * 1.0 / boxPosXYWH[3])
            featherCrop[2] = int(min((fullBBox[2] - boxPos[0]) * 1.0 / boxPosXYWH[2], 1) * FEATHER_WEIGHT_ARRAY.shape[1])
            featherCrop[3] = int(min((fullBBox[3] - boxPos[1]) * 1.0 / boxPosXYWH[3], 1) * FEATHER_WEIGHT_ARRAY.shape[0])
            if (featherCrop[2] - featherCrop[0] < 1 or
                    featherCrop[3] - featherCrop[1] < 1):
                featherCrop = [int(CROP_SIZE / 2 - 1), int(CROP_SIZE / 2 - 1), int(CROP_SIZE / 2), int(CROP_SIZE / 2)]
            featherCrop = np.round(featherCrop).astype(int)
            featherCrop[[0,2]] = np.clip(featherCrop[[0,2]], 0, FEATHER_WEIGHT_ARRAY.shape[1])
            featherCrop[[1,3]] = np.clip(featherCrop[[1,3]], 0, FEATHER_WEIGHT_ARRAY.shape[0])
            featherWeightOn = cv2.resize(
                    FEATHER_WEIGHT_ARRAY[
                        featherCrop[1]:featherCrop[3],
                        featherCrop[0]:featherCrop[2], :],
                    (cropCoords[2] - cropCoords[0],
                     cropCoords[3] - cropCoords[1])).astype(np.float32) / 255.0
        image[cropCoords[1]:cropCoords[3], cropCoords[0]:cropCoords[2],:] = (
                (newIm * featherWeightOn +
                 currentIm * (1 - featherWeightOn)).astype(np.uint8))
    return image


def measure_occlusion(bbox, trackedObjects, cropSize=CROP_SIZE, cropPad=CROP_PAD):
    image = np.zeros((int(cropSize), int(cropSize)), dtype=np.bool)
    fullBBox = scale_bbox(bbox, cropPad)
    fullBBoxXYWH = xyxy_to_xywh(fullBBox)
    fullBBoxXYWH[[2,3]] = np.maximum(fullBBoxXYWH[[2,3]], 1)

    # Now do all objects
    for obj in trackedObjects:
        boxPos = obj.get_bbox()
        boxPosXYWH = xyxy_to_xywh(boxPos)
        if IOU(boxPos, fullBBox) < 0.001:
            continue
        cropCoords = np.zeros(4)
        cropCoords = np.clip(boxPos - fullBBox[[0,1,0,1]], 0, fullBBoxXYWH[[2,3,2,3]])
        cropCoords *= cropSize * 1.0 / fullBBoxXYWH[[2,3,2,3]]
        cropCoords = np.clip(np.round(cropCoords), 0, cropSize).astype(int)
        if (cropCoords[2] - cropCoords[0] < 1 or
                cropCoords[3] - cropCoords[1] < 1):
            cropCoords[[0,1]] = np.clip(cropCoords[[0,1]] - 1, 0, cropSize).astype(int)
            cropCoords[[2,3]] = np.clip(cropCoords[[2,3]] + 1, 0, cropSize).astype(int)
        image[cropCoords[1]:cropCoords[3], cropCoords[0]:cropCoords[2]] = True
    return np.count_nonzero(image) * 1.0 / image.size


def get_shifted_box_coords(newBox, prevBox, cropPad=CROP_PAD,
        object_pad=OBJECT_PAD):
    originBoxXYWH = xyxy_to_xywh(prevBox)
    originBoxXYWH[2:] *= cropPad
    originBox = xywh_to_xyxy(originBoxXYWH)

    # Tight box for obj in full image's coordinate frame
    newLocXYWH = xyxy_to_xywh(newBox)

    # Need it in the current coordinate frame.
    newLocXYWH[:2] -= originBox[:2]
    newLocXYWH *= CROP_SIZE / originBoxXYWH[[2,3,2,3]]
    return xywh_to_xyxy(newLocXYWH)


# Returns a list of (image, tight bb for object).
def get_image_sequence(seqLen, imCount=0, writeFull=False):
    trackingObj, trackedObjects, background = create_new_track()
    sequence = []
    if writeFull:
        seqInd = imCount

    prevLoc = trackingObj.get_object_box()

    for i in range(seqLen):
        newLocShifted = get_shifted_box_coords(trackingObj.get_object_box(), prevLoc)

        renderPatch = render_patch(prevLoc, background, trackedObjects)
        sequence.append((renderPatch, newLocShifted))

        prevLoc = trackingObj.get_object_box()

        if writeFull:
            newImage = get_image_for_frame(trackedObjects, background)
            cv2.imwrite('images_full/%07d.png' % seqInd, newImage[:,:,::-1])
            seqInd += 1
        step(trackedObjects)
    return sequence


if __name__ == '__main__':
    set_seed(0)
    from re3_utils.util import drawing
    NUM_SEQUENCES = 10
    SEQUENCE_LENGTH = 200

    if not os.path.exists('images'):
        os.mkdir('images')
        os.mkdir('images_full')
    make_paths()
    times = []
    imCount = 0

    for xx in range(NUM_SEQUENCES):
        startTime = time.time()
        sequence = get_image_sequence(SEQUENCE_LENGTH, imCount, True)
        times.append((time.time() - startTime) / SEQUENCE_LENGTH)
        for (image, bbox) in sequence:
            imCount += 1
            drawing.drawRect(image, bbox, 1, [255,0,0])
            cv2.imwrite('images/%07d.png' % imCount, image[:,:,::-1])
        print('average time per frame %.5f' % np.mean(times))

