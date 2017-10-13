import numpy as np
import numbers

LIMIT = 99999999

# BBoxes are [x1, y1, x2, y2]
def clip_bbox(bboxes, minClip, maxXClip, maxYClip):
    bboxesOut = bboxes
    addedAxis = False
    if len(bboxesOut.shape) == 1:
        addedAxis = True
        bboxesOut = bboxesOut[:,np.newaxis]
    bboxesOut[[0,2],...] = np.clip(bboxesOut[[0,2],...], minClip, maxXClip)
    bboxesOut[[1,3],...] = np.clip(bboxesOut[[1,3],...], minClip, maxYClip)
    if addedAxis:
        bboxesOut = bboxesOut[:,0]
    return bboxesOut

# [x1 y1, x2, y2] to [xMid, yMid, width, height]
def xyxy_to_xywh(bboxes, clipMin=-LIMIT, clipWidth=LIMIT, clipHeight=LIMIT,
        round=False):
    addedAxis = False
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes).astype(np.float32)
    if len(bboxes.shape) == 1:
        addedAxis = True
        bboxes = bboxes[:,np.newaxis]
    bboxesOut = np.zeros(bboxes.shape)
    x1 = bboxes[0,...]
    y1 = bboxes[1,...]
    x2 = bboxes[2,...]
    y2 = bboxes[3,...]
    bboxesOut[0,...] = (x1 + x2) / 2.0
    bboxesOut[1,...] = (y1 + y2) / 2.0
    bboxesOut[2,...] = x2 - x1
    bboxesOut[3,...] = y2 - y1
    if clipMin != -LIMIT or clipWidth != LIMIT or clipHeight != LIMIT:
        bboxesOut = clip_bbox(bboxesOut, clipMin, clipWidth, clipHeight)
    if bboxesOut.shape[0] > 4:
        bboxesOut[4:,...] = bboxes[4:,...]
    if addedAxis:
        bboxesOut = bboxesOut[:,0]
    if round:
        bboxesOut = np.round(bboxesOut).astype(int)
    return bboxesOut

# [xMid, yMid, width, height] to [x1 y1, x2, y2]
def xywh_to_xyxy(bboxes, clipMin=-LIMIT, clipWidth=LIMIT, clipHeight=LIMIT,
        round=False):
    addedAxis = False
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes).astype(np.float32)
    if len(bboxes.shape) == 1:
        addedAxis = True
        bboxes = bboxes[:,np.newaxis]
    bboxesOut = np.zeros(bboxes.shape)
    xMid = bboxes[0,...]
    yMid = bboxes[1,...]
    width = bboxes[2,...]
    height = bboxes[3,...]
    bboxesOut[0,...] = xMid - width / 2.0
    bboxesOut[1,...] = yMid - height / 2.0
    bboxesOut[2,...] = xMid + width / 2.0
    bboxesOut[3,...] = yMid + height / 2.0
    if clipMin != -LIMIT or clipWidth != LIMIT or clipHeight != LIMIT:
        bboxesOut = clip_bbox(bboxesOut, clipMin, clipWidth, clipHeight)
    if bboxesOut.shape[0] > 4:
        bboxesOut[4:,...] = bboxes[4:,...]
    if addedAxis:
        bboxesOut = bboxesOut[:,0]
    if round:
        bboxesOut = np.round(bboxesOut).astype(int)
    return bboxesOut

# @bboxes {np.array} 4xn array of boxes to be scaled
# @scalars{number or arraylike} scalars for width and height of boxes
# @in_place{bool} If false, creates new bboxes.
def scale_bbox(bboxes, scalars,
        clipMin=-LIMIT, clipWidth=LIMIT, clipHeight=LIMIT,
        round=False, in_place=False):
    addedAxis = False
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes, dtype=np.float32)
    if len(bboxes.shape) == 1:
        addedAxis = True
        bboxes = bboxes[:,np.newaxis]
    if isinstance(scalars, numbers.Number):
        scalars = np.full((2, bboxes.shape[1]), scalars, dtype=np.float32)
    if not isinstance(scalars, np.ndarray):
        scalars = np.array(scalars, dtype=np.float32)
    if len(scalars.shape) == 1:
        scalars = np.tile(scalars[:,np.newaxis], (1,bboxes.shape[1])).astype(np.float32)

    bboxes = bboxes.astype(np.float32)

    width = bboxes[2,...] - bboxes[0,...]
    height = bboxes[3,...] - bboxes[1,...]
    xMid = (bboxes[0,...] + bboxes[2,...]) / 2.0
    yMid = (bboxes[1,...] + bboxes[3,...]) / 2.0
    if not in_place:
        bboxesOut = bboxes.copy()
    else:
        bboxesOut = bboxes

    bboxesOut[0,...] = xMid - width * scalars[0,...] / 2.0
    bboxesOut[1,...] = yMid - height * scalars[1,...] / 2.0
    bboxesOut[2,...] = xMid + width * scalars[0,...] / 2.0
    bboxesOut[3,...] = yMid + height * scalars[1,...] / 2.0

    if clipMin != -LIMIT or clipWidth != LIMIT or clipHeight != LIMIT:
        bboxesOut = clip_bbox(bboxesOut, clipMin, clipWidth, clipHeight)
    if addedAxis:
        bboxesOut = bboxesOut[:,0]
    if round:
        bboxesOut = np.round(bboxesOut).astype(np.int32)
    return bboxesOut


def make_square(bboxes, in_place=False):
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes).astype(np.float32)
    if len(bboxes.shape) == 1:
        numBoxes = 1
        width = bboxes[2] - bboxes[0]
        height = bboxes[3] - bboxes[1]
    else:
        numBoxes = bboxes.shape[1]
        width = bboxes[2,...] - bboxes[0,...]
        height = bboxes[3,...] - bboxes[1,...]
    maxSize = np.maximum(width, height)
    scalars = np.zeros((2, numBoxes))
    scalars[0,...] = maxSize * 1.0 / width
    scalars[1,...] = maxSize * 1.0 / height
    return scale_bbox(bboxes, scalars, in_place=in_place)


# Converts from the full image coordinate system to range 0:crop_padding. Useful for getting the coordinates
#   of a bounding box from image coordinates to the location within the cropped image.
# @bbox_to_change xyxy bbox whose coordinates will be converted to the new reference frame
# @crop_location xyxy box of the new origin and max points (without padding)
# @crop_padding the amount to pad the crop_location box (1 would be keep it the same, 2 would be doubled)
# @crop_size the maximum size of the coordinate frame of bbox_to_change.
def to_crop_coordinate_system(bbox_to_change, crop_location, crop_padding, crop_size):
    if isinstance(bbox_to_change, list):
        bbox_to_change = np.array(bbox_to_change)
    if isinstance(crop_location, list):
        crop_location = np.array(crop_location)
    bbox_to_change = bbox_to_change.astype(np.float32)
    crop_location = crop_location.astype(np.float32)

    crop_location = scale_bbox(crop_location, crop_padding)
    crop_location_xywh = xyxy_to_xywh(crop_location)
    bbox_to_change -= crop_location[[0,1,0,1]]
    bbox_to_change *= crop_size / crop_location_xywh[[2,3,2,3]]
    return bbox_to_change


# Inverts the transformation from to_crop_coordinate_system
# @crop_size the maximum size of the coordinate frame of bbox_to_change.
def from_crop_coordinate_system(bbox_to_change, crop_location, crop_padding, crop_size):
    if isinstance(bbox_to_change, list):
        bbox_to_change = np.array(bbox_to_change)
    if isinstance(crop_location, list):
        crop_location = np.array(crop_location)
    bbox_to_change = bbox_to_change.astype(np.float32)
    crop_location = crop_location.astype(np.float32)

    crop_location = scale_bbox(crop_location, crop_padding)
    crop_location_xywh = xyxy_to_xywh(crop_location)
    bbox_to_change *= crop_location_xywh[[2,3,2,3]] / crop_size
    bbox_to_change += crop_location[[0,1,0,1]]
    return bbox_to_change


