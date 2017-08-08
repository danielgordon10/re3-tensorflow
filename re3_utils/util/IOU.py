import numpy as np

'''
@rects1 numpy dx4 matrix of bounding boxes
@rect2 single numpy 1x4 matrix of bounding box
@return dx1 IOUs
Rectangles are [x1, y1, x2, y2]
'''
def IOU_numpy(rects1, rect2):
    #intersection = np.fmin(np.zeros((rects1.shape[0],1))
    (d, n) = rects1.shape
    x1s = np.fmax(rects1[:,0], rect2[0])
    x2s = np.fmin(rects1[:,2], rect2[2])
    y1s = np.fmax(rects1[:,1], rect2[1])
    y2s = np.fmin(rects1[:,3], rect2[3])
    ws = np.fmax(x2s - x1s, 0)
    hs = np.fmax(y2s - y1s, 0)
    intersection = ws * hs
    rects1Area = (rects1[:,2] - rects1[:,0]) * (rects1[:,3] - rects1[:,1])
    rect2Area = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
    union = np.fmax(rects1Area + rect2Area - intersection, .00001)
    return intersection * 1.0 / union

def IOU_lists(rects1, rects2):
    (d, n) = rects1.shape
    x1s = np.fmax(rects1[:,0], rects2[:,0])
    x2s = np.fmin(rects1[:,2], rects2[:,2])
    y1s = np.fmax(rects1[:,1], rects2[:,1])
    y2s = np.fmin(rects1[:,3], rects2[:,3])
    ws = np.fmax(x2s - x1s, 0)
    hs = np.fmax(y2s - y1s, 0)
    intersection = ws * hs
    rects1Area = (rects1[:,2] - rects1[:,0]) * (rects1[:,3] - rects1[:,1])
    rects2Area = (rects2[:,2] - rects2[:,0]) * (rects2[:,3] - rects2[:,1])
    union = np.fmax(rects1Area + rects2Area - intersection, .00001)
    return intersection * 1.0 / union

# Rectangles are [x1, y1, x2, y2]
def IOU(rect1, rect2):
    if not isinstance(rect1, np.ndarray):
        rect1 = np.array(rect1)
    if not isinstance(rect2, np.ndarray):
        rect2 = np.array(rect2)
    rect1 = [min(rect1[[0,2]]), min(rect1[[1,3]]),
            max(rect1[[0,2]]), max(rect1[[1,3]])]
    rect2 = [min(rect2[[0,2]]), min(rect2[[1,3]]),
            max(rect2[[0,2]]), max(rect2[[1,3]])]
    intersection = (max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0])) *
        max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1])))

    union = ((rect1[2] - rect1[0]) * (rect1[3] - rect1[1]) +
        (rect2[2] - rect2[0]) * (rect2[3] - rect2[1]) -
        intersection)

    return intersection * 1.0 / max(union, .00001)

def intersection(rect1, rect2):
    return (max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0])) *
        max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1])))

'''
@rects1 numpy dx5 matrix of bounding boxes
@rect2 single numpy 1x4 matrix of bounding box
@return nx5 rects where n is number of rects over overlapThresh
Rectangles are [x1, y1, x2, y2, 0]
'''
def get_overlapping_boxes(rects1, rect2, overlapThresh=.001):
    x1s = np.fmax(rects1[:,0], rect2[0])
    x2s = np.fmin(rects1[:,2], rect2[2])
    y1s = np.fmax(rects1[:,1], rect2[1])
    y2s = np.fmin(rects1[:,3], rect2[3])
    ws = np.fmax(x2s - x1s, 0)
    hs = np.fmax(y2s - y1s, 0)
    intersection = ws * hs
    rects1Area = (rects1[:,2] - rects1[:,0]) * (rects1[:,3] - rects1[:,1])
    rect2Area = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
    union = np.fmax(rects1Area + rect2Area - intersection, .00001)
    ious = intersection * 1.0 / union
    rects1[:,4] = ious
    rects1 = rects1[ious > overlapThresh, :]
    return rects1

'''
@rects1 numpy dx4 matrix of bounding boxes
@rect2 single numpy 1x4 matrix of bounding box
@return number of rects over overlapThresh
Rectangles are [x1, y1, x2, y2]
'''
def count_overlapping_boxes(rects1, rect2, overlapThresh=.001):
    if rects1.shape[1] == 0:
        return 0
    x1s = np.fmax(rects1[:,0], rect2[0])
    x2s = np.fmin(rects1[:,2], rect2[2])
    y1s = np.fmax(rects1[:,1], rect2[1])
    y2s = np.fmin(rects1[:,3], rect2[3])
    ws = np.fmax(x2s - x1s, 0)
    hs = np.fmax(y2s - y1s, 0)
    intersection = ws * hs
    rects1Area = (rects1[:,2] - rects1[:,0]) * (rects1[:,3] - rects1[:,1])
    rect2Area = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
    union = np.fmax(rects1Area + rect2Area - intersection, .00001)
    ious = intersection * 1.0 / union
    return np.sum(ious > overlapThresh)
