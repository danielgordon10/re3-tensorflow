import re3_tracker
import vot

print('before first frame')
handle = vot.VOT("rectangle")
print('got handle')
initRegion = handle.region()
initBox = [initRegion.x, initRegion.y, initRegion.x + initRegion.width, initRegion.y + initRegion.height]

tracker = re3_tracker.Re3Tracker()
imageFile = handle.frame()

if not imageFile:
    sys.exit(0)

tracker.track('vot_object', imageFile, initBox)
print('initialized')

frameNum = 0
while True:
    imageFile = handle.frame()
    frameNum += 1
    if not imageFile:
        break
    bbox = tracker.track('vot_object', imageFile)
    bbox[[2,3]] = bbox[[2,3]] - bbox[[0,1]]
    region = vot.Rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
    handle.report(region)

print('finished', frameNum, 'frames')
