import cv2
import numpy as np

BORDER = 0
CV_FONT = cv2.FONT_HERSHEY_DUPLEX

# plots: array of numpy array images to plot. Can be of different sizes and dimensions as long as they are 2 or 3 dimensional.
# rows: int number of rows in subplot. If there are fewer images than rows, it will add empty space for the blanks.
#       if there are fewer rows than images, it will not draw the remaining images.
# cols: int number of columns in subplot. Similar to rows.
# outputWidth: int width in pixels of a single subplot output image.
# outputHeight: int height in pixels of a single subplot output image.
# border: int amount of border padding pixels between each image.
# titles: titles for each subplot to be rendered on top of images.
# fancy_text: if true, uses a fancier font than CV_FONT, but takes longer to render.
def subplot(plots, rows, cols, outputWidth, outputHeight, border=BORDER,
        titles=None, fancy_text=False):
    returnedImage = np.full((
        (outputHeight + 2 * border) * rows,
        (outputWidth + 2 * border) * cols,
        3), 191, dtype=np.uint8)
    if fancy_text:
        from PIL import Image, ImageDraw, ImageFont
        FANCY_FONT = ImageFont.truetype(
            '/usr/share/fonts/truetype/roboto/hinted/Roboto-Bold.ttf', 20)
    for row in range(rows):
        for col in range(cols):
            if col + cols * row >= len(plots):
                return returnedImage
            im = plots[col + cols * row]
            if im is None:
                continue
            if im.dtype != np.uint8 or len(im.shape) < 3:
                im = im.astype(np.float32)
                im -= np.min(im)
                im *= 255 / max(np.max(im), 0.0001)
                im = 255 - im.astype(np.uint8)
            if len(im.shape) < 3:
                im = cv2.applyColorMap(
                        im, cv2.COLORMAP_JET)
            if im.shape != (outputHeight, outputWidth, 3):
                imWidth = im.shape[1] * outputHeight / im.shape[0]
                if imWidth > outputWidth:
                    imWidth = outputWidth
                    imHeight = im.shape[0] * outputWidth / im.shape[1]
                else:
                    imWidth = im.shape[1] * outputHeight / im.shape[0]
                    imHeight = outputHeight
                imWidth = int(imWidth)
                imHeight = int(imHeight)
                im = cv2.resize(
                        im, (imWidth, imHeight),
                        interpolation=cv2.INTER_NEAREST)
                if imWidth != outputWidth:
                    pad0 = int(np.floor((outputWidth - imWidth) * 1.0 / 2))
                    pad1 = int(np.ceil((outputWidth - imWidth) * 1.0 / 2))
                    im = np.lib.pad(
                            im, ((0, 0), (pad0, pad1), (0, 0)),
                            'constant', constant_values=0)
                elif imHeight != outputHeight:
                    pad0 = int(np.floor((outputHeight - imHeight) * 1.0 / 2))
                    pad1 = int(np.ceil((outputHeight - imHeight) * 1.0 / 2))
                    im = np.lib.pad(
                            im, ((pad0, pad1), (0, 0), (0, 0)),
                            'constant', constant_values=0)
            if (titles is not None and len(titles) > 1 and
                    len(titles) > col + cols * row and
                    len(titles[col + cols * row]) > 0):
                if fancy_text:
                    if im.dtype != np.uint8:
                        im = im.astype(np.uint8)
                    im = Image.fromarray(im)
                    draw = ImageDraw.Draw(im)
                    for x in range(9,12):
                        for y in range(9, 12):
                            draw.text((x, y), titles[col + cols * row], (0,0,0),
                                    font=FANCY_FONT)
                    draw.text((10, 10), titles[col + cols * row], (255,255,255),
                            font=FANCY_FONT)
                    im = np.array(im)
                else:
                    cv2.putText(im, titles[col + cols * row], (10, 30), CV_FONT, .5, [0,0,0], 4)
                    cv2.putText(im, titles[col + cols * row], (10, 30), CV_FONT, .5, [255,255,255], 1)
            returnedImage[
                border + (outputHeight + border) * row :
                        (outputHeight + border) * (row + 1),
                border + (outputWidth + border) * col :
                        (outputWidth + border) * (col + 1),:] = im
    im = returnedImage
    # for one long title
    if titles is not None and len(titles) == 1:
        if fancy_text:
            if im.dtype != np.uint8:
                im = im.astype(np.uint8)
            im = Image.fromarray(im)
            draw = ImageDraw.Draw(im)
            for x in range(9,12):
                for y in range(9, 12):
                    draw.text((x, y), titles[0], (0,0,0),
                            font=FANCY_FONT)
            draw.text((10, 10), titles[0], (255,255,255),
                    font=FANCY_FONT)
            im = np.array(im)
        else:
            cv2.putText(im, titles[0], (10, 30), CV_FONT, .5, [0,0,0], 4)
            cv2.putText(im, titles[0], (10, 30), CV_FONT, .5, [255,255,255], 1)

    return im

# BBoxes are [x1 y1 x2 y2]
def drawRect(image, bbox, padding, color):
    from my_utils.util import bb_util
    imageHeight = image.shape[0]
    imageWidth = image.shape[1]
    bbox = np.round(np.array(bbox)) # mostly just for copying
    bbox = bb_util.clip_bbox(bbox, padding, imageWidth - padding, imageHeight - padding).astype(int).squeeze()
    padding = int(padding)
    image[bbox[1]-padding:bbox[3]+padding+1,
            bbox[0]-padding:bbox[0]+padding+1] = color
    image[bbox[1]-padding:bbox[3]+padding+1,
            bbox[2]-padding:bbox[2]+padding+1] = color
    image[bbox[1]-padding:bbox[1]+padding+1,
            bbox[0]-padding:bbox[2]+padding+1] = color
    image[bbox[3]-padding:bbox[3]+padding+1,
            bbox[0]-padding:bbox[2]+padding+1] = color
    return image

def drawPoint(image, point, size, padding, color):
    if not isinstance(point, np.ndarray):
        point = np.array(point)
    point = tuple(point.astype(int).tolist())
    cv2.circle(image, point, int(size), color, int(padding))
    '''
    bbox = xywh_to_xyxy([point[0], point[1], size, size])
    drawRect(image, bbox, padding, color)
    '''
    return image

def images_to_sprite(data, padsize=1, padval=0):
    # Expects NxHxWx3.
    data = data.astype(np.float64)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize),
            (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
            constant_values=(padval, padval))
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
            + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data
