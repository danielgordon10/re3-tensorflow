import cv2
import numpy as np
import glob
import xml.etree.ElementTree as ET
import os
import sys

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(
    basedir,
    os.path.pardir,
    os.path.pardir,
    os.path.pardir)))

from re3_utils.util import drawing
from re3_utils.util.im_util import get_image_size

DEBUG = False

def main(label_type):
    wildcard = '/*/*/' if label_type == 'train' else '/'
    dataset_path = 'data/ILSVRC2015/'
    annotationPath = dataset_path + 'Annotations/'
    imagePath = dataset_path + 'Data/'

    if not os.path.exists(os.path.join('labels', label_type)):
        os.makedirs(os.path.join('labels', label_type))
    imageNameFile = open('labels/' + label_type + '/image_names.txt', 'w')

    labels = []
    labels = glob.glob(annotationPath + 'DET/' + label_type + wildcard + '*.xml')
    labels.sort()
    images = [label.replace('Annotations', 'Data').replace('xml', 'JPEG') for label in labels]

    bboxes = []
    for ii,imageName in enumerate(images):
        if ii % 100 == 0:
            print('iter %d of %d = %.2f%%' % (ii, len(images), ii * 1.0 / len(images) * 100))
        if not DEBUG:
            imageNameFile.write(imageName + '\n')
        imOn = ii
        label = labels[imOn]
        labelTree = ET.parse(label)
        imgSize = get_image_size(images[imOn])
        area_cutoff = imgSize[0] * imgSize[1] * 0.01
        if DEBUG:
            print('\nimage name\n\n%s\n' % images[imOn])
            image = cv2.imread(images[imOn])
            print('image size', image.shape)
            print(label)
            print(labelTree)
            print(labelTree.findall('object'))
        for obj in labelTree.findall('object'):
            bbox = obj.find('bndbox')
            bbox = [int(bbox.find('xmin').text),
                    int(bbox.find('ymin').text),
                    int(bbox.find('xmax').text),
                    int(bbox.find('ymax').text),
                    imOn]
            if (bbox[3] - bbox[1]) * (bbox[2] - bbox[0]) < area_cutoff:
                continue
            if DEBUG:
                print('name', obj.find('name').text, '\n')
                print(bbox)
                image = image.squeeze()
                if len(image.shape) < 3:
                    image = np.tile(image[:,:,np.newaxis], (1,1,3))
                drawing.drawRect(image, bbox[:-1], 3, [0, 0, 255])
            bboxes.append(bbox)

        if DEBUG:
            if len(image.shape) == 2:
                image = np.tile(image[:,:,np.newaxis], (1,1,3))
            cv2.imshow('image', image)
            cv2.waitKey(0)

    bboxes = np.array(bboxes)
    if not DEBUG:
        np.save('labels/' + label_type + '/labels.npy', bboxes)

if __name__ == '__main__':
    main('train')
    main('val')

