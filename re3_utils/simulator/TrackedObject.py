import math
import numpy as np
import random

import sys
import os
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.path.pardir,
    os.path.pardir
    )))

from re3_utils.util import IOU

VELOCITY_SCALAR = .5
SIZE_VELOCITY_SCALAR = 40
SPEED = 1
ANGLE_SCALAR = math.pi / 32
DISTRACTOR_VELOCITY_SCALAR = 1
SIZE_SCALE = 3
NEXT_ID = 0
DISTRACTOR_SCALE = .3 # scale factor for ratio from object to distractor size.

class TrackedObject(object):

    def __init__(self, image_width, image_height,
            texture=None, object_box=None, distractor_max_size=0):
        global NEXT_ID
        self.id = NEXT_ID
        NEXT_ID += 1
        self.image_width = image_width
        self.image_height = image_height
        self.image_size = min(image_width, image_height)
        self.distractor_max_size = min(distractor_max_size, image_width * DISTRACTOR_SCALE)
        self.min_size = image_width  * .01 * DISTRACTOR_SCALE
        self.occluder_boxes = []
        if self.distractor_max_size > 0:
            self.maxSize = self.distractor_max_size * DISTRACTOR_SCALE
        else:
            self.maxSize = np.array([self.image_width, self.image_height]) / (4 * SIZE_SCALE)

        if texture is not None:
            if len(texture.shape) < 3:
                texture = np.tile(texture[:,:,np.newaxis], (1,1,3))
            if texture.shape[2] > 3:
                texture = texture[:,:,:3]
            if texture.shape[0] == 0 or texture.shape[1] == 0:
                texture = np.zeros((1,1,3))
            self.texture = texture
            # The subset of the texture that is the actual object.
            self.object_box = object_box
            if object_box is None:
                self.object_box = np.array(
                        [0,0,texture.shape[1],texture.shape[0]])
            self.object_box = self.object_box.astype(float)
            self.aspect_ratio = ((self.object_box[2] - self.object_box[0]) *
                    1.0 / (self.object_box[3] - self.object_box[1]))

        else:
            self.texture = None
            self.color = int(random.random() * 255)
            self.aspect_ratio = 1.0

        self.size = np.zeros(2)
        if self.distractor_max_size > 0:
            size_scalar = ((random.random() * .3 + .7) *
                    self.distractor_max_size * DISTRACTOR_SCALE *
                    SIZE_SCALE)
            self.size[0] =  np.clip(size_scalar,
                    self.min_size * SIZE_SCALE * DISTRACTOR_SCALE,
                    self.distractor_max_size * SIZE_SCALE * DISTRACTOR_SCALE)
        else:
            size_scalar = (random.random() * .5) * self.image_size
            self.size[0] =  np.clip(size_scalar,
                    self.image_size * .1 * SIZE_SCALE,
                    self.image_size * .5 * SIZE_SCALE)

        self.history = []
        self.bbox_init()

    def bbox_init(self):
        for xx in range(100):
            # Try 100 times to get something that doesn't immediately occlude
            # other things.
            self.position = np.zeros(2)
            self.size[1] = self.size[0] / self.aspect_ratio
            self.velocity = min(np.abs(np.random.laplace(0, 1.0/15)), 1.5)
            self.w_velocity = 0
            self.h_velocity = 0
            self.position[0] = random.random() * self.image_width
            self.position[1] = random.random() * self.image_height
            self.angle = random.random() * math.pi * 2
            self.step()
            self.step()
            if not self.is_occluded():
                break


    def step(self):
        self.velocity += (np.random.laplace(0, .01))
        self.velocity = np.clip(self.velocity, 0, 1.5)
        self.angle += (random.gauss(0, 1)) * ANGLE_SCALAR
        self.w_velocity += (random.gauss(0, .01))
        self.h_velocity += (random.gauss(0, .01))
        self.w_velocity = np.clip(self.w_velocity, -.1, .1)
        self.h_velocity = np.clip(self.h_velocity, -.1, .1)
        self.size[0] += (self.w_velocity) * SIZE_VELOCITY_SCALAR * SPEED
        self.size[1] += (self.h_velocity) * SIZE_VELOCITY_SCALAR * SPEED
        if self.size[0] / self.size[1] < self.aspect_ratio / 2:
            self.size[1] = self.size[0] / (self.aspect_ratio / 2)
            self.w_velocity = 0
            self.h_velocity = 0
        elif self.size[0] / self.size[1] > self.aspect_ratio * 2:
            self.size[0] = self.size[1] * self.aspect_ratio * 2
            #self.size[1] = self.size[0] / (self.aspect_ratio * 2)
            self.w_velocity = 0
            self.h_velocity = 0
        newSize = np.clip(
                self.size, self.min_size * SIZE_SCALE, self.maxSize * SIZE_SCALE)
        if (newSize != self.size).any():
            self.size = newSize
            self.w_velocity = 0
            self.h_velocity = 2
        velocityScalar = VELOCITY_SCALAR * SPEED
        if self.distractor_max_size > 0:
            velocityScalar *= DISTRACTOR_VELOCITY_SCALAR
        self.position[0] += math.cos(self.angle) * self.velocity * self.size[0] * velocityScalar
        self.position[1] += math.sin(self.angle) * self.velocity * self.size[1] * velocityScalar
        if not self.in_bounds():
            box = self.get_bbox()
            xV = math.cos(self.angle)
            yV = math.sin(self.angle)
            if box[0] < 0 or box[2] > self.image_width:
                # X reflection bounce.
                xV *= -1
            else:
                # Y reflection bounce.
                yV *= -1
            self.angle = math.atan2(yV, xV)
            self.velocity *= .5
            self.position[0] = np.clip(self.position[0], self.size[0] / 2, self.image_width - self.size[0] / 2)
            self.position[1] = np.clip(self.position[1], self.size[1] / 2, self.image_height - self.size[1] / 2)
        self.history.append((self.position[0], self.position[1], self.size[0], self.size[1]))

    def step_back(self, back_ind):
        if len(self.history) > back_ind:
            (pos0, pos1, size0, size1) = self.history[-1 - back_ind]
            self.position[0] = pos0
            self.position[1] = pos1
            self.size[0] = size0
            self.size[1] = size1

    def reset_step(self):
        if len(self.history) > 0:
            self.step_back(0)

    # [x1, y1, x2, y2]
    def get_bbox(self):
        bbox = np.zeros((4))
        bbox[0] = self.position[0] - self.size[0] / 2
        bbox[1] = self.position[1] - self.size[1] / 2
        bbox[2] = self.position[0] + self.size[0] / 2
        bbox[3] = self.position[1] + self.size[1] / 2
        return bbox

    def get_bounded_bbox(self):
        bbox = np.round(self.get_bbox()).astype(int)
        bbox[[0,2]] = np.clip(bbox[[0,2]], 0, self.image_width)
        bbox[[1,3]] = np.clip(bbox[[1,3]], 1, self.image_height)
        return bbox

    def get_bounded_bbox_scaled(self):
        bbox = self.get_bounded_bbox().astype(float)
        bbox[[0,2]] /= self.image_width
        bbox[[0,2]] /= self.image_height

    def get_object_box(self):
        bbox = self.get_bbox()
        if self.texture is not None:
            bbox[[0,2]] = bbox[0] + self.size[0] * (self.object_box[[0,2]] / self.texture.shape[1])
            bbox[[1,3]] = bbox[1] + self.size[1] * (self.object_box[[1,3]] / self.texture.shape[0])
        return bbox

    def in_bounds(self):
        box = self.get_bbox()
        return not (
            box[0] < 0 or
            box[1] < 0 or
            box[2] > self.image_width or
            box[3] > self.image_height)

    def is_occluded(self):
        occluder_boxes = np.array([obj.get_bbox() for obj in self.occluder_boxes])
        if len(occluder_boxes) == 0:
            return False
        return IOU.count_overlapping_boxes(occluder_boxes, self.get_bounded_bbox()) > 0

    def __str__(self):
        return '(%.3f, %.3f), V = %.3f, A = %.3f S = [%.3f, %.3f], C = %d ' % (
                self.position[0], self.position[1], self.velocity, self.angle,
                self.size[0], self.size[1], self.color)

