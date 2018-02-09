#from .voc0712 import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .data_custom import FISHdetection, detection_collate
from .config import *
import cv2
import numpy as np


def base_transform(image, size, mean):
    if len(image.shape) == 3:
        x = cv2.resize(image, (size, size)).astype(np.float32)
        # x = cv2.resize(np.array(image), (size, size)).astype(np.float32)
        x -= mean
        x = x.astype(np.float32)
        return x
    elif len(image.shape) == 4:
        x = np.zeros((image.shape[0], size, size, image.shape[3])).astype(np.float32)
        for idx in range(image.shape[0]):
            img_phase = cv2.resize(image[idx], (size, size)).astype(np.float32)
            img_phase -= mean
            img_phase = img_phase.astype(np.float32)
            x[idx] = img_phase
        return x


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels
