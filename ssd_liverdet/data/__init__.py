#from .voc0712 import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .data_custom import FISHdetection, detection_collate
from .data_custom_v2 import *
from .config import *
import cv2
import numpy as np
from PIL import Image


def base_transform(image, size, mean, use_normalize=False, p_only=False):
    if len(image.shape) == 3:
        x = cv2.resize(image, (size, size)).astype(np.float32)
        # x = cv2.resize(np.array(image), (size, size)).astype(np.float32)
        x -= mean
        x = x.astype(np.float32)
    elif len(image.shape) == 4:
        x = np.zeros((image.shape[0], size, size, image.shape[3])).astype(np.float32)
        for idx in range(image.shape[0]):
            img_phase = cv2.resize(image[idx], (size, size)).astype(np.float32)
            img_phase -= mean
            img_phase = img_phase.astype(np.float32)
            x[idx] = img_phase
    if p_only:
        x = np.repeat(np.expand_dims(x[2], 0), 4, axis=0)
    if use_normalize:
        x_min = x.min()
        x_max = x.max()
        assert x_min != x_max, "all-black image detected during Normalizing. check preprocessing"
        x = (x - x_min) / (x_max - x_min)
    return x


def base_transform_fast(image, size, mean, use_normalize=False, p_only=False):
    if len(image.shape) == 3:
        x = cv2.resize(image, (size, size)).astype(np.float32)
        # x = cv2.resize(np.array(image), (size, size)).astype(np.float32)
        x -= mean
        x = x.astype(np.float32)
    elif len(image.shape) == 4:
        x = np.zeros((image.shape[0], size, size, image.shape[3])).astype(np.float32)
        for idx in range(image.shape[0]):
            img_phase = Image.fromarray(image[idx])
            img_phase = img_phase.resize((size, size))
            img_phase = np.asarray(img_phase).astype(np.float32)
            img_phase -= mean
            x[idx] = img_phase
    if p_only:
        x = np.repeat(np.expand_dims(x[2], 0), 4, axis=0)
    if use_normalize:
        x_min = x.min()
        x_max = x.max()
        assert x_min != x_max, "all-black image detected during Normalizing. check preprocessing"
        x = (x - x_min) / (x_max - x_min)
    return x


class BaseTransform:
    def __init__(self, size, mean, use_normalize=False, p_only=False):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)
        self.use_normalize = use_normalize
        self.p_only = p_only

    def __call__(self, image, boxes=None, labels=None):
        # return base_transform(image, self.size, self.mean, self.use_normalize, self.p_only), boxes, labels
        return base_transform_fast(image, self.size, self.mean, self.use_normalize, self.p_only), boxes, labels
