from torch.utils.data import Dataset
from PIL import Image
import torch
import codecs
import random
import math
import copy
import time
import cv2
import os
import numpy as np
from torchvision import transforms


def label_to_mask_and_pixel_pos_weight(label, img_size, version="2s", neighbors=8):
    """
    8 neighbors:
        0 1 2
        7 - 3
        6 5 4
    """
    factor = 2 if version == "2s" else 4
    # ignore = label["ignore"]
    # label = label["coor"]
    # assert len(ignore) == len(label)
    label = np.array(label)
    label = label.reshape([-1, 1, 4, 2])
    pixel_mask_size = [int(i / factor) for i in img_size]
    link_mask_size = [neighbors, ] + pixel_mask_size

    pixel_mask = np.zeros(pixel_mask_size, dtype=np.uint8)
    pixel_weight = np.zeros(pixel_mask_size, dtype=np.float)
    link_mask = np.zeros(link_mask_size, dtype=np.uint8)
    # if label.shape[0] == 0:
    # return torch.LongTensor(pixel_mask), torch.Tensor(pixel_weight), torch.LongTensor(link_mask)
    label = (label / factor).astype(int)  # label's coordinate value should be divided

    # cv2.drawContours(pixel_mask, label, -1, 1, thickness=-1)
    real_box_num = 0
    # area_per_box = []
    for i in range(label.shape[0]):
        pixel_mask_tmp = np.zeros(pixel_mask_size, dtype=np.uint8)
        cv2.drawContours(pixel_mask_tmp, label[i], -1, 1, thickness=-1)
        pixel_mask += pixel_mask_tmp
    neg_pixel_mask = (pixel_mask == 0).astype(np.uint8)
    pixel_mask[pixel_mask != 1] = 0
    # assert not (pixel_mask>1).any()
    pixel_mask_area = np.count_nonzero(pixel_mask)  # total area

    for i in range(label.shape[0]):
        pixel_mask_tmp = np.zeros(pixel_mask_size, dtype=np.uint8)
        cv2.drawContours(pixel_mask_tmp, label[i], -1, 1, thickness=-1)
        pixel_mask_tmp *= pixel_mask
        if np.count_nonzero(pixel_mask_tmp) > 0:
                real_box_num += 1
    if real_box_num == 0:
        # print("box num = 0")
        return pixel_mask, neg_pixel_mask, pixel_weight, link_mask
    avg_weight_per_box = pixel_mask_area / real_box_num

    for i in range(label.shape[0]):  # num of box
        pixel_weight_tmp = np.zeros(pixel_mask_size, dtype=np.float)
        cv2.drawContours(pixel_weight_tmp, [label[i]], -1, avg_weight_per_box, thickness=-1)
        pixel_weight_tmp *= pixel_mask
        area = np.count_nonzero(pixel_weight_tmp)  # area per box
        if area <= 0:
            # print("area label: " + str(label[i]))
            # print("area:" + str(area))
            continue
        pixel_weight_tmp /= area
        # print(pixel_weight_tmp[pixel_weight_tmp>0])
        pixel_weight += pixel_weight_tmp

        # link mask
        weight_tmp_nonzero = pixel_weight_tmp.nonzero()
        # pixel_weight_nonzero = pixel_weight.nonzero()
        link_mask_tmp = np.zeros(pixel_mask_size, dtype=np.uint8)
        # for j in range(neighbors): # neighbors directions
        link_mask_tmp[weight_tmp_nonzero] = 1
        link_mask_shift = np.zeros(link_mask_size, dtype=np.uint8)
        w_index = weight_tmp_nonzero[1]
        h_index = weight_tmp_nonzero[0]
        w_index1 = np.clip(w_index + 1, a_min=None, a_max=link_mask_size[1] - 1)
        w_index_1 = np.clip(w_index - 1, a_min=0, a_max=None)
        h_index1 = np.clip(h_index + 1, a_min=None, a_max=link_mask_size[2] - 1)
        h_index_1 = np.clip(h_index - 1, a_min=0, a_max=None)
        link_mask_shift[0][h_index1, w_index1] = 1
        link_mask_shift[1][h_index1, w_index] = 1
        link_mask_shift[2][h_index1, w_index_1] = 1
        link_mask_shift[3][h_index, w_index_1] = 1
        link_mask_shift[4][h_index_1, w_index_1] = 1
        link_mask_shift[5][h_index_1, w_index] = 1
        link_mask_shift[6][h_index_1, w_index1] = 1
        link_mask_shift[7][h_index, w_index1] = 1

        for j in range(neighbors):
            # +0 to convert bool array to int array
            link_mask[j] += np.logical_and(link_mask_tmp, link_mask_shift[j]).astype(np.uint8)
    return [pixel_mask, neg_pixel_mask, pixel_weight, link_mask]