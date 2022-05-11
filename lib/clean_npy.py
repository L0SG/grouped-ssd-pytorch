import skimage
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from imageio import imwrite

# following previous labeling method
lesion_class_label = 0

def convert(warped_mask):
    bboxes = []

    if warped_mask.max() == 1:
        mask_contour, _ = cv2.findContours(warped_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        label_slice = label(warped_mask)
        props = regionprops(label_slice)

        for prop in props:
            # additionally construct bbox coordinates format matching miccai2018 notations
            x_start, y_start, x_end, y_end = prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]
            # use [x_min, y_min, x_max, y_max] type
            coordinate = [x_start, y_start, x_end, y_end, lesion_class_label]
            # append zero class label for lesion
            coordinate.append(lesion_class_label)
            bboxes.append(coordinate)
        # print(len(props))
    else:
        bboxes.append(None)

    return bboxes


if __name__ == '__main__':
    metadata_path = os.path.join('/home/eunji/hdd/eunji/Data/liver_year1_dataset_extended_1904_preprocessed/ml_ready_phaselabel', "metadata.txt")
    src_path = '/home/eunji/hdd/eunji/Data/liver_year1_dataset_extended_1904_preprocessed/ml_ready_phaselabel_align'
    out_path = '/home/eunji/hdd/eunji/Data/liver_year1_dataset_extended_1904_preprocessed/ml_ready_phaselabel_align_clean'
    data = []
    with open(metadata_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split("|")
            data.append(line[0])
    os.makedirs(out_path, exist_ok=True)

    for i in range(len(data)):
        os.makedirs(os.path.join(out_path, data[i].split('/')[0]), exist_ok=True)
        datapoint_name_relative = data[i]
        datapoint_name_ct = datapoint_name_relative + '_ct.npy'
        datapoint_name_mask = datapoint_name_relative + '_mask.npy'
        datapoint_name_bbox = datapoint_name_relative + '_bbox.npy'

        if os.path.isfile(os.path.join(src_path, datapoint_name_ct)):
            ct_data = np.load(os.path.join(src_path, datapoint_name_ct))
            mask_data = np.load(os.path.join(src_path, datapoint_name_mask))
            orig_bbox = np.load(
                os.path.join('/home/eunji/hdd/eunji/Data/liver_year1_dataset_extended_1904_preprocessed/ml_ready_phaselabel',
                             datapoint_name_bbox))
        else:
            continue

        ct_data[ct_data < 0] = 0
        ct_data[ct_data > 1] = 1
        mask_data[mask_data > 0.5] = 1
        mask_data[mask_data < 1] = 0

        bbox_data = convert(mask_data)
        if (len(bbox_data) != orig_bbox.shape[0]):
            print("{}: {} -> {}".format(data[i], orig_bbox.shape[0], len(bbox_data)))
        np.save(os.path.join(out_path, datapoint_name_ct), ct_data)
        np.save(os.path.join(out_path, datapoint_name_mask), mask_data)
        np.save(os.path.join(out_path, datapoint_name_bbox), bbox_data)
