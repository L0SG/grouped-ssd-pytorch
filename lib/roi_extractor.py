# used to extract roi from image file
# input: 512x512x3 bmp image with box-roi (yellow or red)
# output: text file containing [x_start, x_delta, y_start, y_delta]

# strategy: sum over channels to 512x512 -> yellow or red has specific value
# make them 1, and 0 for other areas -> extract roi from binary image

import numpy as np
import glob
import os
from PIL import Image

roi_image_path = '/media/hdd/tkdrlf9202/Datasets/liver_lesion/roi_image'
roi_coordinate_path = '/media/hdd/tkdrlf9202/Datasets/liver_lesion/roi_coordinate'

# yellow roi have value of 510, red have 765
# yellow have 2-pixel thick line, red have 1-pixel line
rgb_value_yellow = 1020
rgb_value_red = 765

# traverse over subjects
for subject in glob.glob(os.path.join(roi_image_path, '*')):
    # A86 have red roi
    if os.path.basename(os.path.normpath(subject)) != 'A86':
        continue

    # traverse over phases
    for phase in glob.glob(os.path.join(subject, '*')):
        # traverse over slices
        for slice in glob.glob(os.path.join(phase, '*')):
            # get roi image and convert to numpy
            roi_image = Image.open(slice, 'r')
            # use uint32 to multiply scalar over red channels: exceed 255
            roi_image_tensor = np.uint32(np.array(roi_image))
            # sum over channels with 3 * red channel for clear cut
            roi_image_tensor[:, :, 0] = np.multiply(3, roi_image_tensor[:, :, 0])
            temp = roi_image_tensor[:, :, 0]
            roi_image_matrix = np.sum(roi_image_tensor, axis=2)


            print ''
