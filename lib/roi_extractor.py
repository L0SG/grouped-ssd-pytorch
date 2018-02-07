
# used to extract roi from image file
# input: 512x512x3 bmp image with box-roi (yellow or red)
# output: text file containing [x_start, x_delta, y_start, y_delta]

# strategy: sum over channels to 512x512 -> yellow or red has specific value
# make them 1, and 0 for other areas -> extract roi from binary image

"""obsolete code, use rgb version"""

import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
    if os.path.basename(os.path.normpath(subject)) != 'A25':
        continue
    basename_subject = os.path.basename(os.path.normpath(subject))
    path_subject = os.path.join(roi_coordinate_path, basename_subject)
    if not os.path.exists(path_subject):
        os.mkdir(path_subject)
    # traverse over phases
    for phase in glob.glob(os.path.join(subject, '*')):
        basename_phase = os.path.basename(os.path.normpath(phase))
        path_phase = os.path.join(path_subject, basename_phase)
        if not os.path.exists(path_phase):
            os.mkdir(path_phase)
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
            index_yellow = np.where(roi_image_matrix == rgb_value_yellow)
            index_red = np.where(roi_image_matrix == rgb_value_red)

            # roi line should have continuously increasing values in x-axis [1, :]
            # check for 4 successive elements
            continuity_checker = 8
            # -1 if not found: SHOULD NOT HAPPEN since all slices have roi box
            x_start, y_start = -1, -1
            # check for yellow
            for idx_x in xrange(0, len(index_yellow[1])-continuity_checker):
                x_start_candidate = index_yellow[1][range(idx_x, idx_x + continuity_checker)]
                x_start_subtractor = np.append(x_start_candidate[1:], x_start_candidate[-1])
                x_start_identifier = x_start_candidate - x_start_subtractor
                # all elems except the last index in the identifier should have -1
                comparator = np.full(len(x_start_candidate)-1, -1)
                if np.array_equal(x_start_identifier[:-1], comparator):
                    y_start = index_yellow[0][idx_x]
                    x_start = index_yellow[1][idx_x]
                    break

            # check for red
            for idx_x in xrange(0, len(index_red[1])-continuity_checker):
                x_start_candidate = index_red[1][range(idx_x, idx_x + continuity_checker)]
                x_start_subtractor = np.append(x_start_candidate[1:], x_start_candidate[-1])
                x_start_identifier = x_start_candidate - x_start_subtractor
                # all elems except the last index in the identifier should have -1
                comparator = np.full(len(x_start_candidate)-1, -1)
                if np.array_equal(x_start_identifier[:-1], comparator):
                    y_start = index_red[0][idx_x]
                    x_start = index_red[1][idx_x]
                    break

            # check if bad things happened
            assert x_start != -1 and y_start != -1

            # find x_end and y_end : reverse the index lists and do the same with comparator value 1
            x_end, y_end = -1, -1
            index_yellow_reverse = np.flip(index_yellow, axis=1)
            index_red_reverse = np.flip(index_red, axis=1)

            # check for yellow
            for idx_x in xrange(0, len(index_yellow_reverse[1]) - continuity_checker):
                x_end_candidate = index_yellow_reverse[1][range(idx_x, idx_x + continuity_checker)]
                x_end_subtractor = np.append(x_end_candidate[1:], x_end_candidate[-1])
                x_end_identifier = x_end_candidate - x_end_subtractor
                # all elems except the last index in the identifier should have +1
                comparator = np.full(len(x_end_candidate) - 1, 1)
                if np.array_equal(x_end_identifier[:-1], comparator):
                    y_end = index_yellow_reverse[0][idx_x]
                    x_end = index_yellow_reverse[1][idx_x]
                    break

            # check for red
            for idx_x in xrange(0, len(index_red_reverse[1]) - continuity_checker):
                x_end_candidate = index_red_reverse[1][range(idx_x, idx_x + continuity_checker)]
                x_end_subtractor = np.append(x_end_candidate[1:], x_end_candidate[-1])
                x_end_identifier = x_end_candidate - x_end_subtractor
                # all elems except the last index in the identifier should have +1
                comparator = np.full(len(x_end_candidate) - 1, 1)
                if np.array_equal(x_end_identifier[:-1], comparator):
                    y_end = index_red_reverse[0][idx_x]
                    x_end = index_red_reverse[1][idx_x]
                    break

            # check if bad things happened
            assert x_end != -1 and y_end != -1

            x_delta = x_end - x_start
            y_delta = y_end - y_start
            coordinate = [x_start, x_delta, y_start, y_delta]

            # write coordinate to text file
            suffix = slice[-8:-4]
            path_slice = os.path.join(path_phase, str(basename_phase)+'_'+str(suffix)+'.txt')
            output_coordinate = open(path_slice, 'w+')
            output_coordinate.write(str(coordinate))
            output_coordinate.close()
            """
            # debug: draw extracted bounding box and save to image
            output_image = roi_image.copy()
            fig, ax = plt.subplots(1)
            ax.imshow(output_image)
            rect = patches.Rectangle((x_start, y_start), x_delta, y_delta, linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            plt.savefig(os.path.join(path_phase, str(basename_phase)+'_'+str(suffix)+'.png'))
            """
    print('subject ' + str(os.path.basename(os.path.normpath(subject))) +
          ' passed: [x_start, y_start, x_end, y_end] of last slice = ' + str([x_start, y_start, x_end, y_end]))
