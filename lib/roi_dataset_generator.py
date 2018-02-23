import numpy as np
import glob
import os
from PIL import Image
import dicom
import pickle
import h5py
from scipy.misc import imsave

def preprocess_img_slc_for_detection(img_slc, subject):
    """
    modified version of dicom preprocess code from cascaded FCN
    Preprocesses the image 3d volumes by performing the following :
    0- subtract 1024 to have value -1024~1024 (since our custom dataset have 0~2048)
    1- Rotate the input volume so the the liver is on the left, spine is at the bottom of the image
    2- Set pixels with hounsfield value great than 1200, to zero.
    3- Clip all hounsfield values to the range [-100, 400]
    4- Normalize values to [0, 1]

    Args:
        img_slc: raw image slice
    Return:
        Preprocessed image slice
    """

    # some subjects have difference value range: A391 & A452
    # use exceptional processing for these one
    if subject == 'A391' or subject == 'A452':
        img_slc = img_slc.astype(np.float32)
        img_slc[img_slc > 1200] = 0
        img_slc = np.clip(img_slc, -100, 400)
        img_slc = normalize_image(img_slc)

        return img_slc

    if np.amax(img_slc) < 1700:
        ValueError('ERROR: value range is different for this subject')
        exit()

    img_slc = img_slc.astype(np.float32)
    img_slc = np.add(img_slc, -1024)
    img_slc[img_slc > 1200] = 0
    img_slc = np.clip(img_slc, -100, 400)
    img_slc = normalize_image(img_slc)

    return img_slc


def normalize_image(img):
    """ Normalize image values to [0,1] """
    min_, max_ = float(np.min(img)), float(np.max(img))
    return (img - min_) / (max_ - min_)


def generate_roi_dataset(ct_path, roi_coordinate_path):
    USE_P_ONLY = True

    ct_data_master = []
    coordinate_data_master = []

    # traverse over subjects, sorted
    for subject in sorted(glob.glob(os.path.join(roi_coordinate_path, '*'))):
        ct_data_subject = []
        coordinate_data_subject = []

        basename_subject = os.path.basename(os.path.normpath(subject))
        path_subject = os.path.join(ct_path, basename_subject)
        ct_data_oneslice_4phase = []
        coordinate_oneslice_4phase = []

        # for calculating value range of each subject
        mean_min_value = []
        mean_max_value = []

        # traverse over phases
        for phase in sorted(glob.glob(os.path.join(subject, '*'))):
            basename_phase = os.path.basename(os.path.normpath(phase))
            path_phase = os.path.join(path_subject, basename_phase)
            if USE_P_ONLY:
                # use P phase only
                if basename_phase != 'P':
                    continue
            # get name of all ct slices
            path_slices = sorted(os.listdir(path_phase))
            path_slices_basename = [name[:-4] for name in path_slices]
            # get name of coordinate files
            path_coordinates = sorted(os.listdir(phase))
            #path_coordinates = sorted(os.listdir(os.path.join(roi_coordinate_path,
            #                                                  basename_subject,
            #                                                  basename_phase)))
            path_coordinates_basename = [name[:-4] for name in path_coordinates]

            # obtain intersection (data that have both image & coordinate)
            intersection = sorted(set(path_slices_basename).intersection(path_coordinates_basename))
            path_slices_filtered = [name + str('.DCM') for name in list(intersection)]
            path_coordinates_filtered = [name + str('.txt') for name in list(intersection)]
            assert len(path_slices_filtered) == len(path_coordinates_filtered)

            # load matching ct file and coordinate, then append to temporary list
            ct_data_oneslice = []
            coordinate_oneslice = []
            for idx in range(len(path_slices_filtered)):
                ct_file = dicom.read_file(os.path.join(ct_path, basename_subject, basename_phase,
                                                        path_slices_filtered[idx]))

                ct_image = ct_file.pixel_array

                mean_min_value.append(np.amin(ct_image))
                mean_max_value.append(np.amax(ct_image))

                ct_image_preprocessed = preprocess_img_slc_for_detection(ct_image, basename_subject)
                coordinate_file = open(os.path.join(roi_coordinate_path, basename_subject, basename_phase,
                                                        path_coordinates_filtered[idx]), 'rb')
                coordinate = pickle.load(coordinate_file)
                ct_data_oneslice.append(ct_image_preprocessed)
                coordinate_oneslice.append(coordinate)
            assert len(ct_data_oneslice) == len(coordinate_oneslice)

            ct_data_oneslice_4phase.append(ct_data_oneslice)
            coordinate_oneslice_4phase.append(coordinate_oneslice)

        # if generating one-phase data, copy the data 4 times for the model
        if USE_P_ONLY:
            ct_data_oneslice_4phase = ct_data_oneslice_4phase * 4
            coordinate_oneslice_4phase = coordinate_oneslice_4phase * 4

        ct_data_oneslice_4phase = np.array(ct_data_oneslice_4phase)
        coordinate_oneslice_4phase = np.array(coordinate_oneslice_4phase)
        # make 3 continuous slides as a single data point to enable the CNN to recognize vertical info
        # consider 3 slides to additional channel (like RGB)
        for idx_slice in range(ct_data_oneslice_4phase.shape[1] - 2):
            ct_data_threeslices = np.array(ct_data_oneslice_4phase[:, idx_slice:idx_slice+3, :, :])
            coordinate_threeslices = np.array(coordinate_oneslice_4phase[:, idx_slice:idx_slice+3, :])
            #ct_data_threeslices = np.array(ct_data_oneslice[idx_slice:idx_slice+3])
            #coordinate_threeslices = np.array(coordinate_oneslice[idx_slice:idx_slice+3])
            # append to subject list
            ct_data_subject.append(ct_data_threeslices)
            coordinate_data_subject.append(coordinate_threeslices)

        """ for debug """
        # printout the first preprocessed image per subject
        if True:
            printout_subject = (ct_data_subject[0][0][1] * 255).astype(np.uint8)
            imsave(os.path.join('debug', str(basename_subject) + '.png'), printout_subject)

        mean_min_value = np.array(mean_min_value).mean()
        mean_max_value = np.array(mean_max_value).mean()

        """ to make debug data with one data per subject, use [0] at this line"""
        ct_data_subject = np.array(ct_data_subject)
        coordinate_data_subject = np.array(coordinate_data_subject)

        # append to master list
        ct_data_master.append(ct_data_subject)
        coordinate_data_master.append(coordinate_data_subject)

        print(subject + ' mean val: ' + str(mean_min_value) + ' max val: ' + str(mean_max_value))
        if mean_min_value < 0 or mean_max_value < 1700:
            print('WARNING: value range for this subject is out of range, double check the data')

    assert len(ct_data_master) == len(coordinate_data_master)
    return ct_data_master, coordinate_data_master

# generate roi dataset
# each ct image slice have 1 or more (n) bounding boxes [n, y_start, x_start, y_delta, x_delta]
# if using P phase only, input ct should be [n, 1, 512, 512] (1 is greyscale)

ct_path = '/media/ssd/tkdrlf9202/Datasets/liver_lesion_aligned/ct'
roi_coordinate_path = '/media/ssd/tkdrlf9202/Datasets/liver_lesion_aligned/roi_coordinate'
dataset_save_location = '/home/tkdrlf9202/Datasets/liver_lesion_aligned/lesion_dataset_ponly_aligned.h5'

CT_IMAGE_SIZE = (512, 512)


ct_data, coordinate_data = generate_roi_dataset(ct_path, roi_coordinate_path)

"""
# convert [y_start, x_start, y_delta, x_delta] to [center_x, center_y, height, width]
# the latter format is used for SSD application
for idx in range(len(coordinate_data)):
    x_start, y_start, x_delta, y_delta = coordinate_data[idx]
    center_x = int((x_start + (x_start + x_delta)) / 2)
    center_y = int((y_start + (y_start + y_delta)) / 2)
    height = x_delta
    width = y_delta
    coordinate_data[idx] = np.array([center_x, center_y, height, width])
"""
# handle coord info in roi_extractor_rgb, not here
"""
# the above is WRONG: it uses [x_min, y_min, x_max, y_max]
# convert [y_start, x_start, y_delta, x_delta] to [x_min, y_min, x_max, y_max]
for idx in range(len(coordinate_data)):
    for idx_slice in range(coordinate_data[idx].shape[0]):
        y_start, x_start, y_delta, x_delta = coordinate_data[idx][idx_slice]
        x_min, y_min = x_start, y_start
        x_max, y_max = x_start + x_delta, y_start + y_delta
        coordinate_data[idx][idx_slice] = np.array([x_min, y_min, x_max, y_max])
"""

print(np.array(ct_data).shape, np.array(coordinate_data).shape)
# save the dataset as python list type
# is converting to numpy type better?
# but since lesion bbox will have variable length (2 or more lesions), maybe inefficient?
print('dumping dataset...')
with h5py.File(dataset_save_location, 'w') as dump_h5:
    group_ct = dump_h5.create_group('ct')
    group_coordinate = dump_h5.create_group('coordinate')
    for idx in range(len(ct_data)):
        group_ct.create_dataset('ct_' + str(idx), data=ct_data[idx])
        group_coordinate.create_dataset('coordinate_' + str(idx), data=coordinate_data[idx])
    dump_h5.close()