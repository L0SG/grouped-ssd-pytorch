import numpy as np
import glob
import os
from PIL import Image
import dicom
import pickle
import h5py

def preprocess_img_slc_for_detection(img_slc):
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

    ct_data = []
    coordinate_data = []

    # traverse over subjects
    for subject in glob.glob(os.path.join(ct_path, '*')):
        basename_subject = os.path.basename(os.path.normpath(subject))
        path_subject = os.path.join(roi_coordinate_path, basename_subject)
        # traverse over phases
        for phase in glob.glob(os.path.join(subject, '*')):
            basename_phase = os.path.basename(os.path.normpath(phase))
            path_phase = os.path.join(path_subject, basename_phase)

            # use P phase only
            if basename_phase != 'P':
                continue

            # get name of all slices
            path_slices = sorted(os.listdir(phase))
            path_slices_basename = [name[:-4] for name in path_slices]
            # get name of coordinate files
            path_coordinates = sorted(os.listdir(os.path.join(roi_coordinate_path,
                                                              basename_subject,
                                                              basename_phase)))
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
                ct_image_preprocessed = preprocess_img_slc_for_detection(ct_image)
                coordinate_file = open(os.path.join(roi_coordinate_path, basename_subject, basename_phase,
                                                        path_coordinates_filtered[idx]), 'rb')
                coordinate = pickle.load(coordinate_file)
                ct_data_oneslice.append(ct_image_preprocessed)
                coordinate_oneslice.append(coordinate)
        assert len(ct_data_oneslice) == len(coordinate_oneslice)
        # make 3 continuous slides as a single data point to enable the CNN to recognize vertical info
        # consider 3 slides to additional channel (like RGB)
        for idx_slice in range(len(ct_data_oneslice) - 2):
            ct_data_threeslices = np.array(ct_data_oneslice[idx_slice:idx_slice+3])
            coordinate_threeslices = np.array(coordinate_oneslice[idx_slice:idx_slice+3])
            # append to master list
            ct_data.append(ct_data_threeslices)
            coordinate_data.append(coordinate_threeslices)

    assert len(ct_data) == len(coordinate_data)
    return ct_data, coordinate_data

# generate roi dataset
# each ct image slice have 1 or more (n) bounding boxes [n, y_start, x_start, y_delta, x_delta]
# if using P phase only, input ct should be [n, 1, 512, 512] (1 is greyscale)

ct_path = '/media/hdd/tkdrlf9202/Datasets/liver_lesion/ct'
roi_coordinate_path = '/media/hdd/tkdrlf9202/Datasets/liver_lesion/roi_coordinate'
dataset_save_location = '/home/tkdrlf9202/Datasets/liver_lesion/lesion_dataset_Ponly_1332.h5'

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

# the above is WRONG: it uses [x_min, y_min, x_max, y_max]
# convert [y_start, x_start, y_delta, x_delta] to [x_min, y_min, x_max, y_max]
for idx in range(len(coordinate_data)):
    for idx_slice in range(coordinate_data[idx].shape[0]):
        y_start, x_start, y_delta, x_delta = coordinate_data[idx][idx_slice]
        x_min, y_min = x_start, y_start
        x_max, y_max = x_start + x_delta, y_start + y_delta
        coordinate_data[idx][idx_slice] = np.array([x_min, y_min, x_max, y_max])

print(np.array(ct_data).shape, np.array(coordinate_data).shape)
# save the dataset as python list type
# is converting to numpy type better?
# but since lesion bbox will have variable length (2 or more lesions), maybe inefficient?
print('dumping dataset...')
with h5py.File(dataset_save_location, 'w') as dump_h5:
    dump_h5.create_dataset('ct', data=ct_data)
    dump_h5.create_dataset('coordinate', data=coordinate_data)
    dump_h5.close()