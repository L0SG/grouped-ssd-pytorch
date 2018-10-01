import pydicom as dicom
import os
import numpy as np
import glob
import natsort
import scipy
import scipy.misc
import unittest
from PIL import Image
# base code borrowed from CascadedFCN: https://github.com/IBBM/Cascaded-FCN

# global flag of data structure for ct image and seg mask
IMG_DTYPE = np.float32
SEG_DTYPE = np.uint8


def read_dicom_series(directory, filepattern="P_*"):
    """ Reads a DICOM Series files in the given directory.
    Only filesnames matching filepattern will be considered"""

    if not os.path.exists(directory) or not os.path.isdir(directory):
        raise ValueError("Given directory does not exist or is a file : " + str(directory))
    # print('\tRead Dicom', directory)
    lstFilesDCM = natsort.natsorted(glob.glob(os.path.join(directory, filepattern)))
    # print('\tLength dicom series', len(lstFilesDCM))
    # Get ref file
    RefDs = dicom.read_file(lstFilesDCM[0])
    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
    # The array is sized based on 'ConstPixelDims'
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

    # loop through all the DICOM files
    for filenameDCM in lstFilesDCM:
        # read the file
        ds = dicom.read_file(filenameDCM)
        # store the raw image data
        ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array

    return ArrayDicom


def read_liver_seg_masks_raw(masks_dirname, img_shape):
    """
    read 3d liver segmentation raw file
    0's for background pixels, 1's for liver pixels
    :param masks_dirname:
    :return:
    """
    label_volume = None
    rawfile = np.fromfile(masks_dirname, dtype='uint8', sep="")

    # raw file assumes height as first dimension
    # permute img_shape and reshape the raw image
    shape_raw = np.array(img_shape)
    order = [2, 0, 1]
    shape_raw = shape_raw[order]
    num_slice = rawfile.shape[0] / shape_raw[1] / shape_raw[2]
    print(os.path.basename(masks_dirname) + ' slices raw vs dicom: ' + str(num_slice) + ' '+ str(shape_raw[0]))

    label_volume = rawfile.reshape(shape_raw)
    label_volume = label_volume.transpose([1, 2, 0])

    # raw seg image is upside-down: do vertical flip
    label_volume = np.flipud(label_volume)

    return label_volume


def load_liver_seg_dataset(data_path, num_data_to_load):
    """
    load the liver dataset
    :param data_path:
    :return: list of 3D CT data (variable height) & list of 3D binary segmentation data
    """

    # get list of subfolder names
    dir_list = os.listdir(data_path)
    dir_list = sorted(dir_list)

    # empty list for appending data for each subject
    list_ct = []
    list_mask = []
    count_for_early_stop = 0

    # traverse through subfolders
    # each subfolder contains one raw seg file, and another subfolder 'P' containing slices of dicom files
    for dir_name in dir_list:
        # obtain absolute path of the subject
        path_subject = os.path.join(data_path, dir_name)
        # load dicom image of the subject
        dicom_image = read_dicom_series(os.path.join(path_subject, 'P'), "P_*")
        # load mask image of the subject, which needs shape information
        dicom_shape = dicom_image.shape
        mask_path = os.path.join(path_subject, str(dir_name)+'.raw')
        mask_image = read_liver_seg_masks_raw(mask_path, img_shape=dicom_shape)
        # print unique elements of mask tensor
        # for binary segmentation, it should only contain 0 or 1
        # if the tensor contains other number, data structure is not right
        mask_unique_elems = np.unique(mask_image)
        ground_truth_reference = [0, 1] # to assert binary data
        print('unique elements of mask image: ' + str(mask_unique_elems))
        if sorted(mask_unique_elems) != sorted(ground_truth_reference):
            print('incorrect label detected, forcing incorrect labels to zero...')
            # currently hard-coding (10 and 20)
            # TODO: make this generic
            mask_image[mask_image == 10] = 0
            mask_image[mask_image == 20] = 0
        # check label sanity again and assert
        mask_unique_elems = np.unique(mask_image)
        assert len(mask_unique_elems) == len(ground_truth_reference)
        assert sorted(mask_unique_elems) == sorted(ground_truth_reference)
        # append to list
        list_ct.append(dicom_image)
        list_mask.append(mask_image)

        """
        # DEBUG ONLY: save mask images for given directory
        # for checking if the label data has any artifacts
        # save mask data as image
        mask_image_save_path = '/home/vision/tkdrlf9202/Datasets/liver_mask_image'
        if not os.path.exists(os.path.join(mask_image_save_path, str(dir_name))):
            os.makedirs(os.path.join(mask_image_save_path, str(dir_name)))
        # traverse through slices
        for idx in range(mask_image.shape[2]):
            mask_slice_save_path = os.path.join(mask_image_save_path, str(dir_name), str(idx)+'.jpg')
            mask_slice = np.copy(mask_image[:,:,idx])
            # if 0, stay 0; if 1, convert to 255, if other, convert to 128(grey)
            mask_slice[mask_slice == 1] = 255
            mask_slice[mask_slice == 10] = 100
            mask_slice[mask_slice == 20] = 180
            scipy.misc.imsave(mask_slice_save_path, mask_slice)
        """
        # partial load defined in num_data_to_load
        count_for_early_stop += 1
        if num_data_to_load is None:
            continue
        if count_for_early_stop == num_data_to_load:
            break

    return list_ct, list_mask


def preprocess_liver_dataset(list_ct, list_mask):

    list_ct_preprocessed = []
    list_mask_preprocessed = []

    for idx_subject in range(len(list_ct)):
        img = list_ct[idx_subject]
        lbl = list_mask[idx_subject]
        # since our data has range of 0~2048 (instead of -1024~1024) subtract 1024
        img = np.add(img, -1024)

        # preproess the image slice
        img_p = np.zeros((572, 572, img.shape[2]), dtype=IMG_DTYPE)
        lbl_p = np.zeros((388, 388, lbl.shape[2]), dtype=SEG_DTYPE)
        for idx in xrange(img.shape[2]):
            img_p[..., idx] = step1_preprocess_img_slice(img[..., idx])
            lbl_p[..., idx] = preprocess_lbl_slice(lbl[..., idx])

        # append to list
        list_ct_preprocessed.append(img_p)
        list_mask_preprocessed.append(lbl_p)

    return list_ct_preprocessed, list_mask_preprocessed


def step1_preprocess_img_slice(img_slc):
    """
    Preprocesses the image 3d volumes by performing the following :
    1- Rotate the input volume so the the liver is on the left, spine is at the bottom of the image
    2- Set pixels with hounsfield value great than 1200, to zero.
    3- Clip all hounsfield values to the range [-100, 400]
    4- Normalize values to [0, 1]
    5- Rescale img and label slices to 388x388
    6- Pad img slices with 92 pixels on all sides (so total shape is 572x572)

    Args:
        img_slc: raw image slice
    Return:
        Preprocessed image slice
    """
    img_slc = img_slc.astype(IMG_DTYPE)
    img_slc[img_slc > 1200] = 0
    img_slc = np.clip(img_slc, -100, 400)
    img_slc = normalize_image(img_slc)
    img_slc = to_scale(img_slc, (388, 388))
    img_slc = np.pad(img_slc, ((92, 92), (92, 92)), mode='reflect')

    return img_slc


def preprocess_lbl_slice(lbl_slc):
    """ Preprocess ground truth slice to match output prediction of the network in terms
    of size and orientation.

    Args:
        lbl_slc: raw label/ground-truth slice
    Return:
        Preprocessed label slice"""
    lbl_slc = lbl_slc.astype(SEG_DTYPE)
    # downscale the label slc for comparison with the prediction
    lbl_slc = to_scale(lbl_slc, (388, 388))
    return lbl_slc


def normalize_image(img):
    """ Normalize image values to [0,1] """
    min_, max_ = float(np.min(img)), float(np.max(img))
    return (img - min_) / (max_ - min_)


def to_scale(img, shape=None):
    height, width = shape
    if img.dtype == SEG_DTYPE:
        return scipy.misc.imresize(img, (height, width), interp="nearest").astype(SEG_DTYPE)
    elif img.dtype == IMG_DTYPE:
        max_ = np.max(img)
        factor = 255.0 / max_ if max_ != 0 else 1
        return (scipy.misc.imresize(img, (height, width), interp="nearest") / factor).astype(IMG_DTYPE)
    else:
        raise TypeError(
            'Error. To scale the image array, its type must be np.uint8 or np.float64. (' + str(img.dtype) + ')')
